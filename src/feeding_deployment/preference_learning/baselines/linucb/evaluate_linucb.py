#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LinUCB baseline requires numpy. Install it with: pip install numpy"
    ) from e

# Ensure repo root (src/) is on sys.path so `feeding_deployment` imports work.
# This file lives at: src/feeding_deployment/preference_learning/baselines/linucb/evaluate_linucb.py
# so parents[4] is the `src/` directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feeding_deployment.preference_learning import config as root_config  # type: ignore
from feeding_deployment.preference_learning.baselines.linucb.configs import DEFAULT_CONFIG, LinUCBConfig
from feeding_deployment.preference_learning.baselines.linucb.context_encoder import ContextEncoder
from feeding_deployment.preference_learning.baselines.linucb.linucb import LinUCBPreferencePredictor


PREF_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]
PREF_OPTIONS: Dict[str, List[str]] = {name: opts for (name, _, opts) in root_config.PREFERENCE_BUNDLE}

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> None:
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _extract_truth_bundle(day_or_meal_rec: Dict[str, Any]) -> Dict[str, str]:
    prefs = day_or_meal_rec.get("preferences", {}) or {}
    out: Dict[str, str] = {}
    for field in PREF_FIELDS:
        val = prefs.get(field, {})
        if isinstance(val, dict):
            out[field] = str(val.get("choice", "")).strip()
        else:
            out[field] = str(val).strip()
    return out


def _iter_meals(days: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return a list of (day_idx, meal_record_like) where each meal has 'context' and 'preferences'.

    Supports either:
    - days[] items each representing a single meal (current default in memory evaluator)
    - days[] items containing a 'meals' list
    """
    out: List[Tuple[int, Dict[str, Any]]] = []
    for day_rec in days:
        day = int(day_rec.get("day", 0))
        if isinstance(day_rec.get("meals"), list):
            for meal_rec in day_rec["meals"]:
                if isinstance(meal_rec, dict):
                    out.append((day, meal_rec))
        else:
            out.append((day, day_rec))
    return out


def _simulate_interactive_metrics(
    predictor: LinUCBPreferencePredictor,
    x: np.ndarray,
    truth: Dict[str, str],
    rng: random.Random,
    *,
    max_corrections: int = 19,
) -> Tuple[int, int, int]:
    """
    Simulate an interactive correction loop (for metrics only) by applying
    per-dimension semi-bandit updates after each revealed correction, and
    re-predicting the remaining (unrevealed) dimensions.

    Returns:
      (mismatches_m0, mismatches_m1, m_star)
    where:
      mismatches_m0: # wrong dims at m=0
      mismatches_m1: # wrong dims after 1 correction step (if any)
      m_star: corrections needed to stop under this simulated interaction
    """
    sim = copy.deepcopy(predictor)
    corrected_fields: set[str] = set()

    def _mismatches(curr_pred: Dict[str, str]) -> List[str]:
        return [
            f
            for f in PREF_FIELDS
            if f not in corrected_fields and curr_pred.get(f) != truth.get(f)
        ]

    pred0 = sim.predict_bundle(x)
    mm0 = _mismatches(pred0)
    mismatches_m0 = len(mm0)
    mismatches_m1 = mismatches_m0
    m = 0

    while m < max_corrections:
        curr_pred = sim.predict_bundle(x)
        mm = _mismatches(curr_pred)
        if not mm:
            break

        # Reveal one incorrect dimension and update only that dimension (semi-bandit).
        f_corr = rng.choice(mm)
        pred_arm = curr_pred.get(f_corr, "")
        truth_arm = truth.get(f_corr, "")
        if pred_arm in PREF_OPTIONS.get(f_corr, []):
            sim.update(f_corr, pred_arm, x, 0.0)
        if truth_arm in PREF_OPTIONS.get(f_corr, []):
            sim.update(f_corr, truth_arm, x, 1.0)

        corrected_fields.add(f_corr)
        m += 1

        if m == 1:
            pred1 = sim.predict_bundle(x)
            mismatches_m1 = len(_mismatches(pred1))

    return mismatches_m0, mismatches_m1, m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LinUCB baseline on synthetic datasets.")
    p.add_argument("--data-file", help="Path to one JSON dataset file.")
    p.add_argument("--data-dir", help="Directory containing JSON dataset files.")
    p.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha, help="LinUCB exploration alpha.")
    p.add_argument("--max-meals", type=int, default=0, help="Max meals per user to evaluate (0=all).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (correction ordering + shuffles).")
    p.add_argument("--include-affective-state", action="store_true", default=DEFAULT_CONFIG.include_affective_state)
    p.add_argument("--no-include-affective-state", action="store_false", dest="include_affective_state")
    p.add_argument("--include-profile-label", action="store_true", default=DEFAULT_CONFIG.include_physical_profile_label)
    p.add_argument("--no-include-profile-label", action="store_false", dest="include_profile_label")
    p.add_argument(
        "--ours-summary-metrics",
        default=str(Path(__file__).resolve().parents[3] / "methods" / "comparison_metrics" / "summary_metrics.json"),
        help="Path to methods/comparison_metrics/summary_metrics.json for comparison plots.",
    )
    return p.parse_args()


def _mean_and_sem(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(max(var, 0.0))
    sem = std / math.sqrt(n)
    return float(mean), float(sem)


def _compute_summary_metrics(users: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    all_m_star: List[float] = []
    first5_m_star: List[float] = []
    last5_m_star: List[float] = []

    all_acc_m0: List[float] = []
    first5_acc_m0: List[float] = []
    last5_acc_m0: List[float] = []

    for user_report in users:
        per_day_metrics = user_report.get("per_day_metrics", [])
        if not isinstance(per_day_metrics, list):
            continue

        sorted_metrics = sorted(
            [rec for rec in per_day_metrics if "day" in rec],
            key=lambda rec: int(rec["day"]),
        )
        first5 = sorted_metrics[:5]
        last5 = sorted_metrics[-5:] if len(sorted_metrics) >= 5 else sorted_metrics

        for rec in sorted_metrics:
            if "m_star" in rec:
                all_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                all_acc_m0.append(float(rec["acc_m0"]))

        for rec in first5:
            if "m_star" in rec:
                first5_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                first5_acc_m0.append(float(rec["acc_m0"]))

        for rec in last5:
            if "m_star" in rec:
                last5_m_star.append(float(rec["m_star"]))
            if "acc_m0" in rec:
                last5_acc_m0.append(float(rec["acc_m0"]))

    mean_m_star_all_days, sem_m_star_all_days = _mean_and_sem(all_m_star)
    mean_m_star_first_5_days, sem_m_star_first_5_days = _mean_and_sem(first5_m_star)
    mean_m_star_last_5_days, sem_m_star_last_5_days = _mean_and_sem(last5_m_star)

    mean_acc_m0_all_days, sem_acc_m0_all_days = _mean_and_sem(all_acc_m0)
    mean_acc_m0_first_5_days, sem_acc_m0_first_5_days = _mean_and_sem(first5_acc_m0)
    mean_acc_m0_last_5_days, sem_acc_m0_last_5_days = _mean_and_sem(last5_acc_m0)

    return {
        "mean_m_star_all_days": mean_m_star_all_days,
        "sem_m_star_all_days": sem_m_star_all_days,
        "mean_m_star_first_5_days": mean_m_star_first_5_days,
        "sem_m_star_first_5_days": sem_m_star_first_5_days,
        "mean_m_star_last_5_days": mean_m_star_last_5_days,
        "sem_m_star_last_5_days": sem_m_star_last_5_days,
        "mean_acc_m0_all_days": mean_acc_m0_all_days,
        "sem_acc_m0_all_days": sem_acc_m0_all_days,
        "mean_acc_m0_first_5_days": mean_acc_m0_first_5_days,
        "sem_acc_m0_first_5_days": sem_acc_m0_first_5_days,
        "mean_acc_m0_last_5_days": mean_acc_m0_last_5_days,
        "sem_acc_m0_last_5_days": sem_acc_m0_last_5_days,
    }


def _plot_summary_comparison(
    output_dir: Path,
    ours: Dict[str, Any],
    linucb: Dict[str, Any],
) -> None:
    if not HAS_MATPLOTLIB:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    acc_labels = ["Acc(m=0) all", "Acc(m=0) first5", "Acc(m=0) last5"]
    ours_acc = [
        ours.get("mean_acc_m0_all_days"),
        ours.get("mean_acc_m0_first_5_days"),
        ours.get("mean_acc_m0_last_5_days"),
    ]
    lin_acc = [
        linucb.get("mean_acc_m0_all_days"),
        linucb.get("mean_acc_m0_first_5_days"),
        linucb.get("mean_acc_m0_last_5_days"),
    ]

    def _barplot(labels: List[str], ours_vals: List[float], lin_vals: List[float], title: str, ylabel: str, out_path: Path) -> None:
        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - w / 2, ours_vals, w, label="Ours", color="#4C78A8")
        ax.bar(x + w / 2, lin_vals, w, label="LinUCB", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    _barplot(
        labels=acc_labels,
        ours_vals=[float(x) for x in ours_acc],
        lin_vals=[float(x) for x in lin_acc],
        title="Summary Accuracy: Ours vs LinUCB",
        ylabel="Acc(m=0)",
        out_path=output_dir / "comparison_summary_acc_m0.png",
    )

    m_labels = ["m* all", "m* first5", "m* last5"]
    ours_m = [
        ours.get("mean_m_star_all_days"),
        ours.get("mean_m_star_first_5_days"),
        ours.get("mean_m_star_last_5_days"),
    ]
    lin_m = [
        linucb.get("mean_m_star_all_days"),
        linucb.get("mean_m_star_first_5_days"),
        linucb.get("mean_m_star_last_5_days"),
    ]

    _barplot(
        labels=m_labels,
        ours_vals=[float(x) for x in ours_m],
        lin_vals=[float(x) for x in lin_m],
        title="Summary Correction Burden: Ours vs LinUCB",
        ylabel="m* (corrections to stop)",
        out_path=output_dir / "comparison_summary_m_star.png",
    )


def _compute_mean_sem_by_day(
    users: List[Dict[str, Any]],
    value_fn,
) -> Tuple[List[int], List[float], List[float]]:
    sum_by_day: Dict[int, float] = {}
    sq_sum_by_day: Dict[int, float] = {}
    n_by_day: Dict[int, int] = {}

    for user_report in users:
        per_day_metrics = user_report.get("per_day_metrics", [])
        if not isinstance(per_day_metrics, list):
            continue

        for rec in per_day_metrics:
            day, value = value_fn(rec)
            if day is None or value is None:
                continue
            sum_by_day[day] = sum_by_day.get(day, 0.0) + float(value)
            sq_sum_by_day[day] = sq_sum_by_day.get(day, 0.0) + float(value) * float(value)
            n_by_day[day] = n_by_day.get(day, 0) + 1

    xs = sorted(sum_by_day.keys())
    means: List[float] = []
    sems: List[float] = []

    for day in xs:
        n = n_by_day[day]
        mean = sum_by_day[day] / n
        var = (sq_sum_by_day[day] / n) - (mean * mean)
        std = math.sqrt(max(var, 0.0))
        sem = std / math.sqrt(n) if n > 0 else 0.0
        means.append(mean)
        sems.append(sem)

    return xs, means, sems


def _plot_with_band(
    method_to_users: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    value_fn,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(9, 5))
    for method_name, users in method_to_users.items():
        xs, means, sems = _compute_mean_sem_by_day(users, value_fn)
        if not xs:
            continue

        lower = [m - e for m, e in zip(means, sems)]
        upper = [m + e for m, e in zip(means, sems)]

        line = plt.plot(xs, means, marker="o", label=method_name)
        color = line[0].get_color()
        plt.fill_between(xs, lower, upper, alpha=0.25, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_linucb_comparison_plots(
    report_dir: Path,
    user_reports: List[Dict[str, Any]],
) -> None:
    """Write the 4 comparison-style plots for LinUCB (no external methods needed)."""
    if not HAS_MATPLOTLIB:
        return

    out_dir = report_dir / "comparison_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_to_users = {"LinUCB": user_reports}

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=out_dir / "comparison_acc_m0_by_day.png",
        title="Initial Prediction Accuracy Over Time (LinUCB)",
        xlabel="Day",
        ylabel="Initial Accuracy (m=0)",
        value_fn=lambda rec: (int(rec["day"]), float(rec["acc_m0"]))
        if "day" in rec and "acc_m0" in rec
        else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=out_dir / "comparison_m_star_by_day.png",
        title="Per-Day Mean Corrections to Stop (LinUCB)",
        xlabel="Day",
        ylabel="Mean Corrections to Stop (m*)",
        value_fn=lambda rec: (int(rec["day"]), float(rec["m_star"]))
        if "day" in rec and "m_star" in rec
        else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=out_dir / "comparison_interactive_benefit_by_day.png",
        title="Benefits of Interactive Prediction (LinUCB simulation)",
        xlabel="Day",
        ylabel="Corrections Avoided (mismatches_m0 - m*)",
        value_fn=lambda rec: (int(rec["day"]), float(rec["mismatches_m0"]) - float(rec["m_star"]))
        if "day" in rec and "mismatches_m0" in rec and "m_star" in rec
        else (None, None),
    )

    _plot_with_band(
        method_to_users=method_to_users,
        output_path=out_dir / "comparison_single_correction_gain_by_day.png",
        title="Information Gain with Single Correction (LinUCB simulation)",
        xlabel="Day",
        ylabel="|mismatches_m0 - mismatches_m1|",
        value_fn=lambda rec: (int(rec["day"]), abs(float(rec["mismatches_m0"]) - float(rec["mismatches_m1"])))
        if "day" in rec and "mismatches_m0" in rec and "mismatches_m1" in rec
        else (None, None),
    )


def main() -> int:
    args = parse_args()
    cfg = LinUCBConfig(
        alpha=float(args.alpha),
        include_affective_state=bool(args.include_affective_state),
        include_physical_profile_label=bool(args.include_profile_label),
        add_bias=True,
    )

    files: List[str] = []
    if args.data_file:
        files.append(args.data_file)
    if args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(args.data_dir, "*.json"))))
    if not files:
        print("No input files provided. Use --data-file or --data-dir.")
        return 1

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Always write LinUCB outputs inside the LinUCB baseline folder so results
    # are easy to find regardless of current working directory.
    linucb_root = Path(__file__).resolve().parent
    report_dir = linucb_root / "reports" / f"linucb_eval_{run_ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_txt_path = report_dir / "report.txt"
    report_txt_file = open(report_txt_path, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report_txt_file)

    logs_dir = report_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pred_log = open(logs_dir / "predictions.jsonl", "a", encoding="utf-8")
    upd_log = open(logs_dir / "updates.jsonl", "a", encoding="utf-8")

    try:
        user_reports: List[Dict[str, Any]] = []

        for path in files:
            print(f"Evaluating {path} ...", flush=True)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            user = str(data.get("user", "unknown"))
            physical_profile_label = str(data.get("physical_profile_label", "")).strip()
            days: List[Dict[str, Any]] = list(data.get("days", []))
            days.sort(key=lambda r: int(r.get("day", 0)))

            meals = _iter_meals(days)
            if args.max_meals and args.max_meals > 0:
                meals = meals[: int(args.max_meals)]

            contexts = [mrec.get("context", {}) or {} for (_d, mrec) in meals]
            profile_labels = [physical_profile_label] * len(contexts)

            # Seed vocabularies from config when available; extend from dataset values.
            encoder = ContextEncoder.from_records(
                contexts=contexts,
                physical_profile_labels=profile_labels,
                include_affective_state=cfg.include_affective_state,
                include_physical_profile_label=cfg.include_physical_profile_label,
                add_bias=cfg.add_bias,
                default_meals=list(getattr(root_config, "MEALS", []) or []),
                default_settings=list(getattr(root_config, "SETTINGS", []) or []),
                default_times=list(getattr(root_config, "TIMES_OF_DAY", []) or []),
                default_affects=list(getattr(root_config, "AFFECTIVE_STATES", []) or []),
            )

            predictor = LinUCBPreferencePredictor(
                dimension_options=PREF_OPTIONS,
                context_dim=encoder.dim,
                alpha=cfg.alpha,
            )

            rng = random.Random(args.seed)

            # Metrics accumulators
            acc_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m0_n_by_day: Dict[int, int] = defaultdict(int)
            m_star_sum_by_day: Dict[int, float] = defaultdict(float)
            m_star_n_by_day: Dict[int, int] = defaultdict(int)
            mismatches_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m0_n_by_day: Dict[int, int] = defaultdict(int)
            mismatches_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m1_n_by_day: Dict[int, int] = defaultdict(int)
            per_dim_correct: Dict[str, int] = defaultdict(int)
            per_dim_total: Dict[str, int] = defaultdict(int)

            for (day, meal_rec) in meals:
                ctx = meal_rec.get("context", {}) or {}
                truth = _extract_truth_bundle(meal_rec)
                x = encoder.encode(ctx, physical_profile_label=physical_profile_label)

                pred = predictor.predict_bundle(x)
                mismatches = [f for f in PREF_FIELDS if pred.get(f) != truth.get(f)]
                acc_m0 = 1.0 - (len(mismatches) / float(len(PREF_FIELDS)))

                # Simulated interactive metrics (for plot comparability)
                mismatches_m0, mismatches_m1, m_star = _simulate_interactive_metrics(
                    predictor=predictor,
                    x=x,
                    truth=truth,
                    rng=rng,
                    max_corrections=len(PREF_FIELDS),
                )

                acc_m0_sum_by_day[day] += acc_m0
                acc_m0_n_by_day[day] += 1
                m_star_sum_by_day[day] += m_star
                m_star_n_by_day[day] += 1
                mismatches_m0_sum_by_day[day] += mismatches_m0
                mismatches_m0_n_by_day[day] += 1
                mismatches_m1_sum_by_day[day] += mismatches_m1
                mismatches_m1_n_by_day[day] += 1

                for f in PREF_FIELDS:
                    per_dim_total[f] += 1
                    if pred.get(f) == truth.get(f):
                        per_dim_correct[f] += 1

                # Log prediction step
                pred_log.write(
                    json.dumps(
                        {
                            "run_timestamp": run_ts,
                            "dataset_file": path,
                            "user": user,
                            "day": day,
                            "context": ctx,
                            "physical_profile_label": physical_profile_label,
                            "x_dim": int(encoder.dim),
                            "pred": pred,
                            "truth": truth,
                            "mismatches": mismatches,
                            "mismatches_m0": mismatches_m0,
                            "mismatches_m1": mismatches_m1,
                            "m_star": m_star,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Update after the meal using semi-bandit feedback
                updates = predictor.update_from_truth(x=x, pred_bundle=pred, truth_bundle=truth)
                upd_log.write(
                    json.dumps(
                        {
                            "run_timestamp": run_ts,
                            "dataset_file": path,
                            "user": user,
                            "day": day,
                            "updates": [{"field": f, "arm": a, "reward": r} for (f, a, r) in updates],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            all_days = sorted(set(acc_m0_sum_by_day.keys()) | set(m_star_sum_by_day.keys()))
            per_day_metrics: List[Dict[str, Any]] = []
            for d in all_days:
                rec: Dict[str, Any] = {"day": d}
                if acc_m0_n_by_day[d]:
                    rec["acc_m0"] = acc_m0_sum_by_day[d] / float(acc_m0_n_by_day[d])
                if m_star_n_by_day[d]:
                    rec["m_star"] = m_star_sum_by_day[d] / float(m_star_n_by_day[d])
                if mismatches_m0_n_by_day[d]:
                    rec["mismatches_m0"] = mismatches_m0_sum_by_day[d] / float(mismatches_m0_n_by_day[d])
                if mismatches_m1_n_by_day[d]:
                    rec["mismatches_m1"] = mismatches_m1_sum_by_day[d] / float(mismatches_m1_n_by_day[d])
                per_day_metrics.append(rec)

            per_dimension_m0_accuracy = {
                f: (per_dim_correct[f] / float(per_dim_total[f]) if per_dim_total[f] else 0.0)
                for f in PREF_FIELDS
            }

            user_reports.append(
                {
                    "file": path,
                    "user": user,
                    "meals": len(meals),
                    "config": {
                        "alpha": cfg.alpha,
                        "include_affective_state": cfg.include_affective_state,
                        "include_physical_profile_label": cfg.include_physical_profile_label,
                        "x_dim": int(encoder.dim),
                    },
                    "per_day_metrics": per_day_metrics,
                    "per_dimension_m0_accuracy": per_dimension_m0_accuracy,
                }
            )

        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "users": user_reports,
                    "run_timestamp": run_ts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Write the four standard comparison-style plots for LinUCB.
        _write_linucb_comparison_plots(report_dir, user_reports)

        # Write summary metrics comparable to methods/comparison_metrics/summary_metrics.json
        linucb_summary = _compute_summary_metrics(user_reports)
        summary_path = report_dir / "summary_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"LinUCB": linucb_summary}, f, ensure_ascii=False, indent=2)

        # Optional comparison against Ours summary metrics + plot
        comparison_path = report_dir / "comparison_with_ours.json"
        ours_summary: Optional[Dict[str, Any]] = None
        try:
            with open(args.ours_summary_metrics, "r", encoding="utf-8") as f:
                ours_all = json.load(f)
            if isinstance(ours_all, dict) and isinstance(ours_all.get("Ours"), dict):
                ours_summary = ours_all["Ours"]
        except Exception:
            ours_summary = None

        if ours_summary is not None:
            with open(comparison_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"Ours": ours_summary, "LinUCB": linucb_summary},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            _plot_summary_comparison(
                output_dir=report_dir,
                ours=ours_summary,
                linucb=linucb_summary,
            )

        print(f"\nWrote report: {report_path}")
        print(f"Wrote summary metrics: {summary_path}")
        if ours_summary is not None:
            print(f"Wrote comparison metrics: {comparison_path}")
            if HAS_MATPLOTLIB:
                print(f"Wrote comparison plots: {report_dir / 'comparison_summary_acc_m0.png'} and {report_dir / 'comparison_summary_m_star.png'}")
        print(f"Terminal output saved to: {report_txt_path}")
        print(f"Logs written under: {logs_dir}")
        return 0
    finally:
        sys.stdout = real_stdout
        report_txt_file.close()
        pred_log.close()
        upd_log.close()


if __name__ == "__main__":
    raise SystemExit(main())

