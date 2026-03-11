from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from feeding_deployment.preference_learning.methods.metrics import _generate_metrics
from feeding_deployment.preference_learning.methods.utils import (
    _extract_truth_bundle,
    _resolve_api_key,
    _retry_on_rate_limit,
)

from feeding_deployment.preference_learning.methods.prediction_model import PredictionModel
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS


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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LLM-based interactive predictor on synthetic datasets.")
    p.add_argument("--data-file", help="Path to one JSON dataset file.")
    p.add_argument("--data-dir", help="Directory containing JSON dataset files.")
    p.add_argument("--k-retrieve", type=int, default=10)
    p.add_argument("--max-corrections", type=int, default=18)
    p.add_argument("--max-meals", type=int, default=0)
    p.add_argument("--num-rollouts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--openai-model", default="gpt-5.4")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--api-key", default="")
    p.add_argument("--ablation", choices=["full", "ltm_only", "em_only", "no_memory"], default="full")
    p.add_argument("--days", type=int, default=0, help="Evaluate only the first N days (0 = use all days in dataset).")

    return p.parse_args()


def _load_files(args: argparse.Namespace) -> List[str]:
    files: List[str] = []
    if args.data_file:
        files.append(args.data_file)
    if args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(args.data_dir, "*.json"))))
    return files


def main() -> int:
    args = parse_args()

    files = _load_files(args)
    if not files:
        print("No input files provided. Use --data-file or --data-dir.")
        return 1

    run_ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    report_dir = Path(__file__).parent / "reports" / f"run_{run_ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_txt_path = report_dir / "report.txt"
    report_txt_file = open(report_txt_path, "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report_txt_file)

    logs_dir = report_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    use_long_term_memory = args.ablation not in ("em_only", "no_memory")
    use_episodic_memory = args.ablation not in ("ltm_only", "no_memory")

    try:
        client = OpenAI(api_key=_resolve_api_key(args.api_key))

        user_reports: List[Dict[str, Any]] = []

        for path in files:
            print(f"Evaluating {path} ...", flush=True)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            user = str(data.get("user", "unknown"))
            physical_profile_label = str(data.get("physical_profile_label", "")).strip()
            if not physical_profile_label:
                raise SystemExit(f"Dataset missing required field: 'physical_profile_label' in {path}")

            days: List[Dict[str, Any]] = list(data.get("days", []))
            days.sort(key=lambda r: int(r.get("day", 0)))
            if args.days > 0:
                days = [d for d in days if int(d.get("day", 0)) <= args.days]

            # metrics
            total_meals = 0
            total_corrections = 0
            acc_after_m_sum: Dict[int, float] = {}
            acc_after_m_n: Dict[int, int] = {}

            acc_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m0_n_by_day: Dict[int, int] = defaultdict(int)
            acc_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m1_n_by_day: Dict[int, int] = defaultdict(int)

            mismatches_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m0_n_by_day: Dict[int, int] = defaultdict(int)
            mismatches_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m1_n_by_day: Dict[int, int] = defaultdict(int)

            m_star_sum_by_day: Dict[int, float] = defaultdict(float)
            m_star_n_by_day: Dict[int, int] = defaultdict(int)
            affective_state_by_day: Dict[int, str] = {}

            acc_m0_sum_by_state: Dict[str, float] = defaultdict(float)
            acc_m0_n_by_state: Dict[str, int] = defaultdict(int)
            m_star_sum_by_state: Dict[str, float] = defaultdict(float)
            m_star_n_by_state: Dict[str, int] = defaultdict(int)

            per_dim_correct: Dict[str, int] = defaultdict(int)
            per_dim_total: Dict[str, int] = defaultdict(int)

            zero_correction_meals_total = 0
            zero_correction_meals_final_week = 0

            for rollout_idx in range(args.num_rollouts):
                rng = random.Random(args.seed + rollout_idx)
                
                prediction_model = PredictionModel(
                    user=user,
                    physical_profile_label=physical_profile_label,
                    client=client,
                    chat_model=args.openai_model,
                    embed_model=args.embed_model,
                    retry_fn=_retry_on_rate_limit,
                    logs_dir=logs_dir,
                    use_long_term_memory=use_long_term_memory,
                    use_episodic_memory=use_episodic_memory,
                    k_retrieve=args.k_retrieve,
                )

                meals_this_rollout = 0

                for day_rec in days:
                    day = int(day_rec.get("day", 0))
                    context = day_rec.get("context", {}) or {}
                    truth = _extract_truth_bundle(day_rec)

                    corrected: Dict[str, str] = {}
                    m = 0
                    affective_state = str(context.get("transient_affective_state") or "unknown").strip() or "unknown"
                    affective_state_by_day[day] = affective_state

                    print(
                        f"[Day {day}] Meal: {context.get('meal', 'unknown')} | "
                        f"Setting: {context.get('setting', 'unknown')} | "
                        f"Time: {context.get('time_of_day', 'unknown')} | "
                        f"Affective state: {affective_state}",
                        flush=True,
                    )
                    print(f"  Ground truth bundle: {json.dumps(truth, indent=2)}", flush=True)

                    while True:
                        print(f"  [Predict] Calling OpenAI for bundle (day {day}, m={m}) ...", flush=True)
                        
                        pred = prediction_model.predict_bundle(
                            context=context,
                            corrected=corrected,
                        )
                        print(f"  [Predict] Prediction: {json.dumps(pred, indent=2)}", flush=True)

                        unrevealed = [f for f in PREF_FIELDS if f not in corrected]
                        acc = (
                            (sum(1 for f in unrevealed if pred.get(f) == truth.get(f)) / float(len(unrevealed)))
                            if unrevealed
                            else 1.0
                        )

                        mismatches = [f for f in PREF_FIELDS if f not in corrected and pred.get(f) != truth.get(f)]
                        num_mismatches = len(mismatches)

                        acc_after_m_sum[m] = acc_after_m_sum.get(m, 0.0) + acc
                        acc_after_m_n[m] = acc_after_m_n.get(m, 0) + 1

                        if m == 0:
                            acc_m0_sum_by_day[day] += acc
                            acc_m0_n_by_day[day] += 1
                            mismatches_m0_sum_by_day[day] += num_mismatches
                            mismatches_m0_n_by_day[day] += 1

                            acc_m0_sum_by_state[affective_state] += acc
                            acc_m0_n_by_state[affective_state] += 1
                            for f in unrevealed:
                                per_dim_total[f] += 1
                                if pred.get(f) == truth.get(f):
                                    per_dim_correct[f] += 1
                        elif m == 1:
                            acc_m1_sum_by_day[day] += acc
                            acc_m1_n_by_day[day] += 1
                            mismatches_m1_sum_by_day[day] += num_mismatches
                            mismatches_m1_n_by_day[day] += 1

                        print(
                            f"  [user={user}] rollout {rollout_idx+1}/{args.num_rollouts} "
                            f"day {day} m={m}: acc_unrevealed={acc:.3f} "
                            f"mismatches={num_mismatches}",
                            flush=True,
                        )

                        if not mismatches or m >= args.max_corrections:
                            total_meals += 1
                            total_corrections += m
                            meals_this_rollout += 1
                            m_star_sum_by_day[day] += m
                            m_star_n_by_day[day] += 1
                            m_star_sum_by_state[affective_state] += m
                            m_star_n_by_state[affective_state] += 1

                            if m == 0 and not mismatches:
                                acc_m1_sum_by_day[day] += 1.0
                                acc_m1_n_by_day[day] += 1
                                mismatches_m1_sum_by_day[day] += 0.0
                                mismatches_m1_n_by_day[day] += 1
                                zero_correction_meals_total += 1
                                if day >= 24:
                                    zero_correction_meals_final_week += 1
                            print(
                                f"  Meal finished after {m} corrections\n",
                                flush=True,
                            )
                            break

                        f_corr = rng.choice(mismatches)
                        corrected_value = truth.get(f_corr, "")

                        print(
                            f"    correcting {f_corr} -> {corrected_value}",
                            flush=True,
                        )

                        corrected[f_corr] = corrected_value
                        m += 1

                    # Build episode text (used for long_term_memory_model update and retrieval history)
                    prediction_model.update(day, context, corrected, truth)

                    if args.max_meals and meals_this_rollout >= args.max_meals:
                        break

            mean_corr = (total_corrections / float(total_meals)) if total_meals else 0.0
            acc_after_m = {str(mm): (acc_after_m_sum[mm] / float(acc_after_m_n[mm])) for mm in sorted(acc_after_m_sum)}

            all_days = sorted(
                set(acc_m0_sum_by_day)
                | set(acc_m1_sum_by_day)
                | set(mismatches_m0_sum_by_day)
                | set(mismatches_m1_sum_by_day)
                | set(m_star_sum_by_day)
            )

            per_day_metrics: List[Dict[str, Any]] = []
            for d in all_days:
                rec: Dict[str, Any] = {"day": d, "affective_state": affective_state_by_day.get(d, "unknown")}
                if acc_m0_n_by_day[d]:
                    rec["acc_m0"] = acc_m0_sum_by_day[d] / float(acc_m0_n_by_day[d])
                if acc_m1_n_by_day[d]:
                    rec["acc_m1"] = acc_m1_sum_by_day[d] / float(acc_m1_n_by_day[d])
                if mismatches_m0_n_by_day[d]:
                    rec["mismatches_m0"] = mismatches_m0_sum_by_day[d] / float(mismatches_m0_n_by_day[d])
                if mismatches_m1_n_by_day[d]:
                    rec["mismatches_m1"] = mismatches_m1_sum_by_day[d] / float(mismatches_m1_n_by_day[d])
                if m_star_n_by_day[d]:
                    rec["m_star"] = m_star_sum_by_day[d] / float(m_star_n_by_day[d])
                per_day_metrics.append(rec)

            by_affective_state: Dict[str, Dict[str, float]] = {}
            for state in sorted(set(acc_m0_n_by_state) | set(m_star_n_by_state)):
                by_affective_state[state] = {}
                if acc_m0_n_by_state[state]:
                    by_affective_state[state]["acc_m0"] = acc_m0_sum_by_state[state] / float(acc_m0_n_by_state[state])
                if m_star_n_by_state[state]:
                    by_affective_state[state]["mean_m_star"] = m_star_sum_by_state[state] / float(m_star_n_by_state[state])

            per_dimension_m0_accuracy: Dict[str, float] = {
                f: (per_dim_correct[f] / float(per_dim_total[f]) if per_dim_total[f] else 0.0)
                for f in PREF_FIELDS
            }

            summary_statistics = {
                "zero_correction_meals_total": zero_correction_meals_total,
                "zero_correction_meals_final_week": zero_correction_meals_final_week,
            }

            user_reports.append(
                {
                    "file": path,
                    "user": user,
                    "physical_profile_label": physical_profile_label,
                    "meals": total_meals,
                    "num_rollouts": args.num_rollouts,
                    "mean_corrections_to_stop": mean_corr,
                    "accuracy_after_m": acc_after_m,
                    "per_day_metrics": per_day_metrics,
                    "by_affective_state": by_affective_state,
                    "per_dimension_m0_accuracy": per_dimension_m0_accuracy,
                    "summary_statistics": summary_statistics,
                }
            )

        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "users": user_reports,
                    "run_timestamp": run_ts,
                    "ablation": args.ablation,
                    "k_retrieve": args.k_retrieve,
                    "num_rollouts": args.num_rollouts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nWrote report: {report_path}")
        print(f"Terminal output saved to: {report_txt_path}")

        _generate_metrics(user_reports, report_dir)
        return 0

    finally:
        sys.stdout = real_stdout
        report_txt_file.close()


if __name__ == "__main__":
    raise SystemExit(main())