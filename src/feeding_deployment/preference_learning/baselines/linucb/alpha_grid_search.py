#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SUMMARY_PATH_RE = re.compile(r"^Wrote summary metrics:\s+(?P<path>.+)$")


@dataclass(frozen=True)
class GridResult:
    alpha: float
    summary_path: Path
    report_dir: Path
    summary: Dict[str, Any]


def _parse_summary_path(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        m = _SUMMARY_PATH_RE.match(line.strip())
        if m:
            return Path(m.group("path").strip())
    return None


def _score(summary: Dict[str, Any], metric: str) -> Optional[float]:
    # LinUCB evaluator writes {"LinUCB": {...}}.
    inner = summary.get("LinUCB")
    if not isinstance(inner, dict):
        return None
    val = inner.get(metric)
    return float(val) if val is not None else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search LinUCB alpha and report best.")
    p.add_argument("--data-file", help="Path to one JSON dataset file.")
    p.add_argument("--data-dir", help="Directory containing JSON dataset files.")
    p.add_argument("--max-meals", type=int, default=0, help="Max meals per user (0=all).")
    p.add_argument("--seed", type=int, default=0, help="Seed (passed to evaluator).")
    p.add_argument(
        "--metric",
        default="mean_acc_m0_all_days",
        help="Metric key inside LinUCB summary (default: mean_acc_m0_all_days).",
    )
    p.add_argument(
        "--objective",
        choices=["max", "min"],
        default="max",
        help="Whether to maximize or minimize the metric (default: max).",
    )
    p.add_argument(
        "--alpha-start",
        type=float,
        default=0.1,
        help="Alpha grid start (default: 0.1).",
    )
    p.add_argument(
        "--alpha-end",
        type=float,
        default=0.9,
        help="Alpha grid end, inclusive (default: 0.9).",
    )
    p.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Alpha grid step (default: 0.1).",
    )
    p.add_argument(
        "--include-affective-state",
        action="store_true",
        default=True,
        help="Include affective state feature (default: true).",
    )
    p.add_argument(
        "--no-include-affective-state",
        action="store_false",
        dest="include_affective_state",
        help="Disable affective state feature.",
    )
    p.add_argument(
        "--include-profile-label",
        action="store_true",
        default=True,
        help="Include physical_profile_label feature (default: true).",
    )
    p.add_argument(
        "--no-include-profile-label",
        action="store_false",
        dest="include_profile_label",
        help="Disable physical_profile_label feature.",
    )
    p.add_argument(
        "--ours-summary-metrics",
        default="",
        help="Optional path to methods/comparison_metrics/summary_metrics.json (passed to evaluator).",
    )
    return p.parse_args()


def _frange(start: float, end: float, step: float) -> List[float]:
    # Inclusive end with float-safe rounding.
    vals: List[float] = []
    x = start
    for _ in range(1000):
        if x > end + 1e-9:
            break
        vals.append(round(x, 10))
        x += step
    return vals


def main() -> int:
    args = parse_args()

    if not args.data_file and not args.data_dir:
        print("Error: provide --data-file or --data-dir", file=sys.stderr)
        return 2

    if args.alpha_step <= 0:
        print("Error: --alpha-step must be > 0", file=sys.stderr)
        return 2

    alphas = _frange(args.alpha_start, args.alpha_end, args.alpha_step)
    if not alphas:
        print("Error: empty alpha grid", file=sys.stderr)
        return 2

    this_dir = Path(__file__).resolve().parent
    eval_script = this_dir / "evaluate_linucb.py"
    if not eval_script.exists():
        print(f"Error: evaluator not found at {eval_script}", file=sys.stderr)
        return 2

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = this_dir / "reports" / f"alpha_grid_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[GridResult] = []

    for alpha in alphas:
        cmd = [
            sys.executable,
            "-u",
            str(eval_script),
            "--alpha",
            str(alpha),
            "--seed",
            str(args.seed),
        ]
        if args.data_file:
            cmd += ["--data-file", str(args.data_file)]
        if args.data_dir:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.max_meals:
            cmd += ["--max-meals", str(args.max_meals)]
        if args.include_affective_state:
            cmd += ["--include-affective-state"]
        else:
            cmd += ["--no-include-affective-state"]
        if args.include_profile_label:
            cmd += ["--include-profile-label"]
        else:
            cmd += ["--no-include-profile-label"]
        if args.ours_summary_metrics:
            cmd += ["--ours-summary-metrics", str(args.ours_summary_metrics)]

        print(f"[grid] alpha={alpha:.3f} running ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Save raw stdout/stderr for debugging.
        (out_dir / f"alpha_{alpha:.3f}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (out_dir / f"alpha_{alpha:.3f}.stderr.txt").write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            print(f"[grid] alpha={alpha:.3f} failed (exit={proc.returncode}). See logs in {out_dir}", flush=True)
            continue

        summary_path = _parse_summary_path(proc.stdout)
        if summary_path is None or not summary_path.exists():
            print(f"[grid] alpha={alpha:.3f} could not find summary_metrics.json in stdout.", flush=True)
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report_dir = summary_path.parent
        results.append(GridResult(alpha=float(alpha), summary_path=summary_path, report_dir=report_dir, summary=summary))

        score = _score(summary, args.metric)
        print(f"[grid] alpha={alpha:.3f} {args.metric}={score}", flush=True)

    if not results:
        print("No successful runs in grid search.", file=sys.stderr)
        return 1

    scored: List[Tuple[float, GridResult]] = []
    for r in results:
        s = _score(r.summary, args.metric)
        if s is None:
            continue
        scored.append((s, r))

    if not scored:
        print(f"No runs produced metric {args.metric}.", file=sys.stderr)
        return 1

    best = max(scored, key=lambda t: t[0]) if args.objective == "max" else min(scored, key=lambda t: t[0])
    best_score, best_res = best

    payload = {
        "grid": {
            "alpha_start": args.alpha_start,
            "alpha_end": args.alpha_end,
            "alpha_step": args.alpha_step,
            "alphas": alphas,
        },
        "selection": {
            "metric": args.metric,
            "objective": args.objective,
            "best_alpha": best_res.alpha,
            "best_score": best_score,
            "best_report_dir": str(best_res.report_dir),
        },
        "runs": [
            {
                "alpha": r.alpha,
                "report_dir": str(r.report_dir),
                "summary_path": str(r.summary_path),
                "metric_value": _score(r.summary, args.metric),
            }
            for r in results
        ],
    }

    out_json = out_dir / "grid_search_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBest alpha:", best_res.alpha)
    print(f"Best {args.metric} ({args.objective}):", best_score)
    print("Best report dir:", best_res.report_dir)
    print("Saved grid results:", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

