#!/usr/bin/env python3
"""
Batch run the LLM dataset generator for N people.

Each person gets one 30-day JSON file. Physical profiles are cycled so you get
a mix of user types (e.g. 50 people → ~12–13 per profile).

Usage:
  python scripts/generate_dataset_batch_llm.py --num-users 50 --output-dir out
  python scripts/generate_dataset_batch_llm.py --num-users 50 --seed-base 42

Requires OPENAI_API_KEY in the environment (or set in generate_dataset_llm.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Physical profile keys from config (must match generate_dataset_llm.py)
PROFILES = [
    "severe_paralysis_clear_speech",
    "moderate_motor_unreliable_speech",
    "high_fatigue_swallowing_risk",
    "inconsistent_mouth_control",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LLM datasets for N people (one JSON per person)."
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=50,
        help="Number of people (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Output directory for JSON files (default: out)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Base random seed; each user gets seed_base + user_index (optional)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days per deployment (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not run",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "generate_dataset_llm.py"
    if not script.exists():
        print(f"Error: {script} not found", file=sys.stderr)
        return 1

    for i in range(1, args.num_users + 1):
        user_name = f"User{i}"
        profile = PROFILES[(i - 1) % len(PROFILES)]
        seed = (args.seed_base + i) if args.seed_base is not None else None

        cmd = [
            sys.executable,
            str(script),
            "--user",
            user_name,
            "--deployment-id",
            "dep1",
            "--physical-profile",
            profile,
            "--days",
            str(args.days),
            "--output-dir",
            args.output_dir,
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if args.dry_run:
            print(" ".join(cmd))
            continue

        print(f"[{i}/{args.num_users}] {user_name} ({profile})...")
        ret = subprocess.run(cmd, cwd=repo_root)
        if ret.returncode != 0:
            print(f"Failed: {user_name}", file=sys.stderr)
            return ret.returncode

    print(f"Done. {args.num_users} files in {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
