#!/usr/bin/env python3
"""
Interactive synthetic dataset generator for long-term deployments.

For each user:
  - Generate 30 "days" (rows).
  - On day 1 and day 16, prompt for slow latent preference U^pref (piecewise
    over two 15-day windows). Stored as ltpref_1_* / ltpref_2_*. Not U^phys.
  - Each day, randomly select:
      - meal (part of X_t)
      - setting/context (part of X_t)
      - transient affective state Y_t
  - Prompt for the 20-item preference bundle P_t using fast single-key selection.
  - Write one CSV row per day.

See FORMULATION.md for notation (U^pref, Y_t, P_t, etc.).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# -----------------------------
# Configuration (edit as needed)
# -----------------------------

DEFAULT_MEALS: List[str] = [
    "buffalo chicken bites, potato wedges, and ranch dressing",
    "strawberries with whipped cream",
    "chicken nuggets, broccoli, and ketchup",
    "general tso’s chicken and broccoli",
    "chicken breast strips and hash brown",
    "cantaloupes, bananas, watermelon",
    "bananas, brownies and chocolate sauce",
    "bite-sized sandwiches",
    "breaded fish bites, roasted potatoes, tartar sauce",
    "bite-sized pizza and broccoli",
]

DEFAULT_SETTINGS: List[str] = [
    "Personal",
    "Social",
    "Watching TV",
    "Social + Watching TV",
    "Working on laptop",
]

DEFAULT_AFFECTIVE_STATES: List[str] = [
    "Neutral",
    "Hurried",
    "Slower",
    "Fatigued",
    "Energetic",
]


# Preference bundle schema (each entry: (field_name, prompt, options))
PREFERENCE_BUNDLE: List[Tuple[str, str, List[str]]] = [
    ("microwave_time", "Microwave time", ["no microwave", "1 min", "2 min", "3 min"]),
    (
        "occlusion_relevance",
        "Occlusion relevance",
        ["minimize left occlusion", "minimize front occlusion", "do not consider occlusion"],
    ),
    ("robot_speed", "Robot speed", ["slow", "medium", "fast"]),
    (
        "skewering_axis",
        "Skewering axis selection",
        ["parallel to major axis", "perpendicular to major axis"],
    ),
    (
        "transfer_mode",
        "Outside mouth vs inside mouth transfer",
        ["outside mouth transfer", "inside mouth transfer"],
    ),
    (
        "outside_mouth_distance",
        "For outside-mouth transfer: distance from the mouth",
        ["near", "medium", "far"],
    ),
    (
        "robot_ready_cue",
        "[Robot→Human] Convey ready for initiating transfer",
        ["speech", "LED", "speech + LED", "no cue"],
    ),
    # Human→Robot: Bite initiation interaction (per task)
    (
        "bite_initiation_feeding",
        "[Human→Robot] Bite initiation for FEEDING",
        ["open mouth", "button", "autocontinue"],
    ),
    (
        "bite_initiation_drinking",
        "[Human→Robot] Bite initiation for DRINKING",
        ["open mouth", "button", "autocontinue"],
    ),
    (
        "bite_initiation_wiping",
        "[Human→Robot] Bite initiation for MOUTH WIPING",
        ["open mouth", "button", "autocontinue"],
    ),
    (
        "robot_bite_available_cue",
        "[Robot→Human] Convey user can take a bite",
        ["speech", "LED", "speech + LED", "no cue"],
    ),
    # Human→Robot: Bite completion interaction (per task)
    (
        "bite_completion_feeding",
        "[Human→Robot] Bite completion for FEEDING",
        ["perception", "button", "autocontinue"],
    ),
    (
        "bite_completion_drinking",
        "[Human→Robot] Bite completion for DRINKING",
        ["perception", "button", "autocontinue"],
    ),
    (
        "bite_completion_wiping",
        "[Human→Robot] Bite completion for MOUTH WIPING",
        ["perception", "button", "autocontinue"],
    ),
    (
        "web_interface_confirmation",
        "Web interface confirmation",
        ["yes", "no"],
    ),
    (
        "retract_between_bites",
        "Retract between bites",
        ["yes", "no"],
    ),
    (
        "bite_ordering_preference",
        "Bite ordering preference (X/Y)",
        ["alternate X and Y", "start with X then Y", "start with Y then X"],
    ),
    (
        "bite_dipping_preference",
        "Bite dipping preference (X/Y in Z)",
        ["dip X in Z", "dip Y in Z", "dip both X and Y in Z", "do not dip"],
    ),
    (
        "amount_to_dip",
        "Amount to dip",
        ["less", "more"],
    ),
    (
        "wait_before_autocontinue_seconds",
        "Time to wait before autocontinue",
        ["10", "100", "1000"],
    ),
    (
        "confirm_bundle",
        "Confirm this preference bundle is correct",
        ["confirm", "re-enter bundle"],
    ),
]


KEYS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class DeploymentConfig:
    user_name: str
    deployment_id: str
    seed: int | None
    days: int = 30
    u_update_days: Tuple[int, int] = (1, 16)


def _clear_screen() -> None:
    # Works on macOS/Linux; on Windows it will just print newlines.
    if os.name == "posix":
        os.system("clear")
    else:
        print("\n" * 50)


def _safe_input(prompt: str) -> str:
    """
    Wrapper around input() that exits cleanly on Ctrl-C / Ctrl-D.
    """
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\n\nExiting (input cancelled). No further rows will be collected.")
        raise SystemExit(1)


def _prompt_line(prompt: str, default: str | None = None) -> str:
    if default is None:
        return _safe_input(f"{prompt}: ").strip()
    val = _safe_input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _prompt_choice(prompt: str, options: Sequence[str]) -> str:
    """
    Fast choice prompt:
      - shows A/B/C...
      - accepts letter (case-insensitive) or full option text
      - accepts numeric index (1..N)
    """
    if not options:
        raise ValueError("options must be non-empty")

    if len(options) > len(KEYS):
        raise ValueError(f"Too many options ({len(options)}); max is {len(KEYS)}")

    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(options):
            print(f"  {KEYS[i]}) {opt}")

        raw = _safe_input("Select (letter / number / text): ").strip()
        if not raw:
            print("Please enter a selection.")
            continue

        # letter
        if len(raw) == 1 and raw.upper() in KEYS[: len(options)]:
            return options[KEYS.index(raw.upper())]

        # number
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]

        # text exact-ish
        lowered = raw.lower()
        matches = [opt for opt in options if opt.lower() == lowered]
        if len(matches) == 1:
            return matches[0]

        # text prefix
        prefix_matches = [opt for opt in options if opt.lower().startswith(lowered)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        print("Invalid selection. Try again.")


def _random_choice(rng: random.Random, items: Sequence[str]) -> str:
    if not items:
        raise ValueError("items must be non-empty")
    return items[rng.randrange(0, len(items))]


def collect_preference_bundle(
    meal: str,
    setting: str,
    transient_affective_state: str,
    u_behavioral: Dict[str, str] | None,
) -> Dict[str, str]:
    """
    Collect 20 preferences for the current day, given the context (X, Y) and
    the long-term preferences for each dimension (if available).
    If the user chooses "re-enter bundle" at the end, restart collection.
    """
    while True:
        bundle: Dict[str, str] = {}
        # count how many actual preference fields (exclude confirm_bundle)
        pref_fields = [f for (f, _, _) in PREFERENCE_BUNDLE if f != "confirm_bundle"]
        total = len(pref_fields)
        idx_counter = 0

        for field, prompt, options in PREFERENCE_BUNDLE:
            if field == "confirm_bundle":
                # handled after loop
                continue

            idx_counter += 1

            # Show context for every preference selection to keep it salient.
            print("\n=== Current context ===")
            print(f"  Meal (X_t): {meal}")
            print(f"  Setting (X_t): {setting}")
            print(f"  Transient affective state (Y_t): {transient_affective_state}")
            if u_behavioral:
                lt_desc = u_behavioral.get(field, "")
                if lt_desc:
                    print(f"  Slow latent preference U^pref (this dimension): {lt_desc}")

            bundle[field] = _prompt_choice(f"[{idx_counter}/{total}] {prompt}", options)

        confirm = _prompt_choice(
            "Confirm this preference bundle is correct?",
            ["confirm", "re-enter bundle"],
        )
        if confirm == "confirm":
            return bundle
        print("\nRe-entering bundle...\n")


def collect_long_term_preferences(label: str) -> Dict[str, str]:
    """
    Collect text descriptions of long-term preferences for all 20 dimensions,
    e.g., once on day 1 (for days 1-15) and once on day 16 (for days 16-30).
    """
    prefs: Dict[str, str] = {}
    fields = [(field, prompt) for (field, prompt, _) in PREFERENCE_BUNDLE if field != "confirm_bundle"]
    total = len(fields)

    print(f"\n=== Long-term preferences: {label} ===")
    for idx, (field, prompt) in enumerate(fields, start=1):
        text = _safe_input(f"[{idx}/{total}] Describe long-term preference for: {prompt}\n> ").strip()
        prefs[field] = text
    return prefs


def run_deployment(
    cfg: DeploymentConfig,
    meals: Sequence[str],
    settings: Sequence[str],
    affective_states: Sequence[str],
    output_dir: str,
) -> str:
    rng = random.Random(cfg.seed)

    # Long-term preferences per dimension, for two phases.
    u_behavioral_day1: Dict[str, str] = {}
    u_behavioral_day16: Dict[str, str] = {}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cfg.user_name}__{cfg.deployment_id}__30d.csv")

    # CSV columns
    pref_fields = [field for (field, _, _) in PREFERENCE_BUNDLE if field != "confirm_bundle"]
    base_cols = [
        "user",
        "deployment_id",
        "day",
        "meal",
        "setting",
        "transient_affective_state",
    ]
    # Slow latent preference U^pref (one column per dimension per phase); see FORMULATION.md
    lt1_cols = [f"ltpref_1_{field}" for field in pref_fields]
    lt2_cols = [f"ltpref_2_{field}" for field in pref_fields]
    # Daily realized preferences (choice selections)
    pref_cols = pref_fields
    cols = base_cols + lt1_cols + lt2_cols + pref_cols

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for day in range(1, cfg.days + 1):
            _clear_screen()
            print(f"User: {cfg.user_name} | Deployment: {cfg.deployment_id} | Day {day}/{cfg.days}\n")

            if day == cfg.u_update_days[0]:
                u_behavioral_day1 = collect_long_term_preferences("days 1–15")
            if day == cfg.u_update_days[1]:
                u_behavioral_day16 = collect_long_term_preferences("days 16–30")

            meal = _random_choice(rng, meals)
            setting = _random_choice(rng, settings)
            affect = _random_choice(rng, affective_states)

            print("Auto-selected context for today:")
            print(f"  - Meal (X_t): {meal}")
            print(f"  - Setting (X_t): {setting}")
            print(f"  - Transient affective state (Y_t): {affect}")

            # Decide which long-term preference text is currently in effect.
            if day < cfg.u_update_days[1]:
                u_current = u_behavioral_day1
            else:
                u_current = u_behavioral_day16

            _safe_input("\nPress Enter to input today's preference bundle...")
            bundle = collect_preference_bundle(
                meal=meal,
                setting=setting,
                transient_affective_state=affect,
                u_behavioral=u_current,
            )

            row: Dict[str, str] = {
                "user": cfg.user_name,
                "deployment_id": cfg.deployment_id,
                "day": str(day),
                "meal": meal,
                "setting": setting,
                "transient_affective_state": affect,
            }
            # Attach long-term preference text (same values copied for every day in the phase).
            for field in pref_fields:
                row[f"ltpref_1_{field}"] = u_behavioral_day1.get(field, "")
                row[f"ltpref_2_{field}"] = u_behavioral_day16.get(field, "")
            for k in pref_cols:
                row[k] = bundle.get(k, "")

            writer.writerow(row)
            # Make sure the row is visible on disk immediately (so users can open
            # the CSV mid-run and see progress).
            f.flush()
            os.fsync(f.fileno())

            print("\nSaved day to CSV.")
            _safe_input("Press Enter for next day...")

    return out_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive dataset generator (30-day deployments).")
    p.add_argument("--user", required=True, help="User name (e.g., Aimee)")
    p.add_argument("--deployment-id", default="dep1", help="Deployment identifier (e.g., permA)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    p.add_argument("--output-dir", default="out", help="Directory to write CSVs into (default: out)")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    cfg = DeploymentConfig(
        user_name=args.user,
        deployment_id=args.deployment_id,
        seed=args.seed,
        days=args.days,
    )

    out_path = run_deployment(
        cfg=cfg,
        meals=DEFAULT_MEALS,
        settings=DEFAULT_SETTINGS,
        affective_states=DEFAULT_AFFECTIVE_STATES,
        output_dir=args.output_dir,
    )

    print("\nDone.")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

