#!/usr/bin/env python3
"""
LLM-based synthetic dataset generator for long-term (30-day) deployments.

Generates a 30-day dataset using LLM reasoning based on:
- User physical capability profile (U^phys)
- A single, fixed user preference encoding for the full deployment (U^pref)
- Context (X_t): meal, setting, time of day
- Transient affective state (Y_t)

For each day, makes ONE joint LLM call (all preference dimensions at once) to
generate the preference bundle for that day.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
    sys.exit(1)

from feeding_deployment.preference_learning.config.affective_state import AFFECTIVE_STATES
from feeding_deployment.preference_learning.config.mealtime_context import (
    MEAL_CONTENTS_BY_LABEL,
    MEALS,
    SETTINGS,
    TIMES_OF_DAY,
)
from feeding_deployment.preference_learning.config.physical_capabilities import PHYSICAL_CAPABILITY_PROFILES
from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.data_generation.generate_user_preference_encoding import (
    generate_user_preference_encoding_llm,
)
from feeding_deployment.preference_learning.data_generation.prompts.preference_generation import (
    get_preference_generation_prompt,
)

DEFAULT_MODEL = "gpt-4o"


def _strip_json_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw


def get_meal_info(meal_label: str) -> Dict[str, Any]:
    meal = MEAL_CONTENTS_BY_LABEL[meal_label]
    dippable = meal.dippable_items or []
    sauces = meal.sauces or []
    return {
        "meal": meal.label,
        "dippable_items": dippable,
        "sauces": sauces,
        "storage_condition": meal.storage_condition,
        "intended_serving_temp": meal.intended_serving_temp,
        "has_dippable": bool(dippable),
        "has_sauce": bool(sauces),
        "num_dippable": len(dippable),
        "num_sauces": len(sauces),
    }


def apply_hard_rules(preferences: Dict[str, str], meal_info: Dict[str, Any]) -> Dict[str, str]:
    prefs = dict(preferences)

    # Rule 1: If inside mouth transfer, distance must be "near" (represents 0cm)
    if prefs.get("transfer_mode") == "inside mouth transfer":
        prefs["outside_mouth_distance"] = "near"

    # Rule 2: If no dippable items or no sauces, must be "do not dip"
    if (not meal_info["has_dippable"]) or (not meal_info["has_sauce"]):
        prefs["bite_dipping_preference"] = "do not dip"

    # Rule 3: If do not dip, amount_to_dip must be "not applicable"
    if prefs.get("bite_dipping_preference") == "do not dip":
        prefs["amount_to_dip"] = "not applicable"

    return prefs


def _validate_joint_output_strict(data: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Strict schema:
      {"preferences": {"<field>": {"choice": "<exact option>", "rationale": "<string>"}, ...}}

    Returns:
      (choices, rationales)
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object (dict). Got {type(data).__name__}")

    if set(data.keys()) != {"preferences"}:
        raise KeyError(f'Top-level JSON must have exactly one key: "preferences". Got: {sorted(data.keys())}')

    prefs_obj = data["preferences"]
    if not isinstance(prefs_obj, dict):
        raise TypeError(f'"preferences" must be an object (dict). Got {type(prefs_obj).__name__}')

    allowed_map: Dict[str, List[str]] = {dim.field: dim.options for dim in PREFERENCE_BUNDLE}
    expected_fields = set(allowed_map.keys())
    got_fields = set(prefs_obj.keys())

    missing = sorted(expected_fields - got_fields)
    extra = sorted(got_fields - expected_fields)
    if missing:
        raise KeyError(f"Missing fields in joint LLM output: {missing}")
    if extra:
        raise KeyError(f"Unexpected extra fields in joint LLM output: {extra}")

    choices: Dict[str, str] = {}
    rationales: Dict[str, str] = {}

    for field in sorted(expected_fields):
        entry = prefs_obj[field]
        if not isinstance(entry, dict):
            raise TypeError(f'Field "{field}" must be an object with keys ["choice","rationale"].')

        if set(entry.keys()) != {"choice", "rationale"}:
            raise KeyError(
                f'Field "{field}" must have exactly keys ["choice","rationale"]. Got: {sorted(entry.keys())}'
            )

        choice = entry["choice"]
        rationale = entry["rationale"]

        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(f'Field "{field}.choice" must be a non-empty string.')
        if not isinstance(rationale, str):
            raise TypeError(f'Field "{field}.rationale" must be a string.')

        allowed = allowed_map[field]
        if choice not in allowed:
            raise ValueError(f'Invalid choice for "{field}": {choice!r}. Allowed: {allowed}')

        choices[field] = choice
        rationales[field] = rationale.strip()

    return choices, rationales


def generate_joint_preferences_with_llm(
    client: OpenAI,
    physical_profile_label: str,
    user_preference_encoding: Dict[str, Any],
    meal_info: Dict[str, Any],
    setting: str,
    time_of_day: str,
    affective_state: str,
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate ALL preferences jointly using ONE LLM call."""
    prompt = get_preference_generation_prompt(
        physical_profile_label=physical_profile_label,
        user_preference_encoding=json.dumps(user_preference_encoding, ensure_ascii=False, indent=2),
        meal=meal_info["meal"],
        dippable_items=", ".join(meal_info["dippable_items"]) if meal_info["dippable_items"] else "None",
        sauces=", ".join(meal_info["sauces"]) if meal_info["sauces"] else "None",
        storage_condition=meal_info["storage_condition"],
        intended_serving_temp=meal_info["intended_serving_temp"],
        setting=setting,
        time_of_day=time_of_day,
        affective_state=affective_state,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at reasoning about robotic mealtime-assistance preferences based on "
                    "user capabilities, user preference encodings, context, and affective state. "
                    "Return ONLY valid JSON (no markdown, no extra text). "
                    'The JSON must match exactly: {"preferences": {field: {"choice": <exact option>, "rationale": <string>}}} '
                    "Include all dimensions exactly once and no extra fields."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise RuntimeError("LLM returned empty content.")

    parsed = json.loads(_strip_json_fences(raw))
    return _validate_joint_output_strict(parsed)


def run_deployment(
    user_name: str,
    deployment_id: str,
    physical_profile_label: str,
    seed: Optional[int] = None,
    days: int = 30,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "out",
) -> str:
    """Generate a deployment dataset using LLM and save as JSON."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY env var or pass --api-key.")

    client = OpenAI(api_key=api_key)
    rng = random.Random(seed)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{user_name}__{deployment_id}__{days}d.json")

    print(f"Generating user preference encoding for: {physical_profile_label}")
    user_preference_encoding = generate_user_preference_encoding_llm(client, physical_profile_label, model=model)

    day_records: List[Dict[str, Any]] = []

    for day in range(1, days + 1):
        print(f"\n=== Day {day}/{days} ===")

        meal = rng.choice(MEALS)
        setting = rng.choice(SETTINGS)
        time_of_day = rng.choice(TIMES_OF_DAY)
        affective_state = rng.choice(AFFECTIVE_STATES)

        meal_info = get_meal_info(meal)
        print(f"Context: {meal} | {setting} | {time_of_day} | {affective_state}")

        pref_choices, pref_rationales = generate_joint_preferences_with_llm(
            client,
            physical_profile_label=physical_profile_label,
            user_preference_encoding=user_preference_encoding,
            meal_info=meal_info,
            setting=setting,
            time_of_day=time_of_day,
            affective_state=affective_state,
            model=model,
        )

        pref_choices_after = apply_hard_rules(pref_choices, meal_info)
        for k, v_after in pref_choices_after.items():
            v_before = pref_choices.get(k)
            if v_before != v_after:
                note = f" [HARD RULE OVERRIDE: {v_before!r} -> {v_after!r}]"
                pref_rationales[k] = (pref_rationales.get(k, "") + note).strip()
        pref_choices = pref_choices_after

        day_records.append(
            {
                "day": day,
                "context": {
                    "meal": meal,
                    "setting": setting,
                    "time_of_day": time_of_day,
                    "transient_affective_state": affective_state,
                },
                "preferences": {
                    field: {"choice": pref_choices[field], "rationale": pref_rationales.get(field, "")}
                    for field in sorted(pref_choices.keys())
                },
            }
        )
        print(f"=== Day {day} generated ===")

    data: Dict[str, Any] = {
        "user": user_name,
        "deployment_id": deployment_id,
        "physical_profile_label": physical_profile_label,
        "config": {"days": days, "seed": seed, "model": model},
        "user_preference_encoding": user_preference_encoding,
        "days": day_records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    physical_labels = [p.label for p in PHYSICAL_CAPABILITY_PROFILES]

    parser = argparse.ArgumentParser(description="LLM-based dataset generator (long-term deployment).")
    parser.add_argument("--user", required=True, help="User name (e.g., User1)")
    parser.add_argument("--deployment-id", default="dep1", help="Deployment identifier")
    parser.add_argument(
        "--physical-profile",
        required=True,
        choices=physical_labels,
        help="Physical capability profile label",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    parser.add_argument("--output-dir", default="out", help="Output directory (default: out)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides env)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        out_path = run_deployment(
            user_name=args.user,
            deployment_id=args.deployment_id,
            physical_profile_label=args.physical_profile,
            seed=args.seed,
            days=args.days,
            api_key=args.api_key,
            model=args.model,
            output_dir=args.output_dir,
        )
        print(f"\n✓ Done. Wrote: {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))