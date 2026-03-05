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
from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.data_generation.prompts.preference_generation import (
    get_preference_generation_prompt,
)

DEFAULT_MODEL = "gpt-5.4"


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

    # Rule 1: If inside mouth transfer, distance must be "not applicable"
    if prefs.get("transfer_mode") == "inside mouth transfer":
        prefs["outside_mouth_distance"] = "not applicable"

    # Rule 2: If no dippable items or no sauces, must be "do not dip"
    if (not meal_info["has_dippable"]) or (not meal_info["has_sauce"]):
        prefs["bite_dipping_preference"] = "do not dip"

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
        max_completion_tokens=5000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise RuntimeError("LLM returned empty content.")

    parsed = json.loads(_strip_json_fences(raw))
    return _validate_joint_output_strict(parsed)


def _load_encoding_payload(path: str) -> Dict[str, Any]:
    """
    Load a user encoding payload. Expected schema:
      {
        "user_index": <int or str>,
        "physical_profile": <str>,
        "encoding": <dict>
      }
    Returns the whole payload dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Encoding file must be a JSON object. Got {type(data).__name__}: {path}")

    for k in ("physical_profile", "encoding"):
        if k not in data:
            raise KeyError(f'Missing key "{k}" in encoding file: {path}')

    if not isinstance(data["physical_profile"], str) or not data["physical_profile"].strip():
        raise ValueError(f'"physical_profile" must be a non-empty string in: {path}')

    if not isinstance(data["encoding"], dict) or not data["encoding"]:
        raise ValueError(f'"encoding" must be a non-empty object in: {path}')

    return data


def run_deployment(
    client: OpenAI,
    user_name: str,
    deployment_id: str,
    physical_profile_label: str,
    user_preference_encoding: Dict[str, Any],
    seed: Optional[int] = None,
    days: int = 30,
    model: str = DEFAULT_MODEL,
    output_dir: str = "out",
    output_filename: Optional[str] = None,
) -> str:
    """Generate a deployment dataset using LLM and save as JSON."""
    rng = random.Random(seed)

    os.makedirs(output_dir, exist_ok=True)
    if output_filename:
        out_path = os.path.join(output_dir, output_filename)
    else:
        out_path = os.path.join(output_dir, f"{user_name}__{deployment_id}__{days}d.json")

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
    parser = argparse.ArgumentParser(description="LLM-based dataset generator (deployment).")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--user-encoding-file", help="Path to a single user encoding JSON file.")
    src.add_argument("--user-encodings-dir", help="Directory containing user encoding JSON files.")

    parser.add_argument("--deployment-id", default="dep1", help="Deployment identifier")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (varied per file in dir mode).")
    parser.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    parser.add_argument("--output-dir", default="out", help="Output directory (default: out)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides env)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    try:
        client = OpenAI(api_key=api_key)
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        def process_one(path: str, output_filename: str, seed: Optional[int]) -> str:
            payload = _load_encoding_payload(path)
            physical_profile_label = payload["physical_profile"]
            user_preference_encoding = payload["encoding"]

            # Prefer user_index for naming if present, otherwise fallback to basename
            user_name = str(payload.get("user_index") or os.path.splitext(os.path.basename(path))[0])

            print(f"\n=== Processing: {os.path.basename(path)} ===")
            print(f"User: {user_name} | Physical profile: {physical_profile_label} | Output: {output_filename}")

            return run_deployment(
                client=client,
                user_name=user_name,
                deployment_id=args.deployment_id,
                physical_profile_label=physical_profile_label,
                user_preference_encoding=user_preference_encoding,
                seed=seed,
                days=args.days,
                model=args.model,
                output_dir=out_dir,
                output_filename=output_filename,
            )

        # Single file mode
        if args.user_encoding_file:
            in_path = os.path.abspath(args.user_encoding_file)
            if not os.path.isfile(in_path):
                raise ValueError(f"--user-encoding-file does not exist: {in_path}")

            out_name = os.path.basename(in_path)
            out_path = process_one(in_path, out_name, args.seed)
            print(f"\n✓ Done. Wrote: {out_path}")
            return 0

        # Directory mode
        enc_dir = os.path.abspath(args.user_encodings_dir)
        if not os.path.isdir(enc_dir):
            raise ValueError(f"--user-encodings-dir is not a directory: {enc_dir}")

        files = sorted(
            fn
            for fn in os.listdir(enc_dir)
            if not fn.startswith(".")
            and os.path.isfile(os.path.join(enc_dir, fn))
            and fn.lower().endswith(".json")
        )
        if not files:
            raise ValueError(f"No .json files found in: {enc_dir}")

        print(f"Batch mode: {len(files)} encoding files found in {enc_dir}")
        print(f"Writing outputs to: {out_dir}")

        base_seed = args.seed
        for idx, fn in enumerate(files, start=1):
            in_path = os.path.join(enc_dir, fn)
            seed_i = None if base_seed is None else (base_seed + idx)
            out_path = process_one(in_path, fn, seed_i)
            print(f"✓ Wrote: {out_path}")

        print("\n✓ Done (batch).")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))