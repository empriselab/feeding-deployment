#!/usr/bin/env python3
"""
LLM-based synthetic dataset generator for long-term deployments.

Generates 30-day datasets using LLM reasoning based on:
- User physical capability profile (U^phys)
- Slow latent preference (U^pref) - two windows with random variation
- Context (X): meal, setting, time of day
- Transient affective state (Y_t)

For each day, makes ONE joint LLM call (all 19 preference dimensions at once) to
generate the preference bundle based on physical constraints, long-term preferences,
context, and affect.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from feeding_deployment.preference_learning.config import (
    MEAL_STRUCTURE,
    MEALS,
    SETTINGS,
    TIMES_OF_DAY,
    AFFECTIVE_STATES,
    PREFERENCE_BUNDLE,
    PHYSICAL_CAPABILITY_PROFILES,
)
from feeding_deployment.preference_learning.data_generation.generate_user_preference_encoding import (
    generate_user_preference_encoding_llm,
)

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
    sys.exit(1)

OPENAI_API_KEY: Optional[str] = ""
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"

from pathlib import Path

# NOTE: You said you'll edit the prompt later.
# For now we keep a single template path; update it to your new joint prompt when ready.
PROMPT_PATH = Path(__file__).parent / "prompts" / "preference_prompt.txt"


def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


PREFERENCE_PROMPT_TEMPLATE = load_prompt_template()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_meal_info(meal: str) -> Dict[str, Any]:
    """Extract meal structure information."""
    info = MEAL_STRUCTURE.get(
        meal,
        {
            "dippable_items": [],
            "sauces": [],
            "storage_condition": "refrigerated_leftover",
            "intended_serving_temp": "hot",
        },
    )
    dippable = info.get("dippable_items", [])
    sauces = info.get("sauces", [])
    return {
        "meal": meal,
        "dippable_items": dippable,
        "sauces": sauces,
        "storage_condition": info.get("storage_condition", "refrigerated_leftover"),
        "intended_serving_temp": info.get("intended_serving_temp", "hot"),
        "has_dippable": len(dippable) > 0,
        "has_sauce": len(sauces) > 0,
        "num_dippable": len(dippable),
        "num_sauces": len(sauces),
    }


def apply_hard_rules(preferences: Dict[str, str], meal_info: Dict[str, Any]) -> Dict[str, str]:
    """Apply hard rules to preferences after LLM generation."""
    prefs = preferences.copy()

    # Rule 1: If inside mouth transfer, distance must be "near" (represents 0cm)
    if prefs.get("transfer_mode") == "inside mouth transfer":
        prefs["outside_mouth_distance"] = "near"

    # Rule 2: If no dippable items or no sauces, must be "do not dip"
    if not meal_info["has_dippable"] or not meal_info["has_sauce"]:
        prefs["bite_dipping_preference"] = "do not dip"

    return prefs


# ============================================================================
# JOINT LLM GENERATION (ONE CALL PER DAY)
# ============================================================================

def _strip_json_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw


def create_joint_preference_prompt(
    physical_profile: str,
    long_term_prefs: Dict[str, Any],
    context: Dict[str, str],
    meal_info: Dict[str, Any],
    affective_state: str,
) -> str:
    """
    Create a prompt for generating ALL 19 preferences jointly.

    NOTE: This currently reuses PREFERENCE_PROMPT_TEMPLATE. You said you'll edit the prompt later.
    When you update the prompt, make sure it instructs the model to output a single JSON object
    containing all 19 fields.
    """
    meal = context["meal"]
    setting = context["setting"]
    time_of_day = context.get("time_of_day", "unknown")

    # Provide options for each dimension (so the joint call can pick valid strings)
    # We'll embed a compact spec the prompt can reference.
    bundle_spec_lines: List[str] = []
    for field, label, options in PREFERENCE_BUNDLE:
        bundle_spec_lines.append(f"- {field} ({label}): {options}")

    bundle_spec = "\n".join(bundle_spec_lines)

    # Put U^pref in the prompt as JSON so the model can reference default + user_tendencies per field.
    long_term_pref_json = json.dumps(long_term_prefs, ensure_ascii=False, indent=2)

    prompt = PREFERENCE_PROMPT_TEMPLATE.format(
        physical_profile=physical_profile,
        long_term_pref=long_term_pref_json,
        meal=meal,
        dippable_items=", ".join(meal_info["dippable_items"]) if meal_info["dippable_items"] else "None",
        sauces=", ".join(meal_info["sauces"]) if meal_info["sauces"] else "None",
        storage_condition=meal_info["storage_condition"],
        intended_serving_temp=meal_info["intended_serving_temp"],
        setting=setting,
        time_of_day=time_of_day,
        affective_state=affective_state,
        previous_preferences="N/A (joint generation)",
        preference_label="ALL_DIMENSIONS_JOINT",
        options=bundle_spec,
    )
    return prompt


def _validate_joint_output_strict(
    data: Any,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Validate that the model returned all preference fields and each choice is one of the allowed options.

    Expected JSON shape (strict):
      {
        "<field>": {"choice": "<exact option>", "rationale": "<string>"},
        ...
      }

    Returns:
      (choices, rationales)

    Raises:
      KeyError / TypeError / ValueError on any violation.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object (dict). Got {type(data).__name__}")

    allowed_map: Dict[str, List[str]] = {field: options for field, _label, options in PREFERENCE_BUNDLE}
    expected_fields = set(allowed_map.keys())
    got_fields = set(data.keys())

    missing = sorted(expected_fields - got_fields)
    extra = sorted(got_fields - expected_fields)
    if missing:
        raise KeyError(f"Missing fields in joint LLM output: {missing}")
    if extra:
        raise KeyError(f"Unexpected extra fields in joint LLM output: {extra}")

    choices: Dict[str, str] = {}
    rationales: Dict[str, str] = {}

    for field in sorted(expected_fields):
        entry = data[field]
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

        # Strict exact matching (case-sensitive) against allowed options list
        allowed = allowed_map[field]
        if choice not in allowed:
            raise ValueError(f'Invalid choice for "{field}": {choice!r}. Allowed: {allowed}')

        choices[field] = choice
        rationales[field] = rationale.strip()

    return choices, rationales


def generate_joint_preferences_with_llm(
    client: OpenAI,
    physical_profile: str,
    long_term_prefs: Dict[str, Any],
    context: Dict[str, str],
    meal_info: Dict[str, Any],
    affective_state: str,
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate ALL 19 preferences jointly using ONE LLM call.

    Returns:
      (pref_choices, pref_rationales)

    Strict behavior:
      - Requires valid JSON
      - Requires exactly the 19 expected fields
      - Requires each choice to exactly match an allowed option string
      - Raises on any violation
    """
    prompt = create_joint_preference_prompt(
        physical_profile=physical_profile,
        long_term_prefs=long_term_prefs,
        context=context,
        meal_info=meal_info,
        affective_state=affective_state,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at reasoning about robotic mealtime-assistance preferences based on "
                    "user capabilities, long-term tendencies, context, and affective state. "
                    "Return ONLY valid JSON (no markdown, no extra text). "
                    "Output schema: {field: {\"choice\": <exact option string>, \"rationale\": <string>}} "
                    "for all provided fields, with no missing or extra fields."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
        # If supported by your SDK/version, this helps enforce JSON-only outputs.
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    print("Raw LLM output:")
    print(raw)

    raw = _strip_json_fences(raw)
    parsed = json.loads(raw)

    return _validate_joint_output_strict(parsed)


# ============================================================================
# MAIN GENERATION
# ============================================================================

@dataclass
class DeploymentConfig:
    user_name: str
    deployment_id: str
    physical_capability_profile: str
    seed: Optional[int] = None
    days: int = 30
    u_update_days: Tuple[int, int] = (1, 16)
    variation_level: float = 0.3  # (kept for config/logging)


def run_deployment(
    cfg: DeploymentConfig,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "out",
) -> str:
    """Generate a deployment dataset using LLM and save as JSON."""

    if api_key is None:
        api_key = OPENAI_API_KEY

    if api_key is None:
        raise ValueError(
            "OpenAI API key not found. Please set it in one of these ways:\n"
            "1. Set OPENAI_API_KEY environment variable\n"
            "2. Modify OPENAI_API_KEY in this script\n"
            "3. Pass --api-key argument"
        )

    client = OpenAI(api_key=api_key)
    rng = random.Random(cfg.seed)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cfg.user_name}__{cfg.deployment_id}__30d.json")

    # Long-term preferences (two windows)
    print("Generating long-term preferences for days 1-15...")
    u_pref_window1 = generate_user_preference_encoding_llm(client, cfg.physical_capability_profile, model)

    print("Generating long-term preferences for days 16-30...")
    u_pref_window2 = generate_user_preference_encoding_llm(client, cfg.physical_capability_profile, model)

    days: List[Dict[str, object]] = []

    for day in range(1, cfg.days + 1):
        print(f"\n=== Day {day}/{cfg.days} ===")

        if day < cfg.u_update_days[1]:
            u_current = u_pref_window1
            window_num = 1
        else:
            u_current = u_pref_window2
            window_num = 2

        # Random context
        meal = rng.choice(MEALS)
        setting = rng.choice(SETTINGS)
        time_of_day = rng.choice(TIMES_OF_DAY)
        affective_state = rng.choice(AFFECTIVE_STATES)

        meal_info = get_meal_info(meal)
        context = {
            "meal": meal,
            "setting": setting,
            "time_of_day": time_of_day,
            "transient_affective_state": affective_state,
        }

        print(f"Context: {meal} | {setting} | {time_of_day} | {affective_state}")

        # ONE joint call for all 19 dimensions
        pref_choices, pref_rationales = generate_joint_preferences_with_llm(
            client=client,
            physical_profile=cfg.physical_capability_profile,
            long_term_prefs=u_current,
            context=context,
            meal_info=meal_info,
            affective_state=affective_state,
            model=model,
        )

        # Apply hard rules (may override choices)
        pref_choices_after = apply_hard_rules(pref_choices, meal_info)

        # If hard rules changed anything, keep rationale but append a note for traceability.
        for k, v_after in pref_choices_after.items():
            v_before = pref_choices.get(k)
            if v_before != v_after:
                note = f" [HARD RULE OVERRIDE: {v_before!r} -> {v_after!r}]"
                pref_rationales[k] = (pref_rationales.get(k, "") + note).strip()

        pref_choices = pref_choices_after

        # Add bite_ordering_preference (not in the 19-item list; keep random for now)
        ordering_choice = rng.choice(["alternate X and Y", "start with X then Y", "start with Y then X"])
        pref_choices["bite_ordering_preference"] = ordering_choice
        pref_rationales.setdefault("bite_ordering_preference", "")

        day_record: Dict[str, object] = {
            "day": day,
            "window": window_num,
            "context": context,
            "preferences": {
                name: {"choice": pref_choices.get(name, ""), "rationale": pref_rationales.get(name, "")}
                for name in pref_choices.keys()
            },
        }
        days.append(day_record)

        print(f"✓ Day {day} generated")

    data = {
        "user": cfg.user_name,
        "deployment_id": cfg.deployment_id,
        "physical_capability_profile": cfg.physical_capability_profile,
        "config": {
            "days": cfg.days,
            "u_update_days": list(cfg.u_update_days),
            "variation_level": cfg.variation_level,
            "model": model,
        },
        "long_term_preferences": {
            "window_1_days_1_15": u_pref_window1,
            "window_2_days_16_30": u_pref_window2,
        },
        "days": days,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based dataset generator (30-day deployments).")
    parser.add_argument("--user", required=True, help="User name (e.g., User1)")
    parser.add_argument("--deployment-id", default="dep1", help="Deployment identifier")
    parser.add_argument(
        "--physical-profile",
        required=True,
        choices=list(PHYSICAL_CAPABILITY_PROFILES.keys()),
        help="Physical capability profile key",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    parser.add_argument(
        "--variation-level",
        type=float,
        default=0.3,
        help="U^pref variation level between windows (0.0-1.0, default: 0.3)",
    )
    parser.add_argument("--output-dir", default="out", help="Output directory (default: out)")
    parser.add_argument("--api-key", help="OpenAI API key (overrides env/config)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    physical_profile = PHYSICAL_CAPABILITY_PROFILES[args.physical_profile]

    cfg = DeploymentConfig(
        user_name=args.user,
        deployment_id=args.deployment_id,
        physical_capability_profile=physical_profile,
        seed=args.seed,
        days=args.days,
        variation_level=args.variation_level,
    )

    try:
        out_path = run_deployment(cfg, api_key=args.api_key, model=args.model, output_dir=args.output_dir)
        print(f"\n✓ Done. Wrote: {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))