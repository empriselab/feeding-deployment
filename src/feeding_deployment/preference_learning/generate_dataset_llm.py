#!/usr/bin/env python3
"""
LLM-based synthetic dataset generator for long-term deployments.

Generates 30-day datasets using LLM reasoning based on:
- User physical capability profile (U^phys)
- Slow latent preference (U^pref) - two windows with random variation
- Context (X): meal, setting, time of day
- Transient affective state (Y_t)

For each day, makes 19 LLM calls (one per preference dimension) to reason about
the preference bundle based on physical constraints, long-term preferences, context, and affect.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import MEAL_STRUCTURE, MEALS, SETTINGS, TIMES_OF_DAY, AFFECTIVE_STATES, PREFERENCE_BUNDLE, PHYSICAL_CAPABILITY_PROFILES

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
    sys.exit(1)

OPENAI_API_KEY: Optional[str] = ""  
DEFAULT_MODEL = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo", etc.

from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompts" / "preference_prompt.txt"

def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

PREFERENCE_PROMPT_TEMPLATE = load_prompt_template()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_meal_info(meal: str) -> Dict[str, any]:
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


def generate_long_term_preference_variation(
    base_prefs: Dict[str, str],
    variation_level: float,
    rng: random.Random,
) -> Dict[str, str]:
    """
    Generate a variation of long-term preferences.
    variation_level: 0.0 = no change, 1.0 = maximum change
    """
    new_prefs = {}
    pref_fields = [f[0] for f in PREFERENCE_BUNDLE]
    
    for field in pref_fields:
        if rng.random() < variation_level:
            # Change this preference - randomly select from options
            options = next((opt[2] for opt in PREFERENCE_BUNDLE if opt[0] == field), [])
            if options:
                new_prefs[field] = rng.choice(options)
            else:
                new_prefs[field] = base_prefs.get(field, "")
        else:
            # Keep the same
            new_prefs[field] = base_prefs.get(field, "")
    
    return new_prefs


# ============================================================================
# LLM GENERATION
# ============================================================================

def create_preference_prompt(
    preference_name: str,
    preference_label: str,
    options: List[str],
    physical_profile: str,
    long_term_pref: Optional[str],
    context: Dict[str, str],
    meal_info: Dict[str, any],
    affective_state: str,
    previous_preferences: Dict[str, str],
    idx: int,
    total: int,
) -> str:
    """Create a prompt for generating a single preference dimension."""

    meal = context["meal"]
    setting = context["setting"]
    time_of_day = context.get("time_of_day", "unknown")

    previous_str = (
        "\n".join(f"- {k}: {v}" for k, v in previous_preferences.items())
        if previous_preferences else "None yet"
    )

    prompt = PREFERENCE_PROMPT_TEMPLATE.format(
        physical_profile=physical_profile,
        long_term_pref=long_term_pref if long_term_pref else "Not specified",
        meal=meal,
        dippable_items=", ".join(meal_info["dippable_items"]) if meal_info["dippable_items"] else "None",
        sauces=", ".join(meal_info["sauces"]) if meal_info["sauces"] else "None",
        storage_condition=meal_info["storage_condition"],
        intended_serving_temp=meal_info["intended_serving_temp"],
        setting=setting,
        time_of_day=time_of_day,
        affective_state=affective_state,
        previous_preferences=previous_str,
        preference_label=preference_label,
        options=", ".join(options),
    )

    return prompt

def generate_preference_with_llm(
    client: OpenAI,
    preference_name: str,
    preference_label: str,
    options: List[str],
    physical_profile: str,
    long_term_pref: Optional[str],
    context: Dict[str, str],
    meal_info: Dict[str, any],
    affective_state: str,
    previous_preferences: Dict[str, str],
    idx: int,
    total: int,
    model: str = DEFAULT_MODEL,
) -> Tuple[str, str]:
    """Generate a single preference using LLM.

    Returns:
        (choice, rationale)
    """
    
    prompt = create_preference_prompt(
        preference_name, preference_label, options,
        physical_profile, long_term_pref, context, meal_info,
        affective_state, previous_preferences, idx, total
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at reasoning about robotic feeding assistance preferences based on user capabilities, context, and affective state. Always follow hard rules strictly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more consistent reasoning
            max_tokens=100,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON if wrapped in ```json``` fences
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(raw)
        choice_raw = str(data.get("choice", "")).strip()
        rationale = str(data.get("rationale", "")).strip()

        # Validate choice against options (case-insensitive)
        choice_lower = choice_raw.lower()
        for opt in options:
            if opt.lower() == choice_lower:
                return opt, rationale

        # Loose contains matching as fallback
        for opt in options:
            if opt.lower() in choice_lower or choice_lower in opt.lower():
                return opt, rationale

        print(
            f"Warning: LLM returned invalid choice '{choice_raw}' for {preference_name}. Using '{options[0]}'",
            file=sys.stderr,
        )
        return options[0], rationale

    except Exception as e:
        print(f"Error calling LLM for {preference_name}: {e}. Using first option.", file=sys.stderr)
        return options[0], ""


def apply_hard_rules(preferences: Dict[str, str], meal_info: Dict[str, any]) -> Dict[str, str]:
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
    variation_level: float = 0.3  # How much U^pref changes between windows (0.0-1.0)


def generate_long_term_preferences_llm(
    client: OpenAI,
    physical_profile: str,
    window_label: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Generate long-term preferences for a window using LLM."""
    
    prompt_path = Path(__file__).parent / "prompts" / "long_term_preferences_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(
        physical_profile=physical_profile,
        window_label=window_label,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at generating safe and appropriate default preferences for robotic feeding assistance based on user physical capabilities. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1000,
        )
        
        result = response.choices[0].message.content.strip()
        # Try to extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        prefs = json.loads(result)
        return prefs
    
    except Exception as e:
        print(f"Error generating long-term preferences: {e}. Using defaults.", file=sys.stderr)
        # Return sensible defaults
        return {
            "microwave_time": "2 min",
            "occlusion_relevance": "minimize front occlusion",
            "robot_speed": "medium",
            "skewering_axis": "perpendicular to major axis",
            "transfer_mode": "outside mouth transfer",
            "outside_mouth_distance": "medium",
            "robot_ready_cue": "speech",
            "bite_initiation_feeding": "open mouth",
            "bite_initiation_drinking": "open mouth",
            "bite_initiation_wiping": "open mouth",
            "robot_bite_available_cue": "speech",
            "bite_completion_feeding": "perception",
            "bite_completion_drinking": "perception",
            "bite_completion_wiping": "perception",
            "web_interface_confirmation": "yes",
            "retract_between_bites": "no",
            "bite_dipping_preference": "do not dip",
            "amount_to_dip": "more",
            "wait_before_autocontinue_seconds": "100",
        }


def run_deployment(
    cfg: DeploymentConfig,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "out",
) -> str:
    """Generate a 30-day deployment dataset using LLM and save as JSON."""
    
    # Get API key
    if api_key is None:
        api_key = OPENAI_API_KEY
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
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
    
    # Generate long-term preferences for window 1
    print(f"Generating long-term preferences for days 1-15...")
    u_pref_window1 = generate_long_term_preferences_llm(
        client, cfg.physical_capability_profile, "days 1-15", model
    )
    
    # Generate long-term preferences for window 2 with variation
    print(f"Generating long-term preferences for days 16-30 (with variation level {cfg.variation_level})...")
    u_pref_window2_base = generate_long_term_preferences_llm(
        client, cfg.physical_capability_profile, "days 16-30", model
    )
    u_pref_window2 = generate_long_term_preference_variation(
        u_pref_window1, cfg.variation_level, rng
    )
    # Merge: use window2_base but apply variation from window1
    for k in u_pref_window2:
        if rng.random() < 0.5:  # 50% chance to use window2_base instead
            u_pref_window2[k] = u_pref_window2_base.get(k, u_pref_window2[k])
    
    # Collect all days into an in-memory structure, then dump to JSON.
    days: List[Dict[str, object]] = []

    for day in range(1, cfg.days + 1):
        print(f"\n=== Day {day}/{cfg.days} ===")

        # Determine which window and U^pref is in effect
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

        # Generate preferences one by one using LLM (19 dimensions)
        pref_choices: Dict[str, str] = {}
        pref_rationales: Dict[str, str] = {}
        total = len(PREFERENCE_BUNDLE)  # 19 preferences

        for idx, (field, label, options) in enumerate(PREFERENCE_BUNDLE, 1):
            print(f"  [{idx}/{total}] Generating {label}...", end=" ", flush=True)
            long_term_val = u_current.get(field, "")

            choice, rationale = generate_preference_with_llm(
                client,
                field,
                label,
                options,
                cfg.physical_capability_profile,
                long_term_val,
                context,
                meal_info,
                affective_state,
                pref_choices,
                idx,
                total,
                model,
            )
            pref_choices[field] = choice
            pref_rationales[field] = rationale
            print(f"→ {choice}")

        # Apply hard rules
        pref_choices = apply_hard_rules(pref_choices, meal_info)

        # Add bite_ordering_preference (not in user's 19-item list, generate randomly)
        ordering_choice = rng.choice(
            ["alternate X and Y", "start with X then Y", "start with Y then X"]
        )
        pref_choices["bite_ordering_preference"] = ordering_choice
        pref_rationales.setdefault("bite_ordering_preference", "")

        day_record: Dict[str, object] = {
            "day": day,
            "window": window_num,
            "context": context,
            "preferences": {
                name: {
                    "choice": pref_choices.get(name, ""),
                    "rationale": pref_rationales.get(name, ""),
                }
                for name in pref_choices.keys()
            },
        }
        days.append(day_record)

        print(f"✓ Day {day} generated")

    # Top-level JSON structure
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
        # Slow latent preferences U^pref for each window
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
    parser = argparse.ArgumentParser(
        description="LLM-based dataset generator (30-day deployments)."
    )
    parser.add_argument("--user", required=True, help="User name (e.g., User1)")
    parser.add_argument("--deployment-id", default="dep1", help="Deployment identifier")
    parser.add_argument(
        "--physical-profile",
        required=True,
        choices=list(PHYSICAL_CAPABILITY_PROFILES.keys()),
        help="Physical capability profile key"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    parser.add_argument(
        "--variation-level",
        type=float,
        default=0.3,
        help="U^pref variation level between windows (0.0-1.0, default: 0.3)"
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
        out_path = run_deployment(
            cfg,
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
