"""
Generate a structured “user preference encoding” for a robot-assisted mealtime system
using an LLM, conditioned on a care recipient’s physical functioning profile.

This script:
- Loads a prompt template from `prompts/user_preference_encoding_prompt.txt` and fills
  in `{physical_profile}`.
- Calls the OpenAI Chat Completions API to produce a JSON object describing the user’s
  default preferences and how those preferences vary across dining context, food type,
  and affective state.
- Enforces a STRICT output schema:
    - The output must be valid JSON (no extra text/markdown).
    - The JSON must contain exactly the expected preference fields (no missing/extra keys).
    - Each field must be an object with exactly:
        {"default": <string>, "user_tendencies": <non-empty string>}
    - Each "default" must be one of the allowed options in `PREFERENCE_SCHEMA`.

If the LLM output violates the schema (e.g., truncated JSON, wrong keys, invalid option),
the script raises an error rather than silently falling back to defaults.

Usage:
  python generate_user_preference_encoding.py --physical-profile <profile_key> [--model <model_name>]

Requirements:
- `OPENAI_API_KEY` must be set in the environment (or add wiring to use --api-key).
- `openai` Python package installed.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
    sys.exit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo", etc.

from feeding_deployment.preference_learning.config import MEAL_STRUCTURE, MEALS, SETTINGS, TIMES_OF_DAY, AFFECTIVE_STATES, PREFERENCE_BUNDLE, PHYSICAL_CAPABILITY_PROFILES

import json
import re
from pathlib import Path
from typing import Any, Dict, List


# Canonical schema (fields + allowed options) based on your final prompt
PREFERENCE_SCHEMA: Dict[str, List[str]] = {
    "microwave_time": ["no microwave", "1 min", "2 min", "3 min"],
    "occlusion_relevance": ["minimize left occlusion", "minimize front occlusion", "do not consider occlusion"],
    "robot_speed": ["slow", "medium", "fast"],
    "skewering_axis": ["parallel to major axis", "perpendicular to major axis"],
    "web_interface_confirmation": ["yes", "no"],
    "transfer_mode": ["outside mouth transfer", "inside mouth transfer"],
    "outside_mouth_distance": ["near", "medium", "far"],
    "robot_ready_cue": ["speech", "LED", "speech + LED", "no cue"],
    "bite_initiation_feeding": ["open mouth", "button", "autocontinue"],
    "bite_initiation_drinking": ["open mouth", "button", "autocontinue"],
    "bite_initiation_wiping": ["open mouth", "button", "autocontinue"],
    "robot_ready_for_transfer_completion_cue": ["speech", "LED", "speech + LED", "no cue"],
    "bite_completion_feeding": ["perception", "button", "autocontinue"],
    "bite_completion_drinking": ["perception", "button", "autocontinue"],
    "bite_completion_wiping": ["perception", "button", "autocontinue"],
    "retract_between_bites": ["yes", "no"],
    "bite_dipping_preference": ["dip food in sauce", "do not dip"],
    "amount_to_dip": ["not applicable", "less", "more"],
    # Keep as strings to match your prompt exactly
    "wait_before_autocontinue_seconds": ["10", "100", "1000"],
}


def _extract_json_object(text: str) -> str:
    """
    Extract a JSON object from model output.
    Accepts:
      - raw JSON
      - ```json ... ```
      - ``` ... ```
    Rejects if no JSON object is found.
    """
    text = (text or "").strip()

    # Prefer fenced JSON code block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Any fenced code block
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: first {...} region
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    raise ValueError("No JSON object found in model output.")


def _validate_preferences_strict(prefs: Any) -> Dict[str, Dict[str, str]]:
    """
    Enforce STRICT schema:

    prefs must be:
      Dict[field_name, Dict{"default": <allowed>, "user_tendencies": <non-empty str>}]

    - No missing fields
    - No extra fields
    - No old flat format
    - default must be one of allowed options
    """
    if not isinstance(prefs, dict):
        raise TypeError(f"Expected a JSON object (dict). Got: {type(prefs).__name__}")

    expected_fields = set(PREFERENCE_SCHEMA.keys())
    got_fields = set(prefs.keys())

    missing = sorted(expected_fields - got_fields)
    extra = sorted(got_fields - expected_fields)

    if missing:
        raise KeyError(f"Missing fields in LLM output: {missing}")
    if extra:
        raise KeyError(f"Unexpected extra fields in LLM output: {extra}")

    out: Dict[str, Dict[str, str]] = {}

    for field, allowed in PREFERENCE_SCHEMA.items():
        entry = prefs[field]
        if not isinstance(entry, dict):
            raise TypeError(
                f'Field "{field}" must be an object with keys "default" and "user_tendencies". '
                f"Got: {type(entry).__name__}"
            )

        if set(entry.keys()) != {"default", "user_tendencies"}:
            raise KeyError(
                f'Field "{field}" must have exactly keys ["default", "user_tendencies"]. Got keys: {sorted(entry.keys())}'
            )

        default_val = entry["default"]
        tendencies = entry["user_tendencies"]

        if not isinstance(default_val, str):
            raise TypeError(f'Field "{field}.default" must be a string. Got: {type(default_val).__name__}')
        if default_val not in allowed:
            raise ValueError(
                f'Field "{field}.default" must be one of {allowed}. Got: {default_val!r}'
            )

        if not isinstance(tendencies, str) or not tendencies.strip():
            raise ValueError(f'Field "{field}.user_tendencies" must be a non-empty string.')

        out[field] = {"default": default_val, "user_tendencies": tendencies.strip()}

    return out


def generate_user_preference_encoding_llm(
    client,
    physical_profile: str,
    model: str = "gpt-4o",
) -> Dict[str, Dict[str, str]]:
    """
    Generate user preference encoding for a physical profile using an LLM.

    STRICT behavior:
      - Requires valid JSON object output
      - Requires exact schema, no missing/extra fields
      - Requires each default to be in allowed options
      - Raises on any violation

    Returns:
      Dict[field_name, {"default": <allowed option>, "user_tendencies": <string>}]
    """
    prompt_path = Path(__file__).parent / "prompts" / "user_preference_encoding_prompt.txt"
    template = prompt_path.read_text(encoding="utf-8")

    prompt = template.format(
        physical_profile=physical_profile,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate structured user preference encodings for a robot-assisted mealtime system. "
                    "Return ONLY valid JSON (no markdown, no extra text). "
                    'Schema: dict[field] = {"default": <one allowed option>, "user_tendencies": <non-empty string>}. '
                    "You must include all fields and no extra fields."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=5000,
        # If your SDK supports it, this strongly enforces JSON
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    print("Raw LLM output:")
    print(raw)
    json_str = _extract_json_object(raw)
    parsed = json.loads(json_str)

    return _validate_preferences_strict(parsed)

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based dataset generator (30-day deployments)."
    )
    parser.add_argument(
        "--physical-profile",
        required=True,
        choices=list(PHYSICAL_CAPABILITY_PROFILES.keys()),
        help="Physical capability profile key"
    )
    parser.add_argument("--api-key", help="OpenAI API key (overrides env/config)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    
    physical_profile = PHYSICAL_CAPABILITY_PROFILES[args.physical_profile]
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Generate user preference encoding for the physical profile 
    print(f"Generating user preference encoding for physical profile: {args.physical_profile}")
    user_preference_encoding = generate_user_preference_encoding_llm(
        client, physical_profile, args.model
    )
    
    print("Generated user preference encoding:")
    print(json.dumps(user_preference_encoding, indent=2))

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
