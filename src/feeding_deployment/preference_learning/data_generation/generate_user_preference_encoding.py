"""
Generate a structured “user preference encoding” for a robot-assisted mealtime system
using an LLM, conditioned on a care recipient’s physical capability profile.

This script:
- Builds the prompt via `get_user_preference_encoding_prompt(...)` (recommended) and
  validates the output against `PREFERENCE_BUNDLE` (single source of truth).
- Calls the OpenAI Chat Completions API to produce a JSON object describing the user’s
  default preferences and how those preferences vary across dining context, food type,
  and affective state.
- Enforces a STRICT output schema:
    - The output must be valid JSON (no extra text/markdown).
    - The JSON must contain exactly the expected preference fields (no missing/extra keys).
    - Each field must be an object with exactly:
        {"default": <string>, "user_tendencies": <non-empty string>}
    - Each "default" must be one of the allowed options from `PREFERENCE_BUNDLE`.

If the LLM output violates the schema (e.g., truncated JSON, wrong keys, invalid option),
the script raises an error rather than silently falling back to defaults.

Usage:
  python generate_user_preference_encoding.py --physical-profile <profile_key> [--model <model_name>]

Requirements:
- `OPENAI_API_KEY` must be set in the environment (or pass --api-key).
- `openai` Python package installed.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
    sys.exit(1)

from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.config.physical_capabilities import PHYSICAL_CAPABILITY_PROFILES
from feeding_deployment.preference_learning.data_generation.prompts.user_preference_encoding import get_user_preference_encoding_prompt

DEFAULT_MODEL = "gpt-4o"


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
    Validate LLM output against PREFERENCE_BUNDLE (single source of truth).

    Expected schema:
      prefs[field] = {"default": <allowed option>, "user_tendencies": <non-empty str>}

    Strict checks:
      - Must be a dict
      - Must contain exactly all fields in PREFERENCE_BUNDLE (no missing/extra)
      - Each field must be an object with exactly keys {"default","user_tendencies"}
      - default must be one of dim.options
      - user_tendencies must be a non-empty string
    """
    if not isinstance(prefs, dict):
        raise TypeError(f"Expected JSON object (dict). Got: {type(prefs).__name__}")

    expected_fields = {dim.field for dim in PREFERENCE_BUNDLE}
    got_fields = set(prefs.keys())

    missing = sorted(expected_fields - got_fields)
    extra = sorted(got_fields - expected_fields)

    if missing:
        raise KeyError(f"Missing fields in LLM output: {missing}")
    if extra:
        raise KeyError(f"Unexpected extra fields in LLM output: {extra}")

    out: Dict[str, Dict[str, str]] = {}

    for dim in PREFERENCE_BUNDLE:
        field = dim.field
        allowed = dim.options

        entry = prefs[field]
        if not isinstance(entry, dict):
            raise TypeError(
                f'Field "{field}" must be an object with keys "default" and "user_tendencies". '
                f"Got: {type(entry).__name__}"
            )

        if set(entry.keys()) != {"default", "user_tendencies"}:
            raise KeyError(
                f'Field "{field}" must have exactly keys ["default", "user_tendencies"]. '
                f"Got keys: {sorted(entry.keys())}"
            )

        default_val = entry["default"]
        tendencies = entry["user_tendencies"]
        
        if isinstance(default_val, int):
            default_val = str(default_val)  # Convert integers to strings for leniency

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
    client: OpenAI,
    physical_profile_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Dict[str, str]]:
    """
    Generate a user preference encoding for the given physical capability profile key.

    STRICT behavior:
      - Requires valid JSON object output
      - Requires exact schema, no missing/extra fields
      - Requires each default to be in allowed options (from PREFERENCE_BUNDLE)
      - Raises on any violation
    """
    prompt = get_user_preference_encoding_prompt(physical_profile_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate structured user preference encodings for a robot-assisted mealtime system.\n"
                    "Return ONLY valid JSON (no markdown, no extra text).\n"
                    'Schema: dict[field] = {"default": <one allowed option>, "user_tendencies": <non-empty string>}.\n'
                    f"You must include all fields and no extra fields.\n"
                    f"Fields (exact): {[dim.field for dim in PREFERENCE_BUNDLE]}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        # Slight randomness is fine for "personality", but keep it controlled.
        temperature=0.5,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    print("Raw LLM output:")
    print(raw)

    json_str = _extract_json_object(raw)
    parsed = json.loads(json_str)

    return _validate_preferences_strict(parsed)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a user preference encoding via an LLM.")
    parser.add_argument(
        "--physical-profile",
        required=True,
        choices=[p.label for p in PHYSICAL_CAPABILITY_PROFILES],
        help="Physical capability profile key",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Set it in the environment or pass --api-key.")

    client = OpenAI(api_key=api_key)

    print(f"Generating user preference encoding for physical profile: {args.physical_profile}")
    enc = generate_user_preference_encoding_llm(
        client=client,
        physical_profile_key=args.physical_profile,
        model=args.model,
    )

    print("Generated user preference encoding:")
    print(json.dumps(enc, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))