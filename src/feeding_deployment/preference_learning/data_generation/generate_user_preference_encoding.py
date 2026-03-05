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
from feeding_deployment.preference_learning.data_generation.prompts.user_preference_encoding import (
    get_user_preference_encoding_prompt,
)

DEFAULT_MODEL = "gpt-5.4"


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    raise ValueError("No JSON object found in model output.")


def _validate_preferences_strict(prefs: Any) -> Dict[str, Dict[str, str]]:
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
            default_val = str(default_val)

        if not isinstance(default_val, str):
            raise TypeError(f'Field "{field}.default" must be a string. Got: {type(default_val).__name__}')
        if default_val not in allowed:
            raise ValueError(f'Field "{field}.default" must be one of {allowed}. Got: {default_val!r}')

        if not isinstance(tendencies, str) or not tendencies.strip():
            raise ValueError(f'Field "{field}.user_tendencies" must be a non-empty string.')

        out[field] = {"default": default_val, "user_tendencies": tendencies.strip()}

    return out


def generate_user_preference_encoding_llm(
    client: OpenAI,
    physical_profile_key: str,
    model: str = DEFAULT_MODEL,
    print_raw: bool = False,
) -> Dict[str, Dict[str, str]]:
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
                    "You must include all fields and no extra fields.\n"
                    f"Fields (exact): {[dim.field for dim in PREFERENCE_BUNDLE]}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_completion_tokens=5000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    if print_raw:
        print("Raw LLM output:")
        print(raw)

    json_str = _extract_json_object(raw)
    parsed = json.loads(json_str)
    return _validate_preferences_strict(parsed)


def _ensure_output_dir(path: str) -> str:
    out_dir = os.path.abspath(path)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _profile_labels() -> List[str]:
    # Preserve ordering as defined in PHYSICAL_CAPABILITY_PROFILES
    return [p.label for p in PHYSICAL_CAPABILITY_PROFILES]


def _write_user_file(
    output_dir: str,
    user_idx: int,
    physical_profile_key: str,
    encoding: Dict[str, Dict[str, str]],
) -> str:

    filename = f"user_{user_idx}__profile_{physical_profile_key}.json"
    filepath = os.path.join(output_dir, filename)

    payload = {
        "user_index": user_idx,
        "physical_profile": physical_profile_key,
        "encoding": encoding,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return filepath


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate user preference encodings via an LLM.")
    parser.add_argument(
        "--physical-profile",
        required=False,
        choices=_profile_labels(),
        help="Physical capability profile key. If omitted, cycle through all profiles round-robin.",
    )
    parser.add_argument("--num-users", type=int, required=True, help="Number of user encodings to generate (>= 1).")
    parser.add_argument("--output-dir", required=True, help="Directory to write output files into.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides env var)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model output for debugging.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.num_users < 1:
        raise ValueError("--num-users must be >= 1")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Set it in the environment or pass --api-key.")

    output_dir = _ensure_output_dir(args.output_dir)
    client = OpenAI(api_key=api_key)

    all_profiles = _profile_labels()
    if not all_profiles:
        raise ValueError("No physical capability profiles are available in PHYSICAL_CAPABILITY_PROFILES.")

    print(f"Model: {args.model}")
    print(f"Output dir: {output_dir}")
    print(f"Num users: {args.num_users}")

    if args.physical_profile:
        print(f"Using single physical profile for all users: {args.physical_profile}")
    else:
        print(f"Cycling through profiles round-robin: {all_profiles}")

    for i in range(1, args.num_users + 1):
        if args.physical_profile:
            profile_key = args.physical_profile
        else:
            # Round-robin: user 1 -> profiles[0], user 2 -> profiles[1], ...
            profile_key = all_profiles[(i - 1) % len(all_profiles)]

        print(f"\n=== Generating user {i}/{args.num_users} (profile={profile_key}) ===")
        enc = generate_user_preference_encoding_llm(
            client=client,
            physical_profile_key=profile_key,
            model=args.model,
            print_raw=args.print_raw,
        )
        path = _write_user_file(output_dir, i, profile_key, enc)
        print(f"Wrote: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))