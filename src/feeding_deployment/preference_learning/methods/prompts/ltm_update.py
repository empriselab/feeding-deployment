from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from feeding_deployment.preference_learning.config.physical_capabilities import (
    PHYSICAL_CAPABILITY_PROFILES,
)
from feeding_deployment.preference_learning.data_generation.prompts.system_description import (
    get_system_description_prompt,
)
from feeding_deployment.preference_learning.methods.utils import (
    _episode_text,
    _extract_truth_bundle,
)

LTM_UPDATE_PROMPT_PATH = Path(__file__).parent / "ltm_update.txt"

_PHYSICAL_CAPABILITY_BY_LABEL = {p.label: p for p in PHYSICAL_CAPABILITY_PROFILES}


def get_ltm_update_prompt(
    physical_profile_label: str,
    previous_ltm_summary: str,
    new_episode: str,
    *,
    physical_profile_description: str | None = None,
) -> str:
    template = LTM_UPDATE_PROMPT_PATH.read_text(encoding="utf-8")
    system_description = get_system_description_prompt()

    if physical_profile_description is not None:
        desc = physical_profile_description.strip()
        if not desc:
            raise ValueError("physical_profile_description is empty")
    else:
        if physical_profile_label not in _PHYSICAL_CAPABILITY_BY_LABEL:
            valid = ", ".join(sorted(_PHYSICAL_CAPABILITY_BY_LABEL.keys()))
            raise ValueError(f"Unknown physical_profile_label={physical_profile_label!r}. Valid: {valid}")
        desc = _PHYSICAL_CAPABILITY_BY_LABEL[physical_profile_label].description

    return template.format(
        system_description=system_description,
        physical_profile=desc,
        previous_ltm_summary=previous_ltm_summary,
        new_episode=new_episode,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print the LTM update prompt for one day from a dataset.")
    p.add_argument("--data-file", required=True, help="Path to one JSON dataset file.")
    p.add_argument(
        "--day",
        type=int,
        required=True,
        help="Day number to render (e.g., 1, 10, 30). This matches the 'day' field in the JSON.",
    )
    p.add_argument(
        "--previous-summary",
        default="N/A",
        help='Previous LTM summary. Use "N/A" or empty to simulate cold-start.',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    user = str(data.get("user", "unknown"))
    physical_profile_label = str(data.get("physical_profile_label", "")).strip()
    days: List[Dict[str, Any]] = list(data.get("days", []))

    # Find record whose "day" field matches args.day
    day_rec = next((r for r in days if int(r.get("day", -1)) == args.day), None)
    if day_rec is None:
        available = sorted({int(r.get("day", 0)) for r in days})
        raise SystemExit(f"Day {args.day} not found in dataset. Available days: {available}")

    day = int(day_rec.get("day", 0))
    print(f"User={user} | physical_profile_label={physical_profile_label} | day={day}", flush=True)

    ctx = day_rec.get("context", {}) or {}
    truth = _extract_truth_bundle(day_rec)
    ep_txt = _episode_text(day, ctx, truth)

    prompt = get_ltm_update_prompt(
        physical_profile_label=physical_profile_label,
        previous_ltm_summary=args.previous_summary,
        new_episode=ep_txt,
    )

    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())