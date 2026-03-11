from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

from openai import OpenAI

from feeding_deployment.preference_learning.methods.prompts.ltm_update import (
    get_ltm_update_prompt,
)
from feeding_deployment.preference_learning.methods.utils import (
    _retry_on_rate_limit,
    _resolve_api_key,
    _episode_text,
    _extract_truth_bundle,
)


class LongTermMemoryModel:
    """
    Stateful LTM that updates EVERY meal (online).
    """

    def __init__(self, physical_profile_label: str, client: OpenAI, chat_model: str, retry_fn, logs_dir: Path = None) -> None:
        self.client = client
        self.chat_model = chat_model
        self._retry = retry_fn
        self.logs_dir = logs_dir

        self.physical_profile_label = physical_profile_label
        self._ltm_summary: str = ""
        self._initialized: bool = False

    def get_ltm(self) -> str:
        return self._ltm_summary

    def add_episode(self, episode_text: str) -> None:
        # Single-prompt behavior: if we don't have a previous summary, use N/A (cold-start).
        previous = self._ltm_summary.strip() if self._initialized and self._ltm_summary.strip() else "N/A"

        prompt = get_ltm_update_prompt(
            physical_profile_label=self.physical_profile_label,
            previous_ltm_summary=previous,
            new_episode=episode_text,
        )

        def _call() -> Any:
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You write concise, faithful preference summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=15000,
            )

        resp = self._retry(_call)
        new_ltm_summary = (resp.choices[0].message.content or "").strip()
    
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                log_file.write_text(f"===PROMPT===\n{prompt}\n\n===RESPONSE===\n{json.dumps(json.loads(new_ltm_summary), indent=2)}", encoding="utf-8")
            except Exception:
                log_file.write_text(f"===PROMPT===\n{prompt}\n\n===RESPONSE===\nFailed to parse response as JSON. Raw response:\n{resp}", encoding="utf-8")

        
        # check if the summary is a valid JSON object, if not, print a warning and the whole summary for debugging
        try:
            json.loads(new_ltm_summary)
            self._ltm_summary = new_ltm_summary
        except Exception:
            print(f"Warning: LTM summary is not valid JSON so skipping. Response from llm:\n{resp}\n", flush=True)
        self._initialized = True


def parse_args_ltm() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and log LTM summaries day-by-day from a dataset file.")
    p.add_argument("--data-file", required=True, help="Path to one JSON dataset file.")
    p.add_argument("--log-dir", required=True, help="Directory to write logs (will be created).")
    p.add_argument("--openai-model", default="gpt-5.4", help="Chat model for LTM (default: gpt-5.4).")
    p.add_argument("--api-key", default="", help="OpenAI API key (optional).")
    return p.parse_args()


def main() -> int:
    args = parse_args_ltm()

    os.makedirs(args.log_dir, exist_ok=True)
    updates_dir = os.path.join(args.log_dir, "ltm_updates")
    os.makedirs(updates_dir, exist_ok=True)

    checkpoints_path = os.path.join(args.log_dir, "ltm_checkpoints.json")

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    physical_profile_label = str(data.get("physical_profile_label", "")).strip()
    if not physical_profile_label:
        raise SystemExit("Dataset missing required field: 'physical_profile_label'")

    days: List[Dict[str, Any]] = list(data.get("days", []))
    days.sort(key=lambda r: int(r.get("day", 0)))

    client = OpenAI(api_key=_resolve_api_key(args.api_key))
    ltm = LongTermMemoryModel(
        client=client,
        chat_model=args.openai_model,
        retry_fn=_retry_on_rate_limit,
    )
    ltm.reset(physical_profile_label)

    wanted = {10, 20, 30}
    seen: set[int] = set()

    checkpoints: Dict[str, Any] = {
        "dataset_file": args.data_file,
        "openai_model": args.openai_model,
        "checkpoints": {},
    }

    for day_rec in days:
        day = int(day_rec.get("day", 0))
        print(f"Processing day {day} ...", flush=True)

        ctx = day_rec.get("context", {}) or {}
        truth = _extract_truth_bundle(day_rec)
        ep_txt = _episode_text(day, ctx, truth)

        ltm.add_episode(ep_txt)
        summary_raw = ltm.get_ltm()

        # Log summary as JSON if possible (but keep raw string if parsing fails).
        try:
            summary_logged: Any = json.loads(summary_raw)
        except Exception:
            print(f"Warning: LTM summary for day {day} is not valid JSON. Logging raw string.", flush=True)
            summary_logged = summary_raw

        record = {
            "day": day,
            "context": {
                "meal": ctx.get("meal"),
                "setting": ctx.get("setting"),
                "time_of_day": ctx.get("time_of_day"),
                "transient_affective_state": ctx.get("transient_affective_state"),
            },
            "episode_text": ep_txt,
            "ltm_summary": summary_logged,
        }

        # One file per update (nice to browse)
        out_path = os.path.join(updates_dir, f"day_{day:04d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if day in wanted:
            seen.add(day)
            checkpoints["checkpoints"][str(day)] = summary_logged
            print(f"\n=== LTM checkpoint day {day} ===\n{summary_logged}\n", flush=True)

    missing = sorted(wanted - seen)
    if missing:
        print(f"Warning: dataset missing checkpoint days: {missing}", flush=True)

    with open(checkpoints_path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)

    print(f"Wrote per-day LTM update files to: {updates_dir}")
    print(f"Wrote checkpoint summaries: {checkpoints_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())