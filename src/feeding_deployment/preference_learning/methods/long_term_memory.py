from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from feeding_deployment.preference_learning.methods.prompts.ltm_summary_cold_start import (
    get_ltm_cold_start_prompt,
)
from feeding_deployment.preference_learning.methods.prompts.ltm_summary_update import (
    get_ltm_update_prompt,
)

from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE as _PREF_BUNDLE_DIMS,
)
import feeding_deployment.preference_learning.config as root_config  # type: ignore

from feeding_deployment.preference_learning.methods.utils import (
    _retry_on_rate_limit,
    _resolve_api_key,
    _episode_text,
    _extract_truth_bundle,
)


def _build_options_block() -> str:
    pref_fields = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]
    pref_options = {name: opts for (name, _, opts) in root_config.PREFERENCE_BUNDLE}
    pref_descriptions = {dim.field: dim.description for dim in _PREF_BUNDLE_DIMS}

    lines: List[str] = []
    for field in pref_fields:
        opts_str = ", ".join(pref_options[field])
        desc = pref_descriptions.get(field, "").strip()
        if desc:
            lines.append(f"- {field}: [{opts_str}] — {desc}")
        else:
            lines.append(f"- {field}: [{opts_str}]")
    return "\n".join(lines)


class LongTermMemoryModel:
    """
    Stateful LTM that updates EVERY meal (online).
    """

    def __init__(
        self,
        client: OpenAI,
        chat_model: str,
        cache_path: Path,
        retry_fn,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.cache_path = cache_path
        self._retry = retry_fn

        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache: Dict[str, str] = json.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

        self.user: str = ""
        self.physical_profile: str = ""
        self._ltm_summary: str = ""
        self._initialized: bool = False

    def reset(self, user: str, physical_profile: str) -> None:
        self.user = user
        self.physical_profile = physical_profile
        self._ltm_summary = ""
        self._initialized = False

    def get_ltm(self) -> str:
        return self._ltm_summary

    def ensure_initialized(self) -> bool:
        if self._initialized and self._ltm_summary.strip():
            return False

        opts_block = _build_options_block()
        prompt = get_ltm_cold_start_prompt(
            physical_profile=self.physical_profile,
            options_block=opts_block,
        )

        key = self._cold_start_cache_key()
        if key in self._cache:
            self._ltm_summary = self._cache[key]
            self._initialized = True
            return False

        def _call() -> Any:
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You write concise, faithful preference summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=350,
            )

        resp = self._retry(_call)
        self._ltm_summary = (resp.choices[0].message.content or "").strip()
        self._cache[key] = self._ltm_summary
        self._initialized = True
        return True

    def add_episode(self, episode_text: str) -> tuple[bool, str]:
        self.ensure_initialized()

        key = self._update_cache_key(self._ltm_summary, episode_text)
        if key in self._cache:
            self._ltm_summary = self._cache[key]
            return True, episode_text

        opts_block = _build_options_block()
        prompt = get_ltm_update_prompt(
            physical_profile=self.physical_profile,
            previous_ltm_summary=self._ltm_summary,
            new_episodes_block=episode_text,
            options_block=opts_block,
        )

        def _call() -> Any:
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You write concise, faithful preference summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
            )

        resp = self._retry(_call)
        self._ltm_summary = (resp.choices[0].message.content or "").strip()
        self._cache[key] = self._ltm_summary
        return False, episode_text

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _cold_start_cache_key(self) -> str:
        prof_hash = hashlib.sha256(self.physical_profile.encode("utf-8")).hexdigest()[:12]
        # include model so caches don't collide across models
        return f"{self.user}|model={self.chat_model}|cold|prof_{prof_hash}"

    def _update_cache_key(self, prev_summary: str, episode_text: str) -> str:
        prev_hash = hashlib.sha256(prev_summary.encode("utf-8")).hexdigest()[:12] if prev_summary.strip() else "cold"
        ep_hash = hashlib.sha256(episode_text.encode("utf-8")).hexdigest()[:12]
        return f"{self.user}|model={self.chat_model}|prev_{prev_hash}|ep_{ep_hash}"


def parse_args_ltm() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and log LTM summaries day-by-day from a dataset file.")
    p.add_argument("--data-file", required=True, help="Path to one JSON dataset file.")
    p.add_argument("--log-dir", required=True, help="Directory to write logs (will be created).")
    p.add_argument("--openai-model", default="gpt-4o", help="Chat model for LTM (default: gpt-4o).")
    p.add_argument("--api-key", default="", help="OpenAI API key (optional).")
    p.add_argument(
        "--cache-file",
        default=".cache/llm_memory_eval/ltm_summaries.json",
        help="Path to LTM cache JSON (default: .cache/llm_memory_eval/ltm_summaries.json).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args_ltm()

    os.makedirs(args.log_dir, exist_ok=True)
    per_day_log_path = os.path.join(args.log_dir, "ltm_day_by_day.jsonl")
    checkpoints_path = os.path.join(args.log_dir, "ltm_checkpoints.json")

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    user = str(data.get("user", "unknown"))
    physical_profile = str(data.get("physical_capability_profile", "")).strip()
    days: List[Dict[str, Any]] = list(data.get("days", []))
    days.sort(key=lambda r: int(r.get("day", 0)))

    client = OpenAI(api_key=_resolve_api_key(args.api_key))

    ltm = LongTermMemoryModel(
        client=client,
        chat_model=args.openai_model,
        cache_path=Path(args.cache_file),
        retry_fn=_retry_on_rate_limit,
    )
    ltm.reset(user, physical_profile)
    ltm.ensure_initialized()

    wanted = {10, 20, 30}
    seen: set[int] = set()

    checkpoints: Dict[str, Any] = {
        "user": user,
        "dataset_file": args.data_file,
        "openai_model": args.openai_model,
        "checkpoints": {},
    }

    with open(per_day_log_path, "w", encoding="utf-8") as per_day_f:
        for day_rec in days:
            day = int(day_rec.get("day", 0))
            ctx = day_rec.get("context", {}) or {}
            truth = _extract_truth_bundle(day_rec)
            ep_txt = _episode_text(day, ctx, truth)

            used_cache, _ = ltm.add_episode(ep_txt)
            summary = ltm.get_ltm()

            per_day_f.write(
                json.dumps(
                    {
                        "user": user,
                        "day": day,
                        "used_cache": used_cache,
                        "context": {
                            "meal": ctx.get("meal"),
                            "setting": ctx.get("setting"),
                            "time_of_day": ctx.get("time_of_day"),
                            "transient_affective_state": ctx.get("transient_affective_state"),
                        },
                        "episode_text": ep_txt,
                        "ltm_summary": summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if day in wanted:
                seen.add(day)
                checkpoints["checkpoints"][str(day)] = summary
                print(f"\n=== LTM checkpoint day {day} ===\n{summary}\n", flush=True)

    missing = sorted(wanted - seen)
    if missing:
        print(f"Warning: dataset missing checkpoint days: {missing}", flush=True)

    with open(checkpoints_path, "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)

    ltm.flush()

    print(f"Wrote per-day LTM log: {per_day_log_path}")
    print(f"Wrote checkpoint summaries: {checkpoints_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())