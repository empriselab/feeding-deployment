from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from openai import OpenAI

import feeding_deployment.preference_learning.config as root_config  # type: ignore
from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE as _PREF_BUNDLE_DIMS,
)
from feeding_deployment.preference_learning.methods.episodic_memory import EpisodicMemoryModel
from feeding_deployment.preference_learning.methods.long_term_memory import LongTermMemoryModel
from feeding_deployment.preference_learning.methods.prompts.bundle_prediction import (
    get_bundle_prediction_prompt,
)
from feeding_deployment.preference_learning.methods.utils import _episode_text, PREF_FIELDS

PREF_OPTIONS: Dict[str, List[str]] = {name: opts for (name, _, opts) in root_config.PREFERENCE_BUNDLE}
PREF_DESCRIPTIONS: Dict[str, str] = {dim.field: dim.description for dim in _PREF_BUNDLE_DIMS}


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if "```" not in s:
        return s
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    return s.split("```", 1)[1].split("```", 1)[0].strip()


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _get_meal_info(meal: str) -> Dict[str, Any]:
    known_meal = meal in root_config.MEAL_STRUCTURE
    info = root_config.MEAL_STRUCTURE.get(meal, {})
    dippable = list(info.get("dippable_items", []) or [])
    sauces = list(info.get("sauces", []) or [])
    return {
        "known_meal": known_meal,
        "dippable_items": dippable,
        "sauces": sauces,
        "has_dippable": len(dippable) > 0,
        "has_sauce": len(sauces) > 0,
    }


def _apply_hard_rules(prefs: Dict[str, str], meal: str, corrected: Dict[str, str]) -> Dict[str, str]:
    out = dict(prefs)
    meal_info = _get_meal_info(meal)

    if out.get("transfer_mode") == "inside mouth transfer" and "outside_mouth_distance" not in corrected:
        out["outside_mouth_distance"] = "not applicable"

    if (
        "bite_dipping_preference" not in corrected
        and meal_info.get("known_meal", False)
        and ((not meal_info["has_dippable"]) or (not meal_info["has_sauce"]))
    ):
        out["bite_dipping_preference"] = "do not dip"

    return out

def _build_options_block() -> str:
    """
    Bundle-prediction options block. Match the format used in your prompt-printer script:
    - field: [opt1, opt2, ...]
    """
    lines: List[str] = []
    for field in PREF_FIELDS:
        opts = PREF_OPTIONS[field]
        lines.append(f"- {field}: [{', '.join(opts)}]")
    return "\n".join(lines)


def _format_corrected_block(corrected: Dict[str, str]) -> str:
    """
    Match the format used by your get_bundle_prediction_prompt printer:
    each line is key=value.
    """
    if not corrected:
        return "(none)"
    return "\n".join(f"{k}={v}" for k, v in corrected.items())

def _day_path(dir_path: Path, day: int) -> Path:
    return dir_path / f"day_{day:04d}.json"

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class PredictionModel:
    """
    Combines:
    - LongTermMemoryModel (semantic memory summary, as JSON string)
    - EpisodicMemoryModel (episodic retrieval over history)
    - Working memory (current context + corrected so far)
    And calls the LLM to predict the preference bundle.
    """

    def __init__(
        self,
        user: str,
        physical_profile_label: str,
        client: OpenAI,
        chat_model: str,
        embed_model: str,
        retry_fn,
        logs_dir: Path,
        use_long_term_memory: bool = True,
        use_episodic_memory: bool = True,
        k_retrieve: int = 5,
    ) -> None:
        
        self.user = user
        self.physical_profile_label = physical_profile_label
        self.client = client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self._retry = retry_fn
        self.logs_dir = logs_dir
        self.long_term_memory_model: Optional[LongTermMemoryModel] = None
        self.episodic_memory_model: Optional[EpisodicMemoryModel] = None
        
        if use_long_term_memory:
            self.long_term_memory_dir = self.logs_dir / user / "long_term_memory"
            self.long_term_memory_dir.mkdir(parents=True, exist_ok=True)
            self.long_term_memory_model = LongTermMemoryModel(
                physical_profile_label=self.physical_profile_label,
                client=client,
                chat_model=chat_model,
                retry_fn=retry_fn,
                logs_dir=self.logs_dir / "long_term_memory_llm_calls",
            )
            
        if use_episodic_memory:
            self.episodic_memory_dir = self.logs_dir / user / "episodic_memory"
            self.episodic_memory_dir.mkdir(parents=True, exist_ok=True)
            self.episodic_memory_model = EpisodicMemoryModel(
                client=client,
                embed_model=self.embed_model,
                cache_path=self.logs_dir / "embeddings.json",
                retry_fn=self._retry,
                k_retrieve=k_retrieve,
            )
            
        self.working_memory_dir = self.logs_dir / user / "working_memory"
        self.working_memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.working_memory_calls_dir = self.logs_dir / user / "prediction_model_llm_calls"
        self.working_memory_calls_dir.mkdir(parents=True, exist_ok=True)
        
    def update(self, day: int, context: Dict[str, Any], corrected: Dict[str, str], ground_truth_bundle: Dict[str, str]) -> None:
        
        ep_txt = _episode_text(day=day, context=context, prefs=ground_truth_bundle)
        
        if self.long_term_memory_model:
            print(f"  [long_term_memory_model] Updating summary (day {day}) ...", flush=True)
            self.long_term_memory_model.add_episode(ep_txt)
            long_term_memory = self.long_term_memory_model.get_ltm()  # JSON string (or empty)

            log_ltm_summary = long_term_memory
            try:
                log_ltm_summary = json.loads(long_term_memory)
            except Exception:
                print(
                    f"Warning: long_term_memory_model summary for logging is not valid JSON. "
                    f"Logging raw string. Summary:\n{long_term_memory}\n",
                    flush=True,
                )
                
            # Per-day logs we will write once at the end of the day
            long_term_memory_record = {
                "day": day,
                "context": context,
                "episode_text": ep_txt,
                "ltm_summary_raw": log_ltm_summary,
            }
            
            _write_json(_day_path(self.long_term_memory_dir, day), long_term_memory_record)

        if self.episodic_memory_model:
            # Update retrieval history after finishing the interactive correction loop
            self.episodic_memory_model.add_episode(ep_txt)
            
            episodic_memory_record = {
                "day": day,
                "context": context,
                "corrected": dict(corrected),
                "episode_text": ep_txt,
                "retrieved_episodes": self.episodic_memory_model.get_last_retrieved(),
            }
            _write_json(_day_path(self.episodic_memory_dir, day), episodic_memory_record)
            
        working_memory_record = {
            "day": day,
            "context": context,
            "corrected": dict(corrected)
        }
        _write_json(_day_path(self.working_memory_dir, day), working_memory_record)

    def predict_bundle(
        self,
        context: Dict[str, Any],
        corrected: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Returns predicted_bundle.
        """
        
        episodic_memory = ""
        if self.episodic_memory_model:
            episodic_memory = self.episodic_memory_model.retrieve(context, corrected)

        long_term_memory = ""
        if self.long_term_memory_model:
            long_term_memory = self.long_term_memory_model.get_ltm()  # JSON string (or empty)

        # Prompt blocks
        options_block = _build_options_block()
        corrected_block = _format_corrected_block(corrected)

        prompt = get_bundle_prediction_prompt(
            physical_profile_label=self.physical_profile_label,
            ltm_summary=long_term_memory,
            retrieved_block=episodic_memory,
            context=context,
            corrected_block=corrected_block,
            options_block=options_block,
        )

        def _call() -> Any:
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "Return JSON only. No extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=15000,
            )

        resp = self._retry(_call)
        raw = (resp.choices[0].message.content or "").strip()
        raw = _strip_code_fences(raw)
        data = _safe_json_load(raw) or {}
            
        if self.working_memory_calls_dir:
            log_file = self.working_memory_calls_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if data:
                log_file.write_text(f"===PROMPT===\n{prompt}\n\n===RESPONSE===\n{json.dumps(data, indent=2)}", encoding="utf-8")
            else:
                log_file.write_text(f"===PROMPT===\n{prompt}\n\n===RESPONSE===\nFailed to parse response as JSON. Raw response:\n{resp}", encoding="utf-8")

        # Validate against allowed options, fallback to corrected or default
        out: Dict[str, str] = {}
        for field in PREF_FIELDS:
            val = str(data.get(field, "")).strip()
            if val in PREF_OPTIONS[field]:
                out[field] = val
            else:
                out[field] = corrected.get(field, PREF_OPTIONS[field][0])

        out = _apply_hard_rules(out, meal=str(context.get("meal", "")), corrected=corrected)

        # Corrected always overrides
        for k, v in corrected.items():
            out[k] = v

        # Final validation
        for field in PREF_FIELDS:
            if out[field] not in PREF_OPTIONS[field]:
                out[field] = PREF_OPTIONS[field][0]
        return out