from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from openai import OpenAI

import feeding_deployment.preference_learning.config as root_config  # type: ignore
from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE as _PREF_BUNDLE_DIMS,
)

from feeding_deployment.preference_learning.methods.prompts.bundle_prediction import (
    get_bundle_prediction_prompt,
)

from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS

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
        client: OpenAI,
        chat_model: str,
        retry_fn,
        logs_dir: Path = None,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self._retry = retry_fn
        self.logs_dir = logs_dir

    def predict_bundle(
        self,
        physical_profile_label: str,
        long_term_memory: str,
        episodic_memory: str,
        context: Dict[str, Any],
        corrected: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Returns (predicted_bundle, debug_info).

        - physical_profile_label: label string (used by get_bundle_prediction_prompt)
        - ltm_summary: JSON string from your new LongTermMemoryModel (or "N/A"/"" if absent)
        """

        # Prompt blocks
        options_block = _build_options_block()
        corrected_block = _format_corrected_block(corrected)

        prompt = get_bundle_prediction_prompt(
            physical_profile_label=physical_profile_label,
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
            
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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