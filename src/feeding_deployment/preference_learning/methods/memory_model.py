from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import feeding_deployment.preference_learning.config as root_config  # type: ignore
from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE as _PREF_BUNDLE_DIMS,
)
from feeding_deployment.preference_learning.methods.prompts.bundle_prediction_prompt import (
    get_bundle_prediction_prompt,
)

from feeding_deployment.preference_learning.methods.retrieval import RetrievalModel
from feeding_deployment.preference_learning.methods.long_term_memory import LongTermMemoryModel, _build_options_block

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
        out["outside_mouth_distance"] = "near"
    if (
        "bite_dipping_preference" not in corrected
        and meal_info.get("known_meal", False)
        and ((not meal_info["has_dippable"]) or (not meal_info["has_sauce"]))
    ):
        out["bite_dipping_preference"] = "do not dip"
    return out


def _query_text(context: Dict[str, Any], corrected: Dict[str, str]) -> str:
    ctx = (
        f"meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    corr = "; ".join(f"{k}={v}" for k, v in sorted(corrected.items())) if corrected else "none"
    return f"{ctx}\ncorrected_so_far: {corr}"


class MemoryModel:
    """
    Combines:
    - LongTermMemoryModel (semantic memory summary updated every Δ)
    - RetrievalModel (episodic retrieval over history)
    - Working memory (current context + corrected so far)
    And calls the LLM to predict the preference bundle.
    """

    def __init__(
        self,
        client: OpenAI,
        chat_model: str,
        embed_model: str,
        k_retrieve: int,
        ablation: str,  # full|ltm_only|em_only|no_memory
        ltm: LongTermMemoryModel,
        retriever: RetrievalModel,
        retry_fn,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.k_retrieve = k_retrieve
        self.ablation = ablation
        self.ltm = ltm
        self.retriever = retriever
        self._retry = retry_fn

    def predict_bundle(
        self,
        physical_profile: str,
        ltm_summary: str,
        history_texts: List[str],
        context: Dict[str, Any],
        corrected: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Returns (predicted_bundle, debug_info).
        """
        query = _query_text(context, corrected)

        use_em = self.ablation not in ("ltm_only", "no_memory")
        retrieved: List[str] = []
        if use_em and self.k_retrieve > 0 and history_texts:
            retrieved = self.retriever.retrieve(history_texts, query, self.k_retrieve)

        ltm_for_pred = ltm_summary if self.ablation not in ("em_only", "no_memory") else ""
        retrieved_for_pred = retrieved if self.ablation not in ("ltm_only", "no_memory") else []

        opts_block = _build_options_block()
        retrieved_block = "\n\n".join(retrieved_for_pred) if retrieved_for_pred else "(none)"
        corrected_block = "\n".join(f"- {k}: {v}" for k, v in corrected.items()) if corrected else "(none)"

        prompt = get_bundle_prediction_prompt(
            physical_profile=physical_profile,
            ltm_summary=ltm_for_pred,
            retrieved_block=retrieved_block,
            context=context,
            corrected_block=corrected_block,
            options_block=opts_block,
        )

        def _call() -> Any:
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "Return JSON only. No extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=650,
            )

        resp = self._retry(_call)
        raw = (resp.choices[0].message.content or "").strip()
        raw = _strip_code_fences(raw)
        data = _safe_json_load(raw) or {}

        out: Dict[str, str] = {}
        for field in PREF_FIELDS:
            val = str(data.get(field, "")).strip()
            if val in PREF_OPTIONS[field]:
                out[field] = val
            else:
                out[field] = corrected.get(field, PREF_OPTIONS[field][0])

        out = _apply_hard_rules(out, meal=str(context.get("meal", "")), corrected=corrected)

        for k, v in corrected.items():
            out[k] = v

        for field in PREF_FIELDS:
            if out[field] not in PREF_OPTIONS[field]:
                out[field] = PREF_OPTIONS[field][0]

        debug = {
            "query": query,
            "use_em": use_em,
            "retrieved_episodes": retrieved,
        }
        return out, debug