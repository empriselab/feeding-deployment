#!/usr/bin/env python3
"""
Heuristic implementation of the cognitively-inspired memory architecture
for long-term preference adaptation, plus utilities for evaluation on the
synthetic JSON datasets produced by generate_dataset_llm.py.

This file is kept at the repo root as an experiment module so it can be
used directly without going through the src/ package structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config as root_config  # type: ignore


PREFERENCE_FIELDS: List[str] = [name for (name, _, _) in root_config.PREFERENCE_BUNDLE]
PREFERENCE_FIELDS_WITH_ORDERING: List[str] = [
    name for (name, _, _) in root_config.PREFERENCE_BUNDLE_WITH_ORDERING
]


@dataclass
class Context:
    meal: str
    setting: str
    time_of_day: str
    transient_affective_state: str


@dataclass
class Episode:
    day: int
    window: int
    context: Context
    preferences: Dict[str, str]


@dataclass
class LongTermMemory:
    window_1: Dict[str, Any]
    window_2: Dict[str, Any]

    def for_day(self, day: int, u_update_days: Sequence[int]) -> Dict[str, Any]:
        if day < u_update_days[1]:
            return self.window_1
        return self.window_2


@dataclass
class WorkingMemory:
    context: Context
    corrected_indices: List[int]
    corrected_values: Dict[str, str]


class EpisodicMemory:
    def __init__(self, episodes: Sequence[Episode]) -> None:
        self._episodes: List[Episode] = list(episodes)

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    def retrieve(self, query_context: Context, k: int = 10) -> List[Episode]:
        exact_matches: List[Episode] = [
            e
            for e in self._episodes
            if e.context.meal == query_context.meal
            and e.context.setting == query_context.setting
            and e.context.time_of_day == query_context.time_of_day
        ]
        if len(exact_matches) >= k:
            return exact_matches[:k]

        relaxed: List[Episode] = [
            e
            for e in self._episodes
            if e.context.meal == query_context.meal
            and e.context.setting == query_context.setting
        ]
        relaxed_extra = [e for e in relaxed if e not in exact_matches]
        out = exact_matches + relaxed_extra
        return out[:k]


class CognitivePreferenceModel:
    def __init__(
        self,
        ltm: LongTermMemory,
        em: EpisodicMemory,
        u_update_days: Sequence[int],
    ) -> None:
        self.ltm = ltm
        self.em = em
        self.u_update_days = list(u_update_days)

    def _global_majority(self, field: str) -> Optional[str]:
        counts: Dict[str, int] = {}
        for e in self.em.episodes:
            val = e.preferences.get(field)
            if not val:
                continue
            counts[val] = counts.get(val, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _majority_from_episodes(
        self, episodes: Sequence[Episode], field: str
    ) -> Optional[str]:
        counts: Dict[str, int] = {}
        for e in episodes:
            val = e.preferences.get(field)
            if not val:
                continue
            counts[val] = counts.get(val, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def predict_bundle(
        self,
        day: int,
        wm: WorkingMemory,
        k_retrieve: int = 10,
    ) -> Dict[str, str]:
        u_pref = self.ltm.for_day(day, self.u_update_days)
        retrieved = self.em.retrieve(wm.context, k=k_retrieve)

        corrected_set = set(wm.corrected_values.keys())
        pred: Dict[str, str] = {}

        for field in PREFERENCE_FIELDS_WITH_ORDERING:
            if field in corrected_set:
                pred[field] = wm.corrected_values[field]
                continue

            from_retrieved = self._majority_from_episodes(retrieved, field)
            if from_retrieved is not None:
                pred[field] = from_retrieved
                continue

            if field in u_pref:
                pred[field] = str(u_pref[field])
                continue

            global_val = self._global_majority(field)
            if global_val is not None:
                pred[field] = global_val
                continue

            opt_list = next(
                (opts for (name, _, opts) in root_config.PREFERENCE_BUNDLE_WITH_ORDERING if name == field),
                [],
            )
            pred[field] = opt_list[0] if opt_list else ""

        return pred


def load_user_dataset(path: str) -> Tuple[LongTermMemory, List[Episode], List[int]]:
    import json
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ltm = LongTermMemory(
        window_1=data["long_term_preferences"]["window_1_days_1_15"],
        window_2=data["long_term_preferences"]["window_2_days_16_30"],
    )
    u_update_days = data.get("config", {}).get("u_update_days", [1, 16])

    episodes: List[Episode] = []
    for day_rec in data["days"]:
        ctx_raw = day_rec["context"]
        ctx = Context(
            meal=ctx_raw["meal"],
            setting=ctx_raw["setting"],
            time_of_day=ctx_raw.get("time_of_day", "unknown"),
            transient_affective_state=ctx_raw.get("transient_affective_state", "unknown"),
        )
        prefs = {
            name: str(val.get("choice", ""))
            for (name, val) in day_rec["preferences"].items()
        }
        episodes.append(
            Episode(
                day=day_rec["day"],
                window=day_rec["window"],
                context=ctx,
                preferences=prefs,
            )
        )

    return ltm, episodes, u_update_days


def evaluate_user_dataset(
    path: str,
    k_retrieve: int = 10,
) -> Dict[str, Any]:
    """
    Cognitive model evaluation: uses LTM + EM + WM (context only) with
    episodic retrieval and majority voting.
    """
    from collections import defaultdict

    ltm, episodes, u_update_days = load_user_dataset(path)

    em = EpisodicMemory([])
    model = CognitivePreferenceModel(ltm, em, u_update_days)

    correct_counts: Dict[str, int] = defaultdict(int)
    total_counts: Dict[str, int] = defaultdict(int)

    for ep in episodes:
        wm = WorkingMemory(
            context=ep.context,
            corrected_indices=[],
            corrected_values={},
        )
        pred = model.predict_bundle(day=ep.day, wm=wm, k_retrieve=k_retrieve)

        for field in PREFERENCE_FIELDS_WITH_ORDERING:
            true_val = ep.preferences.get(field)
            if true_val is None:
                continue
            total_counts[field] += 1
            if pred.get(field) == true_val:
                correct_counts[field] += 1

        em.add(ep)

    field_accuracies: Dict[str, float] = {}
    for field, total in total_counts.items():
        if total == 0:
            continue
        field_accuracies[field] = correct_counts[field] / float(total)

    if field_accuracies:
        avg_acc = sum(field_accuracies.values()) / len(field_accuracies)
    else:
        avg_acc = 0.0

    return {
        "file": path,
        "field_accuracies": field_accuracies,
        "avg_accuracy": avg_acc,
    }

