from typing import Dict, List, Tuple

from .preference_bundle import PREFERENCE_BUNDLE as _PREFERENCE_BUNDLE_DATACLASSES
from .mealtime_context import (
    SETTINGS,
    TIMES_OF_DAY,
    MEAL_CONTENTS_BY_LABEL,
    MEALS,
)
from .physical_capabilities import PHYSICAL_CAPABILITY_PROFILES
from .affective_state import AFFECTIVE_STATES


# Public, tuple-based preference spec expected by dataset generation and evaluation
# modules: List[Tuple[field, label, options]].
PREFERENCE_BUNDLE: List[Tuple[str, str, List[str]]] = [
    (dim.field, dim.label, dim.options) for dim in _PREFERENCE_BUNDLE_DATACLASSES
]


# Meal structure map expected by callers (e.g., for checking dippable items/sauces).
MEAL_STRUCTURE: Dict[str, Dict[str, object]] = {
    label: {
        "dippable_items": meal.dippable_items,
        "sauces": meal.sauces,
        "storage_condition": meal.storage_condition,
        "intended_serving_temp": meal.intended_serving_temp,
    }
    for label, meal in MEAL_CONTENTS_BY_LABEL.items()
}


__all__ = [
    "PREFERENCE_BUNDLE",
    "MEAL_STRUCTURE",
    "MEALS",
    "SETTINGS",
    "TIMES_OF_DAY",
    "AFFECTIVE_STATES",
    "PHYSICAL_CAPABILITY_PROFILES",
]

