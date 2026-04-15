"""Build and validate mealtime context for preference prediction.

Only observable context is included (meal, setting, time_of_day). 
"""

from __future__ import annotations

from typing import Any, Mapping

from feeding_deployment.preference_learning.config import MEALS, SETTINGS, TIMES_OF_DAY

PREFERENCE_CONTEXT_KEYS = ("meal", "setting", "time_of_day")


def build_preference_context(meal: str, setting: str, time_of_day: str) -> dict[str, str]:
    """
    Assemble the context dict for the current meal before calling predict_bundle / update.

    Values must match the canonical lists in preference_learning.config.
    """
    context = {
        "meal": meal.strip(),
        "setting": setting.strip(),
        "time_of_day": time_of_day.strip(),
    }
    validate_preference_context(context)
    return context


def validate_preference_context(context: Mapping[str, Any]) -> None:
    """Raise ValueError if any field is missing or not in the allowed vocabulary."""
    for key in PREFERENCE_CONTEXT_KEYS:
        if key not in context:
            raise ValueError(f"preference context missing required key: {key!r}")
        val = context[key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"preference context {key!r} must be a non-empty string")

    meal = context["meal"].strip()
    if meal not in MEALS:
        raise ValueError(
            f"Unknown meal={meal!r}. Must be one of the labels in "
            "preference_learning.config.mealtime_context.MEALS."
        )

    setting = context["setting"].strip()
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting={setting!r}. Allowed: {SETTINGS!r}")

    tod = context["time_of_day"].strip()
    if tod not in TIMES_OF_DAY:
        raise ValueError(f"Unknown time_of_day={tod!r}. Allowed: {TIMES_OF_DAY!r}")
