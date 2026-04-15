"""Terminal-based preference context input and preference correction.

Used when --pref_mode=terminal.  Provides the same contract as the
web-interface path (context dict, corrected bundle) but via stdin/stdout.
"""

from __future__ import annotations

from feeding_deployment.preference_learning.config import MEALS, SETTINGS, TIMES_OF_DAY
from feeding_deployment.preference_learning.methods.prediction_model import PREF_OPTIONS
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS


def _pick_from_list(prompt: str, options: list[str]) -> str:
    """Display numbered options and return the user's choice."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"  Invalid choice. Enter a number between 1 and {len(options)}.")


def terminal_collect_context() -> dict[str, str]:
    """Prompt the operator to choose meal, setting, and time_of_day."""
    print("\n=== Preference Context ===")
    meal = _pick_from_list("Select meal:", MEALS)
    setting = _pick_from_list("Select dining setting:", SETTINGS)
    time_of_day = _pick_from_list("Select time of day:", TIMES_OF_DAY)
    return {"meal": meal, "setting": setting, "time_of_day": time_of_day}


def terminal_correct_preferences(
    predicted_bundle: dict[str, str],
    pref_options: dict[str, list[str]],
) -> dict[str, str]:
    """Show the predicted bundle field-by-field and let the operator correct any field.

    Returns the final bundle (predicted values + any corrections).
    """
    bundle = dict(predicted_bundle)

    print("\n=== Predicted Preferences ===")
    print("Review each field. Press Enter to accept the predicted value, or enter")
    print("a number to change it.\n")

    for i, field in enumerate(PREF_FIELDS, 1):
        options = pref_options[field]
        predicted = bundle[field]
        pred_idx = options.index(predicted) + 1 if predicted in options else "?"

        print(f"[{i}/{len(PREF_FIELDS)}] {field}")
        for j, opt in enumerate(options, 1):
            marker = " <-- predicted" if opt == predicted else ""
            print(f"  {j}. {opt}{marker}")

        while True:
            raw = input(f"Choice [{pred_idx}]: ").strip()
            if raw == "":
                break
            try:
                idx = int(raw)
                if 1 <= idx <= len(options):
                    bundle[field] = options[idx - 1]
                    break
            except ValueError:
                pass
            print(f"  Invalid. Enter a number 1-{len(options)} or press Enter to keep.")

    corrected = {k: v for k, v in bundle.items() if v != predicted_bundle.get(k)}
    print(f"\nCorrections made: {len(corrected)} field(s)")
    if corrected:
        for k, v in corrected.items():
            print(f"  {k}: {predicted_bundle[k]!r} -> {v!r}")
    return bundle
