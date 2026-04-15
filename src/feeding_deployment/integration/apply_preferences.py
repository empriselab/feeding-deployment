"""Translate a ground-truth preference bundle into BT YAML parameter writes
and scene-level configuration changes.

Every bundle field that maps to a BT parameter is declared in _BT_MAPPING.
apply_bundle_to_behavior_trees() iterates that mapping, loads the affected
YAML files, overwrites `value` entries, and saves them back to disk so that
subsequent execute_action() calls pick up the new values (BT YAMLs are
loaded fresh on each tick).
"""

from __future__ import annotations

import warnings as _warnings
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Value translators: bundle option string  →  BT parameter value
# ---------------------------------------------------------------------------

_SPEED_MAP = {"slow": "low", "medium": "medium", "fast": "high"}

_CONFIRMATION_MAP = {"yes": 1, "no": 0}

_RETRACT_MAP = {"yes": 1, "no": 0}

_AUTOCONTINUE_MAP = {"10 sec": 10.0, "100 sec": 100.0, "1000 sec": 1000.0}

_OUTSIDE_MOUTH_DISTANCE_MAP = {"near": 0.1, "medium": 0.15, "far": 0.2}

_CONVEY_READY_MAP = {
    "no cue": "silent",
    "speech": "voice",
    "LED": "led",
    "speech + LED": "voice_led",
}

_INITIATE_TRANSFER_MAP = {
    "open mouth": "open_mouth",
    "button": "button",
    "autocontinue": "auto_timeout",
}

_COMPLETE_TRANSFER_MAP = {
    "perception": "sense",
    "button": "button",
    "autocontinue": "auto_timeout",
}

_TRANSFER_MODE_MAP = {
    "inside mouth transfer": "inside",
    "outside mouth transfer": "outside",
}

_SKEWERING_AXIS_MAP = {
    "parallel to major axis": "horizontal",
    "perpendicular to major axis": "vertical",
}


def _dipping_depth_translate(val: str) -> float | None:
    # NOTE: "do not dip" only skips the FoodDippingDepth BT write — it does NOT
    # prevent FLAIR's autonomous planner from choosing to dip. The dipping
    # decision is made independently in inference_class.py
    # (get_autonomous_action), which looks at plate contents and user
    # preferences. To truly suppress dipping when the user says "do not dip",
    # a flag must be passed into FLAIR's planning logic so that
    # get_autonomous_action never selects a dip action. This is separate from
    # the BT parameter layer and is not yet implemented.
    if val == "do not dip":
        return None
    return {"less": 0.01, "more": 0.03}[val]

# ---------------------------------------------------------------------------
# Declarative mapping:  bundle field  →  list of (yaml_filename, bt_param_name, translator)
#
# "translator" is either a dict (direct lookup) or a callable (value → bt_value).
# A translator may return None to signal "skip this write" (e.g. outside_mouth_distance
# when the bundle value is "not applicable").
# ---------------------------------------------------------------------------

_ALL_BT_YAMLS: list[str] = [
    "acquire_bite.yaml",
    "close_fridge.yaml",
    "close_microwave.yaml",
    "emulate_transfer.yaml",
    "navigate_to_fridge.yaml",
    "navigate_to_microwave.yaml",
    "navigate_to_sink.yaml",
    "navigate_to_table.yaml",
    "open_fridge.yaml",
    "open_microwave.yaml",
    "pick_drink.yaml",
    "pick_plate_from_fridge.yaml",
    "pick_plate_from_holder.yaml",
    "pick_plate_from_microwave.yaml",
    "pick_plate_from_table.yaml",
    "pick_utensil.yaml",
    "pick_wipe.yaml",
    "place_plate_in_fridge.yaml",
    "place_plate_in_microwave.yaml",
    "place_plate_in_sink.yaml",
    "place_plate_on_holder.yaml",
    "place_plate_on_table.yaml",
    "press_microwave_button.yaml",
    "stow_drink.yaml",
    "stow_utensil.yaml",
    "stow_wipe.yaml",
    "transfer_drink.yaml",
    "transfer_utensil.yaml",
    "transfer_wipe.yaml",
]

_TRANSFER_YAMLS = ["transfer_utensil.yaml", "transfer_drink.yaml", "transfer_wipe.yaml"]


def _outside_mouth_translate(val: str) -> float | None:
    if val == "not applicable":
        return None
    return _OUTSIDE_MOUTH_DISTANCE_MAP[val]


def _microwave_duration_translate(val: str) -> float | None:
    """Translate microwave_time bundle value to BT MicrowaveDuration seconds.

    "no microwave" returns None (skip the BT write — the planner already
    excludes the PressMicrowaveButton HLA via the FoodHeated atom).
    """
    if val == "no microwave":
        return None
    return {"1 min": 60.0, "2 min": 120.0, "3 min": 180.0}[val]


# Each entry: (bundle_field, yaml_files, bt_param_name, translator)
_BT_MAPPING: list[tuple[str, list[str], str, dict | Any]] = [
    # Speed — all 29 YAMLs
    ("robot_speed", _ALL_BT_YAMLS, "Speed", _SPEED_MAP),

    # Web-interface confirmation
    ("web_interface_confirmation", ["acquire_bite.yaml"], "TransferAskForConfirmation", _CONFIRMATION_MAP),
    ("web_interface_confirmation", ["transfer_drink.yaml", "transfer_wipe.yaml"],
     "AskForConfirmationInitiatingTransferSequence", _CONFIRMATION_MAP),

    # Autocontinue wait time
    ("wait_before_autocontinue_seconds",
     ["acquire_bite.yaml", "transfer_utensil.yaml", "transfer_drink.yaml"],
     "TimeToWaitBeforeAutocontinue", _AUTOCONTINUE_MAP),

    # Outside-mouth distance
    ("outside_mouth_distance", _TRANSFER_YAMLS, "OutsideMouthDistance", _outside_mouth_translate),

    # Convey robot ready for initiating transfer
    ("convey_robot_ready_for_initiating_transfer", _TRANSFER_YAMLS,
     "ReadyToInitiateTransferInteraction", _CONVEY_READY_MAP),

    # Detect user ready (per-tool)
    ("detect_user_ready_for_initiating_transfer_feeding",
     ["transfer_utensil.yaml"], "InitiateTransferInteraction", _INITIATE_TRANSFER_MAP),
    ("detect_user_ready_for_initiating_transfer_drinking",
     ["transfer_drink.yaml"], "InitiateTransferInteraction", _INITIATE_TRANSFER_MAP),
    ("detect_user_ready_for_initiating_transfer_wiping",
     ["transfer_wipe.yaml"], "InitiateTransferInteraction", _INITIATE_TRANSFER_MAP),

    # Convey robot ready for completing transfer
    ("convey_robot_ready_for_completing_transfer", _TRANSFER_YAMLS,
     "ReadyForTransferInteraction", _CONVEY_READY_MAP),

    # Detect user completed transfer (per-tool)
    ("detect_user_completed_transfer_feeding",
     ["transfer_utensil.yaml"], "TransferCompleteInteraction", _COMPLETE_TRANSFER_MAP),
    ("detect_user_completed_transfer_drinking",
     ["transfer_drink.yaml"], "TransferCompleteInteraction", _COMPLETE_TRANSFER_MAP),
    ("detect_user_completed_transfer_wiping",
     ["transfer_wipe.yaml"], "TransferCompleteInteraction", _COMPLETE_TRANSFER_MAP),

    # Skewering axis
    ("skewering_axis", ["acquire_bite.yaml"], "SkeweringOrientation", _SKEWERING_AXIS_MAP),

    # Bite dipping preference → FoodDippingDepth (see _dipping_depth_translate for "do not dip" caveat)
    ("bite_dipping_preference", ["acquire_bite.yaml"], "FoodDippingDepth", _dipping_depth_translate),

    # Microwave duration
    ("microwave_time", ["press_microwave_button.yaml"], "MicrowaveDuration", _microwave_duration_translate),

    # Retract between bites (utensil only — drink/wipe don't have repeated transfer loops)
    ("retract_between_bites", ["transfer_utensil.yaml"], "RetractAfterTransfer", _RETRACT_MAP),
]


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

class _HlaTag:
    """Placeholder for the !hla YAML tag so we can round-trip without losing it."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"!hla {self.value}"


def _hla_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> _HlaTag:
    return _HlaTag(loader.construct_scalar(node))


def _hla_representer(dumper: yaml.Dumper, tag: _HlaTag) -> yaml.Node:
    return dumper.represent_scalar("!hla", tag.value)


_Loader = type("_Loader", (yaml.SafeLoader,), {})
_Loader.add_constructor("!hla", _hla_constructor)

_Dumper = type("_Dumper", (yaml.Dumper,), {})
_Dumper.add_representer(_HlaTag, _hla_representer)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def _save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=False))


def _set_param_value(bt_data: dict, param_name: str, new_value: Any) -> bool:
    """Find a parameter by name in the YAML dict and set its value.

    Returns True if the parameter was found and updated.
    """
    for param in bt_data.get("parameters", []):
        if param["name"] == param_name:
            param["value"] = new_value
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_bundle_to_behavior_trees(
    bundle: dict[str, str],
    bt_dir: Path,
) -> list[str]:
    """Write preference-bundle values into the BT YAML files on disk.

    Returns a list of warning strings for edge cases (e.g. unsupported
    combined cues).
    """
    warnings: list[str] = []
    # Cache: yaml_filename -> loaded dict (load each file at most once)
    loaded: dict[str, dict] = {}
    dirty: set[str] = set()

    for bundle_field, yaml_files, bt_param, translator in _BT_MAPPING:
        bundle_val = bundle.get(bundle_field)
        if bundle_val is None:
            continue

        # Translate
        if callable(translator) and not isinstance(translator, dict):
            bt_val = translator(bundle_val)
        else:
            bt_val = translator.get(bundle_val)
            if bt_val is None:
                warnings.append(
                    f"No mapping for {bundle_field}={bundle_val!r}; skipping BT write."
                )
                continue

        if bt_val is None:
            # Explicit skip (e.g. outside_mouth_distance="not applicable")
            continue

        for fname in yaml_files:
            fpath = bt_dir / fname
            if not fpath.exists():
                warnings.append(f"BT YAML not found: {fpath}")
                continue
            if fname not in loaded:
                loaded[fname] = _load_yaml(fpath)
            if _set_param_value(loaded[fname], bt_param, bt_val):
                dirty.add(fname)
            else:
                warnings.append(
                    f"Parameter {bt_param!r} not found in {fname}; skipping."
                )

    for fname in dirty:
        _save_yaml(bt_dir / fname, loaded[fname])

    return warnings


def apply_transfer_mode(
    bundle: dict[str, str],
    scene_description: Any,
    hla_map: dict[str, Any],
) -> None:
    """Set scene_description.transfer_type from the bundle and re-init the
    TransferToolHLA transfer object so that subsequent execute_action() calls
    use the correct inside/outside mouth transfer implementation.
    """
    mode = bundle.get("transfer_mode")
    if mode is None:
        return

    new_type = _TRANSFER_MODE_MAP.get(mode)
    if new_type is None:
        raise ValueError(
            f"Unknown transfer_mode={mode!r}. "
            f"Expected one of {list(_TRANSFER_MODE_MAP.keys())}."
        )

    scene_description.transfer_type = new_type

    transfer_hla = hla_map.get("TransferTool")
    if transfer_hla is None:
        return

    # Re-run the inside/outside branch from TransferToolHLA.__init__
    from feeding_deployment.actions.feel_the_bite.inside_mouth_transfer import (
        InsideMouthTransfer,
    )
    from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import (
        OutsideMouthTransfer,
    )

    if new_type == "inside":
        transfer_hla.transfer = InsideMouthTransfer(
            transfer_hla.sim,
            transfer_hla.robot_interface,
            transfer_hla.perception_interface,
            transfer_hla.rviz_interface,
            transfer_hla.no_waits,
            transfer_hla.head_perception_log_dir,
        )
    elif new_type == "outside":
        transfer_hla.transfer = OutsideMouthTransfer(
            transfer_hla.sim,
            transfer_hla.robot_interface,
            transfer_hla.perception_interface,
            transfer_hla.rviz_interface,
            transfer_hla.no_waits,
            transfer_hla.head_perception_log_dir,
        )
    else:
        raise ValueError(f"Unrecognized transfer type: {new_type!r}")


_MICROWAVE_TIME_MAP = {
    "no microwave": None,
    "1 min": 60,
    "2 min": 120,
    "3 min": 180,
}


def apply_microwave_preference(
    bundle: dict[str, str],
    current_atoms: set,
    food_heated_atom: Any,
) -> int | None:
    """Adjust planner atoms based on the microwave_time preference.

    If "no microwave", adds the FoodHeated atom to current_atoms so the PDDL
    planner skips the entire microwave sequence (navigate to microwave, open
    door, place plate, press button, pick plate, close door).

    For "1 min"/"2 min"/"3 min", ensures FoodHeated is absent (the planner
    will include microwave steps).

    Returns the microwave duration in seconds, or None for "no microwave".
    The caller can use this value to set the actual microwave timer when the
    PressMicrowaveButton HLA executes — this wiring is not yet implemented.
    """
    microwave_time = bundle.get("microwave_time")
    if microwave_time is None:
        return None

    duration = _MICROWAVE_TIME_MAP.get(microwave_time)
    if duration is None and microwave_time != "no microwave":
        raise ValueError(
            f"Unknown microwave_time={microwave_time!r}. "
            f"Expected one of {list(_MICROWAVE_TIME_MAP.keys())}."
        )

    if microwave_time == "no microwave":
        current_atoms.add(food_heated_atom)
    else:
        current_atoms.discard(food_heated_atom)

    return duration


def apply_dip_preference(
    bundle: dict[str, str],
    flair: Any,
) -> None:
    """Set FLAIR's allow_dip flag based on bite_dipping_preference.

    When "do not dip", sets allow_dip=False so get_autonomous_action
    deterministically strips any dip the LLM suggests.
    """
    dip_pref = bundle.get("bite_dipping_preference")
    if dip_pref is None or flair is None:
        return
    flair.set_allow_dip(dip_pref != "do not dip")


_OCCLUSION_TEXT_MAP = {
    "do not consider occlusion": None,
    "minimize left occlusion": "When choosing which food to pick, prefer items that minimize the robot arm blocking the user's view from the left.",
    "minimize front occlusion": "When choosing which food to pick, prefer items that minimize the robot arm blocking the user's view from the front.",
    "minimize right occlusion": "When choosing which food to pick, prefer items that minimize the robot arm blocking the user's view from the right.",
}


def apply_occlusion_preference(
    bundle: dict[str, str],
    flair: Any,
) -> None:
    """Append occlusion_relevance as a soft hint to FLAIR's user_preference string.

    This injects text into the LLM preference planner prompt. It's a soft
    hint — the LLM may or may not factor it into food selection. There is
    no geometric enforcement of occlusion minimization.
    """
    occlusion = bundle.get("occlusion_relevance")
    if occlusion is None or flair is None:
        return

    hint = _OCCLUSION_TEXT_MAP.get(occlusion)
    if hint is None:
        return

    current = flair.get_preference()
    if current:
        flair.set_preference(f"{current} {hint}")
    else:
        flair.set_preference(hint)
