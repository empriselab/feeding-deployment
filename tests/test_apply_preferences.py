"""Tests for Step 4: apply_preferences module (BT YAML writes + transfer mode).

Run with:
    PYTHONPATH=src python -m pytest tests/test_apply_preferences.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from feeding_deployment.integration.apply_preferences import (
    apply_bundle_to_behavior_trees,
    apply_transfer_mode,
    apply_microwave_preference,
    apply_dip_preference,
    apply_occlusion_preference,
    _SPEED_MAP,
    _CONFIRMATION_MAP,
    _AUTOCONTINUE_MAP,
    _OUTSIDE_MOUTH_DISTANCE_MAP,
    _CONVEY_READY_MAP,
    _INITIATE_TRANSFER_MAP,
    _COMPLETE_TRANSFER_MAP,
    _TRANSFER_MODE_MAP,
    _SKEWERING_AXIS_MAP,
    _MICROWAVE_TIME_MAP,
    _RETRACT_MAP,
    _dipping_depth_translate,
    _microwave_duration_translate,
    _load_yaml,
)
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS

BT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "feeding_deployment"
    / "actions"
    / "behavior_trees"
)


@pytest.fixture()
def bt_dir(tmp_path: Path) -> Path:
    """Copy all source BT YAMLs into a temp directory for isolated testing."""
    dest = tmp_path / "behavior_trees"
    shutil.copytree(BT_SOURCE, dest)
    return dest


def _load(bt_dir: Path, fname: str) -> dict:
    return _load_yaml(bt_dir / fname)


def _get_param_value(data: dict, param_name: str):
    for p in data.get("parameters", []):
        if p["name"] == param_name:
            return p["value"]
    raise KeyError(f"parameter {param_name!r} not found")


# -------------------------------------------------------------------
# Value translator unit tests
# -------------------------------------------------------------------


class TestValueTranslators:

    def test_speed_map_covers_all_bundle_options(self):
        assert set(_SPEED_MAP.keys()) == {"slow", "medium", "fast"}

    def test_confirmation_map(self):
        assert _CONFIRMATION_MAP == {"yes": 1, "no": 0}

    def test_autocontinue_map_values_are_floats(self):
        for v in _AUTOCONTINUE_MAP.values():
            assert isinstance(v, float)

    def test_outside_mouth_distance_values_in_range(self):
        for v in _OUTSIDE_MOUTH_DISTANCE_MAP.values():
            assert 0.1 <= v <= 0.2

    def test_convey_ready_maps_speech_plus_led_to_voice_led(self):
        assert _CONVEY_READY_MAP["speech + LED"] == "voice_led"

    def test_initiate_transfer_map(self):
        assert _INITIATE_TRANSFER_MAP["open mouth"] == "open_mouth"
        assert _INITIATE_TRANSFER_MAP["autocontinue"] == "auto_timeout"

    def test_complete_transfer_map(self):
        assert _COMPLETE_TRANSFER_MAP["perception"] == "sense"
        assert _COMPLETE_TRANSFER_MAP["autocontinue"] == "auto_timeout"

    def test_skewering_axis_map(self):
        assert _SKEWERING_AXIS_MAP["parallel to major axis"] == "horizontal"
        assert _SKEWERING_AXIS_MAP["perpendicular to major axis"] == "vertical"

    def test_dipping_depth_translate(self):
        assert _dipping_depth_translate("less") == 0.01
        assert _dipping_depth_translate("more") == 0.03
        assert _dipping_depth_translate("do not dip") is None

    def test_microwave_duration_translate(self):
        assert _microwave_duration_translate("1 min") == 60.0
        assert _microwave_duration_translate("2 min") == 120.0
        assert _microwave_duration_translate("3 min") == 180.0
        assert _microwave_duration_translate("no microwave") is None

    def test_retract_map(self):
        assert _RETRACT_MAP["yes"] == 1
        assert _RETRACT_MAP["no"] == 0


# -------------------------------------------------------------------
# apply_bundle_to_behavior_trees: YAML round-trip
# -------------------------------------------------------------------


class TestApplyBundleToBehaviorTrees:

    def test_speed_written_to_all_yamls(self, bt_dir: Path):
        bundle = {"robot_speed": "fast"}
        warnings = apply_bundle_to_behavior_trees(bundle, bt_dir)

        for fname in bt_dir.glob("*.yaml"):
            data = _load(bt_dir, fname.name)
            assert _get_param_value(data, "Speed") == "high", (
                f"Speed not updated in {fname.name}"
            )
        assert not warnings

    def test_speed_slow_maps_to_low(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"robot_speed": "slow"}, bt_dir)
        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "Speed") == "low"

    def test_confirmation_written(self, bt_dir: Path):
        bundle = {"web_interface_confirmation": "no"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        ab = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(ab, "TransferAskForConfirmation") == 0

        td = _load(bt_dir, "transfer_drink.yaml")
        assert _get_param_value(td, "AskForConfirmationInitiatingTransferSequence") == 0

        tw = _load(bt_dir, "transfer_wipe.yaml")
        assert _get_param_value(tw, "AskForConfirmationInitiatingTransferSequence") == 0

    def test_autocontinue_written(self, bt_dir: Path):
        bundle = {"wait_before_autocontinue_seconds": "1000 sec"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        for fname in ["acquire_bite.yaml", "transfer_utensil.yaml", "transfer_drink.yaml"]:
            data = _load(bt_dir, fname)
            assert _get_param_value(data, "TimeToWaitBeforeAutocontinue") == 1000.0

    def test_outside_mouth_distance_near(self, bt_dir: Path):
        bundle = {"outside_mouth_distance": "near"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        for fname in ["transfer_utensil.yaml", "transfer_drink.yaml", "transfer_wipe.yaml"]:
            data = _load(bt_dir, fname)
            assert _get_param_value(data, "OutsideMouthDistance") == 0.1

    def test_outside_mouth_distance_not_applicable_skips(self, bt_dir: Path):
        original = _get_param_value(_load(bt_dir, "transfer_utensil.yaml"), "OutsideMouthDistance")
        bundle = {"outside_mouth_distance": "not applicable"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        after = _get_param_value(_load(bt_dir, "transfer_utensil.yaml"), "OutsideMouthDistance")
        assert after == original

    def test_convey_ready_initiating(self, bt_dir: Path):
        bundle = {"convey_robot_ready_for_initiating_transfer": "LED"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        for fname in ["transfer_utensil.yaml", "transfer_drink.yaml", "transfer_wipe.yaml"]:
            data = _load(bt_dir, fname)
            assert _get_param_value(data, "ReadyToInitiateTransferInteraction") == "led"

    def test_convey_ready_completing(self, bt_dir: Path):
        bundle = {"convey_robot_ready_for_completing_transfer": "no cue"}
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        for fname in ["transfer_utensil.yaml", "transfer_drink.yaml", "transfer_wipe.yaml"]:
            data = _load(bt_dir, fname)
            assert _get_param_value(data, "ReadyForTransferInteraction") == "silent"

    def test_detect_initiate_per_tool(self, bt_dir: Path):
        bundle = {
            "detect_user_ready_for_initiating_transfer_feeding": "button",
            "detect_user_ready_for_initiating_transfer_drinking": "autocontinue",
            "detect_user_ready_for_initiating_transfer_wiping": "open mouth",
        }
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        assert _get_param_value(
            _load(bt_dir, "transfer_utensil.yaml"), "InitiateTransferInteraction"
        ) == "button"
        assert _get_param_value(
            _load(bt_dir, "transfer_drink.yaml"), "InitiateTransferInteraction"
        ) == "auto_timeout"
        assert _get_param_value(
            _load(bt_dir, "transfer_wipe.yaml"), "InitiateTransferInteraction"
        ) == "open_mouth"

    def test_detect_complete_per_tool(self, bt_dir: Path):
        bundle = {
            "detect_user_completed_transfer_feeding": "perception",
            "detect_user_completed_transfer_drinking": "button",
            "detect_user_completed_transfer_wiping": "autocontinue",
        }
        apply_bundle_to_behavior_trees(bundle, bt_dir)

        assert _get_param_value(
            _load(bt_dir, "transfer_utensil.yaml"), "TransferCompleteInteraction"
        ) == "sense"
        assert _get_param_value(
            _load(bt_dir, "transfer_drink.yaml"), "TransferCompleteInteraction"
        ) == "button"
        assert _get_param_value(
            _load(bt_dir, "transfer_wipe.yaml"), "TransferCompleteInteraction"
        ) == "auto_timeout"

    def test_skewering_axis_parallel(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"skewering_axis": "parallel to major axis"}, bt_dir)
        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "SkeweringOrientation") == "horizontal"

    def test_skewering_axis_perpendicular(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"skewering_axis": "perpendicular to major axis"}, bt_dir)
        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "SkeweringOrientation") == "vertical"

    def test_dipping_preference_more(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"bite_dipping_preference": "more"}, bt_dir)
        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "FoodDippingDepth") == 0.03

    def test_dipping_preference_less(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"bite_dipping_preference": "less"}, bt_dir)
        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "FoodDippingDepth") == 0.01

    def test_dipping_preference_do_not_dip_skips(self, bt_dir: Path):
        original = _get_param_value(_load(bt_dir, "acquire_bite.yaml"), "FoodDippingDepth")
        apply_bundle_to_behavior_trees({"bite_dipping_preference": "do not dip"}, bt_dir)
        after = _get_param_value(_load(bt_dir, "acquire_bite.yaml"), "FoodDippingDepth")
        assert after == original

    def test_microwave_duration_2_min(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"microwave_time": "2 min"}, bt_dir)
        data = _load(bt_dir, "press_microwave_button.yaml")
        assert _get_param_value(data, "MicrowaveDuration") == 120.0

    def test_microwave_duration_1_min(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"microwave_time": "1 min"}, bt_dir)
        data = _load(bt_dir, "press_microwave_button.yaml")
        assert _get_param_value(data, "MicrowaveDuration") == 60.0

    def test_microwave_duration_3_min(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"microwave_time": "3 min"}, bt_dir)
        data = _load(bt_dir, "press_microwave_button.yaml")
        assert _get_param_value(data, "MicrowaveDuration") == 180.0

    def test_microwave_no_microwave_skips(self, bt_dir: Path):
        original = _get_param_value(_load(bt_dir, "press_microwave_button.yaml"), "MicrowaveDuration")
        apply_bundle_to_behavior_trees({"microwave_time": "no microwave"}, bt_dir)
        after = _get_param_value(_load(bt_dir, "press_microwave_button.yaml"), "MicrowaveDuration")
        assert after == original

    def test_retract_between_bites_yes(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"retract_between_bites": "yes"}, bt_dir)
        data = _load(bt_dir, "transfer_utensil.yaml")
        assert _get_param_value(data, "RetractAfterTransfer") == 1

    def test_retract_between_bites_no(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"retract_between_bites": "no"}, bt_dir)
        data = _load(bt_dir, "transfer_utensil.yaml")
        assert _get_param_value(data, "RetractAfterTransfer") == 0

    def test_retract_only_affects_utensil(self, bt_dir: Path):
        apply_bundle_to_behavior_trees({"retract_between_bites": "yes"}, bt_dir)
        for fname in ["transfer_drink.yaml", "transfer_wipe.yaml"]:
            data = _load(bt_dir, fname)
            with pytest.raises(KeyError):
                _get_param_value(data, "RetractAfterTransfer")


class TestApplyBundleWarnings:

    def test_speech_plus_led_writes_voice_led(self, bt_dir: Path):
        bundle = {"convey_robot_ready_for_initiating_transfer": "speech + LED"}
        warnings = apply_bundle_to_behavior_trees(bundle, bt_dir)
        assert not any("speech + LED" in w for w in warnings)

        for fname in ["transfer_utensil.yaml", "transfer_drink.yaml", "transfer_wipe.yaml"]:
            data = _load(bt_dir, fname)
            assert _get_param_value(data, "ReadyToInitiateTransferInteraction") == "voice_led"

    def test_missing_yaml_file_produces_warning(self, bt_dir: Path):
        (bt_dir / "acquire_bite.yaml").unlink()
        bundle = {"web_interface_confirmation": "yes"}
        warnings = apply_bundle_to_behavior_trees(bundle, bt_dir)
        assert any("not found" in w for w in warnings)

    def test_unknown_bundle_field_ignored(self, bt_dir: Path):
        bundle = {"nonexistent_field": "whatever"}
        warnings = apply_bundle_to_behavior_trees(bundle, bt_dir)
        assert not warnings

    def test_empty_bundle_no_changes(self, bt_dir: Path):
        before = _load(bt_dir, "acquire_bite.yaml")
        speed_before = _get_param_value(before, "Speed")

        warnings = apply_bundle_to_behavior_trees({}, bt_dir)
        assert not warnings

        after = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(after, "Speed") == speed_before


class TestApplyBundleFullBundle:
    """Apply a realistic full bundle and verify no errors."""

    def test_full_bundle_applies_without_error(self, bt_dir: Path):
        bundle = {
            "robot_speed": "slow",
            "web_interface_confirmation": "yes",
            "wait_before_autocontinue_seconds": "100 sec",
            "outside_mouth_distance": "far",
            "convey_robot_ready_for_initiating_transfer": "speech",
            "detect_user_ready_for_initiating_transfer_feeding": "open mouth",
            "detect_user_ready_for_initiating_transfer_drinking": "button",
            "detect_user_ready_for_initiating_transfer_wiping": "autocontinue",
            "convey_robot_ready_for_completing_transfer": "LED",
            "detect_user_completed_transfer_feeding": "perception",
            "detect_user_completed_transfer_drinking": "autocontinue",
            "detect_user_completed_transfer_wiping": "button",
            "transfer_mode": "inside mouth transfer",
            "microwave_time": "no microwave",
            "occlusion_relevance": "do not consider occlusion",
            "skewering_axis": "parallel to major axis",
            "retract_between_bites": "yes",
            "bite_dipping_preference": "do not dip",
        }
        warnings = apply_bundle_to_behavior_trees(bundle, bt_dir)
        assert not any("Error" in w for w in warnings)

        data = _load(bt_dir, "acquire_bite.yaml")
        assert _get_param_value(data, "Speed") == "low"
        assert _get_param_value(data, "TransferAskForConfirmation") == 1
        assert _get_param_value(data, "TimeToWaitBeforeAutocontinue") == 100.0


# -------------------------------------------------------------------
# apply_transfer_mode
# -------------------------------------------------------------------


class TestApplyTransferMode:
    """Tests for apply_transfer_mode.

    When a TransferTool HLA is present in the map, the function imports
    InsideMouthTransfer / OutsideMouthTransfer (heavy robot deps).  We mock
    those imports to avoid pulling in scipy etc. in the test environment.
    When there is no HLA in the map, only the scene attribute is set.
    """

    def _make_scene(self, current_type: str = "outside") -> MagicMock:
        scene = MagicMock()
        scene.transfer_type = current_type
        return scene

    def _make_hla(self) -> MagicMock:
        hla = MagicMock()
        hla.sim = MagicMock()
        hla.robot_interface = None
        hla.perception_interface = None
        hla.rviz_interface = None
        hla.no_waits = True
        hla.head_perception_log_dir = Path("/tmp/test_head_log")
        return hla

    def test_sets_transfer_type_inside_no_hla(self):
        scene = self._make_scene("outside")
        apply_transfer_mode(
            {"transfer_mode": "inside mouth transfer"}, scene, {},
        )
        assert scene.transfer_type == "inside"

    def test_sets_transfer_type_outside_no_hla(self):
        scene = self._make_scene("inside")
        apply_transfer_mode(
            {"transfer_mode": "outside mouth transfer"}, scene, {},
        )
        assert scene.transfer_type == "outside"

    def test_unknown_transfer_mode_raises(self):
        scene = self._make_scene()
        with pytest.raises(ValueError, match="Unknown transfer_mode"):
            apply_transfer_mode({"transfer_mode": "sideways"}, scene, {})

    def test_no_transfer_mode_in_bundle_is_noop(self):
        scene = self._make_scene("outside")
        apply_transfer_mode({}, scene, {})
        assert scene.transfer_type == "outside"

    def test_no_transfer_hla_only_sets_scene(self):
        scene = self._make_scene("outside")
        apply_transfer_mode(
            {"transfer_mode": "inside mouth transfer"}, scene, {},
        )
        assert scene.transfer_type == "inside"


# -------------------------------------------------------------------
# apply_microwave_preference
# -------------------------------------------------------------------


class _FakeAtom:
    """Minimal stand-in for GroundAtom(FoodHeated, []) so tests don't need
    the heavy relational_structs import."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeAtom) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class TestApplyMicrowavePreference:

    def _food_heated(self) -> _FakeAtom:
        return _FakeAtom("FoodHeated")

    def test_no_microwave_adds_food_heated(self):
        atoms: set = set()
        fh = self._food_heated()
        result = apply_microwave_preference({"microwave_time": "no microwave"}, atoms, fh)
        assert fh in atoms
        assert result is None

    def test_1_min_removes_food_heated(self):
        fh = self._food_heated()
        atoms = {fh}
        result = apply_microwave_preference({"microwave_time": "1 min"}, atoms, fh)
        assert fh not in atoms
        assert result == 60

    def test_2_min_returns_120(self):
        fh = self._food_heated()
        atoms: set = set()
        result = apply_microwave_preference({"microwave_time": "2 min"}, atoms, fh)
        assert fh not in atoms
        assert result == 120

    def test_3_min_returns_180(self):
        fh = self._food_heated()
        atoms: set = set()
        result = apply_microwave_preference({"microwave_time": "3 min"}, atoms, fh)
        assert result == 180

    def test_no_microwave_time_in_bundle_is_noop(self):
        fh = self._food_heated()
        atoms: set = set()
        result = apply_microwave_preference({}, atoms, fh)
        assert fh not in atoms
        assert result is None

    def test_unknown_microwave_time_raises(self):
        fh = self._food_heated()
        with pytest.raises(ValueError, match="Unknown microwave_time"):
            apply_microwave_preference({"microwave_time": "5 min"}, set(), fh)

    def test_microwave_time_map_covers_all_options(self):
        assert set(_MICROWAVE_TIME_MAP.keys()) == {"no microwave", "1 min", "2 min", "3 min"}


# -------------------------------------------------------------------
# apply_dip_preference
# -------------------------------------------------------------------


class _FakeFlair:
    """Minimal stand-in for FLAIR so tests don't need heavy imports."""

    def __init__(self) -> None:
        self._allow_dip = True

    def set_allow_dip(self, allow: bool) -> None:
        self._allow_dip = allow


class TestApplyDipPreference:

    def test_do_not_dip_sets_false(self):
        flair = _FakeFlair()
        apply_dip_preference({"bite_dipping_preference": "do not dip"}, flair)
        assert flair._allow_dip is False

    def test_less_keeps_true(self):
        flair = _FakeFlair()
        apply_dip_preference({"bite_dipping_preference": "less"}, flair)
        assert flair._allow_dip is True

    def test_more_keeps_true(self):
        flair = _FakeFlair()
        apply_dip_preference({"bite_dipping_preference": "more"}, flair)
        assert flair._allow_dip is True

    def test_missing_key_is_noop(self):
        flair = _FakeFlair()
        apply_dip_preference({}, flair)
        assert flair._allow_dip is True

    def test_none_flair_is_noop(self):
        apply_dip_preference({"bite_dipping_preference": "do not dip"}, None)


# -------------------------------------------------------------------
# apply_occlusion_preference
# -------------------------------------------------------------------


class _FakeFlairWithPreference:
    """Minimal stand-in for FLAIR with get/set_preference."""

    def __init__(self, initial: str | None = None) -> None:
        self._preference = initial

    def get_preference(self):
        return self._preference

    def set_preference(self, pref: str):
        self._preference = pref


class TestApplyOcclusionPreference:

    def test_minimize_left_sets_hint(self):
        flair = _FakeFlairWithPreference()
        apply_occlusion_preference({"occlusion_relevance": "minimize left occlusion"}, flair)
        assert "left" in flair._preference

    def test_minimize_front_sets_hint(self):
        flair = _FakeFlairWithPreference()
        apply_occlusion_preference({"occlusion_relevance": "minimize front occlusion"}, flair)
        assert "front" in flair._preference

    def test_minimize_right_sets_hint(self):
        flair = _FakeFlairWithPreference()
        apply_occlusion_preference({"occlusion_relevance": "minimize right occlusion"}, flair)
        assert "right" in flair._preference

    def test_do_not_consider_is_noop(self):
        flair = _FakeFlairWithPreference()
        apply_occlusion_preference({"occlusion_relevance": "do not consider occlusion"}, flair)
        assert flair._preference is None

    def test_appends_to_existing_preference(self):
        flair = _FakeFlairWithPreference("I prefer chicken")
        apply_occlusion_preference({"occlusion_relevance": "minimize left occlusion"}, flair)
        assert flair._preference.startswith("I prefer chicken")
        assert "left" in flair._preference

    def test_missing_key_is_noop(self):
        flair = _FakeFlairWithPreference()
        apply_occlusion_preference({}, flair)
        assert flair._preference is None

    def test_none_flair_is_noop(self):
        apply_occlusion_preference({"occlusion_relevance": "minimize left occlusion"}, None)
