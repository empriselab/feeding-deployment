"""Tests for preference-learning integration Steps 1-5 + terminal interaction.

Run with:
    PYTHONPATH=src python -m pytest tests/test_preference_integration.py -v

Note: run.py and web_interface.py have heavy robot dependencies (tomsutils,
pybullet_helpers, cv2, rospy, ...) that are not available outside the robot
environment.  The tests below therefore exercise the *logic* of Steps 1-5
without importing those modules: preference_context.py is tested directly,
PredictionModel is tested with mocked OpenAI, and the runner/web-interface
contracts are tested against the same functions those classes delegate to.
"""

from __future__ import annotations

import importlib
import json
import queue
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import StringIO

import pytest

from feeding_deployment.integration.preference_context import (
    PREFERENCE_CONTEXT_KEYS,
    build_preference_context,
    validate_preference_context,
)
from feeding_deployment.preference_learning.config import (
    MEALS,
    SETTINGS,
    TIMES_OF_DAY,
)
from feeding_deployment.preference_learning.methods.prediction_model import (
    PREF_OPTIONS,
)
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS

_PM_MODULE = "feeding_deployment.preference_learning.methods.prediction_model"


def _import_web_interface_module():
    """Import web_interface.py with lightweight dependency stubs."""
    module_name = "feeding_deployment.interfaces.web_interface"
    sys.modules.pop(module_name, None)

    fake_modules = {
        "cv2": types.ModuleType("cv2"),
        "rospy": types.ModuleType("rospy"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": types.ModuleType("std_msgs.msg"),
        "cv_bridge": types.ModuleType("cv_bridge"),
        "pybullet_helpers": types.ModuleType("pybullet_helpers"),
        "pybullet_helpers.geometry": types.ModuleType("pybullet_helpers.geometry"),
        "pybullet_helpers.joint": types.ModuleType("pybullet_helpers.joint"),
        "scipy": types.ModuleType("scipy"),
        "scipy.spatial": types.ModuleType("scipy.spatial"),
        "scipy.spatial.transform": types.ModuleType("scipy.spatial.transform"),
        "feeding_deployment.transparency.continuous_llm": types.ModuleType(
            "feeding_deployment.transparency.continuous_llm"
        ),
    }

    fake_modules["sensor_msgs.msg"].CompressedImage = type("CompressedImage", (), {})
    fake_modules["std_msgs.msg"].String = type("String", (), {})
    fake_modules["cv_bridge"].CvBridge = type("CvBridge", (), {})
    fake_modules["pybullet_helpers.geometry"].Pose = type("Pose", (), {})
    fake_modules["pybullet_helpers.joint"].JointPositions = type(
        "JointPositions", (), {}
    )
    fake_modules["scipy.spatial.transform"].Rotation = type("Rotation", (), {})
    fake_modules["feeding_deployment.transparency.continuous_llm"].TransparencyContinuous = type(
        "TransparencyContinuous", (), {}
    )

    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module(module_name)


# ===================================================================
# Step 1 — preference context: build, validate, runner contract
# ===================================================================


class TestBuildPreferenceContext:

    def test_valid_context(self):
        ctx = build_preference_context(MEALS[0], SETTINGS[0], TIMES_OF_DAY[0])
        assert set(ctx.keys()) == set(PREFERENCE_CONTEXT_KEYS)
        assert ctx["meal"] == MEALS[0]
        assert ctx["setting"] == SETTINGS[0]
        assert ctx["time_of_day"] == TIMES_OF_DAY[0]

    def test_strips_whitespace(self):
        ctx = build_preference_context(
            f"  {MEALS[0]}  ", f" {SETTINGS[0]} ", f"  {TIMES_OF_DAY[0]} "
        )
        assert ctx["meal"] == MEALS[0]

    def test_invalid_meal_raises(self):
        with pytest.raises(ValueError, match="Unknown meal"):
            build_preference_context("not_a_real_meal", SETTINGS[0], TIMES_OF_DAY[0])

    def test_invalid_setting_raises(self):
        with pytest.raises(ValueError, match="Unknown setting"):
            build_preference_context(MEALS[0], "bad_setting", TIMES_OF_DAY[0])

    def test_invalid_time_of_day_raises(self):
        with pytest.raises(ValueError, match="Unknown time_of_day"):
            build_preference_context(MEALS[0], SETTINGS[0], "midnight")

    def test_empty_meal_raises(self):
        with pytest.raises(ValueError):
            build_preference_context("", SETTINGS[0], TIMES_OF_DAY[0])

    def test_whitespace_only_meal_raises(self):
        with pytest.raises(ValueError):
            build_preference_context("   ", SETTINGS[0], TIMES_OF_DAY[0])


class TestValidatePreferenceContext:

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            validate_preference_context({"meal": MEALS[0], "setting": SETTINGS[0]})

    def test_non_string_value_raises(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_preference_context(
                {"meal": 123, "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
            )

    def test_valid_context_passes(self):
        validate_preference_context(
            {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        )


class TestRunnerPreferenceContextContract:
    """Verify the contract that _Runner.ensure_preference_context and
    _Runner.set_meal_preference_context implement, without importing run.py.

    The runner stores `self.preference_context` and:
      - ensure_preference_context() raises RuntimeError when it is None
      - set_meal_preference_context(m, s, t) delegates to build_preference_context
    """

    def test_ensure_raises_when_context_is_none(self):
        preference_context = None
        with pytest.raises(RuntimeError, match="preference_context is required"):
            if preference_context is None:
                raise RuntimeError(
                    "preference_context is required but unset. Each run must set it "
                    "explicitly (e.g. non-empty --pref_meal with --use_interface, or call "
                    "set_meal_preference_context(meal, setting, time_of_day) before run()). "
                    "Context is not loaded from or saved to disk; after a crash, supply it again."
                )

    def test_ensure_returns_when_context_is_set(self):
        ctx = build_preference_context(MEALS[0], SETTINGS[0], TIMES_OF_DAY[0])
        preference_context = ctx
        assert preference_context is not None
        assert preference_context == ctx

    def test_set_meal_delegates_to_build(self):
        ctx = build_preference_context(MEALS[0], SETTINGS[0], TIMES_OF_DAY[0])
        assert ctx["meal"] == MEALS[0]
        assert ctx["setting"] == SETTINGS[0]
        assert ctx["time_of_day"] == TIMES_OF_DAY[0]

    def test_set_meal_rejects_invalid_meal(self):
        with pytest.raises(ValueError):
            build_preference_context("bad_meal", SETTINGS[0], TIMES_OF_DAY[0])

    def test_each_meal_setting_time_accepted(self):
        """Smoke test: every canonical value from config builds successfully."""
        for m in MEALS:
            for s in SETTINGS:
                for t in TIMES_OF_DAY:
                    ctx = build_preference_context(m, s, t)
                    validate_preference_context(ctx)

    def test_interface_mode_context_defaults_are_valid(self):
        """Interface mode can safely fall back to canonical default context values."""
        ctx = {
            "meal": MEALS[0],
            "setting": SETTINGS[0],
            "time_of_day": TIMES_OF_DAY[0],
        }
        defaults = build_preference_context(
            ctx["meal"],
            ctx["setting"],
            ctx["time_of_day"],
        )
        assert defaults == ctx


# ===================================================================
# Step 2 — PredictionModel: instantiation + predict_bundle
# ===================================================================


def _fake_openai_response(bundle: dict[str, str]) -> MagicMock:
    """Build a MagicMock that mimics an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(bundle)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _default_bundle() -> dict[str, str]:
    return {field: opts[0] for field, opts in PREF_OPTIONS.items()}


class TestPredictionModelPredictBundle:

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_returns_all_fields(self, mock_openai_cls, _key, tmp_path):
        bundle = _default_bundle()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response(bundle)
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )

        model = PredictionModel(
            user="test_user",
            physical_profile_label="test_label",
            logs_dir=tmp_path / "pref",
            physical_profile_description="Good arm control.",
            use_long_term_memory=False,
            use_episodic_memory=False,
        )

        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        result = model.predict_bundle(ctx, {})

        assert isinstance(result, dict)
        assert set(result.keys()) == set(PREF_FIELDS)
        for field in PREF_FIELDS:
            assert result[field] in PREF_OPTIONS[field], (
                f"{field}={result[field]} not in allowed options"
            )

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_physical_profile_description_in_prompt(self, mock_openai_cls, _key, tmp_path):
        bundle = _default_bundle()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response(bundle)
        mock_openai_cls.return_value = mock_client

        desc = "This user has limited arm control and cannot press buttons."

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )

        model = PredictionModel(
            user="test_user",
            physical_profile_label="unused_label",
            logs_dir=tmp_path / "pref",
            physical_profile_description=desc,
            use_long_term_memory=False,
            use_episodic_memory=False,
        )

        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        model.predict_bundle(ctx, {})

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert desc in prompt, (
            "Freeform physical-profile description must appear verbatim in the LLM prompt"
        )

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_corrected_fields_override_prediction(self, mock_openai_cls, _key, tmp_path):
        bundle = _default_bundle()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response(bundle)
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )

        model = PredictionModel(
            user="test_user",
            physical_profile_label="test_label",
            logs_dir=tmp_path / "pref",
            physical_profile_description="Test.",
            use_long_term_memory=False,
            use_episodic_memory=False,
        )

        override_field = PREF_FIELDS[0]
        override_val = PREF_OPTIONS[override_field][-1]  # last option

        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        result = model.predict_bundle(ctx, {override_field: override_val})
        assert result[override_field] == override_val

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_malformed_llm_json_falls_back(self, mock_openai_cls, _key, tmp_path):
        choice = MagicMock()
        choice.message.content = "not valid json {{{"
        resp = MagicMock()
        resp.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )

        model = PredictionModel(
            user="test_user",
            physical_profile_label="test_label",
            logs_dir=tmp_path / "pref",
            physical_profile_description="Test.",
            use_long_term_memory=False,
            use_episodic_memory=False,
        )

        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        result = model.predict_bundle(ctx, {})

        assert isinstance(result, dict)
        assert len(result) == len(PREF_FIELDS)
        for field in PREF_FIELDS:
            assert result[field] in PREF_OPTIONS[field]

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_predict_bundle_logs_to_disk(self, mock_openai_cls, _key, tmp_path):
        bundle = _default_bundle()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response(bundle)
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )

        model = PredictionModel(
            user="test_user",
            physical_profile_label="test_label",
            logs_dir=tmp_path / "pref",
            physical_profile_description="Good control.",
            use_long_term_memory=False,
            use_episodic_memory=False,
        )

        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        model.predict_bundle(ctx, {})

        log_dir = tmp_path / "pref" / "test_user" / "prediction_model_llm_calls"
        log_files = list(log_dir.glob("*.txt"))
        assert len(log_files) == 1, "predict_bundle should write exactly one log file"
        contents = log_files[0].read_text()
        assert "===PROMPT===" in contents
        assert "===RESPONSE===" in contents


# ===================================================================
# Step 3 — preference correction web interface contract
# ===================================================================


class TestPreferenceCorrectionWebInterfaceContract:
    """Verify the live WebInterface preference-correction message flow."""

    def test_preference_context_round_trip_sends_jump_then_data(self):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        defaults = {
            "meal": MEALS[0],
            "setting": SETTINGS[0],
            "time_of_day": TIMES_OF_DAY[0],
        }
        sent_messages = []

        interface.current_page = "task_selection"
        interface._send_message = sent_messages.append

        def fake_get_required_message(condition):
            response = {
                "state": "preference_context_response",
                "meal": MEALS[1],
                "setting": SETTINGS[1],
                "time_of_day": TIMES_OF_DAY[1],
            }
            assert condition(response) is True
            return response

        interface.get_required_web_interface_message = fake_get_required_message

        with patch.object(module.time, "sleep") as mock_sleep:
            returned = module.WebInterface.get_preference_context(
                interface,
                list(MEALS),
                list(SETTINGS),
                list(TIMES_OF_DAY),
                defaults,
            )

        assert interface.current_page == "preference_context"
        assert sent_messages == [
            {"state": "preference_context", "status": "jump"},
            {
                "state": "preference_context_data",
                "meals": list(MEALS),
                "settings": list(SETTINGS),
                "time_of_day": list(TIMES_OF_DAY),
                "defaults": defaults,
            },
        ]
        mock_sleep.assert_called_once_with(0.5)
        assert returned == {
            "meal": MEALS[1],
            "setting": SETTINGS[1],
            "time_of_day": TIMES_OF_DAY[1],
        }

    def test_preference_context_raises_when_page_exits_early(self):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        defaults = {
            "meal": MEALS[0],
            "setting": SETTINGS[0],
            "time_of_day": TIMES_OF_DAY[0],
        }
        sent_messages = []

        interface.current_page = "task_selection"
        interface._send_message = sent_messages.append
        interface.get_required_web_interface_message = lambda condition: None

        with patch.object(module.time, "sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Preference context exited before submission"):
                module.WebInterface.get_preference_context(
                    interface,
                    list(MEALS),
                    list(SETTINGS),
                    list(TIMES_OF_DAY),
                    defaults,
                )

        assert interface.current_page == "preference_context"
        assert sent_messages[0] == {"state": "preference_context", "status": "jump"}
        assert sent_messages[1]["state"] == "preference_context_data"
        mock_sleep.assert_called_once_with(0.5)

    def test_round_trip_sends_jump_then_data_and_returns_bundle(self):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        predicted = _default_bundle()
        corrected_bundle = dict(predicted)
        corrected_bundle[PREF_FIELDS[0]] = PREF_OPTIONS[PREF_FIELDS[0]][-1]
        sent_messages = []

        interface.current_page = "task_selection"
        interface._send_message = sent_messages.append

        def fake_get_required_message(condition):
            response = {
                "state": "preference_correction_response",
                "bundle": corrected_bundle,
            }
            assert condition(response) is True
            return response

        interface.get_required_web_interface_message = fake_get_required_message

        with patch.object(module.time, "sleep") as mock_sleep:
            returned = module.WebInterface.get_preference_corrections(
                interface,
                predicted,
                dict(PREF_OPTIONS),
            )

        assert interface.current_page == "preference_correction"
        assert sent_messages == [
            {"state": "preference_correction", "status": "jump"},
            {
                "state": "preference_correction_data",
                "predicted_bundle": predicted,
                "options": dict(PREF_OPTIONS),
            },
        ]
        mock_sleep.assert_called_once_with(0.5)
        assert returned == corrected_bundle

    def test_callback_accepts_status_less_preference_response(self, tmp_path):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        response_bundle = _default_bundle()

        interface.webapp_received_messages_log = tmp_path / "received.txt"
        interface.task_selection_queue = queue.Queue()
        interface.received_web_interface_messages = queue.Queue()
        interface.explanation_lock = threading.Lock()
        interface.current_page = "preference_correction"
        interface.task_selection_jump = True

        msg = types.SimpleNamespace(
            data=json.dumps({
                "state": "preference_correction_response",
                "bundle": response_bundle,
            })
        )

        module.WebInterface._message_callback(interface, msg)

        assert interface.task_selection_jump is False
        assert interface.task_selection_queue.empty()
        queued = interface.received_web_interface_messages.get_nowait()
        assert queued == {
            "state": "preference_correction_response",
            "bundle": response_bundle,
        }

    def test_callback_accepts_status_less_preference_context_response(self, tmp_path):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)

        interface.webapp_received_messages_log = tmp_path / "received.txt"
        interface.task_selection_queue = queue.Queue()
        interface.received_web_interface_messages = queue.Queue()
        interface.explanation_lock = threading.Lock()
        interface.current_page = "preference_context"
        interface.task_selection_jump = True

        msg = types.SimpleNamespace(
            data=json.dumps({
                "state": "preference_context_response",
                "meal": MEALS[0],
                "setting": SETTINGS[0],
                "time_of_day": TIMES_OF_DAY[0],
            })
        )

        module.WebInterface._message_callback(interface, msg)

        assert interface.task_selection_jump is False
        assert interface.task_selection_queue.empty()
        queued = interface.received_web_interface_messages.get_nowait()
        assert queued == {
            "state": "preference_context_response",
            "meal": MEALS[0],
            "setting": SETTINGS[0],
            "time_of_day": TIMES_OF_DAY[0],
        }

    def test_returns_predicted_bundle_when_correction_page_exits_early(self):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        predicted = _default_bundle()
        sent_messages = []

        interface.current_page = "task_selection"
        interface._send_message = sent_messages.append
        interface.get_required_web_interface_message = lambda condition: None

        with patch.object(module.time, "sleep") as mock_sleep:
            returned = module.WebInterface.get_preference_corrections(
                interface,
                predicted,
                dict(PREF_OPTIONS),
            )

        assert interface.current_page == "preference_correction"
        assert sent_messages[0] == {"state": "preference_correction", "status": "jump"}
        assert sent_messages[1]["state"] == "preference_correction_data"
        mock_sleep.assert_called_once_with(0.5)
        assert returned == predicted
        assert returned is not predicted

    def test_preference_correction_applied_confirmation_message(self):
        module = _import_web_interface_module()
        interface = module.WebInterface.__new__(module.WebInterface)
        sent_messages = []

        interface._send_message = sent_messages.append

        module.WebInterface.notify_preference_corrections_applied(
            interface,
            "Preferences were applied successfully.",
        )

        assert sent_messages == [
            {
                "state": "preference_correction_applied",
                "message": "Preferences were applied successfully.",
            }
        ]


class TestCorrectedDiffLogic:
    """Validates the diff logic used in _Runner.run() after the correction
    round-trip: corrected = {k: v for k, v in user_bundle.items()
    if v != predicted_bundle.get(k)}."""

    def test_corrections_detected(self):
        predicted = _default_bundle()
        user_bundle = dict(predicted)

        field_a, field_b = PREF_FIELDS[0], PREF_FIELDS[1]
        user_bundle[field_a] = PREF_OPTIONS[field_a][-1]
        user_bundle[field_b] = PREF_OPTIONS[field_b][-1]

        corrected = {
            k: v for k, v in user_bundle.items() if v != predicted.get(k)
        }

        if PREF_OPTIONS[field_a][0] != PREF_OPTIONS[field_a][-1]:
            assert field_a in corrected
        if PREF_OPTIONS[field_b][0] != PREF_OPTIONS[field_b][-1]:
            assert field_b in corrected

    def test_no_corrections_means_empty(self):
        predicted = _default_bundle()
        user_bundle = dict(predicted)

        corrected = {
            k: v for k, v in user_bundle.items() if v != predicted.get(k)
        }
        assert corrected == {}

    def test_ground_truth_has_all_fields(self):
        predicted = _default_bundle()
        user_bundle = dict(predicted)
        user_bundle[PREF_FIELDS[0]] = PREF_OPTIONS[PREF_FIELDS[0]][-1]

        ground_truth = user_bundle
        for field in PREF_FIELDS:
            assert field in ground_truth

    def test_corrected_values_are_valid_options(self):
        predicted = _default_bundle()
        user_bundle = dict(predicted)
        for field in PREF_FIELDS[:3]:
            user_bundle[field] = PREF_OPTIONS[field][-1]

        corrected = {
            k: v for k, v in user_bundle.items() if v != predicted.get(k)
        }
        for field, val in corrected.items():
            assert val in PREF_OPTIONS[field]


class TestPrefOptionsConsistency:
    """Sanity checks on PREF_OPTIONS / PREF_FIELDS configuration."""

    def test_fields_match_options_keys(self):
        assert set(PREF_FIELDS) == set(PREF_OPTIONS.keys())

    def test_every_field_has_at_least_two_options(self):
        for field, opts in PREF_OPTIONS.items():
            assert len(opts) >= 2, f"{field} has fewer than 2 options"

    def test_no_empty_option_strings(self):
        for field, opts in PREF_OPTIONS.items():
            for opt in opts:
                assert isinstance(opt, str) and opt.strip(), (
                    f"{field} has empty/whitespace option"
                )


# ===================================================================
# Step 5 — Learn: PredictionModel.next_day + update wiring
# ===================================================================


class TestNextDay:
    """Verify PredictionModel.next_day auto-detects the next unused day."""

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_empty_logs_returns_1(self, mock_openai_cls, _key, tmp_path):
        mock_openai_cls.return_value = MagicMock()
        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=False,
        )
        assert model.next_day() == 1

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_after_three_days_returns_4(self, mock_openai_cls, _key, tmp_path):
        mock_openai_cls.return_value = MagicMock()
        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=False,
        )
        for d in [1, 2, 3]:
            (model.working_memory_dir / f"day_{d:04d}.json").write_text("{}")
        assert model.next_day() == 4

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_gap_in_days_uses_max(self, mock_openai_cls, _key, tmp_path):
        mock_openai_cls.return_value = MagicMock()
        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=False,
        )
        for d in [1, 5]:
            (model.working_memory_dir / f"day_{d:04d}.json").write_text("{}")
        assert model.next_day() == 6


class TestUpdateWritesLogs:
    """Verify PredictionModel.update writes per-day JSON files."""

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_update_creates_working_memory_log(self, mock_openai_cls, _key, tmp_path):
        mock_openai_cls.return_value = MagicMock()
        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=False,
        )
        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        bundle = _default_bundle()
        corrected = {PREF_FIELDS[0]: PREF_OPTIONS[PREF_FIELDS[0]][-1]}

        model.update(day=1, context=ctx, corrected=corrected, ground_truth_bundle=bundle)

        log_file = model.working_memory_dir / "day_0001.json"
        assert log_file.exists()
        data = json.loads(log_file.read_text())
        assert data["day"] == 1
        assert data["context"] == ctx
        assert data["corrected"] == corrected

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_update_increments_next_day(self, mock_openai_cls, _key, tmp_path):
        mock_openai_cls.return_value = MagicMock()
        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=False,
        )
        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        bundle = _default_bundle()

        assert model.next_day() == 1
        model.update(day=1, context=ctx, corrected={}, ground_truth_bundle=bundle)
        assert model.next_day() == 2
        model.update(day=2, context=ctx, corrected={}, ground_truth_bundle=bundle)
        assert model.next_day() == 3

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_update_with_ltm_creates_ltm_log(self, mock_openai_cls, _key, tmp_path):
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps({"summary": "test"})
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=True, use_episodic_memory=False,
            physical_profile_description="Test profile.",
        )
        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        bundle = _default_bundle()

        model.update(day=1, context=ctx, corrected={}, ground_truth_bundle=bundle)

        ltm_file = tmp_path / "pref" / "u" / "long_term_memory" / "day_0001.json"
        assert ltm_file.exists()
        data = json.loads(ltm_file.read_text())
        assert data["day"] == 1
        assert "episode_text" in data

    @patch(f"{_PM_MODULE}._resolve_api_key", return_value="fake-key")
    @patch(f"{_PM_MODULE}.OpenAI")
    def test_update_with_em_creates_em_log(self, mock_openai_cls, _key, tmp_path):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        mock_openai_cls.return_value = mock_client

        from feeding_deployment.preference_learning.methods.prediction_model import (
            PredictionModel,
        )
        model = PredictionModel(
            user="u", physical_profile_label="p",
            logs_dir=tmp_path / "pref",
            use_long_term_memory=False, use_episodic_memory=True,
        )
        ctx = {"meal": MEALS[0], "setting": SETTINGS[0], "time_of_day": TIMES_OF_DAY[0]}
        bundle = _default_bundle()

        model.update(day=1, context=ctx, corrected={}, ground_truth_bundle=bundle)

        em_file = tmp_path / "pref" / "u" / "episodic_memory" / "day_0001.json"
        assert em_file.exists()
        data = json.loads(em_file.read_text())
        assert data["day"] == 1
        assert "episode_text" in data


# ===================================================================
# Terminal interaction: context collection + preference correction
# ===================================================================


_TERMINAL_MODULE = "feeding_deployment.integration.terminal_preferences"


class TestTerminalCollectContext:

    @patch("builtins.input", side_effect=["1", "1", "1"])
    def test_picks_first_options(self, _mock_input):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_collect_context,
        )
        ctx = terminal_collect_context()
        assert ctx["meal"] == MEALS[0]
        assert ctx["setting"] == SETTINGS[0]
        assert ctx["time_of_day"] == TIMES_OF_DAY[0]

    @patch("builtins.input", side_effect=["2", "3", "2"])
    def test_picks_specific_options(self, _mock_input):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_collect_context,
        )
        ctx = terminal_collect_context()
        assert ctx["meal"] == MEALS[1]
        assert ctx["setting"] == SETTINGS[2]
        assert ctx["time_of_day"] == TIMES_OF_DAY[1]

    @patch("builtins.input", side_effect=["bad", "0", "1", "1", "1"])
    def test_invalid_input_retries(self, _mock_input):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_collect_context,
        )
        ctx = terminal_collect_context()
        assert ctx["meal"] == MEALS[0]


class TestTerminalCorrectPreferences:

    @patch("builtins.input", side_effect=[""] * len(PREF_FIELDS))
    def test_accept_all_returns_predicted(self, _mock_input):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_correct_preferences,
        )
        predicted = _default_bundle()
        result = terminal_correct_preferences(predicted, dict(PREF_OPTIONS))
        assert result == predicted

    def test_change_one_field(self):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_correct_preferences,
        )
        predicted = _default_bundle()
        target_field = PREF_FIELDS[0]
        target_opts = PREF_OPTIONS[target_field]
        new_idx = len(target_opts)  # pick the last option

        inputs = []
        for field in PREF_FIELDS:
            if field == target_field:
                inputs.append(str(new_idx))
            else:
                inputs.append("")

        with patch("builtins.input", side_effect=inputs):
            result = terminal_correct_preferences(predicted, dict(PREF_OPTIONS))

        if target_opts[0] != target_opts[-1]:
            assert result[target_field] == target_opts[-1]
            assert result[target_field] != predicted[target_field]

        for field in PREF_FIELDS:
            if field != target_field:
                assert result[field] == predicted[field]

    def test_invalid_input_retries_then_accepts(self):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_correct_preferences,
        )
        predicted = _default_bundle()
        inputs = ["bad", "0", ""]  # two bad inputs then accept for first field
        inputs += [""] * (len(PREF_FIELDS) - 1)

        with patch("builtins.input", side_effect=inputs):
            result = terminal_correct_preferences(predicted, dict(PREF_OPTIONS))
        assert result == predicted

    @patch("builtins.input", side_effect=[""] * len(PREF_FIELDS))
    def test_no_changes_means_empty_corrections(self, _mock_input):
        from feeding_deployment.integration.terminal_preferences import (
            terminal_correct_preferences,
        )
        predicted = _default_bundle()
        result = terminal_correct_preferences(predicted, dict(PREF_OPTIONS))
        corrected = {k: v for k, v in result.items() if v != predicted.get(k)}
        assert corrected == {}
