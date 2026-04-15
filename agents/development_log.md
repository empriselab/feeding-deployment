## Planned Integration Steps

### 1. Before a meal (or at start of a session) build `context` (`meal`/`setting`/`time_of_day`)

**Implementation**

`integration/preference_context.py`

- `build_preference_context(meal, setting, time_of_day)`
- `validate_preference_context(context)`

`integration/run.py` — `_Runner`

- `preference_context`
- `ensure_preference_context()`
- `set_meal_preference_context(meal, setting, time_of_day)`

**Operational flow (web deployment)**

1. Start the process with `--use_interface` and a non-empty `--pref_meal`, `--pref_setting`, and `--pref_time_of_day`.
2. `__main__` calls `set_meal_preference_context` — context lives on the runner.
3. `run()` calls `ensure_preference_context()` — confirms context is set, then continues.

---

### 2. Predict

- Instantiate `PredictionModel(user, physical_profile_label, logs_dir=log_dir/"preference_learning")`
- Call `predict_bundle(context, corrected={})` → `predicted_bundle`

**Implementation**

1. `PredictionModel` is instantiated in `run()` after `ensure_preference_context()`, with `logs_dir=self.log_dir / "preference_learning"` and the freeform physical profile description loaded from the `.txt` file.
2. `predict_bundle(context, corrected={})` is called, result stored on `self.predicted_bundle` and printed to stdout.
3. `self._prediction_model` is kept on the runner for later use.
4. Physical profile flows from `--physical_profile_file` through `_Runner` → `PredictionModel` → prompt templates as freeform text.
5. CLI requires `--physical_profile_file` and `--pref_meal` with `--use_interface`; missing either raises immediately.

**Modifications**

1. `integration/run.py`
    - `_Runner.__init__`: added new instance variables:
        - `self.deployment_user` — stores the user string.
        - `self.physical_profile_label` — stores the freeform physical profile text.
        - `self._prediction_model: PredictionModel | None` — initialized to `None`.
        - `self.predicted_bundle: dict[str, str] | None` — initialized to `None`.
    - `run()`: added a block between `ensure_preference_context()` and `ready_for_task_selection()` that constructs `PredictionModel` and calls `predict_bundle(dict(ctx), {})`, storing the result on `self.predicted_bundle`.
    - `__main__`: added `--physical_profile_file` argument. Required when `--use_interface` is set.
2. `preference_learning/methods/prediction_model.py` (`PredictionModel`)
    - `__init__`: added optional parameter `physical_profile_description`. Stored on `self.physical_profile_description`. Forwarded into `LongTermMemoryModel(...)` construction.
    - `predict_bundle`: forwards `self.physical_profile_description` into `get_bundle_prediction_prompt(...)`.
3. `preference_learning/methods/long_term_memory.py` (`LongTermMemoryModel`)
    - `__init__`: added optional parameter `physical_profile_description`. Stored on `self._physical_profile_description`.
    - `add_episode`: forwards `self._physical_profile_description` into `get_ltm_update_prompt(...)`.
    - `main()`: fixed a broken call — was constructing `LongTermMemoryModel` without `physical_profile_label` and calling a nonexistent `.reset()` method. Now uses the real `__init__` signature.
4. `preference_learning/methods/prompts/bundle_prediction.py`
    - `get_bundle_prediction_prompt`: added optional keyword-only parameter `physical_profile_description`. When provided and non-empty, uses it directly as the `{physical_profile}` template value instead of looking up `physical_profile_label` in `PHYSICAL_CAPABILITY_PROFILES`. When `None`, falls back to the original label-based lookup.
5. `preference_learning/methods/prompts/ltm_update.py`
    - `get_ltm_update_prompt`: same change as `get_bundle_prediction_prompt` — added optional `physical_profile_description`.

---

### 3. Show prediction + allow correction in the UI

- UI displays the predicted 18 fields.
- User edits any fields they want.
- Collect:
    - `ground_truth_bundle`
    - `corrected`

**Implementation**

1. `interfaces/web_interface.py`
    - `get_preference_corrections(predicted_bundle, pref_options) -> dict[str, str]`: new method. Docstring defines the ROS message contract for the future frontend page:
        - **Sends** to webapp: `{"state": "preference_correction", "status": "jump"}` then `{"state": "preference_correction_data", "predicted_bundle": {...}, "options": {...}}`.
        - **Expects back**: `{"state": "preference_correction_response", "bundle": {...}}` with all 18 fields and the user's final selections.
    - The ROS calls are **commented out** (frontend page does not exist yet). Currently returns `predicted_bundle` unchanged as a stub.
2. `integration/run.py`
    - `run()`: after `predict_bundle`, calls `web_interface.get_preference_corrections(predicted_bundle, PREF_OPTIONS)`. Then computes:
        - `self.ground_truth_bundle` — the full 18-field dict returned from the correction step.
        - `self.corrected` — only the fields where the returned value differs from the prediction.
    - Imports `PREF_OPTIONS` from `prediction_model.py` (already computed there from `PREFERENCE_BUNDLE` config; no config changes needed).

**When frontend is ready**

Uncomment the ROS lines in `get_preference_corrections`, remove the stub return, and build the webapp page that renders 18 dropdowns pre-filled with `predicted_bundle` values and sends back `preference_correction_response`.

---

### 4. Apply to the system

- Translate `ground_truth_bundle` into concrete BT parameter writes in the run's behavior-tree directory (`self.run_behavior_tree_dir`).
    - This makes the next executions use the updated parameters automatically as `execute_action()` loads the YAML fresh each time.
- For planner-level preferences:
    - Adjust `Runner.current_atoms` (e.g. add/remove `FoodHeated`) before calling `process_user_command()` for the first time in that meal.

**Design decisions**

1. **`robot_speed` vocabulary mismatch**: bundle uses `"slow"/"medium"/"fast"` but BT YAML `Speed` enum uses `"low"/"medium"/"high"`. Decision: map in code (`slow`->`low`, `fast`->`high`), keep both vocabularies as-is.
2. **`convey_robot_ready_*` "speech + LED"**: BT YAML enum only allows one of `silent`/`voice`/`led`/`beep`. Decision: skip for now with a runtime warning, fall back to `"voice"`. Handle later when BT supports combined cues.
3. **`wait_before_autocontinue_seconds` "1000 sec"**: exceeds original BT Box upper bound of 100.0. Decision: raise the YAML Box upper bound to 1000.0 in the three source templates.
4. **`outside_mouth_distance` label-to-float mapping**: bundle uses discrete labels, BT uses continuous `[0.1, 0.2]` meters. Decision: `near`=0.1, `medium`=0.15, `far`=0.2. `"not applicable"` skips the write.
5. **6 non-BT fields** (`retract_between_bites`, `bite_dipping_preference`, `microwave_time`, `occlusion_relevance`, `skewering_axis`, `transfer_mode` planner atoms): deferred to a later step.
6. **`transfer_mode`**: `TransferToolHLA.__init__` sets `self.transfer` once at startup. Decision: update `scene_description.transfer_type` and re-initialize the `TransferToolHLA.transfer` object with the correct `InsideMouthTransfer` or `OutsideMouthTransfer`.
7. **`microwave_time` -> `FoodHeated` atom manipulation**: deferred to a later step.

**Implementation**

1. `integration/apply_preferences.py` (new file)
    - Declarative mapping table (`_BT_MAPPING`): each entry maps a bundle field to a list of (YAML filename, BT parameter name, value translator).
    - Value translators: `_SPEED_MAP`, `_CONFIRMATION_MAP`, `_AUTOCONTINUE_MAP`, `_OUTSIDE_MOUTH_DISTANCE_MAP`, `_CONVEY_READY_MAP`, `_INITIATE_TRANSFER_MAP`, `_COMPLETE_TRANSFER_MAP`, `_TRANSFER_MODE_MAP`.
    - `apply_bundle_to_behavior_trees(bundle, bt_dir) -> list[str]`: iterates the mapping table, loads affected YAML files (with a custom loader that round-trips `!hla` tags), overwrites `value` entries, saves back. Returns warnings for edge cases.
    - `apply_transfer_mode(bundle, scene_description, hla_map)`: reads `bundle["transfer_mode"]`, sets `scene_description.transfer_type`, and re-instantiates `TransferToolHLA.transfer` with the correct `InsideMouthTransfer`/`OutsideMouthTransfer`.
2. `actions/behavior_trees/acquire_bite.yaml`, `transfer_utensil.yaml`, `transfer_drink.yaml`
    - Raised `TimeToWaitBeforeAutocontinue` Box upper bound from `100.0` to `1000.0`.
3. `integration/run.py`
    - `run()`: after computing `ground_truth_bundle` and `corrected`, calls `apply_bundle_to_behavior_trees` and `apply_transfer_mode` before `ready_for_task_selection()`. All subsequent `execute_action()` calls pick up the updated YAML values from disk.

**Bundle field -> BT parameter mapping (17 fields)**

| Bundle field | BT parameter | YAML files |
|---|---|---|
| `robot_speed` | `Speed` | all 29 YAMLs |
| `web_interface_confirmation` | `TransferAskForConfirmation` | `acquire_bite.yaml` |
| `web_interface_confirmation` | `AskForConfirmationInitiatingTransferSequence` | `transfer_drink.yaml`, `transfer_wipe.yaml` |
| `wait_before_autocontinue_seconds` | `TimeToWaitBeforeAutocontinue` | `acquire_bite.yaml`, `transfer_utensil.yaml`, `transfer_drink.yaml` |
| `outside_mouth_distance` | `OutsideMouthDistance` | `transfer_utensil.yaml`, `transfer_drink.yaml`, `transfer_wipe.yaml` |
| `convey_robot_ready_for_initiating_transfer` | `ReadyToInitiateTransferInteraction` | `transfer_utensil.yaml`, `transfer_drink.yaml`, `transfer_wipe.yaml` |
| `detect_user_ready_for_initiating_transfer_feeding` | `InitiateTransferInteraction` | `transfer_utensil.yaml` |
| `detect_user_ready_for_initiating_transfer_drinking` | `InitiateTransferInteraction` | `transfer_drink.yaml` |
| `detect_user_ready_for_initiating_transfer_wiping` | `InitiateTransferInteraction` | `transfer_wipe.yaml` |
| `convey_robot_ready_for_completing_transfer` | `ReadyForTransferInteraction` | `transfer_utensil.yaml`, `transfer_drink.yaml`, `transfer_wipe.yaml` |
| `detect_user_completed_transfer_feeding` | `TransferCompleteInteraction` | `transfer_utensil.yaml` |
| `detect_user_completed_transfer_drinking` | `TransferCompleteInteraction` | `transfer_drink.yaml` |
| `detect_user_completed_transfer_wiping` | `TransferCompleteInteraction` | `transfer_wipe.yaml` |
| `skewering_axis` | `SkeweringOrientation` | `acquire_bite.yaml` |
| `bite_dipping_preference` | `FoodDippingDepth` | `acquire_bite.yaml` |
| `microwave_time` | `MicrowaveDuration` | `press_microwave_button.yaml` |
| `retract_between_bites` | `RetractAfterTransfer` | `transfer_utensil.yaml` |

**`skewering_axis`** — clean enum-to-enum mapping via `_SKEWERING_AXIS_MAP`: `"parallel to major axis"` -> `"horizontal"`, `"perpendicular to major axis"` -> `"vertical"`. Targets `SkeweringOrientation` in `acquire_bite.yaml`.

**`bite_dipping_preference`** — label-to-float mapping via `_dipping_depth_translate()`: `"less"` -> `0.01` (Box minimum), `"more"` -> `0.03` (Box maximum). `"do not dip"` -> skip (returns `None`, leaves `FoodDippingDepth` at its default).

**Important caveat for `"do not dip"`**: skipping the `FoodDippingDepth` BT write does **not** prevent FLAIR's autonomous planner from choosing to dip. The dipping decision is made independently in `inference_class.py` (`get_autonomous_action`), which looks at plate contents and user preferences. To truly suppress dipping, a flag must be passed into FLAIR's planning logic — this is not yet implemented. This differs from `outside_mouth_distance = "not applicable"`, which is safe because the `transfer_mode` preference separately switches the execution path to `InsideMouthTransfer` (which never reads `OutsideMouthDistance`).

**`microwave_time`** — dual-layer integration (planner + BT):
- **Planner level** via `apply_microwave_preference(bundle, current_atoms, food_heated_atom)`:
  - `"no microwave"` -> adds `GroundAtom(FoodHeated, [])` to `current_atoms`, causing the PDDL planner to skip the entire microwave sequence. The BT write is also skipped (returns `None`).
  - `"1 min"` / `"2 min"` / `"3 min"` -> discards `FoodHeated` from `current_atoms` so the planner includes microwave steps.
- **BT level** via `_microwave_duration_translate` + `_BT_MAPPING`:
  - Added `MicrowaveDuration` Box parameter (30.0–300.0s, default 60.0) to `press_microwave_button.yaml`.
  - Updated `PressMicrowaveButtonHLA.press_microwave_button(self, speed, duration)` to accept the duration as a second positional argument.
  - `"1 min"` -> 60.0, `"2 min"` -> 120.0, `"3 min"` -> 180.0. Written to the YAML so when `execute_action` loads the BT, the duration flows through the `!hla` function binding into the HLA method.
  - `"no microwave"` -> skip (the HLA is never executed anyway since the planner excludes the action).

**`"speech + LED"` combined cue** — now fully supported:
- Added `"voice_led"` to the `Enum` elements for `ReadyToInitiateTransferInteraction` and `ReadyForTransferInteraction` in all 3 transfer YAMLs (`transfer_utensil.yaml`, `transfer_drink.yaml`, `transfer_wipe.yaml`).
- Added `voice_led` branches in `transfer_tool.py`: `relay_ready_to_initiate_transfer` (turns on LED + speaks the appropriate prompt) and `relay_ready_for_transfer` (turns on LED + speaks "Ready for transfer"). LED turn-off checks in `detect_transfer_initiated` and `detect_transfer_complete` updated to also trigger on `"voice_led"`.
- Updated `_CONVEY_READY_MAP`: `"speech + LED"` now maps to `"voice_led"` (was falling back to `"voice"` with a warning). Removed the fallback warning logic.

**`retract_between_bites`** — BT parameter on `transfer_utensil.yaml` only (bite transfers):
- Added `RetractAfterTransfer` Enum parameter (`[0, 1]`, default 0) to `transfer_utensil.yaml`.
- Updated `TransferToolHLA.transfer_utensil` to accept all 8 BT parameters explicitly. When `retract_after_transfer == 1`, calls `move_to_joint_positions(retract_pos)` at the end of the transfer, moving the robot to its rest position. When `0` (default), stays at the staging position near the user.
- Only applies to utensil (bite) transfers — drink and wipe don't have repeated transfer loops that benefit from a retract toggle.
- `_BT_MAPPING` entry: `retract_between_bites` → `RetractAfterTransfer` via `_RETRACT_MAP` (`"yes"` → 1, `"no"` → 0).

**`"do not dip"` FLAIR suppression** — deterministic override in `get_autonomous_action`:
- Added `allow_dip` boolean flag to `BiteAcquisitionInference` (default `True`).
- In `get_autonomous_action`: if `allow_dip is False` and the LLM returns a two-item list `['food', 'sauce']`, the dip is stripped and only the food item is used.
- Added `FLAIR.set_allow_dip(allow)` method that forwards to the inference server.
- Added `apply_dip_preference(bundle, flair)` in `apply_preferences.py`: when `bite_dipping_preference == "do not dip"`, sets `allow_dip=False`. For `"less"` or `"more"`, keeps `allow_dip=True` (the BT `FoodDippingDepth` parameter handles depth separately).
- Wired into `run.py` after the other preference applications.

**`occlusion_relevance`** — soft LLM hint via FLAIR's `user_preference` string:
- Added `apply_occlusion_preference(bundle, flair)` in `apply_preferences.py`.
- Maps preference values to natural-language hints: e.g. `"minimize left occlusion"` → `"When choosing which food to pick, prefer items that minimize the robot arm blocking the user's view from the left."`.
- `"do not consider occlusion"` → no-op (no hint appended).
- The hint is appended to FLAIR's existing `user_preference` string, which flows into the `PreferencePlanner` LLM prompt. This is a **soft** hint — the LLM may or may not factor it into food selection. There is no geometric enforcement; the current FLAIR architecture has no model of robot arm visibility from the user's perspective.
- **Limitation (pending discussion with Rajat):** This is the same enforcement level as other FLAIR food preferences (all LLM-based, none geometrically enforced). A stronger approach would require a perception/kinematic model to score candidate bites by arm obstruction from a given viewing direction — no such infrastructure exists today.

**All 18 preference fields are now wired.** No remaining deferred items for Step 4.

---

### 5. Learn

- After the correction loop, call `PredictionModel.update(day, context, corrected, ground_truth_bundle)`. This updates LTM/episodic/working memory so next time `predict_bundle()` is closer to the user's intent.

**Implementation**

1. `preference_learning/methods/prediction_model.py` (`PredictionModel`)
    - Added `next_day() -> int`: scans `working_memory_dir` for existing `day_NNNN.json` files and returns `max + 1` (or `1` if empty). Handles gaps in numbering by taking the max, not the count.
2. `integration/run.py`
    - Added `_pref_day: int | None` attribute to `_Runner.__init__` (passed from CLI).
    - In `run()`, after preferences are applied and before the task loop: computes `day` (either from `--pref_day` override or `next_day()` auto-detection), then calls `self._prediction_model.update(day, ctx, corrected, ground_truth_bundle)`.
    - Added `--pref_day` CLI argument (optional integer). When omitted, auto-detects from logs.

**What `update()` does**

1. Builds episode text from `(day, context, ground_truth_bundle)` — a natural-language representation of this meal session's preferences.
2. If LTM is enabled: feeds the episode into `LongTermMemoryModel.add_episode()`, which calls the LLM to update the cumulative preference summary. Writes `day_NNNN.json` to the LTM log directory.
3. If episodic memory is enabled: feeds the episode into `EpisodicMemoryModel.add_episode()`, which stores the episode embedding for future retrieval. Writes `day_NNNN.json` to the EM log directory.
4. Always writes a working memory record (`day_NNNN.json`) with context and corrections.

**Effect on the next session**: When `predict_bundle()` is called in the next meal, it retrieves the updated LTM summary and relevant episodic memories, producing predictions that incorporate what was learned from previous corrections.

---

### Interaction modes (`--pref_mode`)

Controls how the preference prediction/correction flow runs before a meal.

| Mode | Context input | Prediction + correction | Requires |
|---|---|---|---|
| `none` (default) | skipped | skipped | nothing extra |
| `terminal` | interactive numbered menus in terminal | LLM predicts, operator reviews each field and types a number or Enter to accept | `--physical_profile_file` |
| `interface` | from `--pref_meal`, `--pref_setting`, `--pref_time_of_day` CLI flags | LLM predicts, web frontend shows corrections (stub returns unchanged for now) | `--physical_profile_file`, `--pref_meal` |

**Implementation**

1. `integration/terminal_preferences.py` (new file)
    - `terminal_collect_context()`: prompts operator to pick meal, setting, time_of_day from numbered lists. Returns a context dict.
    - `terminal_correct_preferences(predicted_bundle, pref_options)`: shows each of the 18 predicted fields with numbered options. Operator presses Enter to accept or types a number to change. Returns the final bundle.
    - `_pick_from_list(prompt, options)`: helper that displays numbered options and loops until valid input.
2. `integration/run.py`
    - Added `--pref_mode` CLI flag (`none`/`terminal`/`interface`). Default `none`.
    - Added `_pref_mode` attribute to `_Runner.__init__`.
    - `run()`: branches on `_pref_mode`:
      - `none` → skips straight to `ready_for_task_selection()` (no personalization).
      - `terminal` → calls `terminal_collect_context()` + `terminal_correct_preferences()`, then predict → apply → learn.
      - `interface` → uses `web_interface.get_preference_corrections()` (existing stub), then predict → apply → learn.
    - `--physical_profile_file` now required for both `terminal` and `interface` modes.
    - `--pref_meal`/`--pref_setting`/`--pref_time_of_day` only required for `interface` mode; in `terminal` mode, context is collected interactively.

**Full end-to-end flow with `--pref_mode=terminal`:**

1. Operator starts: `python run.py --user alice --use_interface --pref_mode terminal --physical_profile_file alice_profile.txt`
2. Terminal prompts for meal, setting, time_of_day (numbered menus).
3. `PredictionModel` predicts 18-field bundle using LTM + episodic memory.
4. Terminal shows each field with predicted value marked; operator corrects or accepts.
5. `apply_bundle_to_behavior_trees` + `apply_transfer_mode` + FLAIR preferences applied.
6. `PredictionModel.update()` logs the episode for future learning.
7. Meal task loop begins.

**All 5 integration steps are now complete, with terminal interaction mode for testing.**
