## LinUCB baseline pipeline

### Purpose

This folder provides a **non-LLM contextual bandit baseline** for predicting the 19-dim preference bundle from context. It is designed to be comparable to the main evaluator by consuming the same dataset JSON and producing similar per-day metrics, plus detailed logs.

### Inputs

- **Dataset JSON**: one or more `.json` files with:
  - `days`: list of records
    - each record is either a single meal (has `context` + `preferences`) or contains `meals: [...]` where each meal has `context` + `preferences`.
  - `physical_profile_label` (optional but recommended): used as a one-hot feature when enabled.

### Feature encoding (`context_encoder.py`)

`ContextEncoder` builds a deterministic one-hot feature vector \(x_t\) from:

- `meal`
- `setting`
- `time_of_day`
- `transient_affective_state` (optional)
- `physical_profile_label` (optional)
- bias term

Vocabulary is seeded from `feeding_deployment.preference_learning.config` when available and extended from values observed in the dataset.

### Model (`linucb.py`)

`LinUCBPreferencePredictor` maintains **independent LinUCB bandits per dimension**:

- dimension \(d\): one bandit
- arms \(a\): allowed options for that dimension
- parameters per (dimension, arm): \(A_{d,a}\), \(b_{d,a}\)

Prediction chooses the arm with max UCB for each dimension independently.

Update uses a **semi-bandit** rule per meal:

- if predicted option equals ground truth: reward 1 on predicted arm
- else: reward 0 on predicted arm, reward 1 on the ground-truth arm

### Evaluation (`evaluate_linucb.py`)

Produces `reports/linucb_eval_<timestamp>/` containing:

- `report.json`: per-user metrics
- `report.txt`: stdout log
- `summary_metrics.json`: LinUCB summary statistics in the same schema used by `methods/comparison_metrics/summary_metrics.json`
- `comparison_with_ours.json`: (if available) merges `Ours` summary with `LinUCB` summary
- `comparison_summary_acc_m0.png` and `comparison_summary_m_star.png`: (if matplotlib available) separate bar plots (accuracy vs correction burden)
- `comparison_metrics/` (if matplotlib available): 4 plots in the same style/names as the methods comparison figures:
  - `comparison_acc_m0_by_day.png`
  - `comparison_m_star_by_day.png`
  - `comparison_interactive_benefit_by_day.png`
  - `comparison_single_correction_gain_by_day.png`
- `logs/`:
  - `predictions.jsonl`: context + predicted bundle + truth + mismatch list
  - `updates.jsonl`: list of per-dimension arm updates and rewards

### How to run

From `src/feeding_deployment/preference_learning`:

```bash
python3 -u baselines/linucb/evaluate_linucb.py --data-file ./data/<dataset>.json
```

Or:

```bash
python3 -u baselines/linucb/evaluate_linucb.py --data-dir ./data/deployment_datasets
```

### Output location (important)

LinUCB outputs are always written under the LinUCB baseline folder (independent of your current working directory):

- `baselines/linucb/reports/linucb_eval_<timestamp>/...`

### Alpha grid search

Run a simple grid search over \u03b1 (defaults: 0.1..0.9 step 0.1):

```bash
python3 -u baselines/linucb/alpha_grid_search.py \
  --data-dir ./data/deployment_datasets \
  --ours-summary-metrics ./methods/comparison_metrics/summary_metrics.json
```

This writes:

- `baselines/linucb/reports/alpha_grid_<timestamp>/grid_search_results.json`
- per-\u03b1 stdout/stderr logs in the same folder

