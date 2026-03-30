# LinUCB Baseline (Preference Bundle Prediction)

This folder contains a **non-LLM contextual bandit baseline** (LinUCB) for predicting the 19-dimensional user preference bundle from categorical mealtime context (and optionally affective state + physical capability profile label).

## What it predicts

For each meal/day, it outputs a preference bundle:

- one chosen option for every preference dimension in `config/preference_bundle.py`
- using an independent LinUCB bandit per dimension

## Requirements

This baseline uses numpy and (optionally) matplotlib.

```bash
pip install numpy
pip install matplotlib
```

If matplotlib is not installed, the evaluator still writes JSON reports and logs; plots will be skipped.

## Run the evaluator

From `src/feeding_deployment/preference_learning`:

```bash
cd "/Users/frankyang/Desktop/EmPRISE/[Winter 2026 - Present] Long-Term Deployment with Personalization Project (with Rajat)/feeding-deployment/src/feeding_deployment/preference_learning"

python3 -u baselines/linucb/evaluate_linucb.py \
  --data-dir ./data/deployment_datasets \
  --ours-summary-metrics ./methods/comparison_metrics/summary_metrics.json \
  --alpha 0.5
```

Or for a single dataset file:

```bash
python3 -u baselines/linucb/evaluate_linucb.py \
  --data-file ./data/deployment_datasets/user_1__profile_limited_arms_no_trunk_good_head.json \
  --alpha 0.5
```

### Feature toggles

- Include affective state (default): `--include-affective-state`
- Exclude affective state: `--no-include-affective-state`
- Include physical profile label (default): `--include-profile-label`
- Exclude physical profile label: `--no-include-profile-label`

## Outputs

Outputs are always written under this folder (independent of your current working directory):

```text
baselines/linucb/reports/linucb_eval_<timestamp>/
  report.json
  report.txt
  summary_metrics.json
  comparison_with_ours.json            (if ours-summary-metrics is loadable)
  comparison_summary_acc_m0.png
  comparison_summary_m_star.png
  comparison_metrics/
    comparison_acc_m0_by_day.png
    comparison_m_star_by_day.png
    comparison_interactive_benefit_by_day.png
    comparison_single_correction_gain_by_day.png
  logs/
    predictions.jsonl
    updates.jsonl
```

## Alpha grid search

To sweep `alpha` over `0.1..0.9` (step `0.1`) and pick the best run by a chosen metric:

```bash
python3 -u baselines/linucb/alpha_grid_search.py \
  --data-dir ./data/deployment_datasets \
  --ours-summary-metrics ./methods/comparison_metrics/summary_metrics.json \
  --metric mean_acc_m0_all_days \
  --objective max
```

Results are written to:

```text
baselines/linucb/reports/alpha_grid_<timestamp>/grid_search_results.json
```

