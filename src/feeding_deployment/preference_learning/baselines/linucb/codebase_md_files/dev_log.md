## dev_log.md (LinUCB baseline)

### 2026-03-29
- **Added**: `baselines/linucb/` initial pipeline + logging scaffolding.
  - **New files**: `context_encoder.py`, `linucb.py`, `evaluate_linucb.py`, `configs.py`, package `__init__.py`.
  - **Outputs**: evaluator writes `reports/linucb_eval_<timestamp>/report.json`, `report.txt`, and `logs/` JSONL files (`predictions.jsonl`, `updates.jsonl`).
- **Added**: automatic summary-metrics + comparison artifacts on each run.
  - **New outputs**: `summary_metrics.json` (LinUCB-only, comparable schema), and when available `comparison_with_ours.json` + `comparison_summary_metrics.png`.
- **Added**: generation of the 4 per-day comparison-style plots (`comparison_metrics/`) using LinUCB's simulated interactive loop metrics.
- **Added**: `alpha_grid_search.py` to sweep \u03b1 in a configurable grid (defaults 0.1..0.9 step 0.1) and select the best run by a chosen summary metric.

