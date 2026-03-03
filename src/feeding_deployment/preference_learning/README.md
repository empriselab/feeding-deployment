## LLM Dataset Generator

Run the 30-day LLM-based dataset generator with:

```bash
python generate_dataset_llm.py \
  --user User1 \
  --physical-profile severe_paralysis_clear_speech \
  --deployment-id User1 \
  --days 30 \
  --seed 42 \
  --variation-level 0.3 \
  --output-dir out
```

## Evaluate the memory system (full model)

- **Single user / single dataset file**:

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-file generated-data/User1__dep1__30d.json \
  --num-rollouts 1
```

- **All datasets in a directory** (default `generated-data/`):

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-dir generated-data \
  --num-rollouts 1
```

Both commands run the full memory model (LTM + EM + working memory). Reports (plots + JSON + txt log) are written to a timestamped subdirectory under `reports/`.

## Evaluate memory ablations

Run **full vs ablated** models as separate commands over all datasets in `generated-data`:

- **Full memory**:

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-dir generated-data \
  --num-rollouts 1 \
  --ablation full
```

- **LTM-only (no episodic retrieval)**:

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-dir generated-data \
  --num-rollouts 1 \
  --ablation ltm_only
```

- **EM-only (no long-term summary)**:

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-dir generated-data \
  --num-rollouts 1 \
  --ablation em_only
```

- **No-memory (no LTM, no EM; working-memory only)**:

```bash
python3 -u scripts/evaluate_memory_model.py \
  --data-dir generated-data \
  --num-rollouts 1 \
  --ablation no_memory
```

