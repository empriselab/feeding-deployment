## LLM Dataset Generator

Generate user encodings:

```bash
python data_generation/generate_user_preference_encoding.py --num-users 5 --output-dir data/user_encodings
```

Run the 30-day LLM-based dataset generator with:

```bash
python data_generation/generate_deployment_dataset_llm.py --user_encodings-dir data/user_encodings --output-dir data/deployment_datasets
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
python methods/evaluate_memory_model.py --data-dir generated-data \
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

