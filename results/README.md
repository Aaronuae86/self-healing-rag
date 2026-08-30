# Benchmark outputs

`notebooks/06b_squad_retrieval_benchmark.ipynb` writes the reproducible sample
manifest and lightweight retrieval results here at runtime:

- `squad_retrieval_sample_manifest.json`
- `squad_retrieval_metrics.json`
- `squad_retrieval_per_example.csv`

Generated metrics and per-example rows are ignored by Git. The small sample
manifest is intentionally trackable so an executed benchmark sample can be
preserved. Dataset caches, model weights, embeddings, and FAISS indices are
stored under `.cache/` and are also ignored.

`notebooks/06c_squad_stress_benchmark.ipynb` adds Chunk 2 artifacts:

- `squad_stress_manifest.json` — fixed train/validation IDs, stable corpus IDs,
  generated paraphrases, and selected hard-distractor IDs.
- `squad_calibration_diagnostics.csv` — train-only per-example diagnostic
  signals and objective retrieval outcomes.
- `squad_detector_calibration.json` — the original and frozen
  `TRAIN-CALIBRATED` detector configurations plus train-only summaries.
- `squad_stress_per_example.csv` — held-out validation stress outcomes for
  the original and calibrated detectors.
- `squad_stress_metrics.json` — separated train calibration evidence and
  held-out validation metrics.
- `squad_failure_confusion.csv` — objective validation labels against raw
  Phase 5 categorical predictions.

Only the small reproducibility manifests are intended to be committed. Metrics,
per-example diagnostics, and model/index caches are generated locally and
ignored by Git.

`notebooks/06d_squad_generation_evaluation.ipynb` adds the final Chunk 3
generation artifacts:

- `squad_generation_manifest.json` — fixed generation-evaluation IDs, category
  counts, configuration, and hashes of every required frozen Chunk 2 input.
- `squad_generation_per_example.csv` — three-system answers, exact evidence,
  answer/safety/groundedness decisions, recovery metadata, and latency.
- `squad_generation_metrics.json` — overall, per-track, three-way, abstention,
  recovery-conditional, groundedness, and compute-latency summaries.
- `squad_generation_summary.md` — concise human-readable comparison.
- `squad_generation_cache.jsonl` — append-only semantic prompt cache for
  interruption-safe local Qwen generation.
- `squad_nli_per_claim.csv` — transparent per-claim local NLI decisions.

As with Chunk 2, only the small generation manifest is intended to be tracked;
answers, metrics, claim rows, and caches are ignored.
