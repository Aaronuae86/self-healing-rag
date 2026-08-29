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
