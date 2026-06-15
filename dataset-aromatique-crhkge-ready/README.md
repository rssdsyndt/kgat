# dataset-aromatique-crhkge-ready

Dataset ini dibuat oleh `scripts/build_cr_hkge_ready_dataset.py`.

Tujuan utamanya adalah menyelaraskan `train.txt` dan `test.txt` dengan tiga novelty CR-HKGE:

1. Fragrance-specific heterogeneous KG construction.
2. Cross-reference via `inspired_by`.
3. Relation-type priority/attention.

Format file tetap KGAT-compatible, sehingga KGAT, CR-HKGE, dan baseline lain dapat dibandingkan pada split yang sama.

Setiap profile merepresentasikan satu produk sumber/query content. Item positif adalah produk lain dengan skor relevance tertinggi berdasarkan kombinasi local fragrance attributes, global reference enrichment, cross-reference bridge, dan weak `sem_similar` support.

File penting:

- `train.txt`: positive pairs untuk BPR training.
- `test.txt`: held-out positive pairs untuk evaluasi Top-K.
- `profile2product.tsv`: mapping profile ke produk sumber.
- `positive_pair_scores.tsv`: audit skor setiap pasangan positif.
- `summary.json`: statistik dataset dan bobot scoring.
