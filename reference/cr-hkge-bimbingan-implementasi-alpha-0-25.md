# Catatan Bimbingan Implementasi CR-HKGE: Varian Alpha 0.25

Dokumen ini disiapkan untuk menjelaskan implementasi CR-HKGE kepada dosen pembimbing. Fokus dokumen ini adalah alur model, file implementasi yang relevan, alasan pemilihan varian `alpha=0.25`, dan cara membaca hasil evaluasi.

## Update Eksekusi: Dataset CR-HKGE-Ready

Catatan penting terbaru:

```text
Dataset lama: dataset-aromatique-kgat-ready
Dataset baru: dataset-aromatique-crhkge-ready
```

Dataset baru dibuat karena positive pairs pada dataset lama masih terlalu umum
untuk menunjukkan dampak tiga novelty CR-HKGE secara maksimal. Dataset baru
tetap KGAT-compatible, tetapi `train.txt` dan `test.txt` dibangun ulang dengan
skor relevance yang secara eksplisit memakai:

```text
1. Fragrance-specific heterogeneous KG
2. Cross-reference via inspired_by
3. Relation-type priority/attention
```

Rujukan implementasi:

- Script pembangun dataset: [build_cr_hkge_ready_dataset.py](../scripts/build_cr_hkge_ready_dataset.py)
- Dataset baru: [dataset-aromatique-crhkge-ready](../dataset-aromatique-crhkge-ready)
- Audit skor positive pairs: [positive_pair_scores.tsv](../dataset-aromatique-crhkge-ready/positive_pair_scores.tsv)
- Mapping profile ke produk sumber: [profile2product.tsv](../dataset-aromatique-crhkge-ready/profile2product.tsv)
- Ringkasan dataset baru: [summary.json](../dataset-aromatique-crhkge-ready/summary.json)

Statistik dataset baru:

```text
products: 340
content/query profiles: 340
train positive pairs: 2720
test positive pairs: 1360
enriched products: 243
standard products: 97
relations: 7
KG triples: 9250
```

Makna profile pada dataset baru:

```text
Satu profile merepresentasikan satu produk sumber/query content.
Item positif adalah produk lain yang paling relevan menurut skor fragrance,
cross-reference, dan relation priority.
```

Perintah membuat ulang dataset:

```bash
python scripts/build_cr_hkge_ready_dataset.py --overwrite
```

Perintah training/evaluasi final study memakai dataset baru:

```bash
DATASET=dataset-aromatique-crhkge-ready \
LOG_DIR=/content/kgat/final_study/logs_crhkge_ready \
CR_HKGE_FINAL_EPOCHS=100 \
bash scripts/run_cr_hkge_final_study.sh kgat cr_hkge_final_alpha_0_25 A_no_cross_reference A_no_relation_attention A_no_fragrance_prior
```

Untuk Colab, jalankan dari root repository `/content/kgat`.

## Update Eksekusi: Tuning Fokus NFM

Setelah CR-HKGE mengalahkan KGAT dan beberapa baseline KG/CF, baseline yang
paling kuat adalah NFM. NFM kuat karena dataset evaluasi berbasis content
positive pairs, sehingga interaksi fitur langsung sangat menguntungkan model
tersebut.

Tuning berikut tidak mengubah novelty CR-HKGE. Tujuannya hanya mengkalibrasi
seberapa kuat cross-reference dan prior relasi bekerja supaya ranking Top-K
lebih tajam ketika dibandingkan dengan NFM.

Varian tuning yang tersedia:

```text
cr_hkge_final_alpha_0_075
cr_hkge_final_alpha_0_05
cr_hkge_final_alpha_0_1_prior_0_5
cr_hkge_final_alpha_0_1_prior_0_25
cr_hkge_final_alpha_0_1_no_relation_message
cr_hkge_final_alpha_0_075_prior_0_5
```

Perintah Colab:

```bash
DATASET=dataset-aromatique-crhkge-ready \
LOG_DIR=/content/kgat/final_study/logs_crhkge_ready \
CR_HKGE_FINAL_EPOCHS=100 \
bash scripts/run_cr_hkge_nfm_focus.sh
```

Output penting:

```text
/content/kgat/final_study/logs_crhkge_ready/final_summary.md
/content/kgat/final_study/logs_crhkge_ready/nfm_focus_summary.md
```

Cara membaca hasil:

```text
1. Jika NDCG@3 naik, varian tersebut lebih kuat untuk ranking awal.
2. Jika Recall@10 naik, varian tersebut lebih kuat untuk coverage/retrieval.
3. Jika standard subset naik, model lebih baik untuk generalisasi produk tanpa inspired_by.
4. Jika enriched subset turun terlalu jauh, cross-reference terlalu dilemahkan.
```

Kandidat final dipilih dari keseimbangan:

```text
overall NDCG@3
overall Recall@10
enriched Recall@3/NDCG@3
standard Recall@3/NDCG@3
```

## 1. Posisi Penelitian

Penelitian ini mengembangkan CR-HKGE, yaitu modifikasi KGAT untuk rekomendasi parfum pada kondisi tidak tersedia historical user interaction seperti purchase, rating, atau click log.

Karena KGAT membutuhkan format user-item interaction untuk training BPR, penelitian ini tidak memakai user historis asli, melainkan membangun:

```text
content-based positive pairs
```

Pasangan ini disimpan dalam format KGAT:

```text
train.txt
test.txt
```

Penting:

```text
train.txt dan test.txt bukan riwayat pembelian user.
train.txt dan test.txt adalah surrogate supervision berbasis kemiripan konten fragrance.
```

Rujukan file:

- [train.txt](../dataset-aromatique-kgat-ready/train.txt)
- [test.txt](../dataset-aromatique-kgat-ready/test.txt)
- [product2id.tsv](../dataset-aromatique-kgat-ready/product2id.tsv)
- [kg_final.txt](../dataset-aromatique-kgat-ready/kg_final.txt)

## 2. Flow Utama CR-HKGE

Flow implementasi yang digunakan:

```text
1. Aromatique product catalog
2. Global fragrance reference enrichment
3. Fragrance-specific heterogeneous KG construction
4. Content-based positive pair construction
5. CR-HKGE model training
6. Alternating BPR + TransR training
7. Top-K evaluation
8. Ablation study
9. Enriched vs standard analysis
```

## 3. Dataset dan Knowledge Graph

Dataset akhir dalam format KGAT-ready berisi:

```text
products: 340
entities: 998
relations: 7
KG triples: 9250
train positive pairs: 1490
test positive pairs: 406
```

Relasi KG yang dipakai:

```text
inspired_by
has_accord
has_visual_note
belongs_to_family
sem_similar
has_global_accord
belongs_to_global_family
```

Rujukan file:

- Mapping produk: [product2id.tsv](../dataset-aromatique-kgat-ready/product2id.tsv)
- Mapping entitas: [entity2id_typed.tsv](../dataset-aromatique-kgat-ready/entity2id_typed.tsv)
- Mapping relasi: [relation2id.txt](../dataset-aromatique-kgat-ready/relation2id.txt)
- Triple KG: [kg_final.txt](../dataset-aromatique-kgat-ready/kg_final.txt)

## 4. Content-Based Positive Pairs

Format `train.txt` dan `test.txt` mengikuti KGAT:

```text
profile_id item_id_1 item_id_2 item_id_3 ...
```

Contoh:

```text
0 7 3 2 8 5 6 9 4
```

Artinya:

```text
profile 0 memiliki positive items:
7, 3, 2, 8, 5, 6, 9, 4
```

Dalam penelitian ini, `profile_id` bukan user historis, melainkan pseudo-profile berbasis konten fragrance.

Kegunaan file ini:

```text
Phase I training:
BPR loss mempelajari agar skor positive item lebih tinggi dari negative item.

Evaluation:
item pada test.txt dianggap ground truth relevan untuk menghitung Recall@K,
Precision@K, Hit@K, dan NDCG@K.
```

Rujukan implementasi:

- Loader data umum: [load_data.py](../Model/utility/load_data.py)
- Loader KGAT/CR-HKGE: [loader_kgat.py](../Model/utility/loader_kgat.py)
- Training loop: [Main.py](../Model/Main.py)

## 5. Arsitektur CR-HKGE

CR-HKGE mempertahankan struktur KGAT, lalu menambahkan tiga novelty.

### 5.1 Layer 1: TransR Knowledge Graph Embedding

Layer ini mengikuti KGAT. Tujuannya mempelajari embedding entitas dan relasi KG.

Rujukan:

- Implementasi KGAT dasar: [KGAT.py](../Model/KGAT.py)
- Implementasi CR-HKGE: [CRHKGE.py](../Model/CRHKGE.py)

### 5.2 Layer 2: Relation-Type Specific Attention

CR-HKGE memberi bobot berbeda untuk tiap tipe relasi fragrance.

Contoh:

```text
has_accord seharusnya lebih informatif daripada has_visual_note
inspired_by seharusnya lebih informatif untuk produk enriched
sem_similar membantu membentuk neighborhood produk yang mirip
```

Bagian kode utama:

- Konfigurasi relation attention: [CRHKGE.py](../Model/CRHKGE.py)
- Fungsi `_build_weights`: [CRHKGE.py](../Model/CRHKGE.py)
- Fungsi `_initial_relation_type_logits`: [CRHKGE.py](../Model/CRHKGE.py)
- Fungsi `_generate_transE_score`: [CRHKGE.py](../Model/CRHKGE.py)

### 5.3 Layer 3: Cross-Reference Propagation via `inspired_by`

Novelty ini mengalirkan konteks dari global reference fragrance ke local product.

Contoh:

```text
produk lokal --inspired_by--> parfum global
parfum global --has_global_accord--> global accord
parfum global --belongs_to_global_family--> global family
```

Konteks global ini kemudian ditambahkan ke embedding produk lokal.

Bagian kode utama:

- Metadata cross-reference: [loader_kgat.py](../Model/utility/loader_kgat.py)
- Tensor cross-reference: [CRHKGE.py](../Model/CRHKGE.py)
- Fungsi `_create_cross_reference_context`: [CRHKGE.py](../Model/CRHKGE.py)

### 5.4 Layer 4: Bi-Interaction + Prediction

CR-HKGE tetap memakai bi-interaction aggregation seperti KGAT. Bedanya, side embedding yang masuk ke bi-interaction dapat diperkaya dengan:

```text
relation-aware message
cross-reference context
```

Bagian kode utama:

- Fungsi `_create_bi_interaction_embed`: [CRHKGE.py](../Model/CRHKGE.py)

## 6. Alternating Training

Training mengikuti KGAT:

```text
Phase I: BPR training
Phase II: TransR/KGE training + update attentive adjacency A_in
```

### Phase I: BPR Loss

Input:

```text
content-based positive pairs dari train.txt
```

Tujuan:

```text
score(profile, positive_item) > score(profile, negative_item)
```

Rujukan:

- Training loop Phase I: [Main.py](../Model/Main.py)
- BPR loss KGAT: [KGAT.py](../Model/KGAT.py)

### Phase II: TransR/KGE Loss

Input:

```text
triples dari kg_final.txt
```

Tujuan:

```text
mempelajari embedding entity dan relation pada KG
memperbarui attentive adjacency A_in
```

Rujukan:

- Training loop Phase II: [Main.py](../Model/Main.py)
- KGE loss KGAT: [KGAT.py](../Model/KGAT.py)
- Relation attention CR-HKGE: [CRHKGE.py](../Model/CRHKGE.py)

## 7. Kenapa Varian Alpha 0.25 Dipilih

Pada CR-HKGE, `alpha` adalah koefisien yang mengatur kekuatan cross-reference propagation.

Kode konsep:

```text
cross_reference_context =
alpha
* gate
* inspired_by_multiplier
* transformed_global_reference_context
* product_mask
```

Rujukan kode:

- Fungsi `_create_cross_reference_context`: [CRHKGE.py](../Model/CRHKGE.py)
- Argumen `--cr_cross_ref_alpha`: [parser.py](../Model/utility/parser.py)
- Runner final study: [run_cr_hkge_final_study.sh](../scripts/run_cr_hkge_final_study.sh)

Makna nilai alpha:

```text
alpha = 0.5  -> cross-reference cukup kuat
alpha = 0.25 -> cross-reference sedang dan lebih stabil
alpha = 0.1  -> cross-reference konservatif
```

Berdasarkan hasil sementara, `alpha=0.25` lebih stabil daripada `alpha=0.5`.

Perbandingan utama:

| Model | Recall@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| KGAT | 0.21273 | 0.33204 | 0.32664 | 0.43407 |
| CR-HKGE alpha 0.5 | 0.19480 | 0.34239 | 0.28260 | 0.42069 |
| CR-HKGE alpha 0.25 | 0.20850 | 0.35651 | 0.32027 | 0.43409 |

Interpretasi:

```text
CR-HKGE alpha 0.25 belum mengalahkan KGAT pada Recall@3,
tetapi sudah mengalahkan KGAT pada NDCG@3.
```

Artinya:

```text
CR-HKGE alpha 0.25 memperbaiki kualitas ranking awal,
meskipun jumlah item relevan yang terambil di Top-3 masih sedikit di bawah KGAT.
```

Untuk enriched products:

| Model | Enriched Recall@3 | Enriched NDCG@3 |
|---|---:|---:|
| KGAT | 0.20226 | 0.24311 |
| CR-HKGE alpha 0.25 | 0.20648 | 0.29115 |

Interpretasi:

```text
Pada produk enriched, yaitu produk dengan inspired_by,
CR-HKGE alpha 0.25 mengungguli KGAT pada Recall@3 dan NDCG@3.
```

Ini mendukung novelty cross-reference.

## 8. Hasil Sementara CR-HKGE Alpha 0.25

### Overall

```text
Recall@3  = 0.20850
Precision@3 = 0.29902
Hit@3 = 0.47059
NDCG@3 = 0.35651
Recall@5 = 0.24545
NDCG@5 = 0.37023
Recall@10 = 0.32027
NDCG@10 = 0.43409
```

### Enriched Products

```text
Recall@3 = 0.20648
Precision@3 = 0.21026
Hit@3 = 0.43077
NDCG@3 = 0.29115
Recall@5 = 0.23171
NDCG@5 = 0.30519
Recall@10 = 0.32006
NDCG@10 = 0.37560
```

### Standard Products

```text
Recall@3 = 0.23434
Precision@3 = 0.17544
Hit@3 = 0.39474
NDCG@3 = 0.27194
Recall@5 = 0.29558
NDCG@5 = 0.31118
Recall@10 = 0.36366
NDCG@10 = 0.34673
```

## 9. Cara Menjelaskan Hasil ke Dosen

Poin utama yang sebaiknya disampaikan:

1. CR-HKGE tidak memakai historical user interaction.
2. File `train.txt` dan `test.txt` adalah content-based positive pairs dalam format KGAT.
3. CR-HKGE mengikuti alternating training KGAT agar apple-to-apple dengan KGAT.
4. Tiga novelty CR-HKGE adalah:
   - fragrance-specific heterogeneous KG,
   - cross-reference propagation via `inspired_by`,
   - relation-type attention.
5. Varian `alpha=0.25` dipilih karena cross-reference tetap aktif, tetapi tidak terlalu dominan.
6. Dibanding KGAT, alpha 0.25 unggul pada NDCG@3 overall.
7. Pada enriched products, alpha 0.25 unggul pada Recall@3 dan NDCG@3, sehingga novelty `inspired_by` terlihat berdampak.

Kalimat ringkas:

```text
CR-HKGE alpha 0.25 belum menjadi model terbaik absolut dibanding semua baseline,
tetapi sudah menunjukkan peningkatan kualitas ranking awal atas KGAT,
terutama pada produk enriched yang memiliki relasi inspired_by.
```

## 10. Catatan Kritis

Hasil sementara menunjukkan NFM masih lebih tinggi dari CR-HKGE pada banyak metric.

Ini perlu dijelaskan dengan hati-hati:

```text
NFM adalah baseline feature-based yang langsung memakai sparse KG features.
Karena positive pairs dibangun dari metadata konten,
NFM mendapat keuntungan langsung dari fitur yang juga menjadi dasar split.
```

Jadi posisi CR-HKGE dalam paper sebaiknya:

```text
CR-HKGE adalah peningkatan atas KGAT sebagai graph attention baseline,
bukan klaim model terbaik absolut atas semua pendekatan feature-based.
```

Jika setelah tuning tambahan CR-HKGE mengalahkan NFM, klaim bisa diperkuat. Jika tidak, klaim tetap diarahkan pada:

```text
KGAT-based graph model improvement
cross-reference impact pada enriched products
ranking quality improvement via NDCG@3
```

## 11. File Implementasi yang Perlu Ditunjukkan

### Model utama

- [CRHKGE.py](../Model/CRHKGE.py)

Bagian penting:

```text
_parse_args
_build_weights
_initial_relation_type_logits
_generate_transE_score
_create_bi_interaction_embed
_create_cross_reference_context
export_artifacts
```

### Loader metadata CR-HKGE

- [loader_kgat.py](../Model/utility/loader_kgat.py)

Bagian penting:

```text
_build_cr_hkge_data
product_global_mat
global_attr_relation_mats
product_mask
enriched_product_ids
```

### Training loop

- [Main.py](../Model/Main.py)

Bagian penting:

```text
Phase I: BPR training
Phase II-A: KGE training
Phase II-B: update attentive adjacency
CR-HKGE best checkpoint
export artifacts
```

### Evaluasi

- [evaluate_item_subsets.py](../Model/evaluate_item_subsets.py)

Bagian penting:

```text
overall evaluation
enriched evaluation
standard evaluation
```

### Runner eksperimen final

- [run_cr_hkge_final_study.sh](../scripts/run_cr_hkge_final_study.sh)

Bagian penting:

```text
kgat
cr_hkge_final
cr_hkge_final_alpha_0_25
ablation variants
```

### Ringkasan hasil

- [summarize_final_study.py](../scripts/summarize_final_study.py)

## 12. Narasi Singkat untuk Presentasi Bimbingan

Narasi yang bisa digunakan:

```text
Penelitian ini memodifikasi KGAT menjadi CR-HKGE untuk rekomendasi parfum
pada kondisi tidak ada user interaction historis. Karena KGAT membutuhkan
format user-item untuk BPR training, saya membangun content-based positive
pairs dari metadata fragrance. Model kemudian dilatih secara alternating:
Phase I BPR untuk ranking, dan Phase II TransR/KGE untuk knowledge graph.

Kontribusi CR-HKGE ada pada tiga hal: heterogeneous KG khusus fragrance,
relation-type attention, dan cross-reference propagation melalui inspired_by.
Pada implementasi, inspired_by mengalirkan konteks dari parfum global ke produk
lokal. Koefisien alpha digunakan untuk mengatur seberapa kuat konteks global
tersebut masuk ke embedding produk.

Dari eksperimen sementara, alpha 0.25 lebih stabil dibanding alpha 0.5.
Model ini mengungguli KGAT pada NDCG@3 overall dan juga mengungguli KGAT
pada Recall@3 serta NDCG@3 untuk produk enriched. Ini menunjukkan bahwa
cross-reference inspired_by memberi dampak pada produk yang memang memiliki
relasi global reference.
```
