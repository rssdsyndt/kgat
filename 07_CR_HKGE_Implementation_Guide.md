# CR-HKGE Implementation Guide

Tanggal konteks: 2026-05-18  
Workspace utama: `D:\Azam\MTI\TESIS\knowledge_graph_attention_network`  
Repo KGAT: `https://github.com/rssdsyndt/kgat.git`  
Repo sistem Aromatique AI: `https://github.com/rssdsyndt/aromatique-ai.git`

Dokumen ini adalah panduan teknis untuk sesi Codex berikutnya ketika mulai mengimplementasikan model CR-HKGE. Tujuannya adalah membuat implementasi model baru secara bertahap tanpa merusak baseline KGAT yang sudah berhasil berjalan pada dataset Aromatique.

---

## 1. Posisi Saat Ini

### Status KGAT

Baseline KGAT sudah berhasil dijalankan pada dataset `dataset-aromatique-kgat-ready`.

Ringkasan dataset yang sudah valid:

```text
n_users = 68
n_items = 340
n_entities = 998
n_relations = 7
n_train = 1490
n_test = 406
n_triples = 9250
duplicate_triples = 0
```

Mapping penting:

```text
Product/item IDs         : 0..339
Non-product entity IDs   : 340..997
White Sense new item ID  : 68
White Sense old entity ID: 316
```

Baseline KGAT 100 epoch yang sudah berhasil:

```text
Best Iter=[6]@[417.1]
recall    = [0.40364, 0.51594, 0.59071, 0.66108, 0.69774]
precision = [0.11765, 0.08015, 0.06103, 0.05037, 0.04279]
hit       = [0.66176, 0.72059, 0.77941, 0.83824, 0.85294]
ndcg      = [0.35150, 0.39534, 0.42017, 0.43968, 0.45275]
```

Catatan pembacaan hasil:

- Default `Ks` KGAT adalah `[20,40,60,80,100]`.
- Untuk sistem Aromatique yang menampilkan Top-3 rekomendasi, CR-HKGE wajib dievaluasi juga dengan `Ks=[3,5,10]`.
- Baseline KGAT ini harus dipertahankan sebagai pembanding utama, bukan ditimpa.

### Status Integrasi Sistem

Sistem `aromatique-ai` sudah berjalan dengan arsitektur:

- Frontend React/TanStack/Vite.
- Supabase Edge Function `aromatique-chat`.
- Supabase tables untuk conversation, messages, recommendations, feedback, products, KG edges, embeddings, dan experiment sessions.
- GPT/OpenAI digunakan untuk NLU dan explanation generation.
- Rekomendasi produksi sementara boleh memakai KGAT/static retrieval terlebih dahulu.
- CR-HKGE nanti menggantikan retrieval model setelah hasil offline stabil.

Prinsip penting:

```text
KGAT/CR-HKGE repo = training offline dan artifact generator.
aromatique-ai repo = serving, chatbot, UI, Supabase Edge Function.
```

Jangan memindahkan training TensorFlow ke Supabase Edge Function. Edge Function cukup membaca artifact hasil training, seperti product embedding, metadata, KG path, dan ranking score.

---

## 2. Definisi Model CR-HKGE

CR-HKGE adalah modifikasi KGAT untuk domain parfum Aromatique.

Nama penelitian:

```text
Cross-Reference Semantic Enrichment on Heterogeneous Knowledge Graph Embedding
```

Novelty utama:

1. Heterogeneous fragrance KG yang memuat produk, accord, visual note, family, semantic similarity, dan global reference.
2. Relation-Type Specific Attention Weight, yaitu scalar learnable per tipe relasi.
3. Cross-Reference Propagation, yaitu propagation khusus untuk produk yang memiliki relasi `inspired_by` / `revolutionize`.

---

## 3. Formula Arsitektur

### Layer 1: KG Embedding

Layer ini mengikuti KGAT asli.

```text
g(h,r,t) = || W_r e_h + e_r - W_r e_t ||^2

Loss_KG = sum -log sigmoid(g(h,r,t') - g(h,r,t))
```

Tidak perlu mengubah loss TransR/KGE terlebih dahulu. Perubahan utama dimulai dari attention score dan propagation.

### Layer 2: Relation-Type Specific Attention

KGAT asli:

```text
pi(h,r,t) = (W_r e_t)^T tanh(W_r e_h + e_r)
```

CR-HKGE:

```text
pi_CR(h,r,t) = lambda_tilde_r * (W_r e_t)^T tanh(W_r e_h + e_r)
lambda_tilde_r = softmax(lambda_r)
```

Makna:

- `lambda_r` adalah parameter learnable.
- `lambda_tilde_r` adalah bobot normalisasi untuk tipe relasi.
- Relasi yang lebih informatif untuk domain parfum dapat mendapat attention influence lebih besar.

### Layer 3: Cross-Reference Propagation

Untuk produk enriched, yaitu produk yang memiliki relasi `inspired_by`:

```text
e_CR(p) = lambda_tilde_ib * W_CR * (e_g + sum pi_CR(g,r',t') e_t')
e_p^(l) = sigma(W * (e_p^(l-1) + e_N_int(p) + e_CR(p)))
```

Untuk produk standard, yaitu produk tanpa relasi `inspired_by`:

```text
e_p^(l) = sigma(W * (e_p^(l-1) + e_N_int(p)))
```

Interpretasi domain:

- `p` adalah produk lokal Aromatique.
- `g` adalah global reference atau parfum global yang menjadi inspirasi.
- `e_CR(p)` membawa konteks semantik dari global reference ke produk lokal.
- Produk tanpa cross-reference tetap memakai propagation KGAT biasa.

### Layer 4: Prediction

```text
e_p* = e_p^(0) || e_p^(1) || ... || e_p^(L)
score(q,p) = cosine(e_q, e_p*)
Top-3 = argmax score(q,p)
```

Untuk evaluasi offline KGAT, `q` direpresentasikan oleh user interaction dalam `train.txt` dan `test.txt`.

Untuk serving Aromatique, `q` harus dibentuk dari hasil NLU chatbot, bukan dari user ID training. Karena itu query encoder serving sebaiknya dibuat sebagai tahap terpisah setelah CR-HKGE training stabil.

---

## 4. Dataset dan Relasi

File relasi:

```text
dataset-aromatique-kgat-ready/relation2id.txt
```

Isi:

```text
inspired_by                 0
has_accord                  1
has_visual_note             2
belongs_to_family           3
sem_similar                 4
has_global_accord           5
belongs_to_global_family    6
```

Perhatian penting:

KGAT loader tidak memakai ID relasi ini secara langsung di adjacency internal. Loader KGAT menambahkan relasi user-item dan inverse relation.

Untuk dataset dengan 7 relasi asli, expanded relation di loader menjadi 16 relasi:

```text
0  = user -> item
8  = item -> user

1  = inspired_by
9  = inverse inspired_by

2  = has_accord
10 = inverse has_accord

3  = has_visual_note
11 = inverse has_visual_note

4  = belongs_to_family
12 = inverse belongs_to_family

5  = sem_similar
13 = inverse sem_similar

6  = has_global_accord
14 = inverse has_global_accord

7  = belongs_to_global_family
15 = inverse belongs_to_global_family
```

Konsekuensi implementasi:

- Jika memakai `self.n_relations` dari loader, relation weight shape menjadi `[16]`.
- Jika ingin sesuai klaim tesis "satu scalar per tipe relasi", maka inverse relation harus ditied ke semantic relation yang sama.
- `user -> item` dan `item -> user` bukan relasi KG semantik. Keduanya bisa dibuat neutral weight atau relation type terpisah khusus interaction.

Rekomendasi:

Mulai dengan implementasi paling aman:

```text
relation_type_logits shape = [self.n_relations]
relation_type_weights = softmax(relation_type_logits)
```

Setelah model berjalan, naikkan kualitas penelitian dengan tying semantic relation:

```text
expanded relation 1 dan 9  -> semantic inspired_by
expanded relation 2 dan 10 -> semantic has_accord
expanded relation 3 dan 11 -> semantic has_visual_note
expanded relation 4 dan 12 -> semantic belongs_to_family
expanded relation 5 dan 13 -> semantic sem_similar
expanded relation 6 dan 14 -> semantic has_global_accord
expanded relation 7 dan 15 -> semantic belongs_to_global_family
expanded relation 0 dan 8  -> interaction
```

---

## 5. File KGAT yang Perlu Dipahami

Jangan mulai implementasi CR-HKGE sebelum membaca file berikut.

```text
Model/KGAT.py
Model/Main.py
Model/utility/parser.py
Model/utility/loader_kgat.py
Model/utility/batch_test.py
Model/utility/tf_compat.py
```

Titik penting di `Model/KGAT.py`:

```text
class KGAT
_build_weights()
_build_model_phase_I()
_build_model_phase_II()
_create_bi_interaction_embed()
_create_attentive_A_out()
_generate_transE_score()
update_attentive_A()
```

Insertion point paling penting:

```text
_generate_transE_score(h, t, r)
```

Di sinilah score KGAT attention dihitung. Relation-type weight CR-HKGE harus masuk di sini sebelum sparse softmax.

Propagation point paling penting:

```text
_create_bi_interaction_embed()
```

Di sinilah embedding propagation bi-interaction KGAT dibuat. Cross-reference propagation CR-HKGE harus masuk di sini, tetapi lakukan setelah relation-type attention berhasil dulu.

---

## 6. Strategi Implementasi Bertahap

Implementasi harus dilakukan bertahap agar mudah dibuktikan dan mudah dibandingkan.

### Stage 0: Baseline Lock

Tujuan:

- Pastikan KGAT vanilla tetap jalan.
- Jangan ubah perilaku `--model_type kgat`.
- Simpan command dan output baseline sebagai pembanding.

Command smoke test:

```bash
cd /content/kgat/Model
python Main.py \
  --model_type kgat \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 2 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 0
```

Definition of done:

```text
KGAT vanilla masih bisa training tanpa error.
Output menampilkan n_users=68, n_items=340, n_entities=998, n_relations=7, n_triples=9250.
```

### Stage 1: Buat Model Baru CR-HKGE Tanpa Mengubah KGAT

Tujuan:

- Tambahkan `--model_type cr_hkge`.
- Buat class baru yang awalnya identik dengan KGAT.
- Pastikan `cr_hkge` menghasilkan output sama atau sangat dekat dengan KGAT sebelum novelty diaktifkan.

File yang dibuat:

```text
Model/CRHKGE.py
```

File yang diubah:

```text
Model/Main.py
Model/utility/parser.py
Model/utility/batch_test.py
```

Aturan:

- Jangan rename `KGAT.py`.
- Jangan mengganti class `KGAT` untuk kebutuhan CR-HKGE.
- Boleh copy struktur KGAT ke `CRHKGE.py` terlebih dahulu agar eksperimen aman.

Argumen baru yang disarankan:

```text
--model_type cr_hkge
--cr_use_relation_weight 1
--cr_use_cross_ref 0
--cr_relation_weight_mode expanded
--cr_export_embeddings 0
```

Catatan:

- Hindari `type=bool` di argparse karena string `"False"` bisa terbaca truthy.
- Gunakan pola integer `0/1` atau helper `str2bool`.

Definition of done:

```text
python Main.py --model_type cr_hkge ... --epoch 2
```

berjalan tanpa error, walaupun belum ada novelty aktif.

### Stage 2: Relation-Type Specific Attention

Tujuan:

- Tambahkan learnable scalar per relation type.
- Masukkan bobot ini ke attention score KGAT.

Lokasi utama:

```text
Model/CRHKGE.py
_build_weights()
_generate_transE_score()
```

Rancangan minimal:

```text
relation_type_logits: trainable variable shape [self.n_relations]
relation_type_weights = softmax(relation_type_logits)
selected_weight = gather(relation_type_weights, r)
score_cr = selected_weight * score_kgat
```

Urutan kerja:

1. Tambahkan variable `relation_type_logits`.
2. Hitung `relation_type_weights`.
3. Di `_generate_transE_score`, hitung score KGAT asli.
4. Ambil bobot berdasarkan relation id `r`.
5. Kalikan score dengan bobot tersebut.
6. Pastikan `update_attentive_A()` tetap berjalan.
7. Export relation weights setelah training.

Eksperimen:

```text
CR-HKGE RelAttn only:
--model_type cr_hkge
--cr_use_relation_weight 1
--cr_use_cross_ref 0
```

Definition of done:

- Training epoch 2 berhasil.
- Training epoch 100 berhasil.
- File output result muncul.
- Relation weights dapat dicetak atau disimpan.
- KGAT vanilla tetap tidak berubah.

### Stage 3: Dataset Support untuk Cross-Reference

Tujuan:

- Buat struktur data yang mengetahui produk mana yang enriched.
- Buat mapping produk lokal ke global reference.
- Buat adjacency/context untuk global reference.

File yang bisa diubah atau dibuat:

```text
Model/utility/loader_kgat.py
```

atau lebih aman:

```text
Model/utility/loader_cr_hkge.py
```

Data yang dibutuhkan:

```text
enriched_product_ids
product_to_global_ref
global_ref_to_attributes
semantic_relation_map
```

Cara mendeteksi enriched product:

```text
Di kg_final.txt asli KGAT-ready:
relation 0 = inspired_by
head = product lokal
tail = global reference
```

Karena product ID berada pada `0..339`, maka:

```text
enriched product = h < 340 dan r == 0
global reference = t
```

Relasi global reference:

```text
has_global_accord           = relation 5
belongs_to_global_family    = relation 6
```

Perhatian:

- Jangan pakai expanded relation ID loader untuk membaca `kg_final.txt` mentah.
- Expanded relation ID hanya muncul setelah loader membangun adjacency internal.

Definition of done:

- Loader bisa mencetak jumlah enriched products.
- Loader bisa mencetak jumlah product_to_global_ref.
- Loader bisa mencetak jumlah global_ref attributes.
- KGAT vanilla tetap tidak berubah.

### Stage 4: Cross-Reference Propagation

Tujuan:

- Tambahkan extra context dari global reference ke produk enriched.
- Produk standard tetap mengikuti KGAT.

Lokasi utama:

```text
Model/CRHKGE.py
_create_bi_interaction_embed()
```

Rancangan awal yang aman:

```text
1. Hitung ego_embeddings seperti KGAT.
2. Hitung side_embeddings = A * ego_embeddings seperti KGAT.
3. Hitung cr_context untuk product rows saja.
4. Tambahkan cr_context ke side_embeddings hanya untuk enriched products.
5. Lanjutkan bi-interaction transform seperti KGAT.
```

Representasi sparse yang disarankan:

```text
C_pg: sparse matrix product -> global_ref
G_ga: sparse matrix global_ref -> global attributes
```

Skema:

```text
global_context = C_pg * (global_embedding + G_ga * entity_embeddings)
product_context = mask_enriched * global_context
```

Hal yang harus hati-hati:

- KGAT memakai embedding concatenation lintas layer.
- Dengan `embed_size=64` dan `layer_size=[64,32,16]`, final embedding dimension adalah `64+64+32+16 = 176`.
- Jika menambah `W_CR`, dimensinya harus cocok dengan layer saat itu.
- Karena dimensi layer berubah, lebih aman memakai `W_CR_l` per layer atau menambahkan context sebelum projection pada dimensi layer yang sedang aktif.

Rekomendasi teknis:

Mulai dari cross-ref additive context pada dimensi current layer:

```text
ego_embeddings current_dim
cr_context current_dim
side_embeddings = side_embeddings + alpha * cr_context
```

Setelah itu baru masuk ke transform KGAT:

```text
sum_embeddings = activation(W_gc_l * (ego_embeddings + side_embeddings) + b_gc_l)
bi_embeddings  = activation(W_bi_l * (ego_embeddings * side_embeddings) + b_bi_l)
```

Definition of done:

- `--cr_use_cross_ref 1` bisa berjalan epoch 2.
- Tidak ada shape mismatch.
- Jumlah enriched product tercetak benar.
- Metrics dapat dibandingkan dengan RelAttn only.

### Stage 5: Export Artifact untuk Serving

Tujuan:

- Menyimpan hasil CR-HKGE agar bisa dipakai oleh `aromatique-ai`.
- Supabase Edge Function tidak menjalankan TensorFlow.

Artifact yang disarankan:

```text
artifacts/cr_hkge/<run_id>/
  product_embeddings.tsv
  entity_embeddings.tsv
  relation_weights.tsv
  product_scores_debug.tsv
  model_config.json
  run_metrics.json
  kg_paths.jsonl
```

Kolom minimal `product_embeddings.tsv`:

```text
product_id
product_name
embedding_dim
embedding_json
model_version
```

Kolom minimal `relation_weights.tsv`:

```text
relation_id
relation_name
expanded_relation_id
weight
model_version
```

Catatan penting untuk Supabase:

- Jika memakai final KGAT embedding `[64,32,16]`, dimensinya adalah 176.
- Jika tabel Supabase saat ini memakai `vector(64)`, pilih salah satu:
  - Ubah schema menjadi `vector(176)`.
  - Export raw product embedding 64 dimensi.
  - Tambahkan projection layer ke 64 dimensi dan export hasil projection.
- Untuk penelitian, lebih bersih memakai final concatenated embedding 176 dimensi karena itu output representasi KGAT/CR-HKGE.

Definition of done:

- Artifact tersimpan lokal.
- Artifact punya `model_version`.
- Artifact bisa diupload/import ke Supabase tanpa training runtime.

### Stage 6: Evaluasi Penuh

Tujuan:

- Menjawab apakah CR-HKGE benar-benar lebih baik dari KGAT pada dataset Aromatique.
- Menjawab apakah cross-reference memberi manfaat khusus untuk produk enriched.

Model pembanding minimum:

```text
BPRMF
CKE
KGAT vanilla
CR-HKGE RelAttn only
CR-HKGE CrossRef only
CR-HKGE full
```

Metrics utama untuk paper/tesis:

```text
Recall@K
Precision@K
Hit@K
NDCG@K
```

Nilai K wajib:

```text
[3,5,10] untuk use case Top-3 Aromatique
[20,40,60,80,100] untuk perbandingan dengan output KGAT original
```

Evaluasi tambahan yang penting untuk novelty:

```text
1. Overall test set
2. Enriched item subset: item yang punya inspired_by
3. Standard item subset: item yang tidak punya inspired_by
4. Catalog coverage@K
5. Relation weight interpretability
```

Ekspektasi penelitian:

- Relation weights harus menunjukkan relasi mana yang dominan untuk rekomendasi parfum.
- CR propagation seharusnya membantu produk enriched.
- Jika overall KGAT lebih tinggi tetapi enriched subset CR-HKGE lebih baik, hasil itu tetap bernilai karena novelty CR-HKGE memang menargetkan cross-reference semantic enrichment.

---

## 7. Command Colab yang Disarankan

### Install Environment

Gunakan requirement Colab yang sudah cocok dengan patch TensorFlow 2.x.

```bash
pip install -r /content/kgat/requirements-colab.txt
```

Jika import TensorFlow error karena konflik JAX/ml-dtypes:

```bash
pip uninstall -y jax jaxlib
pip install -r /content/kgat/requirements-colab.txt
```

Setelah install package besar, restart runtime Colab jika diminta.

### KGAT Baseline

```bash
cd /content/kgat/Model

python Main.py \
  --model_type kgat \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 100 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 1
```

### CR-HKGE Smoke Test

Dipakai setelah `--model_type cr_hkge` tersedia.

```bash
cd /content/kgat/Model

python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 2 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 0 \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 0
```

### CR-HKGE Full Evaluation Top-3

Dipakai setelah Stage 4 selesai.

```bash
cd /content/kgat/Model

python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 100 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 1 \
  --Ks [3,5,10] \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 1 \
  --cr_export_embeddings 1
```

---

## 8. Integrasi ke Aromatique AI

CR-HKGE tidak langsung menggantikan semua sistem chatbot. Yang diganti hanya komponen retrieval/ranking.

Kontrak output retrieval yang harus dipertahankan:

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_id": 0,
      "product_name": "Moz Art",
      "match_score": 89.4,
      "olfactory_family": "AMBER",
      "main_accords": ["caramel", "sweet", "vanilla"],
      "kg_path": [
        {"relation": "has_accord", "entity": "sweet", "matched": true},
        {"relation": "has_accord", "entity": "vanilla", "matched": true},
        {"relation": "inspired_by", "entity": "JPG Scandal Pour Homme", "matched": false}
      ],
      "model_version": "cr_hkge_v1"
    }
  ]
}
```

Prinsip integrasi:

- `aromatique-ai` mengirim preferensi user ke Edge Function.
- Edge Function membentuk query profile dari NLU result.
- Edge Function mengambil kandidat produk dari Supabase.
- Ranking dapat memakai embedding CR-HKGE yang sudah diimport ke Supabase.
- GPT explanation A/B/C tetap memakai Top-3 recommendation dan KG path.

Jangan lakukan ini di Edge Function:

```text
Training TensorFlow
Loading checkpoint TensorFlow
Running KGAT session
```

Lakukan ini di Edge Function:

```text
Read product embedding
Read product metadata
Read KG paths
Compute lightweight similarity/ranking
Call GPT for explanation
Persist recommendation and feedback
```

---

## 9. Risiko Teknis dan Cara Menghindarinya

### Risiko 1: KGAT Baseline Rusak

Solusi:

- Implementasikan CR-HKGE di file baru `Model/CRHKGE.py`.
- Jangan edit rumus KGAT kecuali benar-benar perlu shared helper.
- Smoke test `--model_type kgat` setelah setiap perubahan besar.

### Risiko 2: Salah Mapping Relation ID

Solusi:

- Bedakan relation ID mentah di `kg_final.txt` dan expanded relation ID di loader.
- Untuk cross-reference detection, pakai relation ID mentah:

```text
inspired_by = 0
has_global_accord = 5
belongs_to_global_family = 6
```

- Untuk attention internal, pakai expanded relation ID.

### Risiko 3: Shape Mismatch pada Cross-Reference

Solusi:

- Jangan langsung memasukkan `W_CR` global tanpa memperhatikan dimensi layer.
- Mulai dari additive context pada current layer dimension.
- Tambahkan assert/log shape di TensorFlow graph bila perlu.

### Risiko 4: Embedding Dimension Tidak Cocok dengan Supabase

Solusi:

- Catat embedding dimension di `model_config.json`.
- Jika final embedding 176 dimensi, schema Supabase harus `vector(176)` atau dibuat projection ke 64.
- Jangan diam-diam memotong vector karena merusak makna embedding.

### Risiko 5: Improvement Tidak Konsisten

Solusi:

- Jalankan minimal 3 seed.
- Laporkan mean dan standard deviation.
- Pakai ablation agar tahu kontribusi relation attention dan cross-reference propagation secara terpisah.

---

## 10. Acceptance Criteria

Implementasi CR-HKGE dianggap siap untuk eksperimen tesis jika semua ini terpenuhi:

```text
1. KGAT vanilla tetap bisa dijalankan.
2. model_type cr_hkge tersedia.
3. CR-HKGE RelAttn only bisa training 100 epoch.
4. CR-HKGE full bisa training 100 epoch.
5. Metrics [3,5,10] dan [20,40,60,80,100] tersedia.
6. Ablation result tersedia.
7. Enriched vs standard subset evaluation tersedia.
8. Relation weights bisa diexport dan diinterpretasi.
9. Product embeddings bisa diexport untuk Supabase.
10. model_version tercatat pada setiap artifact.
```

---

## 11. Prompt untuk Sesi Codex Baru

Gunakan prompt ini saat membuka sesi baru:

```text
Saya ingin mulai implementasi CR-HKGE di repo KGAT.

Tolong baca dulu:
- 05_Implementation_Handoff.md
- 07_CR_HKGE_Implementation_Guide.md
- 03_Research_Azam_Blueprint.md bagian arsitektur CR-HKGE

Target tahap pertama:
1. Jangan rusak model KGAT baseline.
2. Buat model baru Model/CRHKGE.py dari struktur KGAT.
3. Tambahkan --model_type cr_hkge.
4. Tambahkan argumen CR-HKGE minimal:
   --cr_use_relation_weight
   --cr_use_cross_ref
   --cr_relation_weight_mode
   --cr_export_embeddings
5. Implementasikan Stage 1 dulu sampai cr_hkge bisa smoke test epoch 2.
6. Setelah itu implementasikan Stage 2 relation-type specific attention.
7. Jangan implementasikan cross-reference propagation sebelum Stage 2 berhasil dites.

Setelah edit, jalankan smoke test KGAT dan CR-HKGE, lalu jelaskan hasilnya.
```

---

## 12. Urutan Kerja yang Disarankan

Urutan paling aman:

```text
1. Read codebase: KGAT.py, Main.py, parser.py, loader_kgat.py.
2. Add CRHKGE.py as KGAT-equivalent model.
3. Register model_type cr_hkge.
4. Run cr_hkge smoke test epoch 2.
5. Add relation-type weights.
6. Run RelAttn smoke test epoch 2.
7. Run RelAttn full test epoch 100 with Ks=[3,5,10].
8. Add loader support for enriched products.
9. Add cross-reference propagation.
10. Run full CR-HKGE.
11. Export artifact.
12. Integrate artifact into Supabase/aromatique-ai.
```

Prioritas saat pertama eksekusi:

```text
Berhasil jalan dulu, baru optimasi.
Pisahkan novelty agar ablation jelas.
Jangan ubah sistem user testing Raissa sebelum CR-HKGE offline stabil.
```

---

## 13. Status Implementasi Kode

Bagian ini mencatat implementasi yang sudah dimasukkan ke repo KGAT.

File baru:

```text
Model/CRHKGE.py
```

File yang diubah:

```text
Model/Main.py
Model/utility/parser.py
Model/utility/batch_test.py
Model/utility/loader_kgat.py
```

Fitur yang sudah diimplementasikan:

```text
1. model_type cr_hkge.
2. Relation-type specific attention weight.
3. Relation-aware message propagation agar relation weights mendapat gradient dari CF/BPR loss.
4. Semantic relation tying untuk forward dan inverse relation.
5. Cross-reference propagation untuk produk dengan relasi inspired_by.
6. Global reference context dari has_global_accord dan belongs_to_global_family.
7. Artifact export untuk product embeddings, entity embeddings, relation weights, model_config, dan KG paths.
```

Argumen baru:

```text
--cr_use_relation_weight
--cr_use_cross_ref
--cr_relation_weight_mode
--cr_cross_ref_alpha
--cr_export_embeddings
--cr_artifact_path
--cr_model_version
```

Default penting:

```text
--cr_relation_weight_mode semantic
```

Mode `semantic` berarti forward dan inverse relation ditied ke tipe relasi semantik yang sama. Untuk dataset Aromatique, relation type internal CR-HKGE menjadi:

```text
0 interaction
1 inspired_by
2 has_accord
3 has_visual_note
4 belongs_to_family
5 sem_similar
6 has_global_accord
7 belongs_to_global_family
```

Expanded relation KGAT tetap 16 relation, tetapi bobot CR-HKGE pada mode `semantic` hanya 8 tipe.

Metadata cross-reference Aromatique yang diharapkan terbaca oleh loader:

```text
enriched_products via inspired_by = 243
product_global_edges             = 243
has_global_accord edges          = 2017
belongs_to_global_family edges   = 231
```

Command smoke test CR-HKGE:

```bash
cd /content/kgat/Model

python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 2 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 0 \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 1 \
  --cr_relation_weight_mode semantic
```

Command evaluasi penuh dan export artifact:

```bash
cd /content/kgat/Model

python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 100 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 1 \
  --Ks [3,5,10] \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 1 \
  --cr_relation_weight_mode semantic \
  --cr_export_embeddings 1 \
  --cr_artifact_path ../artifacts/cr_hkge \
  --cr_model_version cr_hkge_v1
```

Artifact akan tersimpan di:

```text
artifacts/cr_hkge/<dataset>/<model_type_timestamp>/
```

Isi artifact:

```text
product_embeddings.tsv
entity_embeddings.tsv
relation_weights.tsv
kg_paths.jsonl
model_config.json
```

Catatan verifikasi lokal:

```text
py -3 -m py_compile Model\CRHKGE.py Model\Main.py Model\utility\parser.py Model\utility\loader_kgat.py Model\utility\batch_test.py
```

Sudah berhasil tanpa syntax error. Training smoke test belum dijalankan lokal karena environment Windows saat ini tidak memiliki package Python ML seperti numpy/TensorFlow. Jalankan smoke test di Colab/runtime yang sebelumnya sudah berhasil menjalankan KGAT.
