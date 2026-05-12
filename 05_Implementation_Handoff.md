# Implementation Handoff: KGAT, Supabase, and Aromatique AI

Tanggal konteks: 2026-05-13  
Workspace utama: `D:\Azam\MTI\TESIS\knowledge_graph_attention_network`

Dokumen ini dibuat sebagai handoff untuk sesi Codex baru. Tujuannya agar konteks penelitian Azam/Raissa, status KGAT, dataset Aromatique, koneksi Supabase, dan rencana integrasi dengan repo `aromatique-ai` bisa langsung dipahami tanpa mengulang eksplorasi dari awal.

---

## 1. Gambaran Sistem

Ada dua repository yang saling berhubungan tetapi tidak perlu digabung.

### Repo 1: `kgat`

Lokasi lokal:

```text
D:\Azam\MTI\TESIS\knowledge_graph_attention_network
```

GitHub:

```text
https://github.com/rssdsyndt/kgat.git
```

Peran repo ini:

- Training dan eksperimen offline KGAT.
- Dataset Aromatique KGAT-ready.
- Baseline KGAT untuk tesis Azam.
- Nantinya tempat implementasi CR-HKGE.
- Generator artifact untuk serving, seperti metadata produk, KG path, dan embedding produk.

### Repo 2: `aromatique-ai`

Lokasi clone lokal sementara:

```text
D:\Azam\MTI\TESIS\knowledge_graph_attention_network\_external\aromatique-ai
```

GitHub:

```text
https://github.com/rssdsyndt/aromatique-ai.git
```

Peran repo ini:

- Frontend React/Vite untuk chatbot Aromatique.
- Supabase Edge Function `aromatique-chat`.
- User flow Raissa: welcome, consent, chat, recommendation, feedback.
- Penyimpanan conversation, message, recommendation, feedback.
- Nantinya membaca retrieval result dari Supabase, bukan dari hardcoded catalog.

### Supabase Project

Project URL:

```text
https://esqzngqpfwrlcnapmfpx.supabase.co
```

Project ref:

```text
esqzngqpfwrlcnapmfpx
```

MCP Supabase sudah dikonfigurasi di Codex CLI dan berhasil dites dari sesi ini. Detail ada di bagian 7.

---

## 2. Dokumen Riset Penting

### `02_Dataset_Schema_Validation.md`

Berisi validasi dataset Aromatique:

- 340 produk.
- 998 entitas.
- 7 tipe relasi.
- 9.250 triple setelah deduplicate dan generated accord untuk White Sense.
- Penjelasan `dataset-aromatique` dan `dataset-aromatique-kgat-ready`.

### `03_Research_Azam_Blueprint.md`

Berisi blueprint model usulan Azam:

- CR-HKGE: Cross-Reference Semantic Enrichment on Heterogeneous Knowledge Graph Embedding.
- Model berbasis modifikasi KGAT.
- Novelty utama:
  - Fragrance-specific heterogeneous KG.
  - Cross-reference propagation via `inspired_by` / `revolutionize`.
  - Relation-type specific attention weights.

### `04_Joint_Research_Azam_Raissa.md`

Berisi arsitektur penelitian gabungan Azam dan Raissa:

- Azam meneliti backend intelligence dan kualitas rekomendasi.
- Raissa meneliti XAI explanation type terhadap trust dan purchase intention.
- Sistem akhir:
  - React frontend.
  - Supabase Edge Function.
  - Preference extraction.
  - Retrieval engine.
  - Top-3 recommendations + KG paths.
  - GPT explanation A/B/C.
  - Feedback Likert.

File ini saat handoff masih untracked di repo lokal. Perlu diputuskan apakah akan di-commit.

---

## 3. Dataset Aromatique

### Folder `dataset-aromatique`

Ini dataset asal dari proses konstruksi KG.

File penting:

```text
dataset-aromatique/
├── aromatique_KG_final.xlsx
├── entity2id.txt
├── kg_final.txt
├── relation2id.txt
├── train.txt
└── test.txt
```

Catatan:

- Produk tidak berada pada ID kontigu `0..339`.
- Produk bercampur dengan note, accord, family, global_ref, dan entity lain.
- Ini valid sebagai KG mentah, tetapi tidak aman langsung dipakai ke KGAT vanilla.

### Folder `dataset-aromatique-kgat-ready`

Ini versi remapping teknis agar cocok dengan asumsi KGAT.

File penting:

```text
dataset-aromatique-kgat-ready/
├── README.md
├── entity2id.txt
├── entity2id_typed.tsv
├── entity_list.txt
├── item_list.txt
├── kg_final.txt
├── old_to_new_entity_id.tsv
├── product2id.tsv
├── relation2id.txt
├── summary.json
├── train.txt
└── test.txt
```

Aturan remapping:

- Product/item ID: `0..339`.
- Non-product entity ID: `340..997`.
- `train.txt` dan `test.txt` hanya berisi product ID `<340`.
- `kg_final.txt` memakai ID hasil remapping yang sama.

Validasi akhir:

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

White Sense:

- Old entity ID: `316`.
- New product ID: `68`.
- Main accord awalnya kosong.
- Generated local accords yang ditambahkan:

```text
floral, violet, fresh, fruity, green, powdery, white floral
```

---

## 4. KGAT Compatibility Patch

Repo KGAT asli berbasis TensorFlow 1.x. Colab modern memakai Python 3.12 dan tidak bisa install `tensorflow==1.15.5`. Karena itu repo sudah dipatch agar berjalan dengan TensorFlow 2.x melalui `tf.compat.v1`.

File baru:

```text
Model/utility/tf_compat.py
```

Isi peran:

- Import `tensorflow`.
- Pakai `tf.compat.v1`.
- Disable eager / v2 behavior.
- Sediakan pengganti Xavier initializer dari Keras:
  - `GlorotUniform`
  - `GlorotNormal`

File model yang sudah diarahkan ke compat layer:

```text
Model/Main.py
Model/KGAT.py
Model/BPRMF.py
Model/CKE.py
Model/CFKG.py
Model/NFM.py
```

Patch Python modern:

- `dict_keys` diubah menjadi list untuk `random.sample`.
- `np.mat` diganti `np.column_stack`.
- SparseTensor indices dibuat `int64`.
- Parameter counting memakai `int(dim)`.
- `Main.py` dibuat tahan untuk smoke test pendek seperti `epoch=2`.

Commit penting:

```text
10e8fa8 Add Aromatique KGAT-ready dataset and Colab TF2 compatibility patch
33a3382 Handle short smoke tests without eval crash
```

GitHub repo `rssdsyndt/kgat.git` sudah menerima commit tersebut di branch `master`.

---

## 5. Colab Setup Yang Sudah Berhasil

Dependency file:

```text
requirements-colab.txt
```

Isi:

```text
tensorflow==2.16.1
numpy<2
scipy<1.13
scikit-learn<1.6
```

Di Colab, JAX perlu dihapus karena konflik `ml-dtypes`:

```python
!pip uninstall -y jax jaxlib
!git clone https://github.com/rssdsyndt/kgat.git /content/kgat
!pip install -r /content/kgat/requirements-colab.txt
```

Setelah install, restart runtime.

Smoke test KGAT yang sudah berhasil:

```python
%cd /content/kgat/Model

!python Main.py \
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

Hasil smoke test:

```text
[n_users, n_items]=[68, 340]
[n_train, n_test]=[1490, 406]
[n_entities, n_relations, n_triples]=[998, 7, 9250]
Final Eval muncul dan script tidak crash.
```

---

## 6. Hasil Full KGAT Baseline

Command yang dijalankan:

```python
%cd /content/kgat/Model

!python Main.py \
  --model_type kgat \
  --data_path ../ \
  --proj_path ../ \
  --weights_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --test_flag full \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 100 \
  --verbose 10 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 1
```

Best result:

```text
Best Iter=[6]@[417.1]
recall=[0.40364 0.51594 0.59071 0.66108 0.69774]
precision=[0.11765 0.08015 0.06103 0.05037 0.04279]
hit=[0.66176 0.72059 0.77941 0.83824 0.85294]
ndcg=[0.35150 0.39534 0.42017 0.43968 0.45275]
```

Default `Ks`:

```text
[20, 40, 60, 80, 100]
```

Interpretasi:

- Best Iter `[6]` berarti evaluasi ke-7.
- Karena evaluasi dilakukan setiap 10 epoch, best checkpoint adalah epoch `69`.
- Checkpoint tersimpan sebagai:

```text
/content/kgat/weights/dataset-aromatique-kgat-ready/kgat_si_sum_bi_l3/64-32-16/l0.0001_r1e-05-1e-05/weights-69.*
```

Metrik utama baseline:

```text
Recall@20 = 0.40364
NDCG@20   = 0.35150
Hit@20    = 0.66176
Recall@100 = 0.69774
NDCG@100   = 0.45275
```

Catatan metodologis:

- Ini baseline KGAT pada virtual-user split.
- `n_users=68` adalah virtual users hasil content-based grouping, bukan real user behavior.
- Untuk tesis, jangan klaim sebagai evaluasi perilaku konsumen nyata.

---

## 7. Status Supabase MCP

Supabase MCP sempat tidak terbaca setelah reload karena entry config perlu di-refresh. Perintah remove/add ulang berhasil.

Perintah yang sudah berhasil:

```powershell
codex mcp remove supabase
codex mcp add supabase --url "https://mcp.supabase.com/mcp?project_ref=esqzngqpfwrlcnapmfpx"
codex mcp login supabase
codex mcp list
```

Status CLI setelah re-add:

```text
Name      Url                                                            Status   Auth
supabase  https://mcp.supabase.com/mcp?project_ref=esqzngqpfwrlcnapmfpx  enabled  OAuth
```

Tool MCP Supabase juga sudah berhasil dipakai dari sesi ini:

```text
mcp__supabase__.get_project_url => https://esqzngqpfwrlcnapmfpx.supabase.co
```

Kesimpulan:

- MCP Supabase sudah aktif untuk project `esqzngqpfwrlcnapmfpx`.
- Jika new chat menggunakan environment/config yang sama, MCP seharusnya tersedia.
- Di new chat, langkah pertama yang disarankan:

```text
Panggil get_project_url dari MCP Supabase.
```

Jika tidak tersedia, jalankan ulang:

```powershell
codex mcp list
codex mcp login supabase
```

---

## 8. Repo Aromatique AI: Status Saat Ini

Clone lokal:

```text
_external/aromatique-ai
```

File Edge Function:

```text
_external/aromatique-ai/supabase/functions/aromatique-chat/index.ts
```

Status saat dibaca:

- Function masih memakai hardcoded `CATALOG` 12 produk dummy.
- Model gateway di kode saat ini memakai `google/gemini-2.5-flash` via Lovable AI Gateway, walaupun dokumen joint research menyebut GPT-4o-mini.
- `mode="chat"` streaming ke AI gateway.
- `mode="recommend"` memaksa tool call `recommend_perfumes`.
- Response recommendation lama:

```json
{
  "products": [...],
  "summary": "..."
}
```

File frontend utama:

```text
_external/aromatique-ai/src/components/AromatiqueApp.tsx
```

Frontend saat ini:

- Mengirim chat ke Edge Function.
- Meminta recommendation dengan `mode: "recommend"`.
- Mengharapkan `data.products`.
- Menyimpan recommendation ke tabel `recommendations` hanya pada kolom `products`.
- Belum menyimpan `kg_paths`, `explanation_type`, `condition`, atau `model_version`.

Migration awal:

```text
_external/aromatique-ai/supabase/migrations/20260503041400_8fe0569e-6712-43f8-bb80-35ccd839941f.sql
```

Tabel yang sudah ada:

```text
conversations
messages
recommendations
feedback
```

Tabel yang belum ada dan perlu ditambahkan:

```text
products
product_embeddings
kg_edges
experiment_sessions
```

Kolom yang perlu ditambahkan:

```text
conversations.condition
recommendations.explanation_type
recommendations.kg_paths
recommendations.model_version
feedback.explanation_type
```

---

## 9. Strategi Integrasi KGAT Sementara

Untuk kebutuhan cepat user testing Raissa, gunakan KGAT sebagai interim recommendation engine.

Prinsip:

- Jangan menjalankan TensorFlow/KGAT di Supabase Edge Function.
- KGAT tetap offline di Colab.
- Edge Function harus ringan dan TypeScript/Deno.
- Supabase menjadi layer data/serving.

Arsitektur sementara:

```text
kgat repo / Colab
  train KGAT baseline
  export serving artifact
        ↓
Supabase
  products
  kg_edges
  product_embeddings or product feature rows
        ↓
aromatique-ai Edge Function
  preference extraction
  KGAT-based / hybrid retrieval
  Top-3 + kg_path
        ↓
Frontend Raissa
  explanation A/B/C
  feedback
```

Untuk pilot Raissa:

- `recommendation_engine = "kgat_baseline"`
- `model_version = "kgat_baseline_epoch69_v1"`

Penting:

- Data pilot KGAT boleh dipakai untuk debugging UI/flow.
- Eksperimen utama 90 partisipan harus memakai satu engine yang stabil.
- Jika engine diganti ke CR-HKGE, pisahkan data pilot dan data eksperimen utama.

---

## 10. API Contract Yang Harus Distabilkan

Target response dari `mode="recommend"`:

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_id": "string",
      "product_name": "string",
      "olfactory_family": "string",
      "main_accords": "string",
      "visual_note": "string",
      "revolutionize": "string | null",
      "match_score": 0,
      "kg_path": [
        {
          "relation": "has_accord",
          "entity": "vanilla",
          "matched": true,
          "reason": "Mengandung accord yang sesuai dengan preferensi pengguna."
        }
      ]
    }
  ],
  "summary": "string",
  "recommendation_engine": "kgat_baseline",
  "model_version": "kgat_baseline_epoch69_v1",
  "explanation_type": "A | B | C"
}
```

Untuk compatibility dengan frontend lama, sementara bisa juga return:

```json
{
  "products": [...],
  "recommendations": [...],
  "summary": "..."
}
```

Tetapi arah yang benar adalah frontend memakai `recommendations`.

---

## 11. Langkah Teknis Berikutnya

Urutan kerja yang disarankan di new chat:

1. Verifikasi MCP Supabase:

```text
Call mcp__supabase__.get_project_url.
Expected: https://esqzngqpfwrlcnapmfpx.supabase.co
```

2. Inspect Supabase schema remote:

```text
Use list_tables or execute_sql to inspect public schema.
Do not assume local migration already applied to remote.
```

3. Buat migration baru di repo `aromatique-ai`:

```text
products
kg_edges
product_embeddings or serving_features
experiment_sessions
ALTER existing tables for condition, kg_paths, explanation_type, model_version
```

4. Buat exporter/seed artifact dari repo KGAT:

Minimal yang diperlukan untuk pilot:

```text
340 products metadata
KG edges readable
product accord/family/visual_note/revolutionize fields
model_version
```

Embedding KGAT bisa menyusul jika belum ada exporter checkpoint. Untuk pilot, retrieval hybrid berbasis metadata KG sudah cukup sebagai fallback, dengan label `kgat_baseline`.

5. Patch Edge Function:

```text
Remove hardcoded CATALOG.
Query Supabase products / kg_edges.
Extract preferences from messages.
Rank products.
Build kg_path.
Return recommendations.
Persist model metadata through frontend insert.
```

6. Patch frontend:

```text
Accept recommendations[] response.
Render product_name, accords/family, reason, match_score.
Store kg_paths and model_version in recommendations table.
Store explanation_type in feedback.
```

7. Deploy/test:

```text
Deploy migration.
Seed products and KG edges.
Deploy Edge Function.
Test mode=chat.
Test mode=recommend.
Run one complete session through feedback.
Verify DB rows.
```

---

## 12. Important Constraints

Supabase Edge Function:

- Do not run KGAT/TensorFlow inside Edge Function.
- Keep retrieval deterministic and fast.
- Edge Function should use Supabase data tables and GPT only for NLU/explanation.

Research:

- KGAT baseline is not CR-HKGE.
- KGAT result is valid as baseline and interim pilot engine.
- CR-HKGE must later replace retrieval engine for Azam's final model claim.
- Raissa's main study should not mix KGAT and CR-HKGE participants unless `model_version` is included and analysis is separated.

Data:

- `entity2id.txt` without type is not enough for explanation.
- Use `entity2id_typed.tsv`, `product2id.tsv`, `old_to_new_entity_id.tsv`, and KG readable data when building Supabase seed.
- Duplicate entity names are normal because `amber`, `woody`, etc. can exist as note, accord, family, or global_accord.

---

## 13. Quick Status Checklist

Completed:

- KGAT dataset remap.
- White Sense generated accords.
- Duplicate triple cleanup.
- TensorFlow 2 Colab compatibility patch.
- KGAT smoke test.
- KGAT 100-epoch baseline.
- KGAT GitHub push to `rssdsyndt/kgat`.
- Supabase MCP configured and project URL verified.
- `aromatique-ai` repo inspected.

Not yet completed:

- Remote Supabase schema migration for KGAT/CR-HKGE serving.
- Product metadata seed into Supabase.
- KG edges seed into Supabase.
- Product embedding export from KGAT checkpoint.
- Edge Function integration.
- Frontend response contract update.
- Pilot deployment verification.

Recommended next task:

```text
Implement Supabase-backed recommendation contract in aromatique-ai using products + KG metadata first, with model_version=kgat_baseline_epoch69_v1.
```

