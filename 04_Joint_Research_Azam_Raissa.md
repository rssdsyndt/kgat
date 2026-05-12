# Penelitian Gabungan: Azam × Raissa
## Aromatique AI Conversational Recommendation System

---

## 1. DUA PENELITIAN, SATU SISTEM

Penelitian ini adalah kolaborasi dua tesis S2 yang berbagi satu infrastruktur sistem tapi meneliti pertanyaan yang berbeda secara fundamental.

| Aspek | Azam (DAPI) | Raissa (HcSE/HCI) |
|---|---|---|
| Judul Singkat | CR-HKGE: KG Embedding untuk Fragrance Recommendation | Pengaruh Tipe XAI pada Trust & Purchase Intention |
| Fokus | Backend Intelligence — kualitas teknis rekomendasi | Frontend Experience — efek penjelasan pada perilaku user |
| RQ Utama | Apakah cross-reference enrichment meningkatkan akurasi? | Apakah tipe explanation mempengaruhi trust dan niat beli? |
| Output | Top-3 products + KG paths (JSON) | Explanation yang dipersonalisasi per tipe XAI |
| Evaluasi | Offline: Precision@3, NDCG@3, MRR | Online: User study, Likert scale, ANOVA |

---

## 2. PENELITIAN RAISSA — RINGKASAN

**Judul:**
> "Pengaruh Pengalaman Pengguna Conversational Recommender System Terhadap Tipe Penjelasan XAI: Studi pada Domain Parfum"

**Inti pertanyaan:** Apakah cara sistem menjelaskan rekomendasi mempengaruhi kepercayaan user dan niat belinya?

### Tiga Kondisi Eksperimen (A/B/C)

Raissa menguji tiga tipe penjelasan yang dihasilkan oleh GPT-4o-mini dengan system prompt berbeda:

**Kondisi A — Feature-Based Explanation**
Menyebut atribut teknis secara eksplisit: olfactory family, main accords, dan mengaitkannya ke preferensi user.
- Contoh: *"Moz Art masuk keluarga Oriental Spicy dengan karakter Caramel dan Vanilla — persis yang kamu cari."*
- File system prompt: `system_prompt_feature_based_explanation.md`

**Kondisi B — Narrative-Based Explanation**
Membangun narasi emosional dan sensoris. TIDAK menyebut technical terms.
- Contoh: *"Moz Art adalah parfum untuk malam yang ingin terasa istimewa — hangatnya seperti kopi pahit yang baru diseduh."*
- File system prompt: `system_prompt_narrative_based_explanation.md`

**Kondisi C — Comparative-Based Explanation**
Membandingkan produk satu dengan lainnya, framing sebagai spectrum.
- Contoh: *"Moz Art adalah yang paling dalam di antara ketiganya — lebih hangat dibanding Verve Vale, lebih manis dibanding produk ketiga."*
- File system prompt: `system_prompt_comparative_based_explanation.md`

### Hipotesis Raissa

| ID | Pernyataan |
|---|---|
| H1 | Ada perbedaan signifikan pada perceived trust berdasarkan tipe explanation (A/B/C) |
| H2 | Ada pengaruh signifikan tipe explanation terhadap purchase intention |
| H3 | Ada hubungan positif antara perceived trust dan purchase intention |
| H4 | Familiaritas domain memoderasi pengaruh tipe explanation terhadap trust |

### Instrumen Feedback (8 Item Likert 1-5)

| ID | Konstruk | Pernyataan |
|---|---|---|
| T1 | Trust | Saya percaya rekomendasi chatbot ini sesuai preferensi saya |
| T2 | Trust | Penjelasan chatbot membuat saya yakin sistem memahami kebutuhan saya |
| PI1 | Purchase Intention | Saya berniat membeli parfum yang direkomendasikan |
| PI2 | Purchase Intention | Saya kemungkinan besar akan mempertimbangkan membeli parfum ini |
| U1 | Usefulness | Penjelasan membantu saya memahami mengapa parfum ini cocok |
| U2 | Usefulness | Penjelasan berguna untuk mendukung keputusan pembelian |
| S1 | Satisfaction | Secara keseluruhan saya puas dengan chatbot ini |
| S2 | Satisfaction | Saya akan merekomendasikan chatbot ini kepada orang lain |

### Desain Eksperimen

- **Between-subjects** — setiap user hanya menerima satu kondisi (A, B, atau C)
- **Target partisipan:** 90 orang (30 per kondisi)
- **Lokasi:** Offline store Aromatique (validitas ekologis tinggi — user bisa coba produk fisik)
- **Randomisasi:** Otomatis per session (rotasi A → B → C → A → ...)
- **Analisis:** One-way ANOVA, post-hoc Mann-Whitney, mediasi Trust→PI

---

## 3. ARSITEKTUR SISTEM END-TO-END

```
┌─────────────────────────────────────────────────┐
│                 FRONTEND (React)                 │
│  Welcome → Consent → Chat → Reco → Feedback     │
│                   [RAISSA]                       │
└──────────────────────┬──────────────────────────┘
                       │ Supabase Edge Function
                       │ (aromatique-chat/index.ts)
           ┌───────────┴──────────┐
           │                      │
    mode: "chat"           mode: "recommend"
           │                      │
    GPT-4o-mini              Preference Extraction
    streaming                     │
    [kondisi A/B/C]         CR-HKGE Retrieval
                                  │
                            Top-3 + KG Paths
                                  │
                       XAI Explanation Generator
                       [kondisi A/B/C system prompt]
                                  │
                            GPT-4o-mini
                            (explanation)
                                  │
                         Response JSON ke Frontend
```

### Layer-by-Layer

| Layer | Nama | Domain | Teknologi |
|---|---|---|---|
| L1 | User Interface | Raissa | React + Vite + TanStack Router |
| L2 | NLU & Context | Raissa | GPT-4o-mini (preference extraction) |
| L3 | KG Storage | Azam | Supabase PostgreSQL + pgvector |
| L4 | CR-HKGE Retrieval | Azam | Python training → embeddings di Supabase |
| L5 | XAI Module | Raissa | GPT-4o-mini + system prompt A/B/C |
| L6 | LLM Response | Raissa | GPT-4o-mini streaming |
| L7 | Feedback Loop | Shared | Supabase (tabel feedback) |

---

## 4. APA YANG RAISSA BUTUHKAN DARI AZAM

Raissa membutuhkan **satu output** dari sistem Azam:

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_name": "string",
      "olfactory_family": "string",
      "main_accords": "string (comma-separated)",
      "revolutionize": "string | null",
      "match_score": "float (0-100)",
      "kg_path": [
        {
          "relation": "has_accord | inspired_by | belongs_to_family | ...",
          "entity": "string",
          "matched": "boolean",
          "reason": "string (bahasa Indonesia)"
        }
      ]
    }
  ]
}
```

**`kg_path`** adalah yang paling kritikal untuk Raissa — inilah yang dikonversi oleh XAI module menjadi 3 tipe penjelasan berbeda. Tanpa `kg_path` yang meaningful, explanation Raissa menjadi generatif murni (tidak berbasis KG nyata).

---

## 5. IMPLEMENTASI KONDISI A/B/C DI EDGE FUNCTION

Logika yang harus ada di `supabase/functions/aromatique-chat/index.ts`:

### Assignment kondisi per session
```
Saat session baru pertama kali request:
  1. Hitung total sessions yang sudah ada
  2. condition = ['A','B','C'][total_count % 3]
  3. Simpan ke tabel experiment_sessions
  4. Gunakan kondisi ini untuk seluruh sesi

Saat session lama request:
  1. Ambil kondisi dari tabel experiment_sessions
  2. Gunakan kondisi yang sudah terassign
```

### Penggunaan system prompt
- Mode `chat`: system prompt berisi instruksi percakapan sesuai kondisi (10 pertanyaan, gaya A/B/C)
- Mode `recommend`: system prompt berisi instruksi explanation sesuai kondisi

---

## 6. DATABASE SCHEMA (Lengkap)

### Tabel yang sudah ada (jangan diubah)
```sql
conversations (id, session_id, title, created_at, updated_at)
messages (id, conversation_id, role, content, created_at)
recommendations (id, conversation_id, batch_index, products, created_at)
feedback (id, conversation_id, session_id, answers, created_at)
```

### Tabel baru yang harus ditambahkan
```sql
-- Produk Aromatique asli
products (id TEXT PK, product_name, olfactory_family, main_accords,
          revolutionize, visual_note, meaning, has_enrichment BOOLEAN)

-- Embedding hasil CR-HKGE Python (diisi SEKALI setelah training)
product_embeddings (product_id TEXT PK → products.id,
                    embedding VECTOR(64), enriched BOOLEAN)

-- Tracking kondisi A/B/C per session
experiment_sessions (session_id TEXT PK, condition TEXT CHECK('A','B','C'),
                     assigned_at TIMESTAMPTZ)
```

### Kolom tambahan di tabel yang ada
```sql
ALTER TABLE conversations ADD COLUMN condition TEXT;
ALTER TABLE recommendations ADD COLUMN explanation_type TEXT;
ALTER TABLE recommendations ADD COLUMN kg_paths JSONB;
ALTER TABLE feedback ADD COLUMN explanation_type TEXT;
```

---

## 7. URUTAN INTEGRASI

### Fase 1: Azam selesaikan dulu
1. Training CR-HKGE di Google Colab
2. Upload embeddings ke Supabase `product_embeddings`
3. Build retrieval function (query vector → cosine similarity → Top-3 + kg_paths)
4. Test endpoint `/recommend` menghasilkan JSON yang benar

### Fase 2: Integrasi ke Edge Function
1. Tambah migration SQL (tabel baru + ALTER tabel lama)
2. Modifikasi Edge Function untuk kondisi A/B/C
3. Ganti catalog hardcoded dengan query ke tabel `products`
4. Sambungkan retrieval ke embedding di `product_embeddings`

### Fase 3: Pilot Study
1. Deploy ke production URL yang stabil
2. Test dengan 5-10 partisipan
3. Verifikasi data tersimpan benar di semua tabel
4. Pastikan `explanation_type` tersimpan di `recommendations` dan `feedback`

### Fase 4: Eksperimen Utama
1. Rekrut 90 partisipan di store Aromatique
2. Koleksi data selesai
3. Analisis statistik Raissa (SPSS: ANOVA, mediasi, moderasi)

---

## 8. HAL KRITIKAL YANG HARUS DISINKRONKAN

| Item | Azam | Raissa | Status |
|---|---|---|---|
| Nama kolom dataset | `revolutionize` | `inspired_by` | ⚠️ Perlu disepakati satu nama |
| Jumlah produk | 340 | 340 (sudah update) | ✅ Sama |
| Format kg_path JSON | Azam yang define | Raissa yang consume | 🔄 Perlu API contract |
| explanation_type di DB | Azam insert | Raissa analisis | 🔄 Perlu migration |
| Kondisi A/B/C assignment | Di Edge Function | Di Edge Function | 🔄 Belum diimplementasi |
| Product images di UI | Dari mana? | Perlu untuk UI reco | ❓ Belum dibahas |
