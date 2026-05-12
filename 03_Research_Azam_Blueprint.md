# Penelitian Azam: CR-HKGE
## Cross-Reference Semantic Enrichment on Heterogeneous Knowledge Graph Embedding
### for Personalized Fragrance Recommendation in Conversational Systems

---

## 1. LATAR BELAKANG SINGKAT

Sistem rekomendasi berbasis Knowledge Graph (KG) yang ada — KGAT, HGKR, KGCN, RippleNet — seluruhnya dikembangkan dan dievaluasi pada dataset berskala besar dengan jutaan interaksi pengguna (MovieLens, Amazon-Book, Last.FM). Kondisi nyata bisnis fragrance lokal seperti Aromatique menciptakan skenario yang secara teknis berbeda fundamental:

- Hanya **340 produk aktif**
- **Nol user interaction history**
- Domain sangat terspesialisasi (fragrance)
- Ada atribut unik: kolom `revolutionize` yang menghubungkan produk lokal ke parfum global

Dua kegagalan teknis terjadi bersamaan pada kondisi ini. Pertama, semua model baseline membutuhkan user-item interaction matrix — tanpa ini model tidak bisa dieksekusi secara matematis (crash di `load_data.py`). Kedua, metode embedding standar tidak membedakan bobot antar tipe relasi, padahal relasi `inspired_by` secara semantik jauh lebih informatif dari `has_visual_note`.

---

## 2. RESEARCH QUESTIONS

**RQ Utama:**
> Bagaimana metode heterogeneous KG embedding yang memanfaatkan cross-reference semantic enrichment melalui relasi `inspired_by` meningkatkan kualitas rekomendasi fragrance personal pada small-scale catalog dalam sistem conversational?

**RQ 1:** Bagaimana membangun Heterogeneous Fragrance KG yang mengintegrasikan relasi multi-tipe dengan cross-reference link ke global fragrance knowledge space?

**RQ 2:** Seberapa signifikan kontribusi cross-reference enrichment terhadap kualitas embedding dibandingkan attribute-only embedding pada produk yang sama?

**RQ 3:** Metode hybrid retrieval berbasis KG embedding mana yang paling efektif untuk menghasilkan Top-3 rekomendasi personal dalam alur conversational?

---

## 3. RESEARCH GAPS (dari Paper Original Terindeks)

### Gap 1 — Nature Scientific Reports 2023 (Zhang et al., HGKR)
> *"Experiments show the instability of our model when facing the challenge of the sparser dataset. It is worth verifying whether introducing outstanding KG reasoning and completing techniques into our method will be helpful."*

**Relevansi:** Dataset Aromatique (340 produk, 0 interaction) adalah ekstrem dari sparser dataset yang HGKR akui tidak stabil.

### Gap 2 — Nature Scientific Reports 2024 (Rong et al., ML-KDGATMoco)
> *"The KGAT model may face the challenge of learning effective entity and relationship information in extremely sparse datasets... it has difficulties in allocating attention."*

**Relevansi:** Bukti empiris dari paper original bahwa KGAT gagal pada sparse data. Justifikasi teknis mengapa modifikasi diperlukan.

### Gap 3 — Wiley IJIS 2024 (Wan et al., HN-DKG)
> *"The weight of edges in heterogeneous information networks is set as the same, and the different weights of different types of edges are not taken into account, which makes the recommendation result not ideal."*

**Relevansi:** Langsung dijawab oleh Mekanisme 2 CR-HKGE: relation-type specific attention weights.

### Gap 4 — Springer Applied Intelligence 2025 (Zhang et al., HKGAT)
Paper diuji di Amazon-book (besar). Future work mengakui perlunya ekstensi ke domain baru dan cross-domain semantic bridges.

**Relevansi:** CR-HKGE mengekstensi heterogeneous KG attention dengan cross-reference layer yang tidak ada di HKGAT.

---

## 4. NOVELTY (3 Level)

### Novelty 1: Fragrance-Specific Heterogeneous KG Construction
Pembangunan KG dengan 7 tipe relasi yang memiliki bobot semantik berbeda. Skema ini tidak ada di literatur karena belum ada penelitian yang membangun KG untuk inspired fragrance catalog.

- Menjawab: CKE (2016) future work — exploit heterogeneous information from diverse sources
- Menjawab: DSINS (2025) future work — explore domain-specific graph construction strategies

### Novelty 2: Cross-Reference Semantic Propagation Layer
Mekanisme propagasi khusus yang menggunakan relasi `inspired_by` untuk mengalirkan representasi semantik dari Global Reference node ke product embedding. Mengatasi instabilitas pada sparse dataset.

- Menjawab: HGKR (2023) future work — KG reasoning/completing for sparse stability
- Menjawab: KGBPR (2022) future work — integrate higher-order information from KG

### Novelty 3: Relation-Type Specific Attention Weights
Setiap tipe relasi mendapat learnable scalar parameter `λ_r`. Model belajar sendiri bahwa `inspired_by` lebih informatif dari `has_visual_note` dalam kondisi sparse.

- Menjawab: HN-DKG (2024) future work — different weights for different edge types
- Menjawab: KGAT (2019) future work — hard attention to filter uninformative entities
- Menjawab: RippleNet (2018) future work — non-uniform samplers during propagation

---

## 5. FLOW SISTEM (Layer Azam)

```
User Input (teks percakapan)
        ↓
[Layer 2 — NLU Raissa]
Ekstrak preferensi → structured query {accords, family, occasion}
        ↓
[Layer 3 — KG Storage (Azam)]
Heterogeneous Fragrance KG: 998 entitas, 9.255 triplet, 7 relasi
        ↓
[Layer 4 — CR-HKGE Retrieval (Azam)]
1. Encode query → query vector
2. Cosine similarity dengan 340 product embeddings
3. Filter + ranking
4. Output: Top-3 products + KG paths
        ↓
[Layer 5 — XAI Raissa]
KG paths → 3 tipe explanation
        ↓
OUTPUT: Respons natural + rekomendasi + penjelasan
```

---

## 6. ARSITEKTUR MODEL CR-HKGE

### Layer 1: KG Embedding (TransR — identik KGAT)
```
g(h,r,t) = ||W_r · e_h + e_r - W_r · e_t||²
Loss_KG = Σ -ln σ(g(h,r,t') - g(h,r,t))
```
Tidak ada perubahan dari KGAT asli di layer ini.

### Layer 2: Relation-Type Specific Attention (Novelty 2)
```
Modifikasi formula KGAT:
  π(h,r,t) = (W_r · e_t)ᵀ · tanh(W_r · e_h + e_r)    ← KGAT asli

Menjadi:
  π_CR(h,r,t) = λ̃_r · (W_r · e_t)ᵀ · tanh(W_r · e_h + e_r)   ← CR-HKGE

Di mana λ̃_r = softmax(λ_r), satu scalar per tipe relasi
```

### Layer 3: Cross-Reference Propagation (Novelty 1)
```
Untuk produk WITH revolutionize (enriched):
  e_CR(p) = λ̃_ib · W_CR · (e_g + Σ π_CR(g,r',t') · e_t')
  e_p^(l) = σ(W · (e_p^(l-1) + e_N_int(p) + e_CR(p)))

Untuk produk WITHOUT revolutionize (standard):
  e_p^(l) = σ(W · (e_p^(l-1) + e_N_int(p)))   ← identik KGAT
```

### Layer 4: Prediction
```
e_p* = e_p^(0) || e_p^(1) || ... || e_p^(L)  (concatenation)
score(q, p) = e_q^T · e_p*                     (cosine similarity)
Top-3 = argmax_{p ∈ P} score(q, p)
```

---

## 7. FORMAT OUTPUT (Titik Integrasi dengan Raissa)

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_name": "Moz Art",
      "olfactory_family": "AMBER",
      "main_accords": "Caramel, Sweet, Vanilla, Aromatic, Citrus",
      "revolutionize": "JPG Scandal Pour Homme",
      "match_score": 89.4,
      "kg_path": [
        {"relation": "has_accord", "entity": "sweet", "matched": true,
         "reason": "Mengandung Sweet sesuai preferensimu"},
        {"relation": "has_accord", "entity": "vanilla", "matched": true,
         "reason": "Vanilla sesuai preferensimu"},
        {"relation": "inspired_by", "entity": "JPG Scandal Pour Homme", "matched": false},
        {"relation": "belongs_to_family", "entity": "AMBER", "matched": true}
      ]
    },
    {"rank": 2, "...": "..."},
    {"rank": 3, "...": "..."}
  ]
}
```

---

## 8. REFERENSI UTAMA

| Paper | Journal | Tahun | Peran |
|---|---|---|---|
| Zhang et al. (HGKR) | Nature Scientific Reports | 2023 | Gap 1 — instability on sparse |
| Rong et al. (ML-KDGATMoco) | Nature Scientific Reports | 2024 | Gap 2 — KGAT fails sparse |
| Wan et al. (HN-DKG) | Wiley IJIS | 2024 | Gap 3 — homogeneous edge weights |
| Zhang et al. (HKGAT) | Springer Applied Intelligence | 2025 | Gap 4 — no cross-domain bridge |
| Wang et al. (KGAT) | ACM KDD | 2019 | Base model yang dimodifikasi |
| Liang et al. (KGCN-UP) | Nature Scientific Reports | 2025 | Validasi content-based pairs |
| Ma et al. (KGBPR) | ACM ICCPR | 2022 | Gap — higher-order information |

---

## 9. METHOD & TOOLS

| Komponen | Tool/Library |
|---|---|
| Base Model | KGAT (TensorFlow 1.15) |
| Graph Library | NetworkX (opsional, untuk visualisasi) |
| Training Environment | Google Colab (GPU T4) |
| Embedding Storage | Supabase pgvector |
| Backend API | FastAPI (Python) atau Supabase Edge Function |
| Evaluation | Custom script berbasis KGAT batch_test.py |

---

## 10. METRIK EVALUASI

Semua terstandarisasi, tidak ada metrik custom:

| Metrik | K | Alasan |
|---|---|---|
| Precision@K | 3, 5 | Output adalah Top-3 |
| Recall@K | 3, 5 | Pasangan Precision |
| NDCG@K | 3, 5 | Metrik utama — mempertimbangkan posisi |
| MRR | — | Seberapa tinggi item relevan pertama muncul |
| Hit Rate@K | 3 | Minimal 1 relevan di Top-3 |
| Catalog Coverage | — | Anti-popularity bias untuk 340 produk |
