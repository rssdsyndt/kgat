# Dataset Schema & Validation Guide
## Aromatique Knowledge Graph — Pengecekan dan Mekanisme Data

---

## 1. RINGKASAN DATASET AKHIR

| Metrik | Nilai | Keterangan |
|---|---|---|
| Total produk | **340** | Setelah hapus 22 GENERATED names |
| Produk COMPLETE | 242 | Semua kolom ada termasuk revolutionize |
| Produk NO_REVOLUTIONIZE | 97 | Tidak punya inspirasi parfum global |
| Produk GENERATED_ACCORDS | 1 | White Sense — main accord digenerate dari visual note + global enrichment |
| Produk WITH revolutionize | 243 | Grup Enriched untuk CR-HKGE |
| Produk WITHOUT revolutionize | 97 | Grup Standard untuk CR-HKGE |

---

## 2. SKEMA KOLOM DATASET PRODUK

File referensi: `aromatique_KG_final.xlsx` → Sheet `1_products_clean`

| Kolom | Contoh Nilai | Sumber | Wajib? |
|---|---|---|---|
| `product_name` | Moz Art | Input Aromatique | ✅ WAJIB |
| `visual_note` | Amber / Ginger | Input Aromatique | ✅ WAJIB |
| `visual_note_alt` | Ginger / Amber | Input Aromatique | ❌ Opsional |
| `revolutionize` | LV L'Immensite | Input Aromatique | ❌ Opsional (243/340 ada) |
| `main_accords` | Aromatic, Citrus, Amber | Input Aromatique | ✅ WAJIB* |
| `olfactory_family` | FRESH SPICY | Derived dari visual_note[0] | ✅ WAJIB |
| `meaning` | Perpaduan hangat jahe | Input Aromatique | ❌ Opsional |
| `data_quality` | COMPLETE / NO_REVOLUTIONIZE | Generated | Info saja |

> \*1 produk (White Sense) awalnya tidak punya main_accords lokal. Untuk versi training, accord digenerate dari visual note + global enrichment Ginza M Shiseido: `floral, violet, fresh, fruity, green, powdery, white floral`.

---

## 3. SKEMA KNOWLEDGE GRAPH

### 3.1 Entitas (998 total)

| Tipe Entitas | Jumlah | Contoh | Asal |
|---|---|---|---|
| `product` | 340 | "Moz Art" | Dataset Aromatique |
| `global_ref` | 241 | "JPG Scandal Pour Homme" | Kolom revolutionize |
| `note` | 185 | "Amber", "Pepper" | Kolom visual_note |
| `accord` | 110 | "sweet", "vanilla" | Kolom main_accords |
| `global_accord` | 60 | "caramel", "tobacco" | Fragrantica via global_reference.xlsx |
| `global_family` | 45 | "oriental woody" | Fragrantica via global_reference.xlsx |
| `family` | 17 | "AMBER", "WOODY" | Derived dari visual_note |

### 3.2 Relasi (7 tipe, 9.250 triplet total setelah deduplication + White Sense accord generation)

| ID | Nama Relasi | Jumlah Triplet | Arah | Penjelasan |
|---|---|---|---|---|
| 0 | `inspired_by` | 243 | product → global_ref | Cross-reference bridge (inti novelty) |
| 1 | `has_accord` | 1.629 | product → accord | Accord lokal Aromatique + generated accord White Sense |
| 2 | `has_visual_note` | 680 | product → note | Visual note dari kolom visual_note |
| 3 | `belongs_to_family` | 340 | product → family | Satu per produk |
| 4 | `sem_similar` | 4.110 | product ↔ product | Computed, bidirectional |
| 5 | `has_global_accord` | 2.017 | global_ref → global_accord | Accord dari Fragrantica setelah deduplication |
| 6 | `belongs_to_global_family` | 231 | global_ref → global_family | Family dari Fragrantica setelah deduplication |

### 3.3 Format File KG

**`kg_final.txt`** — format: `head_id SPASI relation_id SPASI tail_id`
```
243  0  9      ← produk ID 243 --inspired_by--> global_ref ID 9 (Interlude Man Amouage)
9    5  10     ← global_ref ID 9 --has_global_accord--> accord ID 10 (amber)
18   2  1      ← produk ID 18 (Berrylicious) --has_visual_note--> note ID 1 (amber)
```

**`entity2id.txt`** — format: `count\n nama TAB id`
```
998
Moz Art    0
Berrylicious   18
interlude man amouage  9
amber  1
...
```

**`relation2id.txt`** — format: `count\n nama TAB id`
```
7
inspired_by    0
has_accord     1
has_visual_note    2
belongs_to_family  3
sem_similar    4
has_global_accord  5
belongs_to_global_family   6
```

---

## 4. FORMAT TRAINING FILES

**`train.txt`** — format: `virtual_user_id SPASI item_id1 SPASI item_id2 ...`
```
0 18 45 67 102 134 201
1 5 88 144 267
...
```
- 68 baris (virtual users)
- 1.490 total product interactions
- Split 80% dari total pasangan

**`test.txt`** — format identik train.txt
- 68 baris
- 406 total product interactions
- Split 20% dari total pasangan

---

## 5. PENGECEKAN KESESUAIAN DENGAN PLAN PENELITIAN

### ✅ Yang sudah sesuai

- [x] 340 produk aktif Aromatique (dikonfirmasi langsung dari Aromatique)
- [x] Semua 22 produk tanpa product_name dihapus
- [x] 0 unresolved revolutionize — semua 243 parfum global berhasil di-resolve via fuzzy matching
- [x] 17 olfactory family sesuai fragrance wheel standar
- [x] Dua grup yang diperlukan untuk natural experiment tersedia: 243 enriched + 97 standard
- [x] Format file identik dengan KGAT asli — `kg_final.txt`, `entity2id.txt`, `relation2id.txt`
- [x] `train.txt` dan `test.txt` sudah 80/20 split
- [x] Semua item ID di train/test valid (verified saat generate)

### ⚠️ Yang perlu perhatian

- [x] **White Sense** (1 produk): main accord sudah digenerate dan ditambahkan ke `kg_final.txt` sebagai 7 triple `has_accord`, sehingga tidak lagi kosong saat training.
- [ ] **22 produk dihapus** mengubah distribusi per family. Perlu dicek apakah family tertentu kehilangan representasi berlebihan.
- [ ] **Sem_similar bidirectional** sudah dimasukkan dua arah. Jika KGAT menambahkan inverse relation secara otomatis, akan ada duplikasi. Cek di `loader_kgat.py` apakah ada `n_relations * 2` logic.

---

## 6. MEKANISME PEMBANGUNAN DATASET (Alur Lengkap)

```
Excel Raw Aromatique (362 baris)
        │
        ▼ Parsing & Cleaning
        ├── Hapus baris tanpa product_name (22 baris GENERATED)
        ├── Clean main_accords (hapus prefix Top:/Heart:/Base:)
        └── Derive olfactory_family dari visual_note[0]
        │
        ▼ 340 produk bersih
        │
        ▼ Entity Extraction
        ├── product nodes (340)
        ├── note nodes dari visual_note
        ├── family nodes dari olfactory_family
        └── accord nodes dari main_accords
        │
        ▼ Cross-Reference Enrichment
        ├── global_ref nodes dari kolom revolutionize (243)
        ├── Fuzzy matching ke global_reference.xlsx
        ├── global_accord nodes dari Fragrantica data
        └── global_family nodes dari Fragrantica data
        │
        ▼ Triple Construction
        ├── inspired_by: product → global_ref
        ├── has_accord: product → accord
        ├── has_visual_note: product → note
        ├── belongs_to_family: product → family
        ├── has_global_accord: global_ref → global_accord
        ├── belongs_to_global_family: global_ref → global_family
        └── sem_similar: computed (≥2 accord sama + family sama)
        │
        ▼ Content-Based Positive Pairs
        ├── Group by family → virtual users
        ├── Group by shared accords (≥3) → virtual users
        └── 80/20 split → train.txt + test.txt
        │
        ▼ OUTPUT FILES
        ├── kg_final.txt (9.250 triplets)
        ├── entity2id.txt (998 entities)
        ├── relation2id.txt (7 relations)
        ├── train.txt (68 virtual users, 1.490 interactions)
        └── test.txt (68 virtual users, 406 interactions)
```

---

## 7. DISTRIBUSI PER OLFACTORY FAMILY

17 family yang ada di dataset:

```
AMBER, AQUATIC, AROMATIC, CITRUS, FLORAL,
FRESH SPICY, FRUITY, GREEN, LEATHER, MUSK POWDERY,
OUD, RED FLORAL, SWEET, VANILLA, WARM SPICY,
WHITE FLORAL, WOODY
```

Setiap produk hanya masuk ke **satu** family (dari visual_note bagian pertama).

---

## 8. FILE REFERENSI DATASET

| File | Isi | Digunakan Untuk |
|---|---|---|
| `aromatique_KG_final.xlsx` | 8 sheet lengkap + visualisasi | Review dan debugging |
| `kg_final.txt` | KG triplets numerik | Input KGAT training |
| `entity2id.txt` | Mapping nama → ID | Lookup produk |
| `relation2id.txt` | Mapping relasi → ID | Referensi relasi |
| `train.txt` | Virtual user pairs (train) | KGAT training |
| `test.txt` | Virtual user pairs (test) | KGAT evaluation |

---

## 9. VERSI KGAT-READY

Folder tambahan: `dataset-aromatique-kgat-ready`

Versi ini dibuat khusus agar sesuai dengan asumsi loader KGAT asli:

- Product entity diremap menjadi item/entity ID `0..339`.
- Entity non-produk diremap mulai dari ID `340..997`.
- `train.txt` dan `test.txt` hanya berisi item ID produk `<340`.
- `kg_final.txt` memakai ruang ID hasil remapping yang sama.
- Mapping tipe entity disimpan di `entity2id_typed.tsv`, `product2id.tsv`, dan `old_to_new_entity_id.tsv`.

Hasil validasi loader:

| Metrik | Nilai |
|---|---:|
| KGAT inferred `n_users` | 68 |
| KGAT inferred `n_items` | 340 |
| KGAT inferred `n_entities` | 998 |
| KG triples | 9.250 |
