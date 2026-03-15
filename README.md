# RAG ThesisDoc System
**Retrieval-Augmented Generation (RAG) untuk Dokumen Skripsi/TA**  

> **Status Terkini:** V2.0 (`v2.0-hybrid`) sudah di-tag di `main`. Branch `dev` aktif untuk pengembangan V3.x.

---

## Roadmap Versi (Bertahap)
- **V1.0 (DONE ✅)** — `v1.0-naive`: Naive Dense RAG baseline + UI + Audit Logging + PDF Viewer/Highlight + Utility Cleanup Runs
- **V1.5 (DONE ✅)** — `v1.5-memory`: Multi-turn Chat (Session Memory) + Query Contextualization + Guardrails + UI/UX Enhancement
- **V2.0 (DONE ✅)** — `v2.0-hybrid`: Hybrid Retrieval (BM25+Sastrawi Sparse + Dense Vector + RRF Fusion) + UI Hybrid Score Display
- **V3.0+ (NEXT 🔄)**: Regex Parsing (narasi vs sitasi), Metadata Filtering, Cross-Encoder Reranking, Post-Hoc Verification

**Pemetaan Ablation Study:**
| Skenario | Versi | Tag | Keterangan |
|----------|-------|-----|------------|
| A | V1.0 | `v1.0-naive` | Naive Dense RAG |
| B | V2.0 | `v2.0-hybrid` | Hybrid RAG (BM25+Sastrawi+RRF) |
| C | V3.x | *(planned)* | Proposed Method (Full Pipeline) |

---

## Summary
Repo ini berisi prototipe sistem **Question Answering** pada kumpulan PDF skripsi/TA.
Versi **V2.0** membangun di atas V1.5 dengan tambahan jalur retrieval hybrid:
- **Hybrid Retrieval**: BM25+Sastrawi (sparse/keyword) + Dense vector (Chroma) digabungkan via Reciprocal Rank Fusion (RRF)
- Dense retrieval (Chroma + embeddings) dengan diversity retrieval
- Multi-turn memory (sliding window, topic shift detection)
- Query contextualization (rewrite sebelum retrieval)
- Generator (LLM) opsional (Groq/OpenAI/Ollama) dengan output guardrails lengkap
- UI Streamlit demo-ready (turn viewer, history expander, styled sections, hybrid score display)
- Coverage retrieval per-metode (untuk pertanyaan multi-target)
- Metadata routing (page count intent)
- Logging lengkap untuk audit (manifest + retrieval.jsonl + answers.jsonl + app.log)
- PDF Viewer berbasis gambar + **highlight keyword** (auto/manual)
- Utility `cleanup_runs.py` (dry-run default, aman)

> Prinsip utama: **transparan & dapat diaudit**. Jawaban selalu disertai konteks (CTX) dan sumber.
> Format output wajib: `Jawaban: ...` + `Bukti: - ... [CTX n]` — sitasi `[CTX n]` hanya di bagian Bukti.

---

## Fitur Utama 

### [V1.0 - Baseline]

#### Retrieval (Dense)
- Vector DB: **Chroma** (lokal/persisten)
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Query: `similarity_search_with_score()`
- UI menampilkan **distance + similarity** + metadata (source_file, page)
- Optimasi: caching embeddings & vectorstore (lebih cepat untuk multi-query)

#### Generation (Opsional)
- Toggle di UI: **Use LLM (Generator)**
- Guardrail prompt:
  - Bahasa Indonesia
  - Hanya dari konteks
  - Jika tidak ada bukti: **"Tidak ditemukan pada dokumen."**
  - Klaim faktual wajib disertai `[CTX n]`
- Provider:
  - `groq` (default untuk development — `llama-3.1-8b-instant`)
  - `openai`
  - `ollama` (offline/privasi — Llama-3 Quantized 4-bit)

#### UI/UX Streamlit (V1.0)
- Sidebar controls: config, top_k, max_chars_per_ctx, toggle LLM, status API key
- Indikator config (INFO/WARNING/SUCCESS) → jelas kapan config sudah diterapkan
- Riwayat session (preview Q/A)
- Sources Top-k:
  - filter sources (doc_id/source_file/text)
  - tombol View/Download PDF per chunk
- PDF Viewer:
  - render per halaman jadi PNG (anti-block browser)
  - highlight keyword (auto dari query/CTX + manual override)

#### Logging & Audit (per-run)
Folder `runs/<run_id>/` berisi:
- `manifest.json` — snapshot config + git hash
- `retrieval.jsonl` — log retrieval (top_k, nodes, distance/similarity, metadata)
- `answers.jsonl` — log jawaban (raw answer, used_context ids)
- `app.log` — log aplikasi
- UI menyediakan download per file + `logs.zip`

---

### [V1.5 — Memory, Contextualization & Guardrails]

#### Multi-turn Memory
- **Sliding window memory** (default 3–5 turns, dapat diatur di sidebar)
- **Topic shift detection** — jika query berubah topik, memory tidak dipaksa disambungkan ke topik lama
- **Anaphora detection** — query seperti *"tahapannya?"* / *"itu apa?"* otomatis dikenali sebagai follow-up
- **History policy** — segmentasi otomatis setelah topic shift; untuk anafora setelah shift, pakai window *pre-shift*

#### Query Contextualization
- Rewrite query lanjutan menjadi **standalone query** sebelum retrieval
  (mis. *"tahapannya?"* → *"Apa saja tahapan dari metode Prototyping?"*)
- Dua mode rewrite: **LLM** (akurat) dan **heuristic** (fallback deterministik, lebih stabil untuk anafora+steps)
- **Quality gate**: jika hasil rewrite terlalu umum/lemah, fallback ke heuristic otomatis
- Badge `contextualized` / `direct` di UI menunjukkan apakah query telah direwrite

#### Coverage Retrieval per-Metode
- Untuk query multi-target (mis. *"RAD dan Prototyping"*), sistem melakukan retrieval tambahan **per metode** agar setiap metode terwakili di konteks
- Steps-aware expansion: untuk pertanyaan tahapan, retrieval diperluas ke semua metode target
- `select_nodes_with_coverage()` menjamin minimal 1 node per metode target

#### Diversity Retrieval
- Mode **Auto / On / Off** — Auto aktif jika query mengandung sinyal lintas dokumen
- `max_per_doc` dan `candidate_pool_multiplier` dapat diatur di sidebar
- Tiga-pass diversification: sebar dokumen dulu, lalu isi sisa slot, lalu relax cap

#### Output Guardrails
- **Format enforcement**: `Jawaban:` + `Bukti:` selalu ada, walau LLM tidak patuh
- **Citation normalization**: variasi `(CTX 2)`, `CTX2`, dll. → standar `[CTX n]`
- **Citation range check**: sitasi yang keluar range CTX tersedia otomatis diperbaiki
- **Grounding check**: setiap bullet Bukti divalidasi terhadap CTX yang dirujuk (anti-halusinasi)
- **Deterministic answer** untuk query tipe METODE — stabil, tidak bergantung format LLM
- **Structured steps answer** untuk query tipe TAHAP + multi-metode — LLM hanya memoles bahasa, bukan mengarang fakta
- **Cap + dedupe Bukti**: maksimal N bullet (default 3, expandable 5), deduplikasi otomatis
- **Post-hoc repair**: format, citation range, grounding — masing-masing 1x repair attempt sebelum fallback

#### Metadata Routing
- Intent **page_count** — *"berapa halaman dokumen X?"* dijawab langsung dari `doc_catalog.json` tanpa retrieval ke Chroma
- Catalog dibangun saat ingest, merge inkremental (tidak overwrite entry lama saat partial ingest)

#### UI/UX Enhancement (V1.5)
- **Turn Viewer** — klik turn historis di history expander untuk melihat ulang hasil turn tersebut
- **History expander** — scrollable, card-style buttons, active turn highlighted dengan border biru
- **Viewing banner** — banner informatif saat sedang melihat turn historis (bukan turn terbaru)
- **Styled section headers** — Jawaban, Bukti, CTX Mapping, Sources, PDF Viewer dengan warna berbeda
- **Processing box** — tampilkan user query vs retrieval query (contextualized/direct badge)
- **CTX Mapping table** — kolom `sim` color-coded (🟢🟡🔴) dengan pandas Styler
- **Source score pills** — sim + dist + CTX label di awal tiap source expander
- **Sidebar: Last Turn Status** — memory used, contextualized, viewing turn
- **Mode indicator** — 🎯 demo / 🧪 eval di info bar utama

---

### [V2.0 — Hybrid Retrieval: BM25+Sastrawi + Dense + RRF Fusion]

#### Hybrid Retrieval Pipeline
- **Dual-path retrieval**: dua jalur berjalan paralel untuk setiap query
  - **Jalur Dense** (ChromaDB): semantic similarity via embeddings — kuat untuk pertanyaan konseptual
  - **Jalur Sparse** (BM25+Sastrawi): keyword/exact-match — kuat untuk istilah teknis spesifik
- **Reciprocal Rank Fusion (RRF)**: menggabungkan ranking dense + sparse menjadi satu ranked list
  - Formula: `score_rrf(d) = Σ 1 / (k + rank_i(d))` untuk setiap retriever i
  - Parameter `k=60` (dikonfigurasi di `v2.yaml`)
  - Node yang muncul di **kedua retriever** mendapat boost skor otomatis
- **BM25 corpus**: dibangun dari ChromaDB `collection.get()` sehingga selalu sinkron dengan vector index
- **BM25 index caching**: dibangun sekali di query pertama, di-cache di memori — query berikutnya instan

#### Sastrawi Stemmer Integration
- **Stemming Bahasa Indonesia** via `PySastrawi` — aktif **hanya** di jalur BM25 sparse
- Jalur dense **tidak di-stemming** tanpa pengecualian (menjaga konteks semantik embedding tetap utuh)
- Corpus di-tokenize + di-stem saat build BM25 index; query di-tokenize + di-stem saat retrieval
- Fallback graceful: jika PySastrawi tidak terinstall, sistem tetap berjalan tanpa stemming (dengan warning)
- Konfigurasi: `bm25.use_sastrawi: true` di `v2.yaml`, `false` di `base.yaml`

#### RetrievedNode Schema (V2.0)
Setiap node hasil hybrid membawa informasi lengkap dari kedua retriever:
- `score` — RRF score final (dipakai untuk sorting & display utama)
- `score_dense` — Chroma distance asli (ada jika node muncul di jalur dense)
- `score_sparse` — BM25 raw score (ada jika node muncul di jalur sparse)
- `rank_dense` — posisi ranking di jalur dense (`None` jika tidak muncul)
- `rank_sparse` — posisi ranking di jalur sparse (`None` jika tidak muncul)

Node yang hanya muncul di satu retriever tetap valid — field retriever lain di-set `None`.

#### JSONL Schema — Backward Compatible
Field `retrieval_mode` di `retrieval.jsonl` membedakan jalur yang dipakai:
- `"dense"` — V1.0/V1.5 behaviour, `distance` + `similarity` terisi, `score_rrf` = `null`
- `"hybrid"` — V2.0, `score_rrf` + `score_dense` + `score_sparse` + `rank_dense` + `rank_sparse` terisi, `distance` + `similarity` = `null`
- `"metadata_router"` — dijawab dari doc_catalog, tidak ada retrieval node

Field `strategy` di `answers.jsonl` mencerminkan pipeline yang dipakai:
- `"hybrid"` — hybrid retrieval tanpa rewrite
- `"hybrid+rewrite"` — hybrid retrieval + query contextualization
- `"dense"`, `"dense+rewrite"`, `"dense+diverse"` — V1.x behaviour

#### UI/UX Enhancement (V2.0) — Hybrid Score Display
- **Info banner dinamis** — menampilkan mode retrieval aktif (`🔀 Retrieval: HYBRID` / `🔵 Retrieval: DENSE`)
- **CTX Mapping table (hybrid schema)** — kolom: `score_rrf` ↑ | `score_dense` ↓ | `score_sparse` ↑ | `rank_dense` | `rank_sparse` | `chunk_id`
  - Kolom rank tampil sebagai integer bersih (`1`, `2`, `3`) menggunakan nullable `Int64` dtype
  - Warna `score_rrf`: 🟢 ≥ 0.025 (Tinggi) | 🟡 ≥ 0.015 (Sedang) | 🔴 < 0.015 (Rendah)
  - Legend informatif di bawah tabel
- **Hybrid source score pills** — di dalam tiap Sources expander:
  - `CTX n` label (biru)
  - `🔀 RRF score` pill — warna dari threshold RRF
  - `🔵 Dense sim=... (label) dist=...` pill — warna dari semantic similarity threshold
  - `🟡 BM25=... (label)` pill — label (Tinggi/Sedang/Rendah) dari `rank_sparse`
  - `rank_dense=#n` dan `rank_sparse=#n` pills (muted abu-abu)
- **Badge 🟢🟡🔴 di title expander** — berbasis RRF score (fair untuk node dense-only, sparse-only, maupun keduanya)
- **Dense mode**: tampilan kembali ke schema V1.5 (`sim`, `dist`, pills lama) saat config `v1_5.yaml` atau `v1.yaml` dipakai

#### Arsitektur Modul (V2.0)
```
src/
├── app_streamlit.py          ← UI utama (mode-aware hybrid/dense pipeline)
├── app_ui_render.py          ← Render & display helpers (hybrid score pills, CTX table)
├── rag/
│   ├── method_detection.py   ← Single source of truth: deteksi metode/query
│   ├── generate.py           ← Thin re-export wrapper 3 fungsi publik
│   ├── generate_utils.py     ← Semua implementasi generate + guardrails (~2800 baris)
│   ├── ingest.py             ← PDF ingestion → ChromaDB + doc_catalog
│   ├── retrieve_dense.py     ← Dense retrieval + caching + diversity (score_dense terisi)
│   ├── retrieve_sparse.py    ← BM25+Sastrawi sparse retrieval + index caching (AKTIF V2.0)
│   ├── fusion_rrf.py         ← RRF fusion: List[RetrievedNode] → List[RetrievedNode] (AKTIF V2.0)
│   └── metadata_router.py    ← Intent router (page_count aktif V1.5)
└── core/
    ├── config.py, run_manager.py, schemas.py
    ├── ui_utils.py            ← Generic Streamlit widgets
    └── cleanup_runs.py
configs/  → base.yaml, v1.yaml, v1_5.yaml, v2.yaml, v3.yaml
```

**Dependency Graph Modul Retrieval (V2.0):**
```
retrieve_dense.py  ←── app_streamlit.py (jalur dense + coverage expansion)
retrieve_sparse.py ←── app_streamlit.py (jalur sparse)
fusion_rrf.py      ←── app_streamlit.py (gabungkan dense + sparse)
                        ↑
                   schemas.RetrievedNode (shared output type)
```

---

## Stack Teknis
| Komponen | Detail |
|----------|--------|
| Language | Python 3.10+ (runtime: 3.11) |
| Framework | LangChain (orchestrator) + Streamlit (UI) |
| Vector DB | ChromaDB (lokal/persisten) |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Sparse (V2.0) | `rank-bm25>=0.2.2` + `PySastrawi>=1.2.0` (hanya jalur BM25 sparse) |
| Reranker (V3.0c) | `BAAI/bge-reranker-base` |
| LLM Primary | Groq API — `llama-3.1-8b-instant` |
| LLM Secondary | Ollama — Llama-3 Quantized 4-bit (offline/privasi) |
| Evaluasi | RAGAS (Faithfulness, Answer Relevance, Context Precision) |
| Hardware | AMD Ryzen 7 5800H, RAM 32GB, RTX 3050 Ti (4GB VRAM) |
 
> **Catatan VRAM:** Cross-Encoder dan LLM tidak boleh dimuat bersamaan. Strategi Sequential Loading wajib dipakai di V3.0c ke atas: load Cross-Encoder → rerank → unload → load LLM → generate.
 
> **Catatan Stemming:** Sastrawi Stemmer HANYA untuk jalur BM25 sparse. Jalur dense TIDAK di-stemming agar konteks semantik embedding tetap utuh. Aturan ini berlaku tanpa pengecualian di semua versi.

---

## Struktur Repo
- `src/`
  - `app_streamlit.py` : UI Streamlit utama (mode-aware: hybrid/dense)
  - `app_ui_render.py` : Render & display helpers (V1.5+, hybrid score pills V2.0)
  - `rag/`
    - `ingest.py` : ingestion/indexing PDF → Chroma + doc_catalog
    - `retrieve_dense.py` : dense retrieval (cached + diversity, `score_dense` terisi)
    - `retrieve_sparse.py` : BM25+Sastrawi sparse retrieval + index caching (V2.0)
    - `fusion_rrf.py` : RRF fusion dense+sparse → `List[RetrievedNode]` (V2.0)
    - `generate.py` : thin re-export wrapper public API
    - `generate_utils.py` : semua implementasi generate + guardrails
    - `method_detection.py` : deteksi metode & query type (single source of truth)
    - `metadata_router.py` : metadata intent routing (page_count)
  - `core/`
    - `config.py` : load config YAML (deep merge base + override)
    - `run_manager.py` : run_id + manifest + logging JSONL
    - `schemas.py` : dataclass schema (RetrievedNode + field hybrid V2.0, dll)
    - `ui_utils.py` : download buttons, PDF viewer+highlight, clipboard, dll
    - `cleanup_runs.py` : cleanup runs utility (manual)
  - `evaluation/` : script pondasi evaluasi (RAGAS, dsb.)
- `configs/` : config per versi (base, v1, v1_5, v2, v3)
- `data_raw/` : PDF mentah *(gitignored)*
- `data_index/` : storage Chroma *(gitignored)*
- `runs/` : output log per-run *(gitignored)*
- `notebooks/` : eksperimen (regex/chunking)

---

## Setup

### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

> **Catatan V2.0:** Pastikan `PySastrawi>=1.2.0` dan `rank-bm25>=0.2.2` terinstall. Verifikasi cepat:
> ```bash
> python -c "from Sastrawi.Stemmer.StemmerFactory import StemmerFactory; print('Sastrawi OK')"
> python -c "from rank_bm25 import BM25Okapi; print('rank_bm25 OK')"
> ```

### 2) Environment (.env)
Gunakan `.env` lokal (jangan commit API key). Contoh variabel:
- `LLM_PROVIDER=groq|openai|ollama`
- `GROQ_API_KEY=...` + `GROQ_MODEL=...`
- `OPENAI_API_KEY=...` + `OPENAI_MODEL=...`
- `OLLAMA_BASE_URL=http://localhost:11434` + `OLLAMA_MODEL=...`

> Catatan: file `env` tidak di-commit. Gunakan `.env.example` sebagai template.

---

## Cara Menjalankan

### 1.) Ingest / Index (offline)
Pastikan PDF diletakkan di `data_raw/`, lalu jalankan:
```bash
# Fresh ingest (pertama kali atau re-index penuh)
python -m src.rag.ingest --config configs/v2.yaml --reset
 
# Ingest tambahan tanpa reset (hanya PDF baru, catalog di-merge otomatis)
python -m src.rag.ingest --config configs/v2.yaml
```

> **Catatan V2.0:** BM25 index dibangun dari ChromaDB secara otomatis saat query pertama — tidak perlu langkah ingest tambahan. Jika corpus berubah (re-ingest), BM25 cache lama otomatis tidak valid; restart app atau panggil `clear_sparse_cache()` secara manual.

### 2.) Jalankan UI (Streamlit)
Cara paling praktis (Windows/PowerShell):
```bash
.\run_app.ps1
```
atau Alternatif manual:
```bash
$env:PYTHONPATH="."
streamlit run src/app_streamlit.py
```

> **Catatan V2.0:** Query pertama di mode hybrid membutuhkan waktu lebih lama (~30–60 detik) karena BM25 index dibangun dari corpus saat itu. Query berikutnya jauh lebih cepat (index ter-cache di memori).

### 3.) Mode Eval vs Demo
Di sidebar UI, tersedia toggle **Mode**:
- **demo** — memory dan contextualization aktif, untuk pengalaman chat natural.
- **eval** — memory dan contextualization dimatikan paksa, untuk evaluasi RAGAS yang konsisten dan reproducible.

### 4.) Memilih Config per Skenario Ablation
| Config | Mode | Keterangan |
|--------|------|------------|
| `configs/v1.yaml` | dense | Skenario A — Naive Dense RAG |
| `configs/v1_5.yaml` | dense | V1.5 dengan memory (bukan skenario ablation) |
| `configs/v2.yaml` | **hybrid** | **Skenario B — Hybrid RAG** |
| `configs/v3.yaml` | *(planned)* | Skenario C — Proposed Method |

---

## Konfigurasi (YAML)
Setiap versi memiliki config sendiri di `configs/`. Config di-load dengan deep merge: `base.yaml` sebagai default, di-override oleh config versi yang dipilih.

Key penting di `base.yaml` (default untuk semua versi):
```yaml
retrieval:
  top_k: 8
  mode: "dense"       # default dense; di-override oleh v2.yaml ke "hybrid"
  rrf_k: 60           # parameter k untuk RRF fusion
 
bm25:
  k1: 1.5
  b: 0.75
  use_sastrawi: false # default off; di-override oleh v2.yaml ke true
```

Key penting di `v1_5.yaml`:
```yaml
memory:
  enabled: true
  window: 4               # jumlah turn terakhir yang dipakai

generation:
  deterministic_method_answer: true   # jawaban METODE stabil
  deterministic_steps_answer: false   # jawaban TAHAP: LLM + deterministik
  humanize_steps_answer: true         # format ulang jika Jawaban == Bukti
  per_method_bullets: 2
  max_total_bullets: 6

coverage:
  enabled: true
  per_method_k: 3
  min_methods: 2

metadata_routing:
  enabled: false          # aktifkan jika ingin routing page_count
```

Key penting di `v2.yaml`:
```yaml
retrieval:
  mode: "hybrid"      # aktifkan jalur hybrid (base: "dense")
 
bm25:
  use_sastrawi: true  # aktifkan Sastrawi stemming untuk jalur BM25 sparse (base: false)
 
memory:
  enabled: true       # aktifkan memory untuk V2.0 demo (base: false)
  window: 4
```

---

## Cleanup `runs/` (manual, aman)
Utility: `src/core/cleanup_runs.py`
- Dry-run (default, tidak menghapus apapun):
    ```bash
    python -m src.core.cleanup_runs --runs-dir runs --keep-last 50 --max-age-days 30 --max-total-mb 5000
    ```
- Apply + archive (run di-zip dulu, lalu folder run dihapus):
    ```bash
    python -m src.core.cleanup_runs --runs-dir runs --keep-last 50 --max-age-days 30 --max-total-mb 5000 --archive-dir runs_archive --apply
    ```

> Catatan: Untuk melindungi run penting, buat file `.keep` (atau `KEEP`) di folder `runs/<run_id>/`.

---

## Catatan Pengembangan

### Git Strategy
- Kerja harian dilakukan di branch `dev`
- Push ke `main` + beri tag hanya saat satu versi dinyatakan final
- Tag convention: `v{versi}-{codename}` (contoh: `v1.0-naive`, `v1.5-memory`, `v2.0-hybrid`)

### JSONL Schema Consistency
Field di `retrieval.jsonl` dan `answers.jsonl` tidak boleh berubah antar run dalam satu versi — ablation study bergantung pada konsistensi ini. Jika ada field baru, pastikan diisi di **semua jalur** (normal retrieval dan metadata-routed).

Field hybrid V2.0 yang ditambahkan ke `retrieved_nodes` di `retrieval.jsonl`:
- `score_rrf` — RRF score final (`null` untuk jalur dense)
- `score_dense` — Chroma distance (terisi untuk semua jalur; `null` jika node hanya dari sparse)
- `score_sparse` — BM25 raw score (`null` jika node hanya dari dense)
- `rank_dense` / `rank_sparse` — posisi ranking per retriever (`null` jika tidak muncul)
- `distance` / `similarity` — `null` untuk jalur hybrid, terisi untuk jalur dense

### Prinsip Modifikasi (V1.5+)
- Logika deteksi metode/query → selalu edit di `method_detection.py`
- Logika render/display UI → selalu edit di `app_ui_render.py`
- CSS history expander → selalu pakai sentinel `#hist-scope` + `:has()` scoping agar style tidak bocor
- Field baru di `turn_data` session_state → pastikan diisi di **semua jalur** (metadata_routed, dense, hybrid)

### Prinsip Modifikasi Tambahan (V2.0)
- Modifikasi jalur sparse → edit di `retrieve_sparse.py`; jangan stem di luar fungsi `_tokenize()`
- Modifikasi RRF fusion → edit di `fusion_rrf.py`; kontrak: input tidak dimutasi, output `List[RetrievedNode]` baru
- Penambahan retriever baru → daftarkan di blok `if retrieval_mode == "hybrid":` di `app_streamlit.py`
- Sastrawi stemming **HANYA** di `retrieve_sparse.py` — tidak boleh dipanggil dari modul lain
 
### Constraint Hardware (VRAM 4GB)
- V2.0: BM25 berjalan di CPU, tidak ada tambahan beban VRAM
- V3.0c ke atas: Sequential Loading wajib — Cross-Encoder dan LLM tidak boleh dimuat bersamaan
  - Urutan: load Cross-Encoder → rerank → **unload** → load LLM → generate

---