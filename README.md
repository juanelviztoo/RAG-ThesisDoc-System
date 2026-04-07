# RAG ThesisDoc System
**Retrieval-Augmented Generation (RAG) untuk Dokumen Skripsi/TA**  

> **Status Terkini:** V3.0c (**DONE ✅ / RELEASED ✅ / tag `v3.0c-rerank`**) di branch `main` dan sudah tersinkron kembali ke `dev`. V3.0c sudah selesai difinalisasi secara code, dokumentasi, validasi, dan git strategy. Fokus pengembangan berikutnya adalah **V3.0d** (Post-Hoc Verification).

> **Status Terkini:** V3.0d (**DONE ✅ / RELEASED ✅ / FINALIZED ✅ / tag `v3.0d-verification`**) di branch `main` dan sudah tersinkron kembali ke `dev`. Final git strategy V3.0d (commit ke `dev` → merge ke `main` → tag release final) sudah diselesaikan. V3.0d menutup payung pengembangan **Post-Hoc Verification + Evaluation Pipeline + Intent-Aware Answer Augmentation + Metadata/Logging Hardening**. Fokus pengembangan berikutnya bergeser ke **V3.1** (dataset expansion + evaluasi correctness/helpfulness jika mini gold set tersedia).

---

## Roadmap Versi (Bertahap)
- **V1.0 (DONE ✅)** — `v1.0-naive`: Naive Dense RAG baseline + UI + Audit Logging + PDF Viewer/Highlight + Utility Cleanup Runs
- **V1.5 (DONE ✅)** — `v1.5-memory`: Multi-turn Chat (Session Memory) + Query Contextualization + Guardrails + UI/UX Enhancement
- **V2.0 (DONE ✅)** — `v2.0-hybrid`: Hybrid Retrieval (BM25+Sastrawi Sparse + Dense Vector + RRF Fusion) + UI Hybrid Score Display
- **V3.0a (DONE ✅)** — `v3.0a-structured`: Structure-Aware PDF Parsing + Dual Stream (Narasi vs Sitasi) + Dual ChromaDB Collection + Citation Routing
- **V3.0b (DONE ✅)** — `v3.0b-selfquery`: Metadata Self-Querying Filtering (Tahun, Prodi, Penulis) + Stable JSONL Self-Query Logging + Pre-check Zero-Result Fallback + Metadata Exact-Match Answer Guard + Metadata Router Natural-Language Alias Resolution
- **V3.0c (DONE ✅)** — `v3.0c-rerank`: Cross-Encoder Reranking (`BAAI/bge-reranker-base`) + Sequential Loading + Dynamic Top-N + Anchor Reservation + Before-vs-After Audit UI + Generation Hardening + Multi-turn Stabilization + Backward-Compatibility Polish
- **V3.0d (DONE ✅)** — `v3.0d-verification`: Post-Hoc Verification (Grounding Gate v1, explainable & rule-based) + Verification UI/Logging + Evaluation Dataset Builder + RAGAS Pipeline + Evaluator-Specific LLM Override + Intent Planner + Ownership-Aware Doc Aggregation + Method Explanation Route + Single-Target Steps Handler + Comparison Formatter + Docmeta Sidecar + Safe Title Fallback + Logging Hardening + Final Audit/UI Sync Polish
- **V3.1 (NEXT 🔄)** — Dataset Expansion + mini gold set / reference answers + evaluasi Answer Correctness / rubric helpfulness (opsional, jika sumber daya tersedia)

**Pemetaan Ablation Study:**
| Skenario | Versi | Tag | Keterangan |
|----------|-------|-----|------------|
| A | V1.0 | `v1.0-naive` | Naive Dense RAG |
| B | V2.0 | `v2.0-hybrid` | Hybrid RAG (BM25+Sastrawi+RRF) |
| C | V3.0c | `v3.0c-verification` | Proposed Method final aktif (Retrieval + Self-Query + Rerank + Verification + Intent-Aware Answer Augmentation + Audit/Eval) |

---

## Summary
Repo ini berisi prototipe sistem **Question Answering** pada kumpulan PDF skripsi/TA.

Versi **V3.0d** membangun di atas V3.0c dengan enam fondasi utama:
1. **Structure-aware dual-stream retrieval** dari V3.0a (narasi vs sitasi, dual collection, citation routing)
2. **Metadata Self-Querying Filtering** dari V3.0b pada primary retrieval narasi
3. **Cross-Encoder Reranking** dari V3.0c di atas candidate pool retrieval untuk mempertajam final contexts sebelum generation
4. **Post-Hoc Verification** di answer layer (Grounding Gate v1, explainable & rule-based)
5. **Evaluation pipeline** (dataset builder + RAGAS + evaluator-specific LLM override)
6. **Intent-aware answer augmentation policy** untuk method explanation / count-list / comparison / single-target steps + metadata/audit hardening

Fitur inti yang telah tersedia hingga **V3.0d**:
- **Structure-Aware PDF Parsing**: deteksi bab (BAB I–V, Daftar Pustaka, Lampiran) via regex + filename routing — setiap chunk "tahu posisinya" dalam hierarki akademik
- **Dual Stream**: dokumen dipecah menjadi dua aliran — narasi (BAB I–V) dan sitasi (Daftar Pustaka) — masing-masing diindeks ke ChromaDB collection terpisah
- **Citation Routing**: sistem mendeteksi otomatis apakah query menanyakan konten (→ narasi) atau referensi/pustaka (→ sitasi), lalu merge hasilnya
- **Hybrid Retrieval**: BM25+Sastrawi (sparse/keyword) + Dense vector (Chroma) digabungkan via Reciprocal Rank Fusion (RRF)
- **Metadata Self-Querying Filtering (V3.0b)**: query natural language dipecah menjadi:
  - `semantic_query` untuk retrieval
  - filter metadata terstruktur (`tahun`, `prodi`, `penulis`) untuk primary dense retrieval narasi
- **Graceful Zero-Result Fallback (V3.0b)**: jika metadata filter valid tetapi pre-check menunjukkan 0 hasil, sistem otomatis fallback ke retrieval penuh tanpa crash
- **Stable Self-Query Logging (V3.0b)**: field `self_query` kini diisi secara konsisten di `retrieval.jsonl` dan `answers.jsonl` pada semua jalur (normal retrieval, metadata-routed, disabled path, fallback path)
- **Answer Guard untuk Exact Metadata Match (V3.0b)**: jika metadata eksplisit diminta user tetapi exact match tidak ditemukan, sistem memaksa jawaban `not_found` agar tidak menghasilkan false positive
- **Metadata Routing (page count intent)**: pertanyaan seperti *"berapa halaman dokumen X?"* dijawab langsung dari `doc_catalog.json`; resolver mendukung alias natural-language seperti `"E-Ticketing"`
- **Cross-Encoder Reranking (V3.0c)**:
  - model: `BAAI/bge-reranker-base`
  - candidate pool default: Top-10
  - final contexts: Top-3 untuk query biasa, adaptif ke Top-5 untuk query kompleks
  - **Sequential Loading wajib**: load reranker → rerank → unload → load LLM → generate
- **Dynamic Top-N (V3.0c)**: query comparison / multi-target / steps dapat memakai `top_n_rerank_complex = 5`
- **Anchor Reservation (V3.0c)**: final set boleh dimodifikasi pasca-rerank secara sadar-konteks agar metode target tidak hilang dari final contexts
- **Before vs After Audit UI (V3.0c)**: panel audit memperlihatkan candidate pool pre-rerank, final contexts post-rerank, source snapshot, rerank rank/score, dan anchor reservation
- **Generation Hardening (V3.0c)**: comparison/steps answer dibuat lebih grounded, lebih jujur pada coverage asimetris, dan lebih faithful terhadap final contexts
- **Multi-turn Stabilization (V3.0c)**: follow-up singkat seperti *"Bisa lebih ringkas lagi langkah utamanya?"* dapat diperlakukan sebagai kelanjutan konteks, bukan topic shift palsu
- **Regression / Backward-Compatibility Polish (V3.0c)**: config legacy (`v1_5.yaml`, `v2.yaml`) tetap sehat tanpa fingerprint reranker palsu di UI/log
- **Post-Hoc Verification (V3.0d)**: verification rule-based, explainable, dan hidup di answer layer; status utama: `verified`, `partially_verified`, `low_verification`, `verification_n_a`
- **Verification Logging + UI (V3.0d)**: object `verification` masuk ke `answers.jsonl`, disimpan ke `turn_data`, tampil stabil di latest turn maupun history replay
- **Evaluation Dataset Builder (V3.0d)**: `build_eval_dataset.py` merekonstruksi final contexts dari run log + index/Chroma, menghasilkan `eval_dataset.json`, `ragas_rows.jsonl`, dan `ragas_rows.pretty.json`
- **RAGAS Evaluation Pipeline (V3.0d)**: `evaluate_ragas.py` menjalankan metric `faithfulness`, `answer_relevancy`, dan `context_precision`, lengkap dengan transparency fields per-item/per-summary
- **Evaluator-Specific LLM Override (V3.0d)**: evaluator dapat memakai provider/model berbeda dari runtime generator utama (Groq/OpenAI/Ollama-ready), sehingga tuning evaluator tidak merusak perilaku chat runtime
- **Intent Planner + Route Names (V3.0d)**: planner membedakan intent seperti `method_explanation`, `doc_count_list`, `method_comparison`, `single_target_steps`, dll. agar jawaban tidak jatuh ke template identifikasi yang terlalu dangkal
- **Ownership-Aware Doc Aggregation (V3.0d)**: query list/count dan doc-locked explanation kini memiliki pemilihan dokumen yang lebih disiplin dan lebih jujur
- **Method Explanation Route + Explanation-Biased Expansion (V3.0d)**: query penjelasan metode dipisahkan dari identification dan diperluas ke konteks alasan, karakteristik, dan tahapan bila tersedia
- **Single-Target Steps Handler (V3.0d)**: query tahapan tunggal menjadi lebih aman/jujur, mengurangi halusinasi langkah yang sebenarnya tidak eksplisit
- **Comparison Formatter (V3.0d)**: jawaban perbandingan menjadi lebih contrastive dan lebih audit-friendly, dengan penggunaan context unik per-metode bila tersedia
- **Docmeta Sidecar + Safe Title Fallback (V3.0d)**: `docmeta_sidecar.json` ditambahkan agar title/author/year display untuk query count/list tidak lagi tercemar metadata noisy
- **Logging Hardening (V3.0d)**: `intent_plan`, `doc_aggregation_meta`, dan `answer_policy_audit_summary` ditambahkan agar policy internal lebih mudah diaudit
- **Final Audit/UI Sync Polish (V3.0d)**: metadata rerank disinkronkan kembali ke final contexts/UI (CTX Mapping / After Rerank / Sources) agar audit panel jujur terhadap final nodes
- Dense retrieval (Chroma + embeddings) dengan diversity retrieval
- Multi-turn memory (sliding window, topic shift detection)
- Query contextualization (rewrite sebelum retrieval)
- Generator (LLM) opsional (Groq/OpenAI/Ollama) dengan output guardrails lengkap
- UI Streamlit demo-ready (turn viewer, history expander, styled sections, hybrid score display, stream pill, rerank audit panel)
- Coverage retrieval per-metode (untuk pertanyaan multi-target)
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
- Pada V3.0b, metadata router ini diaktifkan kembali pada `v3.yaml` dan diperluas agar mendukung alias natural-language (mis. `"dokumen E-Ticketing"`)
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

#### JSONL Schema — Backward Compatible (V2.0)
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

---

### [V3.0a — Structure-Aware Parsing + Dual Stream + Citation Routing]

#### Structure-Aware PDF Parsing
Sistem kini **memahami struktur hierarki** dokumen akademik Indonesia, bukan sekadar memecah teks secara naif:
- **Hybrid routing strategy** — dua strategi yang berjalan otomatis:
  - **Filename routing**: PDF yang sudah terpotong per-BAB langsung di-route berdasarkan nama file (efisien, deterministik — cocok untuk dataset asli)
  - **Regex active parsing**: PDF utuh di-scan bab-headernya untuk memetakan tiap halaman ke bab yang sesuai (cocok untuk dataset sementara / PDF belum terpotong)
- **Bab yang dikenali**: `BAB_I`, `BAB_II`, `BAB_III`, `BAB_IV`, `BAB_V`, `DAFTAR_PUSTAKA`, `LAMPIRAN`, `FRONT_MATTER`, `UNKNOWN`
- Pola regex yang dikenali: `"BAB I"`, `"BAB 1"`, `"CHAPTER I"`, `"DAFTAR PUSTAKA"`, `"REFERENCES"`, `"LAMPIRAN"`, dll. (case-insensitive)
- **Running header detection**: header bab yang muncul berulang di setiap halaman tidak dikira sebagai pergantian bab baru
- **Metadata extraction otomatis**: Judul, Penulis, Tahun, Prodi di-ekstrak dari halaman awal dan diwariskan ke semua chunk

#### Dual Stream — Narasi vs Sitasi
Dokumen akademik mengandung dua jenis konten yang sifatnya berbeda, kini diindeks terpisah:

| Stream | Konten | ChromaDB Collection | Chunking |
|--------|--------|---------------------|---------|
| **Narasi** | BAB I–V (konten inti skripsi) | `skripsi_structured_narasi` | chunk_size=1200, overlap=200 |
| **Sitasi** | Daftar Pustaka / Referensi | `skripsi_structured_sitasi` | chunk_size=500, overlap=50 |

- **Narasi** dipakai untuk menjawab pertanyaan konseptual, metodologi, hasil, dan pembahasan
- **Sitasi** dipakai untuk *citation discovery* — "skripsi apa yang mengutip buku/jurnal X?"
- Pemisahan ini mencegah **context poisoning**: jawaban pertanyaan konseptual tidak tercampur teks bibliografi yang tidak relevan

#### Citation Routing
- Sistem mendeteksi otomatis apakah query membutuhkan sitasi berdasarkan keyword: `"referensi"`, `"daftar pustaka"`, `"sitasi"`, `"kutip"`, `"mengutip"`, `"buku"`, `"jurnal"`, `"penulis"`, `"artikel"`, dll.
- **Jalur hybrid** (v3.yaml): dense narasi + BM25 narasi + RRF → **jika citation query**: tambahkan dense sitasi + BM25 sitasi + RRF, lalu merge & sort
- **Jalur dense** (v1.yaml, v2.yaml): narasi saja → **jika citation query**: tambahkan dense sitasi, lalu merge & sort
- `_merge_and_sort_nodes()`: gabungkan dua list node, dedupe by chunk_id, sort by score descending, ambil top_k

#### RetrievedNode Schema (V3.0a)
Dua field baru ditambahkan ke `RetrievedNode` (backward-compatible, nullable):
- `stream` — asal stream chunk: `"narasi"` | `"sitasi"` | `None`
- `bab_label` — posisi dalam struktur dokumen: `"BAB_I"` | `"BAB_II"` | ... | `"DAFTAR_PUSTAKA"` | `None`

Field ini diteruskan dari metadata ChromaDB melalui semua lapisan: `retrieve_dense` → `fusion_rrf` → `app_streamlit` → JSONL log → UI.

#### JSONL Schema — V3.0a (Backward Compatible)
Field baru di `retrieved_nodes` di `retrieval.jsonl`:
- `stream` — `"narasi"` | `"sitasi"` | `null`
- `bab_label` — `"BAB_I"` | `"BAB_II"` | `"BAB_III"` | `"BAB_IV"` | `"BAB_V"` | `"DAFTAR_PUSTAKA"` | `"LAMPIRAN"` | `"FRONT_MATTER"` | `"UNKNOWN"` | `null`

Semua field V2.0 (`score_rrf`, `score_dense`, `score_sparse`, `rank_dense`, `rank_sparse`, `distance`, `similarity`) tetap ada tanpa perubahan.

#### UI/UX Enhancement (V3.0a) — Stream Awareness
- **Stream pill (violet)** — di dalam tiap Sources expander, tampil pill kecil sebelum score pills:
  - `📖 narasi` — untuk chunk dari collection narasi
  - `🧷 sitasi` — untuk chunk dari collection sitasi
  - Node lama (tanpa field stream) tidak menampilkan pill (backward-compatible)
- **CTX Mapping table** — kolom `stream` dan `bab_label` ditambahkan di awal (setelah `CTX`, sebelum `doc_id`) untuk kedua schema hybrid dan dense
- **Legend CTX diperluas** — keterangan `stream` (narasi = konten inti · sitasi = Daftar Pustaka) dan `bab_label` (posisi bab asal chunk) ditampilkan di bawah tabel

#### Fix & Stabilisasi (V3.0a)
Selain fitur baru, V3.0a memperbaiki sejumlah bug yang ditemukan selama pengerjaan dan code review:

| # | Bug | File | Severity | Status |
|---|-----|------|----------|--------|
| 1 | ChromaDB compaction error saat `--reset` (`shutil.rmtree` fix) | `ingest.py` | CRITICAL | ✅ Fixed |
| 2 | `stream`/`bab_label` tidak terkopi ke fused node di RRF | `fusion_rrf.py` | HIGH | ✅ Fixed |
| 3 | `is_steps_question()` false positive untuk "alur metodologi penelitian" | `method_detection.py` | MEDIUM | ✅ Fixed |
| 4 | `methods_detected` carry-over ke history dari turn off-topic | `app_streamlit.py` | HIGH | ✅ Fixed |
| 5 | `doc_catalog` pre-init overwrite nilai kosong saat PDF gagal parse | `ingest.py` | MEDIUM | ✅ Fixed |
| 6 | `steps_q_now` di blok doc-focus lock masih memakai `retrieval_query` | `app_streamlit.py` | MEDIUM | ✅ Fixed |

Plus 3 fix perilaku pipeline:
- **Fix 1**: reset `method_doc_map` saat `topic_shift` terdeteksi — mencegah doc-focus lock bocor ke topik baru
- **Fix 2**: `steps_q` hanya dari query asli user (bukan `retrieval_query` hasil rewrite LLM) — di blok Coverage Expansion
- **Fix 3**: buang `methods_detected` dari fallback `turn_methods` — hanya pakai `target_methods_for_gen` + scan `answer_raw`

### [V3.0b — Metadata Self-Querying Filtering + Stabilisasi Pipeline]

#### Metadata Self-Querying Filtering
V3.0b menambahkan lapisan **self-querying** di atas pipeline retrieval V3.0a. Query user tidak lagi hanya dipakai sebagai string retrieval mentah, tetapi diproses menjadi dua keluaran:

1. **`semantic_query`** — query semantik yang dibersihkan dari metadata eksplisit dan benar-benar dikirim ke retriever
2. **`filters`** — filter metadata terstruktur untuk primary dense retrieval narasi

Metadata yang saat ini didukung:
- `tahun`
- `prodi`
- `penulis`

Implementasi V3.0b:
- Modul baru: `src/rag/self_query.py`
- Ekstraksi metadata: LLM → JSON → validasi → Chroma `where_filter`
- Batas maksimum filter simultan: `max_filter_fields = 2`
- Prioritas field saat truncation: `tahun > prodi > penulis`

#### Where Filter Integration
Filter metadata diterapkan pada:
- **primary dense retrieval narasi**

Filter **tidak** diterapkan secara end-to-end pada seluruh jalur hybrid. Sparse leg pada mode hybrid masih mengikuti arsitektur V2.0, sehingga hal ini dicatat sebagai **known limitation implementasi V3.0b**, bukan sebagai klaim bahwa seluruh hybrid retrieval sudah hard-filtered metadata.

#### Graceful Pre-check Fallback
Sebelum filter metadata diterapkan, sistem melakukan pre-check:
- jika hasil > 0 → filter dipakai
- jika hasil = 0 → retrieval otomatis fallback ke retrieval penuh

Dengan demikian, V3.0b menghindari kegagalan diam-diam berupa hasil retrieval kosong akibat filter metadata yang terlalu ketat.

#### Stable JSONL Logging
V3.0b menambahkan field `self_query` ke:
- `retrieval.jsonl`
- `answers.jsonl`

Shape field ini dijaga stabil di semua jalur:
- normal retrieval
- `metadata_router`
- disabled path
- eval-disabled path
- `precheck_no_results`
- exception fallback

Field inti `self_query`:
- `enabled`
- `semantic_query`
- `filter_applied`
- `filter_fields_used`
- `filters`
- `extracted_metadata`
- `fallback_reason`

#### Exact-Match Answer Guard
V3.0b menambahkan guard pada answer layer:
- jika user meminta metadata eksplisit
- self-query berhasil mengekstrak metadata
- tetapi `precheck_no_results` terjadi

maka sistem **tidak** membiarkan generator menjawab positif berdasarkan retrieval fallback umum. Sebagai gantinya, sistem memaksa jawaban canonical:
```text
Jawaban: Tidak ditemukan pada dokumen.
Bukti: -
```
Hal ini mencegah false positive seperti:
- query meminta `tahun=2021`
- retrieval fallback menampilkan dokumen 2023
- generator lalu keliru menjawab “Ya”

#### Citation Routing Stabilization
Pada V3.0b, citation routing tetap aktif, tetapi ditambahkan stabilisasi:
- **minimum floor narasi ≥ 60%** saat citation merge aktif
- sort akhir hybrid dan dense dibedakan sesuai semantik skor:
  - hybrid: `score_rrf` → descending
  - dense: Chroma distance → ascending

#### Metadata Router Improvement
Metadata router (`page_count`) tetap digunakan seperti di V1.5, tetapi V3.0b menambahkan:
- natural-language alias resolution untuk nama dokumen
- contoh: `"Berapa halaman dokumen E-Ticketing?"` dapat dipetakan ke entry:
  - `ITERA_2023_ETicketing_ExtremeProgramming.pdf`

#### Known Limitations (V3.0b)
- Sparse leg pada hybrid retrieval masih mengikuti arsitektur V2.0; filter metadata belum diterapkan end-to-end pada seluruh hybrid path
- Citation intent + metadata sudah stabil secara integrasi, tetapi kualitas retrieval sitasi masih dapat bercampur antar-dokumen
- Filter `prodi` dan `penulis` belum seandal `tahun`, terutama ketika metadata dataset belum cukup bersih atau konsisten
- Filter `penulis` sudah dapat diekstrak, tetapi exact-match retrieval untuk metadata penulis masih dicatat sebagai ruang pengembangan lanjutan

---

### [V3.0c — Cross-Encoder Reranking + Generation Hardening + Multi-turn Stabilization]

#### Cross-Encoder Reranking
V3.0c menambahkan tahap **Cross-Encoder reranking** di antara retrieval dan generation:
- Model: `BAAI/bge-reranker-base`
- Candidate pool default: **Top-10**
- Final contexts default:
  - **Top-3** untuk query biasa
  - **Top-5** untuk query comparison / multi-target / steps
- Lifecycle model diisolasi di modul baru `src/rag/reranker.py`
- Public API utama:
  - `load_reranker(...)`
  - `unload_reranker()`
  - `is_reranker_loaded()`
  - `rerank(...)`
  - `get_reranker_info()`

#### Sequential Loading (WAJIB)
Karena keterbatasan VRAM 4GB:
- Cross-Encoder dan LLM **tidak boleh aktif bersamaan**
- Urutan pipeline final:
  1. retrieval (dense / hybrid)
  2. candidate pool
  3. `load_reranker()`
  4. rerank
  5. `unload_reranker()`
  6. load/gunakan LLM
  7. generation

#### Dynamic Top-N
Awalnya V3.0c memakai `Top-3` statis. Setelah validasi T02, diperkenalkan heuristik:
- `top_n_rerank = 3` → base
- `top_n_rerank_complex = 5` → efektif untuk query:
  - comparison
  - steps/tahapan
  - multi-target
  - query kompleks lain yang relevan

Tujuan:
- memberi ruang final contexts yang lebih lebar
- mengurangi risiko hilangnya satu metode target
- menjaga query biasa tetap ringkas

#### Anchor Reservation
Pada query comparison, hasil rerank murni bisa tetap "mengalahkan" salah satu metode target walaupun candidate pool sebenarnya sudah mengandung chunk yang benar.

Karena itu V3.0c menambahkan **anchor reservation**:
- final set boleh dimodifikasi **secara eksplisit dan sadar konteks**
- sistem menjaga agar minimal satu anchor terbaik untuk tiap metode target tetap hadir
- perilaku ini terdokumentasi di UI dan log, bukan sort ulang tersembunyi

#### Generation Hardening
Selama validasi, bottleneck berpindah dari retrieval ke generation. Karena itu V3.0c juga mempatch `generate_utils.py` agar:
- jawaban comparison / steps / multi-target lebih grounded
- generator lebih jujur ketika coverage antar-metode asimetris
- fallback generik berkurang
- format Bukti lebih faithful terhadap final contexts
- separation retrieval audit vs answer synthesis lebih tegas

#### Multi-turn Stabilization (T08/T09)
V3.0c memperbaiki kasus follow-up singkat seperti:
- *"Bisa lebih ringkas lagi langkah utamanya?"*

Patch yang ditambahkan:
- deteksi `compression_followup`
- contextualization heuristik khusus `steps_summary`
- pengecualian topic-shift untuk compression follow-up
- carry-forward doc focus / method context
- replay historis tetap sinkron untuk turn-turn yang sudah direrank

#### UI/UX Enhancement (V3.0c)
Lapisan audit UI yang ditambahkan:
- **Rerank summary box**
- **Before vs After Rerank panel**
- **Source snapshot** yang lebih rapi dan mudah dipindai
- **CTX Mapping** dengan field:
  - `rerank_score`
  - `rerank_rank`
  - `anchor_reserve`
- **Score pills** baru:
  - rerank pill
  - anchor badge
- penjelasan jujur bahwa:
  - `rerank_score` adalah heuristic UI-only
  - nilai `None` bisa valid untuk reserved anchor

#### Logging & Telemetry (V3.0c)
Field baru di `retrieval.jsonl`:
- `reranker_applied`
- `reranker_candidate_count`
- `reranker_top_n`
- `reranker_top_n_base`
- `reranker_top_n_effective`
- `reranker_info`
- `rerank_anchor_reservation`
- `retrieval_stats_pre_rerank`
- `generation_context_stats`

Field baru di `answers.jsonl`:
- `reranker_applied`
- `reranker_model`
- `reranker_top_n`
- `reranker_top_n_base`
- `reranker_top_n_effective`
- `rerank_anchor_reservation`
- `generation_context_stats`

#### Graceful Skip / Fail Path
Jika reranker gagal load:
- pipeline **tidak crash**
- warning UI muncul
- sistem lanjut memakai top-N fallback
- log tetap jujur (`reranker_applied=false`, `load_failed=true`, `error_msg` terisi)

#### Regression / Backward-Compatibility Polish
Pada akhir V3.0c, jalur legacy dipolish agar:
- config `v1_5.yaml` dan `v2.yaml` tetap sehat
- UI legacy tidak memaksakan kolom rerank yang tidak relevan
- telemetry reranker di run legacy menjadi **disabled-clean**
- history viewer dan latest render tidak membawa jejak reranker palsu

#### Ringkasan hasil validasi V3.0c
- **T01** — smoke test dasar rerank → PASS
- **T02** — comparison query RAD vs Prototyping → PASS substantif / PASS bersyarat
- **T03** — citation routing + rerank → PASS
- **T04** — metadata routing path → PASS
- **T05** — self-query zero-result guard tidak rusak → PASS
- **T06** — graceful skip reranker fail path → PASS
- **T07** — dense mode + rerank → PASS
- **T08** — multi-turn anaphora + steps → PASS (setelah patch)
- **T09** — history viewer / turn replay → PASS
- **T10-A** — regression legacy dense (`v1_5.yaml`) → clean PASS
- **T10-B** — regression legacy dense/hybrid(`v2.yaml`) → clean PASS

#### Known Limitation Notes (non-blocking)
- metadata filter dari V3.0b masih belum end-to-end pada seluruh hybrid path
- dynamic Top-N masih heuristik, belum berbasis evaluasi kuantitatif formal
- anchor reservation membuat final set tidak sepenuhnya pure rerank
- beberapa snippet/bukti masih bisa tampak rough atau clipped
- query citation tertentu masih bisa menghasilkan jawaban yang narasi-heavy

---

### [V3.0d — Post-Hoc Verification + Evaluation Pipeline + Intent-Aware Answer Augmentation]

#### Post-Hoc Verification Layer
V3.0d menambahkan **lapisan verifikasi setelah generation selesai**, bukan confidence retrieval/rerank palsu. Prinsipnya:
- verification hidup di **answer layer** terhadap **jawaban final + final contexts**
- verification dibuat **rule-based, explainable, dan audit-friendly**
- target utamanya adalah **groundedness signal**, bukan angka confidence pseudo-terkalibrasi

Komponen utama verification V3.0d:
- `build_verification_meta()` di `generate_utils.py`
- public re-export di `generate.py`
- integrasi end-to-end di `app_streamlit.py`
- render verification box di `app_ui_render.py`

Label verifikasi yang dipakai di UI/log:
- `verified`
- `partially_verified`
- `low_verification`
- `verification_n_a` / `skipped`

Keputusan desain penting:
- verification **tidak** dijadikan numeric UI confidence utama
- `rerank_score` **bukan** verification score
- semantic helper / reuse Cross-Encoder **tidak** dijadikan inti verification release V3.0d

#### Verification Logging + History Replay
Verification V3.0d tidak berhenti di backend; ia juga di-hardening ke log dan replay:
- field `verification` ditambahkan ke `answers.jsonl`
- metadata-routed path diperlakukan jujur sebagai `N/A` / `skipped`
- `last_verification` di session state ikut sinkron
- `turn_data[turn_id]["verification"]` ikut disimpan agar replay historis benar-benar menampilkan status verifikasi turn yang dipilih

#### Evaluation Dataset Builder (V3.0d)
V3.0d menambahkan script evaluasi baru:
- `src/evaluation/build_eval_dataset.py`

Tugas builder ini:
- join `answers.jsonl` + `retrieval.jsonl` untuk satu `run_id`
- merekonstruksi **final contexts** yang benar-benar dipakai answer
- memprioritaskan full chunk dari index/Chroma, bukan preview trimmed dari log
- memisahkan `question_user` vs `question_eval`
- menambahkan `question_source`, `context_source`, `missing_context_chunk_ids`, dst. untuk audit

Output builder:
- `eval_dataset.json`
- `ragas_rows.jsonl`
- `ragas_rows.pretty.json`

#### RAGAS Evaluation Pipeline
V3.0d menambahkan evaluator penuh melalui:
- `src/evaluation/evaluate_ragas.py`

Metric aktif final V3.0d:
- `faithfulness`
- `answer_relevancy`
- `context_precision`

Keputusan scope evaluasi:
- **Answer Correctness belum diaktifkan** karena dataset evaluator belum menyimpan `reference` / `reference_contexts`
- jadi RAGAS pada V3.0d diposisikan sebagai indikator:
  - groundedness
  - relevansi jawaban terhadap pertanyaan
  - kualitas ranking konteks
- RAGAS **bukan** satu-satunya bukti bahwa jawaban sudah conversationally satisfying

#### Evaluator-Specific LLM Override
Agar evaluasi tidak merusak runtime utama, V3.0d menambahkan jalur evaluator LLM override terpisah:
- config: `evaluation.llm_override` di `configs/base.yaml`
- provider evaluator dapat dipilih terpisah dari runtime generator utama
- provider yang disiapkan: **Groq / OpenAI / Ollama**

Final validasi evaluator menunjukkan bahwa model evaluator dapat berbeda dari runtime generator. Pada iterasi akhir V3.0d, evaluator yang paling stabil untuk run final adalah:
- provider: `groq`
- model: `meta-llama/llama-4-scout-17b-16e-instruct`

Ini dipilih karena lebih stabil daripada evaluator awal yang sempat menghasilkan `LLMDidNotFinishException` / row missing pada model Groq yang lebih kecil.

#### Intent Planner + Route Names
Setelah verification/evaluator selesai, fokus V3.0d bergeser ke **quality of answer augmentation**. Karena itu ditambahkan planner intent di `app_streamlit.py` untuk membedakan jalur-jalur seperti:
- `method_explanation`
- `doc_count_list`
- `method_comparison`
- `single_target_steps`
- `metadata_routed`

Tujuan planner:
- menjaga query penjelasan tidak jatuh ke jawaban identifikasi singkat
- menjaga compound-intent seperti *"Ada berapa ...? Jika ada, sebutkan judul dan penulisnya"* tidak direduksi jadi count-only
- memberi audit trail yang lebih jelas untuk answer policy internal

#### Ownership-Aware Doc Aggregation
V3.0d menambahkan **ownership-aware doc aggregation** agar query list/count dan explanation doc-locked lebih disiplin dalam memilih dokumen target. Dampaknya:
- query list/count seperti Prototyping menjadi lebih jujur dan lebih stabil
- `accepted_docs` / `candidate_docs` dapat diaudit
- route explanation/count tidak lagi terlalu mudah “mengambil dokumen salah”

#### Method Explanation Route + Explanation-Biased Expansion
Salah satu kritik terbesar saat review dosen adalah query penjelasan metode sering dijawab seperti query identifikasi. Untuk itu V3.0d memisahkan jalur **method explanation** dari sekadar method identification, lalu menambahkan retrieval expansion yang bias ke:
- penjelasan metode
- alasan penggunaan metode
- karakteristik metode
- tahapan metode
- konteks penggunaan metode pada dokumen yang sedang dibahas

Dampak praktisnya:
- Q1 dan Q3 tidak lagi terlalu mudah jatuh ke jawaban template satu baris
- answer builder menjadi lebih terasa "augmented"

#### Single-Target Steps Handler
V3.0d juga menambahkan handler khusus untuk query tahapan tunggal (single-target steps). Prinsip pentingnya:
- sistem **lebih jujur** jika langkah lengkap tidak eksplisit di final contexts
- lebih baik menjawab partial / safe daripada mengarang langkah
- metadata steps tetap bisa dicatat, tetapi jawaban final tidak boleh melampaui bukti

#### Comparison Formatter
Untuk query comparison seperti RAD vs Prototyping, V3.0d menambahkan formatter comparison yang lebih contrastive:
- per-metode dipisahkan lebih tegas
- context unik per-metode diprioritaskan bila tersedia
- evidence comparison tidak lagi terlalu generik
- status comparison bisa dicatat lebih jujur (`balanced`, `partial`, dll.)

#### Docmeta Sidecar + Safe Title Fallback
Bug metadata display pada query count/list (misalnya title noisy seperti potongan tabel penelitian terdahulu) ditutup lewat:
- `docmeta_sidecar.json`
- heuristik `safe_title_fallback`

Hasilnya:
- title display aman untuk list/count answer
- judul noisy seperti `8. Keluarga Stressed Out ...` tidak lagi dipakai sebagai display title
- jika metadata mentah buruk, sistem fallback ke source-file/doc-id yang aman

#### Logging intent_plan + doc_aggregation_meta
V3.0d menambahkan telemetry policy yang lebih rapi:
- `intent_plan`
- `doc_aggregation_meta`
- `answer_policy_route`
- `answer_policy_audit_summary`

Tujuannya:
- audit trail policy internal lebih mudah dibaca
- progres Tahap 8 / analisis query buruk lebih mudah ditelusuri
- answer policy menjadi lebih explainable untuk kebutuhan skripsi/demo

#### Final Audit/UI Sync Polish
Pada penutupan V3.0d ditemukan bug kecil tetapi nyata: metadata rerank ada di backend/log, tetapi sebagian final contexts/UI masih tidak menampilkannya. Patch final menutup mismatch ini dengan:
- sinkronisasi ulang metadata rerank ke final nodes
- final hardening pada `nodes_for_answer`
- CTX Mapping / After Rerank / Sources kini lebih jujur terhadap final contexts yang benar-benar dipakai answer

Catatan penting:
- `rerank_score` tetap **heuristic UI-only**, bukan calibrated probability
- kombinasi `rerank_rank = #1` tetapi label raw score tampak “Lemah” di UI **bisa valid**, karena label itu hanya membaca nilai mentah dengan threshold heuristik, bukan confidence sejati

#### Ringkasan Hasil Validasi V3.0d
Validasi V3.0d berjalan dalam dua gelombang besar:

**Gelombang 1 — Verification + Evaluator**
- verification backend → PASS
- verification logging di `answers.jsonl` → PASS
- verification UI + history replay → PASS
- build eval dataset → PASS
- evaluator RAGAS → PASS setelah jalur override dimatangkan

**Gelombang 2 — Answer Augmentation Policy (Tahap 1–8)**
- Tahap 1: intent planner + new route names → PASS
- Tahap 2: ownership-aware doc aggregation → PASS
- Tahap 3: method explanation route + explanation-biased expansion → PASS
- Tahap 4: single-target steps handler (+ micro-fix 4.1/4.2/4.3) → PASS
- Tahap 5: comparison formatter (+ micro-fix 5.1/5.2) → PASS
- Tahap 6: docmeta sidecar + safe title fallback → PASS
- Tahap 7: logging `intent_plan` + `doc_aggregation_meta` → PASS
- Tahap 8: rerun 8 query buruk + evaluator final → PASS

#### Residual Issues Non-Blocking (Final V3.0d)
Pada penutupan V3.0d, residual issue yang tersisa diposisikan sebagai **non-blocking**:
- beberapa `title_display` masih safe fallback dari `source_file` / `doc_id`, sehingga aman tetapi belum humanized sempurna
- comparison Q4 sudah contrastive, tetapi evidence masih agak noisy/truncated dan shared CTX pendukung masih bisa ikut muncul
- Q7 single-target steps masih punya inkonsistensi internal metadata, walau jawaban final sudah aman/jujur
- Q8 unresolved-safe answer masih dapat menghasilkan verification low / missing citation, tetapi tidak lagi salah lock ke dokumen yang keliru

Kesimpulan akhir V3.0d:
- dari sisi engineering-praktis, V3.0d **sudah matang untuk dirilis**
- residual issue yang tersisa **tidak menahan release**
- patch runtime besar berikutnya **tidak direkomendasikan** kecuali ada bug kritis baru

#### Arsitektur Modul (V3.0d)
```text
src/
├── app_streamlit.py          ← UI utama + orchestration pipeline final V3.0d
│                               (hybrid/dense, citation routing, self-query integration,
│                               reranker integration, post-hoc verification, intent planner,
│                               ownership-aware doc aggregation, method explanation route,
│                               single-target steps handler, comparison formatter wiring,
│                               docmeta sidecar consumption, logging hardening,
│                               final context / rerank audit sync)
├── app_ui_render.py          ← Render & display helpers
│                               (stream pill, legend CTX, hybrid score pills,
│                               rerank summary box, before/after audit panel,
│                               source snapshot, verification box, history replay render,
│                               rerank/anchor/verification gating, audit UI polish)
├── rag/
│   ├── pdf_parser.py         ← Structure-aware parsing (regex + filename routing)
│   │                           - single source of truth parsing bab/metadata
│   ├── chunking.py           ← Dual strategy chunking (narasi + sitasi)
│   │                           - single source of truth chunking
│   ├── method_detection.py   ← Single source of truth query/method detection
│   │                           + is_citation_query + is_steps_question
│   ├── self_query.py         ← Metadata self-querying module (V3.0b)
│   │                           - LLM → JSON metadata filter + semantic_query + pre-check helper
│   ├── reranker.py           ← Cross-Encoder reranking lifecycle (V3.0c)
│   │                           - load / rerank / unload / status info
│   ├── generate.py           ← Thin re-export wrapper public API
│   │                           (termasuk verification builder yang diexpose dari generate_utils.py)
│   ├── generate_utils.py     ← Semua implementasi generate + guardrails + answer-policy helpers
│   │                           (verification engine, method explanation builder,
│   │                            steps/comparison formatter, deterministic safe answers,
│   │                            post-hoc verification meta)
│   ├── ingest.py             ← PDF ingestion → dual stream → dual ChromaDB collection
│   │                           + doc_catalog + docmeta_sidecar
│   ├── retrieve_dense.py     ← Dense retrieval + where_filter + stream/bab_label
│   ├── retrieve_sparse.py    ← BM25+Sastrawi sparse + collection-aware + stream/bab_label
│   ├── fusion_rrf.py         ← RRF fusion: List[RetrievedNode] → List[RetrievedNode]
│   │                           + propagate stream/bab_label
│   └── metadata_router.py    ← Metadata intent router (page_count)
│                               + natural-language alias resolution
│                               + doc_catalog/docmeta_sidecar loader helpers
├── core/
│   ├── config.py             ← Load config YAML (deep merge base + override)
│   ├── run_manager.py        ← run_id + manifest + logging JSONL
│   ├── schemas.py            ← RetrievedNode
│   │                           (+ stream + bab_label + rerank_score + rerank_rank)
│   ├── ui_utils.py           ← Generic Streamlit widgets (download, PDF helper, dll.)
│   └── cleanup_runs.py       ← Cleanup runs utility (manual)
└── evaluation/
    ├── build_eval_dataset.py ← Builder dataset evaluasi dari runs/<run_id>
    │                           (rekonstruksi final contexts + audit rows)
    └── evaluate_ragas.py     ← Evaluator RAGAS + evaluator-specific LLM override
                                + transparency outputs + partial-failure handling

configs/
├── base.yaml                 ← baseline global config + evaluation.ragas
│                               + evaluation.llm_override
├── v1.yaml
├── v1_5.yaml
├── v2.yaml
├── v3.yaml                   ← config final V3.0d
│                               (dual collection + self_query + reranker + verification
│                                + docmeta + answer_policy + doc_aggregation)
├── v3_dense_test.yaml
└── v3_rerank_fail.yaml
```

**Dependency Graph Modul V3.0d:**
```text
pdf_parser.py        ← ingest.py (structure-aware parsing per-stream)
chunking.py          ← ingest.py (chunking per-stream)

self_query.py        ← app_streamlit.py
                       (LLM metadata extraction → semantic_query + filters)

retrieve_dense.py    ← app_streamlit.py
                       (narasi + sitasi + where_filter, collection-aware)

retrieve_sparse.py   ← app_streamlit.py
                       (narasi + sitasi, collection-aware)

fusion_rrf.py        ← app_streamlit.py
                       (gabungkan dense + sparse, propagate stream/bab_label)

metadata_router.py   ← app_streamlit.py
                       (page_count intent + natural-language alias resolution
                        + loader doc_catalog/docmeta_sidecar)

method_detection.py  ← app_streamlit.py / pdf_parser.py
                       (single source of truth query/type detection)

reranker.py          ← app_streamlit.py
                       (candidate rerank sebelum generation)

generate_utils.py    ← generate.py / app_streamlit.py
                       (answer builder, verification engine, steps/comparison formatter,
                        deterministic safe-answer helpers)

app_ui_render.py     ← app_streamlit.py
                       (verification box, rerank audit UI, source snapshot,
                        CTX mapping, history replay render)

build_eval_dataset.py ← runs/<run_id>/answers.jsonl + retrieval.jsonl
                        + Chroma/index/doc metadata
                        (membangun eval_dataset.json + ragas_rows)

evaluate_ragas.py    ← build_eval_dataset.py output
                       + configs/base.yaml / current project config
                       + evaluator.llm_override
                       (menjalankan faithfulness, answer_relevancy, context_precision)

schemas.RetrievedNode
    ↑
    ├── retrieve_dense.py
    ├── retrieve_sparse.py
    ├── fusion_rrf.py
    ├── reranker.py
    └── app_streamlit.py

_resolve_collection_name()  ← retrieve_dense.py
                               (diimpor oleh retrieve_sparse.py — DRY)

doc_catalog.json / docmeta_sidecar.json
    ↑
    ├── ingest.py             (membangun / menulis)
    ├── metadata_router.py    (load helper)
    └── app_streamlit.py      (dipakai untuk metadata routing + safe display metadata)
```

---

## Stack Teknis
| Komponen | Detail |
|----------|--------|
| Language | Python 3.10+ (runtime: 3.11) |
| Framework | LangChain (orchestrator) + Streamlit (UI) |
| Vector DB | ChromaDB (lokal/persisten) — dual collection sejak V3.0a |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Sparse (V2.0+) | `rank-bm25>=0.2.2` + `PySastrawi>=1.2.0` (hanya jalur BM25 sparse) |
| PDF Parser (V3.0a) | `pymupdf` (fitz) — structure-aware parsing + regex bab detection |
| Reranker (V3.0c) | `BAAI/bge-reranker-base` |
| LLM Primary | Groq API — `llama-3.1-8b-instant` |
| LLM Secondary | Ollama — Llama-3 Quantized 4-bit (offline/privasi) |
| LLM Evaluator Final (V3.0d) | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` (jalur `evaluation.llm_override`) |
| Evaluasi | RAGAS (Faithfulness, Answer Relevancy, Context Precision) + evaluation dataset builder (`build_eval_dataset.py`) |
| Hardware | AMD Ryzen 7 5800H, RAM 32GB, RTX 3050 Ti (4GB VRAM) |

> **Catatan VRAM:** Cross-Encoder dan LLM tidak boleh dimuat bersamaan. Strategi Sequential Loading wajib dipakai di V3.0c ke atas: load Cross-Encoder → rerank → unload → load LLM → generate.

> **Catatan Stemming:** Sastrawi Stemmer HANYA untuk jalur BM25 sparse. Jalur dense TIDAK di-stemming agar konteks semantik embedding tetap utuh. Aturan ini berlaku tanpa pengecualian di semua versi, termasuk pada collection sitasi.

> **Catatan Dual Collection (V3.0a):** Sejak V3.0a, sistem menggunakan dua ChromaDB collection (`skripsi_structured_narasi` dan `skripsi_structured_sitasi`). Config v1.yaml, v1_5.yaml, dan v2.yaml telah diperbarui untuk menunjuk ke `skripsi_structured_narasi` agar backward-compatible dengan dataset baru.

---

## Struktur Repo
- `src/`
  - `app_streamlit.py` : UI Streamlit utama (mode-aware: hybrid/dense, citation routing, self-query integration, reranker integration, multi-turn stabilization, metadata exact-match guard)
  - `app_ui_render.py` : Render & display helpers (V1.5+, hybrid score pills V2.0, stream pill V3.0a, legend CTX V3.0b, rerank audit UI V3.0c)
  - `rag/`
    - `pdf_parser.py` : **[BARU V3.0a]** structure-aware PDF parsing (regex + filename routing, dual stream)
    - `chunking.py` : **[BARU V3.0a]** dual strategy chunking (structure-aware narasi + citation-aware sitasi)
    - `ingest.py` : ingestion/indexing PDF → dual stream → dual ChromaDB collection + doc_catalog
    - `retrieve_dense.py` : dense retrieval (cached + diversity, `score_dense` + `stream` + `bab_label` terisi)
    - `retrieve_sparse.py` : BM25+Sastrawi sparse retrieval + index caching + collection-aware (V2.0+)
    - `fusion_rrf.py` : RRF fusion dense+sparse → `List[RetrievedNode]` + propagate `stream`/`bab_label` (V3.0a)
    - `self_query.py` : **[BARU V3.0b]** self-query metadata filtering (LLM → JSON → validasi → where_filter + semantic_query)
    - `reranker.py` : **[BARU V3.0c]** Cross-Encoder reranking lifecycle (load / rerank / unload / info)
    - `generate.py` : thin re-export wrapper public API
    - `generate_utils.py` : semua implementasi generate + guardrails + comparison/steps hardening (V3.0c)
    - `method_detection.py` : deteksi metode & query type (single source of truth, `is_steps_question` diperketat V3.0a)
    - `metadata_router.py` : metadata intent routing (page_count) + natural-language alias resolution untuk nama dokumen
  - `core/`
    - `config.py` : load config YAML (deep merge base + override)
    - `run_manager.py` : run_id + manifest + logging JSONL
    - `schemas.py` : dataclass schema (RetrievedNode + field hybrid V2.0 + `stream`/`bab_label` V3.0a + `rerank_score`/`rerank_rank` V3.0c)
    - `ui_utils.py` : download buttons, PDF viewer+highlight, clipboard, dll
    - `cleanup_runs.py` : cleanup runs utility (manual)
  - `evaluation/` : script evaluasi V3.0d (`build_eval_dataset.py`, `evaluate_ragas.py`, wrapper RAGAS, transparency outputs)
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

> **Catatan V3.0a:** `pymupdf` (fitz) wajib terinstall untuk structure-aware parsing. Verifikasi:
> ```bash
> python -c "import fitz; print('PyMuPDF OK, version:', fitz.version)"
> ```

### 2) Environment (.env)
Gunakan `.env` lokal (jangan commit API key). Contoh variabel:
- `LLM_PROVIDER=groq|openai|ollama`
- `GROQ_API_KEY=...` + `GROQ_MODEL=...`
- `OPENAI_API_KEY=...` + `OPENAI_MODEL=...`
- `OLLAMA_BASE_URL=http://localhost:11434` + `OLLAMA_MODEL=...`

> Catatan: file `env` tidak di-commit. Gunakan `.env.example` sebagai template.

> **Catatan V3.0d:** Jika evaluator ingin memakai provider berbeda dari runtime utama, siapkan environment / credential yang relevan untuk jalur `evaluation.llm_override` (mis. Groq/OpenAI/Ollama) tanpa harus mengubah provider generator utama.

---

## Cara Menjalankan

### 1.) Ingest / Index (offline)
Pastikan PDF diletakkan di `data_raw/`, lalu jalankan:

**V3.0d / V3.x final (Dual Stream + Self-Query + Rerank + Verification + Answer-Policy Hardening — Direkomendasikan):**
```bash
# Fresh ingest V3.0a — WAJIB pakai --reset untuk pertama kali
# (menghapus collection lama dan membangun dual collection baru)
python -m src.rag.ingest --config configs/v3.yaml --reset

# Verifikasi hasil ingest
python -c "
import chromadb
c = chromadb.PersistentClient('data_index/chroma')
narasi = c.get_collection('skripsi_structured_narasi')
sitasi  = c.get_collection('skripsi_structured_sitasi')
print(f'narasi={narasi.count()}, sitasi={sitasi.count()}')
"

# Ingest tambahan tanpa reset (hanya PDF baru, catalog di-merge otomatis)
python -m src.rag.ingest --config configs/v3.yaml
```

**V1/V2 (dengan dataset V3.0a — tetap kompatibel):**
```bash
# Config v1/v2 sudah diupdate menunjuk ke skripsi_structured_narasi
python -m src.rag.ingest --config configs/v2.yaml --reset
```

> **Catatan V3.0a — Hard Reset:** Sejak V3.0a, `--reset` menggunakan `shutil.rmtree()` untuk menghapus seluruh direktori persist sebelum membuat collection baru. Ini diperlukan karena ChromaDB versi terbaru menyimpan file lock/index yang tidak bisa dihapus hanya dengan `delete_collection()`.

> **Catatan V3.0a — BM25 Cache:** BM25 cache dibangun terpisah per collection (narasi dan sitasi). Setelah re-ingest, restart app atau panggil `clear_sparse_cache()` secara manual agar cache tidak stale.

> **Catatan V3.0d — Docmeta Sidecar:** Setelah patch Tahap 6, proses ingest final V3.0d juga akan menulis `docmeta_sidecar.json` berdampingan dengan `doc_catalog.json`. File ini dipakai untuk safe title fallback dan display metadata yang lebih aman pada query count/list. Jika hendak mengubah heuristik docmeta, lakukan re-ingest agar sidecar sinkron.

> **Catatan V2.0:** BM25 index dibangun dari ChromaDB secara otomatis saat query pertama — tidak perlu langkah ingest tambahan.

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
| Config | Mode | Collection | Keterangan |
|--------|------|-----------|------------|
| `configs/v1.yaml` | dense | `skripsi_structured_narasi` | Skenario A — Naive Dense RAG |
| `configs/v1_5.yaml` | dense | `skripsi_structured_narasi` | V1.5 dengan memory (bukan skenario ablation) |
| `configs/v2.yaml` | **hybrid** | `skripsi_structured_narasi` | **Skenario B — Hybrid RAG** |
| `configs/v3.yaml` | **hybrid + rerank + verification + answer-policy hardening** | narasi + sitasi (dual) | **Skenario C — Proposed Method final V3.0d** |

### 5.) Build Eval Dataset & Evaluator RAGAS (V3.0d)
Setelah run log (`answers.jsonl` + `retrieval.jsonl`) terbentuk, pipeline evaluasi V3.0d dijalankan dalam dua langkah:

```bash
# 1. Bangun dataset evaluator dari run tertentu
python -m src.evaluation.build_eval_dataset --run_id <RUN_ID>

# 2. Jalankan evaluator RAGAS
python -m src.evaluation.evaluate_ragas --run_id <RUN_ID>
```

Output utama di `runs/<run_id>/`:
- `eval_dataset.json`
- `ragas_rows.jsonl`
- `ragas_rows.pretty.json`
- `ragas_results.json`
- `ragas_item_scores.jsonl`
- `ragas_item_scores.pretty.json`

Catatan operasional V3.0d:
- evaluator memakai jalur `evaluation.llm_override`, terpisah dari runtime generator utama
- untuk final validasi V3.0d, konfigurasi evaluator yang paling stabil memakai Groq `meta-llama/llama-4-scout-17b-16e-instruct`
- jika evaluator terkena limit/provider issue, ganti **provider/model evaluator saja**; jangan terburu-buru membuka patch runtime generator utama

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
  enabled: true          # aktifkan jika ingin routing page_count
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

Key penting di `v3.yaml` (V3.0d — Dual Collection + Metadata Self-Querying + Cross-Encoder Reranking + Verification + Answer-Policy Hardening):
```yaml
index:
  collections:
    narasi: "skripsi_structured_narasi"   # collection untuk konten BAB I–V
    sitasi: "skripsi_structured_sitasi"   # collection untuk Daftar Pustaka

chunking:
  narasi:
    chunk_size: 1200    # structure-aware chunking untuk narasi
    chunk_overlap: 200
  sitasi:
    chunk_size: 500     # citation-aware chunking untuk sitasi (lebih kecil)
    chunk_overlap: 50

parsing:
  filename_routing: true         # aktifkan routing by filename (PDF terpotong)
  regex_fallback: true           # aktifkan regex parsing (PDF utuh)
  metadata_scan_pages: 3         # jumlah halaman awal untuk ekstrak metadata
  running_header_threshold: 3    # threshold deteksi running header bab

retrieval:
  default_collection: "narasi"   # default ke narasi; sitasi diaktifkan via citation routing
  mode: "hybrid"

bm25:
  use_sastrawi: true

memory:
  enabled: true
  window: 4

metadata_routing:
  enabled: true
  allow_in_eval: false
  doc_catalog_filename: doc_catalog.json

self_query:
  enabled: true
  allow_in_eval: false
  max_filter_fields: 2
  llm_temperature: 0.0
  llm_max_tokens: 256

reranker:
  enabled: true
  model_name: "BAAI/bge-reranker-base"
  candidate_k: 10
  top_n_rerank: 3
  dynamic_top_n: true
  top_n_rerank_complex: 5
  reserve_method_anchors: true
  show_before_after_panel: true
  before_after_ui_max_pre_rows: 10
  device: "auto"
  min_vram_mb: 800
  allow_in_eval: true

verification:
  enabled: true
  show_box_in_ui: true

docmeta:
  enabled: true
  sidecar_filename: "docmeta_sidecar.json"
  safe_title_fallback: true

answer_policy:
  planner_enabled: true
  log_intent_plan: true

doc_aggregation:
  enabled: true
  log_meta: true

evaluation:
  ragas:
    batch_size: 1
  llm_override:
    enabled: true
    provider: "groq"
    model: "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: 0.0
    max_tokens: 1024
    timeout: 240
    max_retries: 6
```

> **Catatan V3.0d:**  
> - `metadata_routing` tetap aktif di `v3.yaml` untuk mempertahankan jalur page-count intent pada pipeline terbaru  
> - `self_query` tetap hanya diterapkan pada primary dense retrieval narasi  
> - `reranker.allow_in_eval=true` dipilih karena reranking tetap bagian inti proposed method yang diukur pada Skenario C  
> - `verification`, `docmeta`, `answer_policy`, dan `doc_aggregation` kini hidup di config aktif V3.0d  
> - `evaluation.llm_override` adalah jalur resmi untuk evaluator-only; perubahan evaluator tidak perlu memodifikasi runtime generator utama  
> - `v1.yaml`, `v1_5.yaml`, dan `v2.yaml` dipertahankan sebagai baseline historis; fitur V3.0d tidak dipaksakan aktif di config versi lama

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
- Setelah penutupan V3.0d, asumsi README ini adalah: final git strategy **sudah selesai** (`dev` → `main` → tag final V3.0d → `dev` disinkron kembali dari `main`)
- Tag convention: `v{versi}-{codename}` (contoh: `v1.0-naive`, `v1.5-memory`, `v2.0-hybrid`, `v3.0a-structured`, `v3.0b-selfquery`, `v3.0c-rerank`, `v3.0d-verification`, dst...)

### JSONL Schema Consistency
Field di `retrieval.jsonl` dan `answers.jsonl` tidak boleh berubah antar run dalam satu versi — ablation study bergantung pada konsistensi ini. Jika ada field baru, pastikan diisi di **semua jalur** (normal retrieval, metadata-routed, citation-routed).

Field hybrid V2.0 yang ditambahkan ke `retrieved_nodes` di `retrieval.jsonl`:
- `score_rrf` — RRF score final (`null` untuk jalur dense)
- `score_dense` — Chroma distance (terisi untuk semua jalur; `null` jika node hanya dari sparse)
- `score_sparse` — BM25 raw score (`null` jika node hanya dari dense)
- `rank_dense` / `rank_sparse` — posisi ranking per retriever (`null` jika tidak muncul)
- `distance` / `similarity` — `null` untuk jalur hybrid, terisi untuk jalur dense

Field V3.0a yang ditambahkan ke `retrieved_nodes` di `retrieval.jsonl`:
- `stream` — `"narasi"` | `"sitasi"` | `null` (null untuk node dari dataset V1/V2 lama)
- `bab_label` — `"BAB_I"` | `"BAB_II"` | `"BAB_III"` | `"BAB_IV"` | `"BAB_V"` | `"DAFTAR_PUSTAKA"` | `"LAMPIRAN"` | `"FRONT_MATTER"` | `"UNKNOWN"` | `null`

Field V3.0b yang ditambahkan ke `retrieval.jsonl` dan `answers.jsonl`:
- `self_query` — object stabil lintas jalur:
  - `enabled`
  - `semantic_query`
  - `filter_applied`
  - `filter_fields_used`
  - `filters`
  - `extracted_metadata`
  - `fallback_reason`

Contoh nilai `fallback_reason`:
- `not_executed`
- `self_query_disabled`
- `disabled_in_eval`
- `no_valid_filters_extracted`
- `precheck_no_results`
- `metadata_routed`
- `exception:<TypeError>`

Tambahan perilaku V3.0b:
- `strategy` dapat mengandung suffix `+selfquery` bila filter metadata benar-benar diterapkan
- `generator_strategy` dapat bernilai:
  - `llm`
  - `retrieval_only`
  - `metadata_router`
  - `self_query_guard`

Field V3.0c yang ditambahkan ke `retrieval.jsonl`:
- `reranker_applied`
- `reranker_candidate_count`
- `reranker_top_n`
- `reranker_top_n_base`
- `reranker_top_n_effective`
- `reranker_info`
- `rerank_anchor_reservation`
- `retrieval_stats_pre_rerank`
- `generation_context_stats`

Field V3.0c yang ditambahkan ke `answers.jsonl`:
- `reranker_applied`
- `reranker_model`
- `reranker_top_n`
- `reranker_top_n_base`
- `reranker_top_n_effective`
- `rerank_anchor_reservation`
- `generation_context_stats`

Field V3.0c pada `RetrievedNode`:
- `rerank_score`
- `rerank_rank`

Catatan:
- `rerank_score` adalah heuristic UI-only, bukan confidence score
- `None` pada `rerank_score` / `rerank_rank` dapat valid, misalnya pada reserved anchor

Field V3.0d yang ditambahkan ke `answers.jsonl`:
- `verification` — object verification post-hoc (label, explanation, chips, score internal bila relevan)
- `answer_policy_route` — route utama answer-policy / intent primer
- `intent_plan` — object planner intent (jika logging planner aktif)
- `doc_aggregation_meta` — object audit doc aggregation (jika logging aggregation aktif)
- `answer_policy_audit_summary` — compact summary audit untuk planner/doc aggregation

Output builder/evaluator V3.0d di `runs/<run_id>/`:
- `eval_dataset.json`
- `ragas_rows.jsonl`
- `ragas_rows.pretty.json`
- `ragas_results.json`
- `ragas_item_scores.jsonl`
- `ragas_item_scores.pretty.json`

Catatan V3.0d:
- `verification` mengukur groundedness heuristik di answer layer, **bukan** conversational satisfaction
- `intent_plan` dan `doc_aggregation_meta` diposisikan sebagai telemetry audit-friendly, bukan bagian dari prompt jawaban user-facing
- `rerank_score` tetap heuristic UI-only; label raw-score pada pills dapat terlihat "lemah" walau `rerank_rank=#1`, dan itu bukan bug backend

### Prinsip Modifikasi (V1.5+)
- Logika deteksi metode/query → selalu edit di `method_detection.py`
- Logika render/display UI → selalu edit di `app_ui_render.py`
- CSS history expander → selalu pakai sentinel `#hist-scope` + `:has()` scoping agar style tidak bocor
- Field baru di `turn_data` session_state → pastikan diisi di **semua jalur** (metadata_routed, dense, hybrid, citation-routed)

### Prinsip Modifikasi Tambahan (V2.0)
- Modifikasi jalur sparse → edit di `retrieve_sparse.py`; jangan stem di luar fungsi `_tokenize()`
- Modifikasi RRF fusion → edit di `fusion_rrf.py`; kontrak: input tidak dimutasi, output `List[RetrievedNode]` baru
- Penambahan retriever baru → daftarkan di blok `if retrieval_mode == "hybrid":` di `app_streamlit.py`
- Sastrawi stemming **HANYA** di `retrieve_sparse.py` — tidak boleh dipanggil dari modul lain

### Prinsip Modifikasi Tambahan (V3.0a)
- Logika parsing PDF & penentuan stream/bab_label → selalu edit di `pdf_parser.py` (single source of truth)
- Logika chunking per stream → selalu edit di `chunking.py` (single source of truth)
- Resolusi nama collection → selalu via `_resolve_collection_name()` di `retrieve_dense.py`; jangan hardcode nama collection di modul lain
- `is_citation_query()` saat ini ada di `app_streamlit.py` dan `pdf_parser.py` — di V3.0b akan dikonsolidasi ke `method_detection.py`
- Field `stream` dan `bab_label` di `RetrievedNode` wajib diteruskan (di-copy) di setiap lapisan yang membangun node baru — terutama `fusion_rrf.py`
- Saat menambah collection baru → daftarkan di `v3.yaml` → handle di `_resolve_collection_name()` → tambahkan routing di `app_streamlit.py`

### Prinsip Modifikasi Tambahan (V3.0b)
- Logika self-query metadata → selalu edit di `self_query.py`
- `semantic_query` hasil self-query adalah query final retrieval; jangan langsung pakai query mentah bila self-query aktif
- `where_filter` hanya diterapkan ke **primary dense retrieval narasi**
- Jika menambah field baru ke `self_query` log, pastikan shape object tetap sama di semua jalur:
  - normal retrieval
  - fallback 0 hasil
  - disabled path
  - metadata-routed path
  - eval-disabled path
- `+selfquery` pada `strategy` hanya boleh muncul jika filter metadata benar-benar diterapkan
- Untuk query metadata eksplisit yang gagal exact-match (`precheck_no_results`), answer layer harus tetap jujur dan tidak boleh membuat klaim positif berdasarkan retrieval fallback umum
- Perbaikan metadata router → selalu edit di `metadata_router.py`; jangan memindahkan logic alias resolution ke `app_streamlit.py`

### Prinsip Modifikasi Tambahan (V3.0c)
- Lifecycle Cross-Encoder → selalu edit di `reranker.py`
- Integrasi rerank ke pipeline utama → selalu edit di `app_streamlit.py`
- Render before/after panel, snapshot, dan kolom rerank → selalu edit di `app_ui_render.py`
- Patch synthesis comparison/steps/multi-target → fokuskan di `generate_utils.py`
- `+rerank` pada `strategy` hanya boleh muncul jika `reranker_applied=True`
- `rerank_score` adalah **heuristic UI-only**, bukan confidence score
- `None` pada `rerank_score` / `rerank_rank` bisa valid untuk reserved anchor
- Jika config legacy dipakai, UI/log legacy harus tetap **bersih** dan tidak membawa fingerprint reranker palsu

### Prinsip Modifikasi Tambahan (V3.0d)
- Logic verification rule-based → fokuskan di `src/rag/generate_utils.py`; `generate.py` hanya public re-export tipis
- Integrasi verification, planner, doc aggregation, docmeta sidecar consumption, dan final context/UI sync → fokuskan di `src/app_streamlit.py`
- Render verification box, pills audit, CTX Mapping, Before/After panel, dan Sources display → fokuskan di `src/app_ui_render.py`
- Builder dataset evaluator → `src/evaluation/build_eval_dataset.py`
- Evaluator RAGAS + provider/model override → `src/evaluation/evaluate_ragas.py` + `configs/base.yaml`
- `docmeta_sidecar.json` dibangun saat ingest; jika heuristik display metadata diubah, lakukan re-ingest agar sidecar sinkron
- Jangan membuka ulang retrieval/rerank/parser lama kecuali ada bug kritis; V3.0d menempatkan bottleneck utama pada answer policy / audit / evaluator, bukan pada retrieval recall dasar
- Untuk problem evaluator, prioritaskan **override evaluator-only** (provider/model/timeout/max_tokens/retries) sebelum mempertimbangkan perubahan runtime generator utama
- Untuk patch audit UI terakhir, final contexts harus dianggap sumber kebenaran untuk CTX Mapping / After Rerank / Sources; jika metadata rerank ada di backend tetapi hilang di UI, perbaikilah sinkronisasi final-node, bukan rerank backendnya

### Constraint Hardware (VRAM 4GB)
- V2.0–V3.0b: BM25 berjalan di CPU, tidak ada tambahan beban VRAM signifikan dari sparse leg
- V3.0c ke atas: Sequential Loading wajib — Cross-Encoder dan LLM tidak boleh dimuat bersamaan
  - Urutan: load Cross-Encoder → rerank → **unload** → load LLM → generate
- V3.0d: evaluator LLM diposisikan **terpisah** dari runtime generator; tuning provider/model evaluator tidak boleh dipakai sebagai alasan untuk membuka patch runtime generator utama
- Device strategy reranker: `device="auto"`, `min_vram_mb=800`; jika CUDA tidak aman/tidak tersedia, fallback CPU adalah perilaku yang valid

---