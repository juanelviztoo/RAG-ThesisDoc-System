# RAG ThesisDoc System
**Retrieval-Augmented Generation (RAG) untuk Dokumen Skripsi/TA**  

## Roadmap Versi (Bertahap)
- **V1.0 (DONE)**: Naive Dense RAG baseline + UI + Audit Logging + PDF Viewer/Highlight + Utility Cleanup Runs
- **V1.5 (NEXT)**: Multi-turn Chat (Session Memmory) - tanpa ubah DB/indexing
- **V2.0**: Hybrid Retrieval (BM25 + Sastrawi + RRF Fusion)
- **V3.0+**: Regex Parsing (narasi vs sitasi), Metadata Filtering, Cross-encoder Reranking, Post-hoc Verification

---

## Summary
Repo ini berisi prototipe sistem **Question Answering** pada kumpulan PDF skripsi/TA.
Versi **V1.0** sudah stabil untuk demo:
- Dense retrieval (Chroma + embeddings)
- Generator (LLM) opsional (Groq/OpenAI/Ollama)
- UI Streamlit demo-ready (sources, filter, copy answer)
- Logging lengkap untuk audit (manifest + retrieval.jsonl + answers.jsonl + app.log)
- PDF Viewer berbasis gambar + **highlight keyword** (auto/manual)
- Utility `cleanup_runs.py` (dry-run default, aman)

> Prinsip utama: **transparan & dapat diaudit**. Jawaban selalu disertai konteks (CTX) dan sumber.

---

## Fitur Utama (V1.0 Baseline)
### Retrieval (Dense)
- Vector DB: **Chroma**
- Query: `similarity_search_with_score()`
- UI menampilkan **distance + similarity** + metadata (source_file, page)
- Optimasi: caching embeddings & vectorstore (lebih cepat untuk multi-query)

### Generation (Opsional)
- Toggle di UI: **Use LLM (Generator)**
- Guardrail prompt:
  - Bahasa Indonesia
  - Hanya dari konteks
  - Jika tidak ada bukti: **"Tidak ditemukan pada dokumen."**
  - Klaim faktual wajib disertai `[CTX n]`
- Provider:
  - `groq` (default untuk development)
  - `openai`
  - `ollama`

### UI/UX Streamlit
- Sidebar controls: config, top_k, max_chars_per_ctx, toggle LLM, status API key
- Indikator config (INFO/WARNING/SUCCESS) → jelas kapan config sudah diterapkan
- Riwayat session (preview Q/A)
- Sources Top-k:
  - filter sources (doc_id/source_file/text)
  - tombol View/Download PDF per chunk
- PDF Viewer:
  - render per halaman jadi PNG (anti-block browser)
  - highlight keyword (auto dari query/CTX + manual override)

### Logging & Audit (per run)
Folder `runs/<run_id>/` berisi:
- `manifest.json` — snapshot config + git hash
- `retrieval.jsonl` — log retrieval (top_k, nodes, distance/similarity, metadata)
- `answers.jsonl` — log jawaban (raw answer, used_context ids)
- `app.log` — log aplikasi
- UI menyediakan download per file + `logs.zip`

---

## Struktur Repo
- `src/`
  - `app_streamlit.py` : UI Streamlit
  - `rag/`
    - `ingest.py` : ingestion/indexing PDF → Chroma
    - `retrieve_dense.py` : dense retrieval (cached)
    - `generate.py` : LLM generator + prompt guardrail
  - `core/`
    - `config.py` : load config YAML
    - `run_manager.py` : run_id + manifest + logging JSONL
    - `schemas.py` : dataclass schema (RetrievedNode, dll)
    - `ui_utils.py` : download buttons, PDF viewer+highlight, clipboard, dll
    - `cleanup_runs.py` : cleanup runs utility (manual)
  - `evaluation/` : script pondasi evaluasi (RAGAS, dsb.)
- `configs/` : config per versi (v1, v1.5, v2, v3)
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
python -m src.rag.ingest --config configs/v1.yaml
```

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