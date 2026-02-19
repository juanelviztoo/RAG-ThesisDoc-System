# Retrieval-Augmented Generation (RAG) pada Dokumen Skripsi Menggunakan Hybrid Retrieval dan Metadata Filtering dengan Mekanisme Cross-Encoder Reranking serta Post-Hoc Verification

Prototipe sistem RAG untuk dokumen skripsi: Naive RAG -> Hybrid Retrieval -> Proposed Method (bertahap).

## Struktur
- `src/` : source code
- `configs/` : konfigurasi versi (v1, v1.5, v2, v3)
- `runs/` : output log per-run (jsonl + config snapshot) [gitignored]
- `data_raw/` : PDF mentah [gitignored]
- `data_index/` : persistent storage ChromaDB [gitignored]
- `notebooks/` : eksperimen regex/chunking

## Cara jalan (nanti setelah requirements terpasang)
- Ingest/index (offline): `python -m src.rag.ingest --config configs/v1.yaml`
- App UI: `streamlit run src/app_streamlit.py`
- Evaluasi (RAGAS): `python -m src.evaluation.evaluate_ragas --run_id <run_id>`
