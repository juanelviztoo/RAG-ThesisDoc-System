from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.core.config import ConfigPaths, load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager


def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _list_pdfs(data_raw_dir: Path, pattern: str = "*.pdf") -> List[Path]:
    return sorted([p for p in data_raw_dir.glob(pattern) if p.is_file()])


def _load_pdf_pages(pdf_path: Path) -> List:
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()  # per-page documents
    # Tambahkan metadata dasar
    for d in docs:
        d.metadata["source_file"] = pdf_path.name
        d.metadata["doc_id"] = pdf_path.stem
    # Filter page kosong
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    return docs


def _build_embeddings(cfg: Dict[str, Any]) -> HuggingFaceEmbeddings:
    model_name = _cfg_get(
        cfg,
        ["embeddings", "model_name"],
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    device = _cfg_get(cfg, ["embeddings", "device"], "cpu")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _reset_collection(persist_dir: Path, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # kalau collection belum ada, abaikan
        pass


def ingest(cfg: Dict[str, Any], reset: bool = False, pdf_glob: str = "*.pdf") -> None:
    paths = ConfigPaths.from_cfg(cfg)
    data_raw_dir = paths.data_raw_dir
    persist_dir = paths.persist_dir
    collection_name = cfg["index"]["collection_name"]

    # Run logging (kotak hitam) untuk ingest juga
    rm = RunManager(runs_dir=paths.runs_dir, config=cfg).start()
    logger = setup_logger(rm.app_log_path)
    logger.info("=== INGEST START ===")
    logger.info(f"run_id={rm.run_id}")
    logger.info(f"data_raw_dir={data_raw_dir}")
    logger.info(f"persist_dir={persist_dir}")
    logger.info(f"collection_name={collection_name}")
    logger.info(f"reset={reset}")

    if not data_raw_dir.exists():
        raise FileNotFoundError(f"Folder data_raw tidak ditemukan: {data_raw_dir}")

    pdfs = _list_pdfs(data_raw_dir, pdf_glob)
    if not pdfs:
        raise FileNotFoundError(f"Tidak ada PDF di {data_raw_dir} dengan pattern {pdf_glob}")

    if reset:
        logger.info("Resetting collection (delete_collection)...")
        _reset_collection(persist_dir, collection_name)

    # Load docs
    all_docs = []
    logger.info(f"Loading PDFs: {len(pdfs)} file(s)")
    for p in tqdm(pdfs, desc="Loading PDFs"):
        try:
            docs = _load_pdf_pages(p)
            all_docs.extend(docs)
        except Exception as e:
            logger.exception(f"Gagal load PDF: {p.name} | {type(e).__name__}: {e}")

    if not all_docs:
        raise RuntimeError("Tidak ada teks yang berhasil diekstrak dari PDF (semua kosong/gagal).")

    # Chunking
    chunk_size = int(cfg["chunking"]["chunk_size"])
    chunk_overlap = int(cfg["chunking"]["chunk_overlap"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_docs)

    # Assign chunk_id (deterministik) untuk audit + sources
    ids = []
    for i, d in enumerate(chunks):
        doc_id = d.metadata.get("doc_id", "unknown")
        page = d.metadata.get("page", "na")
        chunk_id = f"{doc_id}#p{page}#c{i:06d}"
        d.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)

    logger.info(f"Pages loaded: {len(all_docs)}")
    logger.info(f"Chunks created: {len(chunks)} | chunk_size={chunk_size} overlap={chunk_overlap}")

    # Embedding + Chroma
    embeddings = _build_embeddings(cfg)
    logger.info(f"Embeddings model: {embeddings.model_name}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    try:
        vectorstore.add_documents(chunks, ids=ids)
    except Exception:
        logger.exception(
            "Gagal add_documents ke Chroma. "
            "Jika ini karena duplicate IDs, jalankan ulang dengan --reset."
        )
        raise

    # # Persist (beberapa versi langchain_chroma tidak perlu, tapi aman)
    # try:
    #     vectorstore.persist()
    # except Exception:
    #     pass

    # Count check via chromadb client (lebih pasti)
    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection(name=collection_name)
    logger.info(f"Chroma count: {col.count()}")

    logger.info("=== INGEST DONE ===")
    print(
        f"[OK] Ingest selesai. run_id={rm.run_id} | collection={collection_name} | count={col.count()}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path config yaml (e.g., configs/v1.yaml)")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Hapus collection lama sebelum ingest (disarankan saat re-index)",
    )
    ap.add_argument("--pdf_glob", default="*.pdf", help="Pattern PDF (default: *.pdf)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ingest(cfg, reset=args.reset, pdf_glob=args.pdf_glob)


if __name__ == "__main__":
    main()
