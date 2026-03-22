"""
src/rag/ingest.py
=================
V3.0a — Structure-Aware Ingest: Dual Stream + Dual ChromaDB Collection

Perubahan dari V2.0:
    - Single collection ("skripsi_naive") → Dual collection (narasi + sitasi)
    - Chunking generik → pdf_parser.py + chunking.py (structure-aware)
    - doc_catalog schema diperkaya: judul, penulis, tahun, prodi, streams, n_chunks_*
    - Prefix metadata propagation: metadata dari Halaman_Depan.pdf dipakai
        sebagai fallback untuk file per-BAB yang tidak bisa mengekstrak metadata sendiri
    - Bug fix: existing_catalog tidak lagi di-override saat reset; double-write dihapus
    - BM25 sparse cache di-clear otomatis setelah ingest (via clear_sparse_cache())

Prinsip routing chunk:
    section.stream == "narasi" → Chroma collection narasi
    section.stream == "sitasi" → Chroma collection sitasi
    section.is_skip == True    → diabaikan (tidak di-index)

chunk_id format: {doc_id}#p{page}#c{idx:06d}
    Counter per-doc, shared antara narasi dan sitasi (memastikan chunk_id unik).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.core.config import ConfigPaths, load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager
from src.rag.chunking import chunk_sections
from src.rag.pdf_parser import DocMetadata, detect_routing_mode, parse_pdf

# ══════════════════════════════════════════════════════════════════════════════
# Pola regex — level modul (dikompilasi sekali)
# ══════════════════════════════════════════════════════════════════════════════

# Pola yang mengindikasikan file adalah halaman cover / depan
_COVER_PATTERNS: List[re.Pattern] = [
    re.compile(r"halaman[_\s\-]*depan",  re.I),
    re.compile(r"halaman[_\s\-]*judul",  re.I),
    re.compile(r"halaman[_\s\-]*awal",   re.I),
    re.compile(r"\bcover\b",             re.I),
]

# Pola yang mengindikasikan file adalah bagian terpisah (bukan PDF utuh)
# Dipakai oleh _extract_file_prefix untuk menemukan separator bagian
_PART_PATTERNS: List[re.Pattern] = [
    re.compile(r"[_\-]bab[_\-\s]",           re.I),
    re.compile(r"[_\-]halaman[_\-\s]",        re.I),
    re.compile(r"[_\-]daftar[_\-\s]*pustaka", re.I),
    re.compile(r"[_\-]daftar[_\-\s]*referensi", re.I),
    re.compile(r"[_\-]lampiran\b",            re.I),
    re.compile(r"[_\-]abstrak\b",             re.I),
    re.compile(r"[_\-]penutup\b",             re.I),
    re.compile(r"[_\-]kesimpulan\b",          re.I),
]


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Ambil nilai nested dari dict config secara aman."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_collection_names(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Ambil (narasi_collection_name, sitasi_collection_name) dari config.
    Fallback ke nama default V3.0a jika config tidak menyertakannya.
    """
    collections = _cfg_get(cfg, ["index", "collections"], {})
    narasi = str(collections.get("narasi", "skripsi_structured_narasi"))
    sitasi = str(collections.get("sitasi", "skripsi_structured_sitasi"))
    return narasi, sitasi


# ══════════════════════════════════════════════════════════════════════════════
# File system helpers
# ══════════════════════════════════════════════════════════════════════════════

def _list_pdfs(data_raw_dir: Path, pattern: str = "*.pdf") -> List[Path]:
    """List semua PDF di folder dengan sorting deterministic."""
    return sorted([p for p in data_raw_dir.glob(pattern) if p.is_file()])


def _get_page_count(pdf_path: Path) -> int:
    """Jumlah halaman PDF via fitz. Mengembalikan 0 jika gagal."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            return int(doc.page_count)
    except Exception:
        return 0


def _is_cover_file(pdf_path: Path) -> bool:
    """True jika stem nama file mengindikasikan halaman cover / depan skripsi."""
    stem = pdf_path.stem
    return any(pat.search(stem) for pat in _COVER_PATTERNS)


def _extract_file_prefix(pdf_path: Path) -> Optional[str]:
    """
    Ekstrak prefix dokumen dari nama file per-bagian.

    Heuristic: cari pola separator bagian (_Bab_, _Halaman_, dll.)
    di stem nama file, lalu ambil substring sebelum match tersebut.

    Returns:
        str:   jika file adalah bagian terpisah (ada prefix)
        None:  jika file adalah PDF utuh (tidak ada pola separator)

    Contoh:
        UNS_Halaman_Depan.pdf  → "UNS"
        UNS_Bab_1.pdf          → "UNS"
        UNS_Daftar_Pustaka.pdf → "UNS"
        ITERA_2023_RAG.pdf     → None  (PDF utuh)
    """
    stem = pdf_path.stem
    for pat in _PART_PATTERNS:
        m = pat.search(stem)
        if m:
            raw_prefix = stem[: m.start()]
            prefix = raw_prefix.strip("_- ").strip()
            return prefix if prefix else None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB helpers
# ══════════════════════════════════════════════════════════════════════════════

def _reset_collections(
    persist_dir: Path, narasi_name: str, sitasi_name: str
) -> None:
    """
    Hard reset: hapus folder ChromaDB di disk dulu (paling reliable),
    lalu fallback ke API delete jika folder tidak ada.
    Mencegah ChromaDB Rust backend compaction error dari stale files.
    """
    # ── Hard delete folder (paling reliable untuk Rust backend) ──────────────
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
        persist_dir.mkdir(parents=True, exist_ok=True)
        return   # folder bersih → tidak perlu API delete

    # ── Fallback: API delete (jika folder tidak ada tapi collection terdaftar) ─
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    for name in (narasi_name, sitasi_name):
        try:
            client.delete_collection(name=name)
        except Exception:
            pass # collection belum ada → abaikan


def _build_embeddings(cfg: Dict[str, Any]) -> HuggingFaceEmbeddings:
    """Build HuggingFaceEmbeddings dari config."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Prefix metadata propagation
# ══════════════════════════════════════════════════════════════════════════════

def _build_prefix_metadata_map(
    pdfs: List[Path],
    cfg: Dict[str, Any],
) -> Dict[str, DocMetadata]:
    """
    Scan file cover / halaman depan dalam daftar PDF, ekstrak metadata-nya.
    Hasilnya digunakan sebagai fallback metadata untuk file per-BAB yang
    tidak dapat mengekstrak metadata (judul, penulis, tahun, prodi) dari
    kontennya sendiri — karena file BAB tidak mengandung halaman cover.

    Alur:
        1. Iterasi semua PDF, pilih yang teridentifikasi sebagai cover file
        2. Ekstrak prefix dokumen dari nama file cover
        3. Panggil parse_pdf() pada cover file → ambil doc_meta dari section pertama
        4. Simpan ke dict[prefix → DocMetadata]

    Returns:
        Dict[prefix → DocMetadata]
        Kosong jika tidak ada cover file yang ditemukan.
    """
    running_header_threshold = int(
        _cfg_get(cfg, ["parsing", "running_header_threshold"], 3)
    )
    metadata_scan_pages = int(
        _cfg_get(cfg, ["parsing", "metadata_scan_pages"], 3)
    )

    prefix_meta: Dict[str, DocMetadata] = {}

    for pdf_path in pdfs:
        if not _is_cover_file(pdf_path):
            continue

        prefix = _extract_file_prefix(pdf_path)
        if prefix is None:
            continue

        try:
            sections = parse_pdf(
                pdf_path,
                running_header_threshold=running_header_threshold,
                metadata_scan_pages=metadata_scan_pages,
            )
            # Semua section dalam satu PDF berbagi doc_meta yang sama
            if sections:
                prefix_meta[prefix] = sections[0].doc_meta
        except Exception:
            pass  # gagal parse cover → prefix tidak dapat metadata, skip

    return prefix_meta


def ingest(cfg: Dict[str, Any], reset: bool = False, pdf_glob: str = "*.pdf") -> None:
    """
    Main entry point ingest V3.0a.

    Pipeline per-PDF:
        1. parse_pdf()      → List[ParsedSection]  (pdf_parser.py)
        2. propagate meta   → overwrite doc_meta per-BAB dengan cover metadata
        3. chunk_sections() → List[Document]        (chunking.py)
        4. split by stream  → narasi_chunks / sitasi_chunks
        5. assign chunk_id  → {doc_id}#p{page}#c{idx:06d}
        6. add_documents()  → Chroma narasi / sitasi

    Args:
        cfg      : config dict hasil load_config()
        reset    : jika True, hapus kedua collection + catalog sebelum ingest
        pdf_glob : glob pattern PDF (default "*.pdf")
    """
    paths = ConfigPaths.from_cfg(cfg)
    data_raw_dir  = paths.data_raw_dir
    persist_dir   = paths.persist_dir
    runs_dir      = paths.runs_dir

    # Dual collection names
    narasi_col_name, sitasi_col_name = _get_collection_names(cfg)

    # Parsing params dari config
    running_header_threshold = int(
        _cfg_get(cfg, ["parsing", "running_header_threshold"], 3)
    )
    metadata_scan_pages = int(
        _cfg_get(cfg, ["parsing", "metadata_scan_pages"], 3)
    )

    # ── Run logging ──────────────────────────────────────────────────────────
    rm     = RunManager(runs_dir=runs_dir, config=cfg).start()
    logger = setup_logger(rm.app_log_path)
    logger.info("=== INGEST V3.0a START ===")
    logger.info("run_id=%s", rm.run_id)
    logger.info("data_raw_dir=%s", data_raw_dir)
    logger.info("persist_dir=%s", persist_dir)
    logger.info("collection_narasi=%s", narasi_col_name)
    logger.info("collection_sitasi=%s", sitasi_col_name)
    logger.info("reset=%s", reset)

    if not data_raw_dir.exists():
        raise FileNotFoundError(f"Folder data_raw tidak ditemukan: {data_raw_dir}")

    pdfs = _list_pdfs(data_raw_dir, pdf_glob)
    if not pdfs:
        raise FileNotFoundError(f"Tidak ada PDF di {data_raw_dir} dengan pattern {pdf_glob}")

    logger.info("PDF ditemukan: %d file(s)", len(pdfs))
    for p in pdfs:
        logger.info("  %s  [routing=%s]", p.name, detect_routing_mode(p))

    # ── Reset (jika diminta) ─────────────────────────────────────────────────
    # Load existing catalog dulu (untuk merge inkremental)
    persist_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = persist_dir / "doc_catalog.json"
    if reset:
        logger.info("Resetting dual collections + catalog...")
        _reset_collections(persist_dir, narasi_col_name, sitasi_col_name)
        # Hapus catalog lama agar sinkron dengan collection yang baru di-reset
        if catalog_path.exists():
            catalog_path.unlink()
            logger.info("doc_catalog.json dihapus (sinkron dengan --reset).")
        logger.info("Reset selesai.")

    # ── Load existing catalog (merge inkremental) ────────────────────────────
    existing_catalog: Dict[str, Any] = {}
    if catalog_path.exists():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_catalog = loaded
        except Exception:
            existing_catalog = {}

    # Inisialisasi catalog entry per PDF (Hanya init field yang TIDAK memerlukan parsing)
    doc_catalog: Dict[str, Any] = dict(existing_catalog)
    for p in pdfs:
        doc_id = p.stem
        if doc_id not in doc_catalog:  # hanya buat entry baru jika belum ada
            doc_catalog[doc_id] = {}
        # Update field statis (tidak perlu parsing)
        doc_catalog[doc_id].update({
            "doc_id": doc_id,
            "source_file": p.name,
            "relative_path": p.name,
            "page_count": _get_page_count(p),
            "routing_mode": detect_routing_mode(p),
        })

    logger.info("doc_catalog initialized: %d docs", len(doc_catalog))

    # ── Prefix metadata propagation ──────────────────────────────────────────
    # Scan cover/halaman-depan files untuk ekstrak metadata dokumen.
    # Metadata ini di-propagate ke file per-BAB dari dokumen yang sama.
    logger.info("Building prefix metadata map...")
    prefix_meta_map = _build_prefix_metadata_map(pdfs, cfg)

    if prefix_meta_map:
        logger.info(
            "Prefix metadata map: %s",
            {k: v.judul[:40] for k, v in prefix_meta_map.items()},
        )
    else:
        logger.info(
            "Prefix metadata map: kosong "
            "(semua PDF utuh atau tidak ada file cover)"
        )

    # ── Build embeddings ─────────────────────────────────────────────────────
    embeddings = _build_embeddings(cfg)
    logger.info("Embeddings model: %s", embeddings.model_name)

    # ── Inisialisasi dual vectorstore ────────────────────────────────────────
    vs_narasi = Chroma(
        collection_name=narasi_col_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vs_sitasi = Chroma(
        collection_name=sitasi_col_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    # ── Counter global per-run ───────────────────────────────────────────────
    total_narasi  = 0
    total_sitasi  = 0
    total_skipped = 0   # jumlah section SKIP (cover, lampiran, dll.)
    total_failed  = 0   # jumlah PDF yang gagal diproses

    # chunk_id counter per doc_id — shared antara narasi & sitasi
    # memastikan chunk_id unik dan deterministik dalam satu run
    _chunk_counter: Dict[str, int] = defaultdict(int)

    # ── Proses setiap PDF ────────────────────────────────────────────────────
    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
        doc_id = pdf_path.stem
        logger.info("--- Processing: %s ---", pdf_path.name)

        # ── 1. Parse PDF → List[ParsedSection] ───────────────────────────────
        try:
            sections = parse_pdf(
                pdf_path,
                running_header_threshold=running_header_threshold,
                metadata_scan_pages=metadata_scan_pages,
            )
        except Exception as exc:
            logger.exception("Gagal parse PDF: %s | %s", pdf_path.name, exc)
            total_failed += 1
            continue

        if not sections:
            logger.warning("Tidak ada section dari: %s — skip.", pdf_path.name)
            total_failed += 1
            continue

        # ── 2. Metadata propagation ───────────────────────────────────────────
        # Untuk file per-BAB (bukan cover), overwrite doc_meta dengan
        # metadata dari cover file yang sama (berdasarkan prefix).
        # Cover file itu sendiri tidak di-propagate (sudah benar).
        prefix = _extract_file_prefix(pdf_path)
        is_cover = _is_cover_file(pdf_path)

        if not is_cover and prefix and prefix in prefix_meta_map:
            cover_meta = prefix_meta_map[prefix]
            for sec in sections:
                sec.doc_meta = cover_meta
            logger.info(
                "  Metadata propagated dari prefix '%s': judul='%s...'",
                prefix,
                cover_meta.judul[:40],
            )
        else:
            # Log metadata yang dipakai (hasil ekstraksi dari file itu sendiri)
            dm = sections[0].doc_meta
            logger.info(
                "  doc_meta (self-extracted): judul='%s...' penulis='%s' tahun=%s",
                dm.judul[:40],
                dm.penulis,
                dm.tahun,
            )

        # ── 3. Chunk sections → List[Document] ───────────────────────────────
        try:
            all_chunks = chunk_sections(
                sections=sections,
                cfg=cfg,
                doc_id=doc_id,
                source_file=pdf_path.name,
            )
        except Exception as exc:
            logger.exception("Gagal chunk: %s | %s", pdf_path.name, exc)
            total_failed += 1
            continue

        # ── 4. Split berdasarkan stream ───────────────────────────────────────
        narasi_chunks = [c for c in all_chunks if c.metadata.get("stream") == "narasi"]
        sitasi_chunks = [c for c in all_chunks if c.metadata.get("stream") == "sitasi"]
        skip_count    = sum(1 for s in sections if s.is_skip)
        total_skipped += skip_count

        logger.info(
            "  sections=%d | skip=%d | narasi_chunks=%d | sitasi_chunks=%d",
            len(sections),
            skip_count,
            len(narasi_chunks),
            len(sitasi_chunks),
        )

        # ── 5. Assign chunk_id ────────────────────────────────────────────────
        # Narasi chunks dulu (urutan dari chunk_sections() sudah narasi dulu)
        narasi_ids: List[str] = []
        for doc in narasi_chunks:
            page    = doc.metadata.get("page", "na")
            idx     = _chunk_counter[doc_id]
            chunk_id = f"{doc_id}#p{page}#c{idx:06d}"
            doc.metadata["chunk_id"] = chunk_id
            narasi_ids.append(chunk_id)
            _chunk_counter[doc_id] += 1

        sitasi_ids: List[str] = []
        for doc in sitasi_chunks:
            page    = doc.metadata.get("page", "na")
            idx     = _chunk_counter[doc_id]
            chunk_id = f"{doc_id}#p{page}#c{idx:06d}"
            doc.metadata["chunk_id"] = chunk_id
            sitasi_ids.append(chunk_id)
            _chunk_counter[doc_id] += 1

        # ── 6. Push ke ChromaDB ───────────────────────────────────────────────
        if narasi_chunks:
            try:
                vs_narasi.add_documents(narasi_chunks, ids=narasi_ids)
                total_narasi += len(narasi_chunks)
                logger.info("  → narasi collection: +%d chunks", len(narasi_chunks))
            except Exception as exc:
                logger.exception(
                    "Gagal add_documents narasi: %s | %s", pdf_path.name, exc
                )

        if sitasi_chunks:
            try:
                vs_sitasi.add_documents(sitasi_chunks, ids=sitasi_ids)
                total_sitasi += len(sitasi_chunks)
                logger.info("  → sitasi collection: +%d chunks", len(sitasi_chunks))
            except Exception as exc:
                logger.exception(
                    "Gagal add_documents sitasi: %s | %s", pdf_path.name, exc
                )

        # ── 7. Update doc_catalog ─────────────────────────────────────────────
        dm = sections[0].doc_meta
        streams_used: List[str] = []
        if narasi_chunks:
            streams_used.append("narasi")
        if sitasi_chunks:
            streams_used.append("sitasi")

        doc_catalog[doc_id].update({
            "judul":           dm.judul,
            "penulis":         dm.penulis,
            "tahun":           dm.tahun if dm.tahun is not None else 0,
            "prodi":           dm.prodi if dm.prodi is not None else "",
            "streams":         streams_used,
            "n_chunks_narasi": len(narasi_chunks),
            "n_chunks_sitasi": len(sitasi_chunks),
        })

    # ── Simpan doc_catalog final (sekali saja — tidak ada double write) ──────
    catalog_path.write_text(
        json.dumps(doc_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("doc_catalog saved: %s | n_docs=%d", catalog_path, len(doc_catalog))

    # ── Count check via raw chromadb client ──────────────────────────────────
    client     = chromadb.PersistentClient(path=str(persist_dir))
    col_narasi = client.get_or_create_collection(name=narasi_col_name)
    col_sitasi = client.get_or_create_collection(name=sitasi_col_name)
    count_narasi = col_narasi.count()
    count_sitasi = col_sitasi.count()

    logger.info("=== INGEST V3.0a SUMMARY ===")
    logger.info("PDFs processed  : %d", len(pdfs) - total_failed)
    logger.info("PDFs failed     : %d", total_failed)
    logger.info("Sections skipped: %d", total_skipped)
    logger.info("Chunks narasi   : %d (Chroma count: %d)", total_narasi, count_narasi)
    logger.info("Chunks sitasi   : %d (Chroma count: %d)", total_sitasi, count_sitasi)
    logger.info("=== INGEST V3.0a DONE ===")

    # ── Clear BM25 sparse cache ──────────────────────────────────────────────
    # Corpus narasi berubah setelah ingest → cache stale → harus di-clear
    # agar query pertama pada sesi berikutnya rebuild dari corpus baru.
    try:
        from src.rag.retrieve_sparse import clear_sparse_cache
        clear_sparse_cache()
        logger.info("BM25 sparse cache cleared.")
    except Exception:
        pass  # retrieve_sparse belum di-load dalam konteks ini

    print(
        f"[OK] Ingest V3.0a selesai. "
        f"run_id={rm.run_id} | "
        f"narasi={count_narasi} | "
        f"sitasi={count_sitasi}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest PDF ke dual ChromaDB collection (V3.0a)"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path config yaml (gunakan configs/v3.yaml untuk V3.0a)",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Hapus kedua collection + catalog sebelum ingest. "
            "WAJIB dipakai saat pertama kali atau saat re-index penuh."
        ),
    )
    ap.add_argument(
        "--pdf_glob",
        default="*.pdf",
        help="Glob pattern untuk filter PDF (default: '*.pdf'). "
            "Contoh: 'UNS_*.pdf' untuk ingest hanya dokumen UNS.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ingest(cfg, reset=args.reset, pdf_glob=args.pdf_glob)


if __name__ == "__main__":
    main()
