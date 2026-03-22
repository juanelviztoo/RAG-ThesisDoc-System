"""
src/rag/chunking.py
===================
V3.0a — Structure-Aware dan Citation-Aware Chunking

Responsibility:
    - Menerima List[ParsedSection] untuk SATU dokumen + config
    - Mengembalikan List[LangChain Document] dengan metadata lengkap
    - Dua strategi:
        Narasi : RecursiveCharacterTextSplitter per halaman (hormati batas bab,
                akurasi page metadata terjaga untuk PDF viewer)
        Sitasi : citation-aware splitting (per blok referensi)
    - SKIP sections diabaikan (tidak menghasilkan Document)
    - chunk_id TIDAK di-set di sini — caller (ingest.py) yang set,
    agar counter per-doc tetap shared antara narasi & sitasi

Public API:
    chunk_sections(sections, cfg, doc_id, source_file) -> List[Document]

Kontrak chunk_id (dijaga oleh ingest.py, bukan modul ini):
    Format : {doc_id}#p{page}#c{idx:06d}
    Counter: per-doc, shared antara narasi dan sitasi
    Hal ini memastikan chunk_id unik dan stabil antar run.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.pdf_parser import ParsedSection

# ═════════════════════════════════════════════════════════════════════════════
# Konstanta
# ═════════════════════════════════════════════════════════════════════════════

# Default chunking params (override dari cfg["chunking"]["narasi"/"sitasi"])
_DEFAULT_NARASI_CHUNK_SIZE    = 1200
_DEFAULT_NARASI_CHUNK_OVERLAP = 200
_DEFAULT_SITASI_CHUNK_SIZE    = 500
_DEFAULT_SITASI_CHUNK_OVERLAP = 50

# Halaman dianggap "pendek" jika teksnya < angka ini (chars)
# Halaman pendek akan di-merge dengan halaman berikutnya sebelum di-chunk
_MIN_PAGE_CHARS = 150


# ═════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_narasi_cfg(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Ambil (chunk_size, chunk_overlap) untuk stream narasi dari config."""
    narasi_cfg = cfg.get("chunking", {}).get("narasi", {})
    chunk_size    = int(narasi_cfg.get("chunk_size",    _DEFAULT_NARASI_CHUNK_SIZE))
    chunk_overlap = int(narasi_cfg.get("chunk_overlap", _DEFAULT_NARASI_CHUNK_OVERLAP))
    return chunk_size, chunk_overlap


def _get_sitasi_cfg(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Ambil (chunk_size, chunk_overlap) untuk stream sitasi dari config."""
    sitasi_cfg = cfg.get("chunking", {}).get("sitasi", {})
    chunk_size    = int(sitasi_cfg.get("chunk_size",    _DEFAULT_SITASI_CHUNK_SIZE))
    chunk_overlap = int(sitasi_cfg.get("chunk_overlap", _DEFAULT_SITASI_CHUNK_OVERLAP))
    return chunk_size, chunk_overlap


# ═════════════════════════════════════════════════════════════════════════════
# Metadata builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_chunk_meta(
    *,
    doc_id:      str,
    source_file: str,
    page:        int,
    section:     ParsedSection,
) -> Dict[str, Any]:
    """
    Build metadata dict untuk satu chunk LangChain Document.

    CATATAN ChromaDB compatibility:
      - ChromaDB metadata hanya menerima: str, int, float, bool
      - None tidak diterima → tahun=0 jika None, prodi="" jika None
      - chunk_id TIDAK di-set di sini (diisi oleh ingest.py)

    Field metadata yang dihasilkan:
      doc_id, source_file, page, stream, bab_label,
      judul, penulis, tahun, prodi, routing_mode
    """
    dm = section.doc_meta
    return {
        "doc_id":       doc_id,
        "source_file":  source_file,
        "page":         page,
        "stream":       section.stream if section.stream else "narasi",
        "bab_label":    section.bab_label,
        "judul":        dm.judul,
        "penulis":      dm.penulis,
        "tahun":        dm.tahun if dm.tahun is not None else 0,
        "prodi":        dm.prodi if dm.prodi is not None else "",
        "routing_mode": section.routing_mode,
        # chunk_id sengaja tidak diset — akan diisi oleh ingest.py
        # dengan format: {doc_id}#p{page}#c{idx:06d}
    }


# ═════════════════════════════════════════════════════════════════════════════
# Narasi chunking — structure-aware, per halaman
# ═════════════════════════════════════════════════════════════════════════════

def _merge_short_pages(
    pages: List[Tuple[int, str]],
    min_chars: int = _MIN_PAGE_CHARS,
) -> List[Tuple[int, str]]:
    """
    Merge halaman pendek (< min_chars) ke halaman berikutnya agar tidak
    menghasilkan chunk yang terlalu kecil.
    page_num yang dipakai = page_num halaman pertama dalam merge.

    Contoh: halaman 3 hanya berisi judul bab "BAB II\nTINJAUAN PUSTAKA" (50 chars)
    → di-merge dengan halaman 4 yang berisi isi bab sebenarnya.
    """
    merged: List[Tuple[int, str]] = []
    buffer_page: Optional[int]    = None
    buffer_text: str               = ""

    for page_num, page_text in pages:
        text = page_text.strip()
        if not text:
            continue

        if buffer_page is None:
            buffer_page = page_num
            buffer_text = text
        else:
            buffer_text = buffer_text + "\n\n" + text

        # Flush buffer kalau sudah cukup panjang
        if len(buffer_text) >= min_chars:
            merged.append((buffer_page, buffer_text))
            buffer_page = None
            buffer_text = ""

    # Flush sisa buffer
    if buffer_page is not None and buffer_text.strip():
        if merged:
            # Gabungkan ke blok terakhir agar tidak ada blok pendek tersisa
            last_page, last_text = merged[-1]
            merged[-1] = (last_page, last_text + "\n\n" + buffer_text)
        else:
            merged.append((buffer_page, buffer_text))

    return merged


def _chunk_narasi_section(
    section:      ParsedSection,
    chunk_size:   int,
    chunk_overlap: int,
    doc_id:       str,
    source_file:  str,
) -> List[Document]:
    """
    Chunk satu section narasi.

    Strategi:
    - Merge halaman pendek dulu (_merge_short_pages)
    - Proses setiap blok halaman dengan RecursiveCharacterTextSplitter
    - Setiap chunk mendapat page_num halaman awal blok-nya
    - Overlap hanya terjadi di dalam satu blok halaman, tidak lintas halaman
        (trade-off: presisi page metadata lebih penting untuk PDF viewer)

    Separators ordered by preference untuk teks akademik Indonesia:
        paragraph break > line break > sentence break > word break > char break
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    # Merge halaman pendek
    page_blocks = _merge_short_pages(section.pages, min_chars=_MIN_PAGE_CHARS)

    docs: List[Document] = []
    for page_num, page_text in page_blocks:
        if not page_text.strip():
            continue

        texts = splitter.split_text(page_text)
        for text in texts:
            text = text.strip()
            if not text:
                continue
            meta = _build_chunk_meta(
                doc_id=doc_id,
                source_file=source_file,
                page=page_num,
                section=section,
            )
            docs.append(Document(page_content=text, metadata=meta))

    return docs


# ═════════════════════════════════════════════════════════════════════════════
# Sitasi chunking — citation-aware
# ═════════════════════════════════════════════════════════════════════════════

def _split_citation_text(text: str, chunk_size: int) -> List[str]:
    """
    Split teks Daftar Pustaka menjadi entry referensi individual.

    Strategi bertingkat:
    1. Split by double-newline (paragraph break) — tiap paragraf ≈ satu referensi
    2. Merge paragraf pendek (<= 40 chars) dengan paragraf berikutnya
        (menangani kasus judul baris terpisah dari pengarang)
    3. Split entry yang melebihi chunk_size dengan RecursiveCharacterTextSplitter
        fallback (referensi panjang dengan banyak keterangan tambahan)

    Catatan: tidak mencoba mendeteksi format sitasi (APA/IEEE/Vancouver) secara
    eksplisit. Pendekatan paragraph-based lebih robust untuk berbagai format
    skripsi Indonesia.
    """
    # ── Split by double newline ───────────────────────────────────────────────
    raw_blocks = re.split(r"\n{2,}", text)

    # ── Merge blok pendek ─────────────────────────────────────────────────────
    entries: List[str] = []
    buffer  = ""

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        if buffer and len(buffer) <= 80:
            # Buffer terlalu pendek → merge dengan blok ini
            buffer = buffer + " " + block
        elif buffer:
            entries.append(buffer)
            buffer = block
        else:
            buffer = block

    if buffer.strip():
        entries.append(buffer)

    # ── Split entry yang terlalu panjang ──────────────────────────────────────
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,   # overlap 0 untuk sitasi agar tidak ada referensi duplikat
        separators=["\n", ". ", " ", ""],
    )

    result: List[str] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if len(entry) <= chunk_size:
            result.append(entry)
        else:
            # Referensi terlalu panjang → split, tapi log warning
            warnings.warn(
                f"[chunking] Entri sitasi sangat panjang ({len(entry)} chars > "
                f"{chunk_size} chunk_size). Di-split lebih lanjut.",
                UserWarning,
                stacklevel=4,
            )
            sub_chunks = fallback_splitter.split_text(entry)
            result.extend(sub_chunks)

    return result


def _chunk_sitasi_section(
    section:     ParsedSection,
    chunk_size:  int,
    doc_id:      str,
    source_file: str,
) -> List[Document]:
    """
    Chunk satu section sitasi (Daftar Pustaka).

    Strategi:
    - Gabungkan semua halaman section menjadi satu teks
    - Split menggunakan citation-aware splitting
    - Tiap entry referensi = satu chunk (chunk_overlap = 0 untuk sitasi)
    - page_num mengacu ke halaman pertama section
      (Daftar Pustaka biasanya beberapa halaman, kita assign ke halaman awal)
    """
    # Gabungkan semua halaman section
    parts = [txt for _, txt in section.pages if txt.strip()]
    if not parts:
        return []

    full_text = "\n\n".join(parts)
    entries   = _split_citation_text(full_text, chunk_size)

    docs: List[Document] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Assign page_start sebagai page referensi untuk semua chunk sitasi
        # (lebih akurat tidak diperlukan untuk citation discovery)
        meta = _build_chunk_meta(
            doc_id=doc_id,
            source_file=source_file,
            page=section.page_start,
            section=section,
        )
        docs.append(Document(page_content=entry, metadata=meta))

    return docs


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def chunk_sections(
    sections:    List[ParsedSection],
    cfg:         Dict[str, Any],
    doc_id:      str,
    source_file: str,
) -> List[Document]:
    """
    Convert List[ParsedSection] untuk SATU dokumen → List[LangChain Document].

    Kontrak:
    - SKIP sections (section.is_skip == True) diabaikan
    - chunk_id TIDAK di-set — caller (ingest.py) yang mengisi dengan format
        {doc_id}#p{page}#c{idx:06d} menggunakan counter shared per-doc
    - Stream "narasi" → structure-aware chunking (per halaman, overlap terjaga)
    - Stream "sitasi" → citation-aware chunking (per blok referensi, overlap=0)
    - Urutan output: narasi chunks dulu, sitasi chunks sesudahnya
        (memudahkan counter assignment di ingest.py)

    Args:
        sections    : output dari pdf_parser.parse_pdf() untuk satu PDF
        cfg         : config dict (dari load_config), dibaca ke cfg["chunking"]
        doc_id      : identifier dokumen (biasanya pdf_path.stem)
        source_file : nama file PDF (biasanya pdf_path.name)

    Returns:
        List[Document] dengan metadata lengkap, siap untuk assign chunk_id dan
        dimasukkan ke ChromaDB.
    """
    narasi_cfg  = _get_narasi_cfg(cfg)
    sitasi_cfg  = _get_sitasi_cfg(cfg)

    narasi_chunk_size,   narasi_chunk_overlap = narasi_cfg
    sitasi_chunk_size,   _                    = sitasi_cfg

    narasi_docs: List[Document] = []
    sitasi_docs: List[Document] = []

    for section in sections:
        # Skip section yang tidak perlu di-index
        if section.is_skip:
            continue

        if section.stream == "narasi":
            chunks = _chunk_narasi_section(
                section=section,
                chunk_size=narasi_chunk_size,
                chunk_overlap=narasi_chunk_overlap,
                doc_id=doc_id,
                source_file=source_file,
            )
            narasi_docs.extend(chunks)

        elif section.stream == "sitasi":
            chunks = _chunk_sitasi_section(
                section=section,
                chunk_size=sitasi_chunk_size,
                doc_id=doc_id,
                source_file=source_file,
            )
            sitasi_docs.extend(chunks)

        # stream = None (SKIP) sudah ter-handle oleh is_skip check di atas

    # Narasi dulu, sitasi sesudahnya — memastikan counter per-doc berurutan
    return narasi_docs + sitasi_docs