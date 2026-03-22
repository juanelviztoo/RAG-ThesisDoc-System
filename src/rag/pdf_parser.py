"""
src/rag/pdf_parser.py
=====================
V3.0a — Structure-Aware PDF Parser

Responsibility:
    - Menerima path PDF → mengembalikan List[ParsedSection]
    - Hybrid routing: filename-based routing dulu, regex fallback untuk PDF utuh
    - Ekstraksi metadata dokumen (judul, penulis, tahun, prodi) dari halaman awal
    - Deteksi batas bab menggunakan regex (untuk PDF utuh)
    - TIDAK menahu soal ChromaDB, chunking, atau embedding — murni parsing

Public API:
  parse_pdf(pdf_path, *, running_header_threshold=3, metadata_scan_pages=3)
        -> List[ParsedSection]
  detect_routing_mode(pdf_path)
        -> Literal["filename", "regex"]

Prinsip desain:
    - PyMuPDF (fitz) sebagai primary extractor, pypdf sebagai fallback
    - Filename routing dicheck terlebih dahulu (efisien untuk dataset asli per-BAB)
    - Regex parsing sebagai mekanisme utama untuk PDF utuh (justifikasi akademis)
    - Running header detection mencegah false-positive dari header berulang
    - TOC page detection mencegah false-positive dari halaman Daftar Isi
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

# ── Primary extractor: PyMuPDF ────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except ImportError:  # pragma: no cover
    _FITZ_OK = False
    warnings.warn(
        "[pdf_parser] PyMuPDF (fitz) tidak terinstall. "
        "Akan coba fallback ke pypdf saat parse_pdf() dipanggil.",
        ImportWarning,
        stacklevel=1,
    )

# ── Fallback extractor: pypdf ─────────────────────────────────────────────────
try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF_OK = True
except ImportError:  # pragma: no cover
    _PYPDF_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# Type aliases
# ═════════════════════════════════════════════════════════════════════════════

StreamType  = Literal["narasi", "sitasi"]
BabLabel    = Literal[
    "BAB_I", "BAB_II", "BAB_III", "BAB_IV", "BAB_V",
    "DAFTAR_PUSTAKA", "UNKNOWN", "SKIP",
]
RoutingMode = Literal["filename", "regex"]


# ═════════════════════════════════════════════════════════════════════════════
# Konstanta
# ═════════════════════════════════════════════════════════════════════════════

_MAX_HEADER_LINE_LEN = 80   # baris lebih panjang → tidak dianggap header bab
_MIN_TITLE_LINE_LEN  = 15   # baris lebih pendek  → diabaikan untuk ekstraksi judul

# Mapping numeral yang dikenali → BabLabel standar
_NUMERAL_TO_BAB: Dict[str, BabLabel] = {
    "I":   "BAB_I",
    "1":   "BAB_I",
    "II":  "BAB_II",
    "2":   "BAB_II",
    "III": "BAB_III",
    "3":   "BAB_III",
    "IV":  "BAB_IV",
    "4":   "BAB_IV",
    "V":   "BAB_V",
    "5":   "BAB_V",
}

# Mapping BabLabel → stream
# BAB_I–V  → narasi  |  DAFTAR_PUSTAKA → sitasi  |  UNKNOWN → narasi (conservative)
_LABEL_TO_STREAM: Dict[str, Optional[StreamType]] = {
    "BAB_I":          "narasi",
    "BAB_II":         "narasi",
    "BAB_III":        "narasi",
    "BAB_IV":         "narasi",
    "BAB_V":          "narasi",
    "DAFTAR_PUSTAKA": "sitasi",
    "UNKNOWN":        "narasi",
    "SKIP":           None,
}


# ═════════════════════════════════════════════════════════════════════════════
# Regex patterns
# ═════════════════════════════════════════════════════════════════════════════

# Header BAB — Roman (diurutkan panjang dulu agar tidak partial-match) atau Arab
_BAB_HEADER_RE = re.compile(
    r"^\s*BAB\s+(VIII|VII|VI|IV|V|III|II|I|[1-9])\b",
    re.IGNORECASE,
)

# Hanya "BAB" (baris terpisah — untuk format dua baris "BAB\nI\nPENDAHULUAN")
_BAB_ALONE_RE = re.compile(r"^\s*BAB\s*$", re.IGNORECASE)

# Numerals standalone (untuk baris setelah "BAB" saja)
_NUMERAL_STANDALONE_RE = re.compile(
    r"^(VIII|VII|VI|IV|V|III|II|I|[1-9])\b",
    re.IGNORECASE,
)

# DAFTAR PUSTAKA / REFERENCES
_DAFTAR_PUSTAKA_RE = re.compile(
    r"^\s*(?:DAFTAR\s+(?:PUSTAKA|REFERENSI)|REFERENCES?|BIBLIOGRAPHY)\s*$",
    re.IGNORECASE,
)

# LAMPIRAN / APPENDIX → SKIP
_LAMPIRAN_RE = re.compile(
    r"^\s*(?:LAMPIRAN|APPENDIX|APPENDICES)\b",
    re.IGNORECASE,
)

# Blocklist untuk ekstraksi judul
_JUDUL_BLOCKLIST_RE = re.compile(
    r"skripsi|tugas\s+akhir|tesis|disertasi|thesis"
    r"|universitas|fakultas|jurusan|program\s+studi|prodi"
    r"|nim\b|nip\b|nidn\b|oleh\b|disusun|diajukan"
    r"|sarjana|diploma|strata|gelar"
    r"|halaman|cover|lembar"
    r"|persetujuan|pengesahan|pernyataan|keaslian"
    r"|kata\s+pengantar|daftar\s+isi|abstrak|abstract"
    r"|pembimbing|penguji|dekan|ketua",
    re.IGNORECASE,
)

# Pola tahun
_TAHUN_RE = re.compile(r"\b(20\d{2})\b")

# Pola penulis — trigger keyword pada awal baris
_PENULIS_TRIGGER_RE = re.compile(
    r"^(?:oleh|disusun\s+oleh|dibuat\s+oleh|nama)\s*:?\s*$",
    re.IGNORECASE,
)
_PENULIS_INLINE_RE = re.compile(
    r"(?:oleh|disusun\s+oleh|dibuat\s+oleh|nama)\s*:\s*(.+)",
    re.IGNORECASE,
)
_NIM_LINE_RE = re.compile(r"\bnim\b|\bnip\b", re.IGNORECASE)

# Pola prodi — trigger keyword
_PRODI_TRIGGER_RE = re.compile(
    r"^(?:jurusan|program\s+studi|prodi|departemen|department)\s*:?\s*$",
    re.IGNORECASE,
)
_PRODI_INLINE_RE = re.compile(
    r"(?:jurusan|program\s+studi|prodi|departemen|department)\s*:?\s*(.+)",
    re.IGNORECASE,
)


# ═════════════════════════════════════════════════════════════════════════════
# Filename routing rules
# ═════════════════════════════════════════════════════════════════════════════
# Format: (compiled_pattern, stream, bab_label)
# Diurutkan: paling spesifik dulu (IV sebelum I, sitasi sebelum narasi, dst.)
_FILENAME_RULES: List[Tuple[re.Pattern, Optional[StreamType], BabLabel]] = [
    # ── Sitasi ────────────────────────────────────────────────────────────
    (re.compile(r"daftar[_\s\-]*pustaka",   re.I), "sitasi", "DAFTAR_PUSTAKA"),
    (re.compile(r"daftar[_\s\-]*referensi", re.I), "sitasi", "DAFTAR_PUSTAKA"),
    (re.compile(r"\breference",             re.I), "sitasi", "DAFTAR_PUSTAKA"),
    (re.compile(r"\bbibliograph",           re.I), "sitasi", "DAFTAR_PUSTAKA"),
    # ── Narasi: Roman (panjang dulu agar tidak partial-match) ────────────
    (re.compile(r"\bbab[_\s\-]*viii\b",    re.I), "narasi", "UNKNOWN"),
    (re.compile(r"\bbab[_\s\-]*vii\b",     re.I), "narasi", "UNKNOWN"),
    (re.compile(r"\bbab[_\s\-]*vi\b",      re.I), "narasi", "UNKNOWN"),
    (re.compile(r"\bbab[_\s\-]*iv\b",      re.I), "narasi", "BAB_IV"),
    (re.compile(r"\bbab[_\s\-]*v\b",       re.I), "narasi", "BAB_V"),
    (re.compile(r"\bbab[_\s\-]*iii\b",     re.I), "narasi", "BAB_III"),
    (re.compile(r"\bbab[_\s\-]*ii\b",      re.I), "narasi", "BAB_II"),
    (re.compile(r"\bbab[_\s\-]*i\b",       re.I), "narasi", "BAB_I"),
    # ── Narasi: Arab ──────────────────────────────────────────────────────
    (re.compile(r"\bbab[_\s\-]*1\b",       re.I), "narasi", "BAB_I"),
    (re.compile(r"\bbab[_\s\-]*2\b",       re.I), "narasi", "BAB_II"),
    (re.compile(r"\bbab[_\s\-]*3\b",       re.I), "narasi", "BAB_III"),
    (re.compile(r"\bbab[_\s\-]*4\b",       re.I), "narasi", "BAB_IV"),
    (re.compile(r"\bbab[_\s\-]*5\b",       re.I), "narasi", "BAB_V"),
    # ── Skip sections ─────────────────────────────────────────────────────
    (re.compile(r"halaman[_\s\-]*depan",   re.I), None, "SKIP"), 
    (re.compile(r"halaman[_\s\-]*awal",    re.I), None, "SKIP"),
    (re.compile(r"\bcover\b",              re.I), None, "SKIP"),
    (re.compile(r"halaman[_\s\-]*judul",   re.I), None, "SKIP"),
    (re.compile(r"\blampiran\b",           re.I), None, "SKIP"),
    (re.compile(r"\bappendix\b",           re.I), None, "SKIP"),
    (re.compile(r"\babstrak\b",            re.I), None, "SKIP"),
    (re.compile(r"\babstract\b",           re.I), None, "SKIP"),
    (re.compile(r"kata[_\s\-]*pengantar",  re.I), None, "SKIP"),
    (re.compile(r"daftar[_\s\-]*isi",      re.I), None, "SKIP"),
    (re.compile(r"daftar[_\s\-]*gambar",   re.I), None, "SKIP"),
    (re.compile(r"daftar[_\s\-]*tabel",    re.I), None, "SKIP"),
    (re.compile(r"\bpengesahan\b",         re.I), None, "SKIP"),
    (re.compile(r"\bpernyataan\b",         re.I), None, "SKIP"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DocMetadata:
    """
    Metadata level dokumen — nilainya sama untuk semua section dalam satu PDF.
    Dipakai untuk ChromaDB metadata dan doc_catalog.json.
    """
    judul:   str
    penulis: str
    tahun:   Optional[int]
    prodi:   Optional[str]


@dataclass
class ParsedSection:
    """
    Satu section dokumen hasil parsing (satu BAB, Daftar Pustaka, atau SKIP).

    Attributes:
        stream       : "narasi" | "sitasi" | None  (None jika is_skip=True)
        bab_label    : label standar section ini
        pages        : list of (page_num, page_text) — 0-indexed page number,
                        konsisten dengan PyMuPDF / PyMuPDFLoader di pipeline lama
        page_start   : page_num halaman pertama section (0-indexed)
        page_end     : page_num halaman terakhir section (0-indexed, inclusive)
        doc_meta     : metadata dokumen (judul, penulis, tahun, prodi)
        routing_mode : "filename" atau "regex" — untuk logging/audit di ingest.py
    """
    stream:       Optional[StreamType]
    bab_label:    BabLabel
    pages:        List[Tuple[int, str]]
    page_start:   int
    page_end:     int
    doc_meta:     DocMetadata
    routing_mode: RoutingMode

    @property
    def is_skip(self) -> bool:
        """True jika section ini tidak perlu di-index ke ChromaDB."""
        return self.bab_label == "SKIP" or self.stream is None

    @property
    def full_text(self) -> str:
        """Gabung semua teks halaman menjadi satu string (utility untuk debug)."""
        return "\n\n".join(txt for _, txt in self.pages if txt.strip())

    @property
    def n_pages(self) -> int:
        """Jumlah halaman dalam section ini."""
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        """Total karakter teks dalam section ini (kasar)."""
        return sum(len(txt) for _, txt in self.pages)


# ── Internal boundary event (tidak diekspos ke luar modul) ───────────────────

@dataclass
class _BoundaryEvent:
    page_num:  int
    bab_label: BabLabel
    raw_line:  str  # untuk logging/debug


# ═════════════════════════════════════════════════════════════════════════════
# Page extraction — fitz primary, pypdf fallback
# ═════════════════════════════════════════════════════════════════════════════

def _extract_pages_fitz(pdf_path: Path) -> List[Tuple[int, str]]:
    """Ekstrak (page_num, text) dari semua halaman menggunakan PyMuPDF."""
    pages: List[Tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text: str = page.get_text("text")  # type: ignore[attr-defined]
            if text and text.strip():
                pages.append((page_num, text))
    return pages


def _extract_pages_pypdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Ekstrak (page_num, text) dari semua halaman menggunakan pypdf (fallback)."""
    pages: List[Tuple[int, str]] = []
    reader = _PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num, text))
    return pages


def _extract_pages(path: Path) -> List[Tuple[int, str]]:
    """Coba fitz dulu, fallback ke pypdf jika gagal."""
    if _FITZ_OK:
        try:
            return _extract_pages_fitz(path)
        except Exception as exc:
            warnings.warn(
                f"[pdf_parser] fitz gagal untuk '{path.name}': {exc}. "
                "Mencoba fallback ke pypdf...",
                UserWarning,
                stacklevel=4,
            )

    if _PYPDF_OK:
        try:
            return _extract_pages_pypdf(path)
        except Exception as exc:
            raise RuntimeError(
                f"Kedua extractor (fitz + pypdf) gagal untuk '{path.name}': {exc}"
            ) from exc

    raise RuntimeError(
        "Tidak ada PDF extractor yang tersedia. Install: pip install pymupdf"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Filename routing
# ═════════════════════════════════════════════════════════════════════════════

def _check_filename_routing(
    pdf_path: Path,
) -> Optional[Tuple[Optional[StreamType], BabLabel]]:
    """
    Cek apakah stem nama file cocok dengan rule filename routing.
    Returns (stream, bab_label) untuk match pertama, atau None jika tidak ada.
    """
    stem = pdf_path.stem
    for pattern, stream, bab_label in _FILENAME_RULES:
        if pattern.search(stem):
            return (stream, bab_label)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Boundary detection (untuk regex parsing)
# ═════════════════════════════════════════════════════════════════════════════

def _check_bab_line(line: str) -> Optional[BabLabel]:
    """
    Periksa apakah satu baris adalah header BAB yang valid (satu baris penuh).
    Constraint: panjang baris <= _MAX_HEADER_LINE_LEN untuk menghindari false positive.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADER_LINE_LEN:
        return None

    match = _BAB_HEADER_RE.match(stripped)
    if not match:
        return None

    numeral = match.group(1).upper()
    return _NUMERAL_TO_BAB.get(numeral, "UNKNOWN")


def _check_section_break_line(line: str) -> Optional[BabLabel]:
    """
    Periksa apakah baris adalah penanda DAFTAR PUSTAKA atau LAMPIRAN.
    Returns label-nya atau None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADER_LINE_LEN:
        return None
    if _DAFTAR_PUSTAKA_RE.match(stripped):
        return "DAFTAR_PUSTAKA"
    if _LAMPIRAN_RE.match(stripped):
        return "SKIP"
    return None


def _scan_boundaries(
    pages: List[Tuple[int, str]],
    running_header_threshold: int = 3,
) -> List[_BoundaryEvent]:
    """
    Scan semua halaman untuk mendeteksi boundary events (header BAB / DAFTAR PUSTAKA).

    Fitur:
    - Deteksi header satu baris: "BAB I", "BAB II TINJAUAN PUSTAKA"
    - Deteksi header dua baris: "BAB" (satu baris) + "I" (baris berikutnya)
    - TOC page detection: halaman dengan >= 3 distinct BAB labels = Daftar Isi → skip
    - Running header filtering: header sama muncul > threshold kali → ambil pertama saja
    """
    # ── Scan per halaman, kumpulkan events ───────────────────────────────────
    events_per_page: Dict[int, List[_BoundaryEvent]] = {}

    for page_num, page_text in pages:
        page_events: List[_BoundaryEvent] = []
        lines = page_text.split("\n")
        found_bab_alone = False  # untuk two-line "BAB\nI" detection

        for line in lines:
            stripped = line.strip()
            if not stripped:
                found_bab_alone = False
                continue

            # ── Dua-baris: baris sebelumnya adalah "BAB" saja ────────────
            if found_bab_alone:
                numeral_match = _NUMERAL_STANDALONE_RE.match(stripped)
                if numeral_match and len(stripped) <= _MAX_HEADER_LINE_LEN:
                    numeral = numeral_match.group(1).upper()
                    label   = _NUMERAL_TO_BAB.get(numeral, "UNKNOWN")
                    page_events.append(_BoundaryEvent(page_num, label, f"BAB {stripped}"))
                found_bab_alone = False
                continue

            # ── Cek "BAB" sendiri di satu baris ──────────────────────────
            if _BAB_ALONE_RE.match(stripped):
                found_bab_alone = True
                continue

            # ── Header BAB satu baris ─────────────────────────────────────
            bab_label = _check_bab_line(stripped)
            if bab_label:
                page_events.append(_BoundaryEvent(page_num, bab_label, stripped))
                continue

            # ── DAFTAR PUSTAKA / LAMPIRAN ─────────────────────────────────
            section_label = _check_section_break_line(stripped)
            if section_label:
                page_events.append(_BoundaryEvent(page_num, section_label, stripped))

        events_per_page[page_num] = page_events

    # ── Deteksi TOC pages: banyak distinct BAB labels di satu halaman ────────
    # Heuristic: >= 3 distinct BAB labels dalam 1 halaman = kemungkinan Daftar Isi
    toc_pages: set = set()
    for page_num, evs in events_per_page.items():
        distinct_bab = {
            ev.bab_label for ev in evs
            if ev.bab_label not in ("DAFTAR_PUSTAKA", "SKIP", "UNKNOWN")
        }
        if len(distinct_bab) >= 3:
            toc_pages.add(page_num)

    if toc_pages:
        warnings.warn(
            f"[pdf_parser] Halaman {sorted(toc_pages)} terdeteksi sebagai Daftar Isi "
            f"(>= 3 BAB headers di satu halaman) → boundary events dari halaman ini diabaikan.",
            UserWarning,
            stacklevel=3,
        )

    # ── Kumpulkan semua events, skip halaman TOC ──────────────────────────────
    raw_events: List[_BoundaryEvent] = []
    for page_num in sorted(events_per_page.keys()):
        if page_num in toc_pages:
            continue
        raw_events.extend(events_per_page[page_num])

    # ── Filter running headers ────────────────────────────────────────────────
    # Hitung kemunculan per label
    label_count: Dict[str, int] = {}
    for ev in raw_events:
        label_count[ev.bab_label] = label_count.get(ev.bab_label, 0) + 1

    # Jika label muncul > threshold kali → hanya ambil kemunculan pertama
    filtered: List[_BoundaryEvent] = []
    seen_running: set = set()

    for ev in raw_events:
        count = label_count.get(ev.bab_label, 0)
        if count > running_header_threshold:
            if ev.bab_label not in seen_running:
                filtered.append(ev)
                seen_running.add(ev.bab_label)
            # else: skip — running header duplikat
        else:
            filtered.append(ev)

    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# Section building
# ═════════════════════════════════════════════════════════════════════════════

def _build_sections_from_boundaries(
    pages: List[Tuple[int, str]],
    boundaries: List[_BoundaryEvent],
    doc_meta: DocMetadata,
    routing_mode: RoutingMode,
) -> List[ParsedSection]:
    """
    Bangun List[ParsedSection] dari halaman dan boundary events.

    Logic:
    - Halaman sebelum boundary pertama → SKIP (cover/front matter)
    - Setiap boundary mendefinisikan awal section baru
    - Section terakhir meluas sampai akhir dokumen
    - Jika tidak ada boundary → seluruh dokumen = satu section UNKNOWN (narasi)
    """
    if not pages:
        return []

    all_page_nums   = [pn for pn, _ in pages]
    page_text_map   = {pn: txt for pn, txt in pages}
    first_page      = all_page_nums[0]
    last_page       = all_page_nums[-1]

    # ── Dedup: jika label sama & page sama, ambil pertama (idempoten) ─────────
    seen_keys: set = set()
    deduped: List[_BoundaryEvent] = []
    for ev in boundaries:
        key = (ev.page_num, ev.bab_label)
        if key not in seen_keys:
            deduped.append(ev)
            seen_keys.add(key)
    boundaries = deduped

    sections: List[ParsedSection] = []

    # ── Kasus 0: tidak ada boundary ditemukan ─────────────────────────────────
    if not boundaries:
        warnings.warn(
            "[pdf_parser] Tidak ada header BAB ditemukan. "
            "Seluruh dokumen diperlakukan sebagai satu section UNKNOWN (narasi).",
            UserWarning,
            stacklevel=3,
        )
        all_pages_data = [(pn, page_text_map[pn]) for pn in all_page_nums]
        sections.append(ParsedSection(
            stream="narasi",
            bab_label="UNKNOWN",
            pages=all_pages_data,
            page_start=first_page,
            page_end=last_page,
            doc_meta=doc_meta,
            routing_mode=routing_mode,
        ))
        return sections

    # ── Pre-section (halaman sebelum boundary pertama) → SKIP ─────────────────
    first_boundary_page = boundaries[0].page_num
    pre_page_nums = [pn for pn in all_page_nums if pn < first_boundary_page]
    if pre_page_nums:
        pre_pages = [(pn, page_text_map[pn]) for pn in pre_page_nums]
        sections.append(ParsedSection(
            stream=None,
            bab_label="SKIP",
            pages=pre_pages,
            page_start=pre_page_nums[0],
            page_end=pre_page_nums[-1],
            doc_meta=doc_meta,
            routing_mode=routing_mode,
        ))

    # ── Build section per boundary ─────────────────────────────────────────────
    for i, ev in enumerate(boundaries):
        section_start = ev.page_num
        if i + 1 < len(boundaries):
            section_end = boundaries[i + 1].page_num - 1
        else:
            section_end = last_page

        section_page_nums = [
            pn for pn in all_page_nums
            if section_start <= pn <= section_end
        ]
        if not section_page_nums:
            continue

        section_pages = [(pn, page_text_map[pn]) for pn in section_page_nums]
        bab_label: BabLabel = ev.bab_label
        stream = _LABEL_TO_STREAM.get(bab_label)

        sections.append(ParsedSection(
            stream=stream,  # type: ignore[arg-type]
            bab_label=bab_label,
            pages=section_pages,
            page_start=section_page_nums[0],
            page_end=section_page_nums[-1],
            doc_meta=doc_meta,
            routing_mode=routing_mode,
        ))

    return sections


# ═════════════════════════════════════════════════════════════════════════════
# Metadata extraction
# ═════════════════════════════════════════════════════════════════════════════

def _extract_judul(lines: List[str], pdf_path: Optional[Path]) -> str:
    """
    Ekstrak judul skripsi dari baris-baris teks halaman awal.
    Strategi: ambil baris terpanjang yang bukan metadata (NIM, Universitas, dll.)
    """
    candidates: List[str] = []
    for ln in lines:
        if len(ln) < _MIN_TITLE_LINE_LEN:
            continue
        if _JUDUL_BLOCKLIST_RE.search(ln):
            continue
        # Harus mengandung setidaknya 2 kata
        if len(ln.split()) < 2:
            continue
        # Tidak boleh murni angka / hanya simbol
        if re.match(r"^[\d\s\-_./,:;()]+$", ln):
            continue
        candidates.append(ln)

    if not candidates:
        # Fallback: gunakan nama file
        if pdf_path:
            stem = pdf_path.stem
            return re.sub(r"[_\-]+", " ", stem).title()
        return "Judul Tidak Terdeteksi"

    # Urutkan dari terpanjang
    candidates_sorted = sorted(candidates, key=len, reverse=True)
    judul = candidates_sorted[0]

    # Coba merge dengan kandidat kedua jika judul pertama pendek (< 40 char)
    # dan kandidat kedua cukup panjang (> 60% panjang kandidat pertama)
    if (
        len(candidates_sorted) > 1
        and len(judul) < 40
        and len(candidates_sorted[1]) > 0.6 * len(judul)
    ):
        judul = judul + " " + candidates_sorted[1]

    return judul.strip()


def _extract_penulis(lines: List[str]) -> str:
    """
    Ekstrak nama penulis dari baris-baris teks halaman awal.
    Mencari kata kunci "Oleh", "Disusun oleh", dll.
    """
    for i, ln in enumerate(lines):
        if _PENULIS_TRIGGER_RE.match(ln):
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                # Pastikan bukan baris NIM/NIP
                if not _NIM_LINE_RE.search(candidate) and len(candidate) > 3:
                    return candidate

    # Fallback: cari pola inline "Oleh: NAMA"
    for ln in lines:
        match = _PENULIS_INLINE_RE.search(ln)
        if match:
            val = match.group(1).strip()
            if val and not _NIM_LINE_RE.search(val):
                return val

    return "Penulis Tidak Terdeteksi"


def _extract_tahun(text: str) -> Optional[int]:
    """
    Ekstrak tahun (20XX) dari teks halaman awal.
    Ambil tahun yang paling sering muncul (biasanya tahun terbit).
    """
    matches = _TAHUN_RE.findall(text)
    if not matches:
        return None
    counter = Counter(matches)
    most_common_year = counter.most_common(1)[0][0]
    return int(most_common_year)


def _extract_prodi(lines: List[str]) -> Optional[str]:
    """
    Ekstrak nama program studi dari baris-baris teks halaman awal.
    """
    for i, ln in enumerate(lines):
        if _PRODI_TRIGGER_RE.match(ln):
            if i + 1 < len(lines):
                val = lines[i + 1].strip()
                if val:
                    return val

    for ln in lines:
        match = _PRODI_INLINE_RE.search(ln)
        if match:
            val = match.group(1).strip()
            if val:
                return val

    return None


def _extract_doc_metadata(
    pages: List[Tuple[int, str]],
    scan_pages: int = 3,
    pdf_path: Optional[Path] = None,
) -> DocMetadata:
    """
    Ekstrak metadata level dokumen dari halaman-halaman awal PDF.
    Akurasi tidak harus sempurna untuk V3.0a — yang penting field tersedia.
    V3.0b (metadata self-querying) akan memperkuat ini nanti.
    """
    # Ambil teks dari halaman awal (0-indexed)
    scan_parts = [txt for _, txt in pages[:scan_pages] if txt.strip()]
    if not scan_parts:
        fallback_name = pdf_path.stem if pdf_path else "unknown"
        return DocMetadata(
            judul=fallback_name,
            penulis="Penulis Tidak Terdeteksi",
            tahun=None,
            prodi=None,
        )

    scan_text = "\n".join(scan_parts)
    lines     = [ln.strip() for ln in scan_text.split("\n") if ln.strip()]

    return DocMetadata(
        judul=_extract_judul(lines, pdf_path),
        penulis=_extract_penulis(lines),
        tahun=_extract_tahun(scan_text),
        prodi=_extract_prodi(lines),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Parsing via filename routing
# ═════════════════════════════════════════════════════════════════════════════

def _parse_via_filename(
    pdf_path: Path,
    pages: List[Tuple[int, str]],
    doc_meta: DocMetadata,
    routing_result: Tuple[Optional[StreamType], BabLabel],
) -> List[ParsedSection]:
    """
    Buat single ParsedSection berdasarkan filename routing decision.
    Seluruh isi PDF diperlakukan sebagai satu section dengan label dari routing.
    """
    if not pages:
        return []

    stream, bab_label = routing_result
    all_page_nums     = [pn for pn, _ in pages]

    return [
        ParsedSection(
            stream=stream,
            bab_label=bab_label,
            pages=pages,
            page_start=all_page_nums[0],
            page_end=all_page_nums[-1],
            doc_meta=doc_meta,
            routing_mode="filename",
        )
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def detect_routing_mode(pdf_path: Union[str, Path]) -> RoutingMode:
    """
    Tentukan routing mode untuk file PDF ini tanpa membuka filenya.

    Returns:
        "filename" jika nama file cocok dengan rule filename routing
        "regex"    jika tidak ada match → akan pakai regex parsing saat parse_pdf()
    """
    path   = Path(pdf_path)
    result = _check_filename_routing(path)
    return "filename" if result is not None else "regex"


def parse_pdf(
    pdf_path: Union[str, Path],
    *,
    running_header_threshold: int = 3,
    metadata_scan_pages: int = 3,
) -> List[ParsedSection]:
    """
    Entry point utama pdf_parser — parse satu file PDF menjadi List[ParsedSection].

    Hybrid routing:
      1. Cek nama file → jika match filename rules → route langsung (cepat)
      2. Jika tidak match → regex parsing untuk deteksi batas bab (PDF utuh)

    Args:
        pdf_path                 : Path ke file PDF (str atau Path)
        running_header_threshold : Header bab yang sama muncul > N kali
                                   → dianggap running header, ambil pertama saja.
                                   Default 3, dapat di-override dari cfg["parsing"]
        metadata_scan_pages      : Jumlah halaman awal yang di-scan untuk
                                   metadata (judul/penulis/tahun/prodi). Default 3.

    Returns:
        List[ParsedSection] — termasuk section SKIP (cover, lampiran, dll.)
        Caller (ingest.py) bertanggung jawab mem-filter section.is_skip == True.
        Urutan: pre-section SKIP (jika ada) → BAB_I → ... → DAFTAR_PUSTAKA → ...

    Raises:
        FileNotFoundError : file tidak ditemukan
        RuntimeError      : PDF tidak bisa dibaca sama sekali (semua extractor gagal
                            atau tidak ada teks yang bisa diekstrak)
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"[pdf_parser] File PDF tidak ditemukan: {path}")

    # ── Ekstrak semua halaman ──────────────────────────────────────────────────
    pages = _extract_pages(path)
    if not pages:
        raise RuntimeError(
            f"[pdf_parser] Tidak ada teks yang bisa diekstrak dari '{path.name}'. "
            "Kemungkinan: file rusak, PDF hasil scan tanpa OCR, atau PDF terproteksi."
        )

    # ── Ekstrak metadata dokumen dari halaman awal ────────────────────────────
    doc_meta = _extract_doc_metadata(pages, metadata_scan_pages, path)

    # ── Routing decision ──────────────────────────────────────────────────────
    filename_result = _check_filename_routing(path)

    if filename_result is not None:
        # Jalur B: Filename routing — langsung assign stream tanpa scan teks
        return _parse_via_filename(path, pages, doc_meta, filename_result)
    else:
        # Jalur A: Regex parsing — scan teks untuk deteksi batas bab
        boundaries = _scan_boundaries(pages, running_header_threshold)
        return _build_sections_from_boundaries(pages, boundaries, doc_meta, "regex")