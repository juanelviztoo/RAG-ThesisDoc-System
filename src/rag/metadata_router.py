from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class MetadataRouteResult:
    handled: bool
    intent: str
    answer_text: str
    meta: Dict[str, Any]


_PAGECOUNT_RE = re.compile(r"\b(jumlah|berapa)\b.*\bhalaman\b|\bpage\s*count\b", re.IGNORECASE)
_QUOTED_FILE_RE = re.compile(r'["“](.+?)["”]')  # ambil isi di antara tanda kutip
_PDFNAME_RE = re.compile(r"([a-z0-9_\-]+\.pdf)\b", re.IGNORECASE)

_DOCREF_AFTER_NOUN_RE = re.compile(
    r"\b(?:dokumen|file|pdf|skripsi)\s+([a-z0-9][a-z0-9 _\-/]{1,100})",
    re.IGNORECASE,
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

_MATCH_STOPWORDS: Set[str] = {
    "berapa",
    "jumlah",
    "halaman",
    "dokumen",
    "file",
    "pdf",
    "skripsi",
    "yang",
    "membahas",
    "tentang",
    "pada",
    "di",
    "ke",
    "dari",
    "dan",
    "atau",
}

def _normalize_doc_key(s: str) -> str:
    t = (s or "").strip().strip('"').strip("'")
    t = t.replace("\\", "/")
    # ambil nama file saja kalau ada path
    if "/" in t:
        t = t.split("/")[-1]
    # buang .pdf -> doc_id style
    if t.lower().endswith(".pdf"):
        t = t[:-4]
    return t.strip()


def _normalize_loose_text(s: str) -> str:
    """
    Normalisasi longgar untuk pencocokan alias natural-language:
    - lowercase
    - buang path
    - buang ekstensi .pdf
    - hilangkan semua karakter non-alnum
    """
    t = _normalize_doc_key(s or "").lower()
    t = _NON_ALNUM_RE.sub("", t)
    return t.strip()


def _tokenize_loose_text(s: str) -> Set[str]:
    """
    Token longgar untuk fallback overlap matching.
    """
    raw = re.split(r"[^a-z0-9]+", (s or "").lower())
    out: Set[str] = set()
    for tok in raw:
        tok = tok.strip()
        if not tok or tok in _MATCH_STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        out.add(tok)
    return out


def _catalog_aliases(entry: Dict[str, Any]) -> Set[str]:
    """
    Bangun himpunan alias yang masuk akal dari satu entry katalog.
    Digunakan untuk resolve query alami seperti:
    - "E-Ticketing"
    - "Sistem Monitoring TA"
    - "Information System MUA"
    """
    aliases: Set[str] = set()

    for key in ("doc_id", "source_file", "relative_path", "judul", "penulis"):
        val = str(entry.get(key, "") or "").strip()
        if not val:
            continue

        aliases.add(val)
        aliases.add(_normalize_doc_key(val))

        stem = _normalize_doc_key(val)
        parts = [p for p in re.split(r"[_\-/]+", stem) if p]
        aliases.update(parts)

        if len(parts) >= 2:
            aliases.add(" ".join(parts))

    return {a.strip() for a in aliases if str(a).strip()}


def _score_alias_match(query: str, alias: str) -> int:
    """
    Semakin tinggi skor, semakin kuat kecocokannya.
    """
    q_norm = _normalize_loose_text(query)
    a_norm = _normalize_loose_text(alias)

    if not q_norm or not a_norm:
        return 0

    if q_norm == a_norm:
        return 100

    if q_norm in a_norm or a_norm in q_norm:
        shorter = min(len(q_norm), len(a_norm))
        return 85 if shorter >= 5 else 70

    q_tokens = _tokenize_loose_text(query)
    a_tokens = _tokenize_loose_text(alias)
    if not q_tokens or not a_tokens:
        return 0

    overlap = q_tokens & a_tokens
    if not overlap:
        return 0

    ratio = len(overlap) / max(1, len(q_tokens))
    score = 40 + int(ratio * 40)

    if max(len(t) for t in overlap) >= 8:
        score += 10

    return min(score, 79)


def _extract_natural_doc_ref_candidates(q: str) -> List[str]:
    """
    Ambil kandidat nama dokumen dari query alami.
    Contoh:
    - "Berapa halaman dokumen E-Ticketing?" -> ["E-Ticketing"]
    """
    text = (q or "").strip()
    if not text:
        return []

    candidates: List[str] = []

    m = _DOCREF_AFTER_NOUN_RE.search(text)
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"[\?\!\.,;:]+$", "", cand).strip()
        if cand:
            candidates.append(cand)

    return candidates


def is_page_count_query(q: str) -> bool:
    return bool(_PAGECOUNT_RE.search(q or ""))


def extract_doc_ref(q: str) -> Optional[str]:
    """
    Coba ekstrak doc ref (filename/doc_id) dari query.
    Prioritas:
    1) teks di dalam kutip
    2) token xxx.pdf
    3) fallback: None (minta user menyebutkan nama dokumen)
    """
    text = q or ""

    m = _QUOTED_FILE_RE.search(text)
    if m:
        cand = m.group(1).strip()
        if cand:
            return _normalize_doc_key(cand)

    m2 = _PDFNAME_RE.search(text)
    if m2:
        return _normalize_doc_key(m2.group(1))

    return None


def load_doc_catalog(
    persist_dir: Path, filename: str = "doc_catalog.json"
) -> Tuple[Dict[str, Any], Path]:
    persist_dir = Path(persist_dir)
    catalog_path = persist_dir / filename
    if not catalog_path.exists():
        return {}, catalog_path
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data, catalog_path
        return {}, catalog_path
    except Exception:
        return {}, catalog_path


def load_docmeta_sidecar(
    persist_dir: Path,
    *,
    filename: str = "docmeta_sidecar.json",
) -> Tuple[Dict[str, Any], Path]:
    """
    Loader additive untuk metadata display sidecar Tahap 6.

    Berbeda dari doc_catalog:
    - doc_catalog.json = raw metadata / source of truth legacy
    - docmeta_sidecar.json = metadata display-safe hasil normalisasi
    """
    sidecar_path = Path(persist_dir) / filename

    if not sidecar_path.exists():
        return {}, sidecar_path

    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, sidecar_path

    if not isinstance(data, dict):
        return {}, sidecar_path

    return data, sidecar_path


def _find_doc_in_catalog(catalog: Dict[str, Any], doc_ref: str) -> Optional[Dict[str, Any]]:
    """
    Cari dokumen berdasarkan:
    1) exact key/doc_id match
    2) exact source_file stem match
    3) alias natural-language berbasis doc_id/source_file/judul/penulis
    """
    if not catalog or not doc_ref:
        return None

    key = _normalize_doc_key(doc_ref)

    # 1) direct key match
    if key in catalog and isinstance(catalog[key], dict):
        return catalog[key]

    # 2) exact source_file stem match
    key_l = key.lower()
    for _, v in catalog.items():
        if not isinstance(v, dict):
            continue
        sf = str(v.get("source_file", "") or "")
        sf_key = sf[:-4].lower() if sf.lower().endswith(".pdf") else sf.lower()
        if sf_key == key_l:
            return v

    # 3) alias / fuzzy-ish match
    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0

    for _, v in catalog.items():
        if not isinstance(v, dict):
            continue

        entry_best = 0
        for alias in _catalog_aliases(v):
            entry_best = max(entry_best, _score_alias_match(doc_ref, alias))

        if entry_best > best_score:
            best_score = entry_best
            best_entry = v

    # threshold konservatif supaya tidak terlalu gampang salah resolve
    if best_score >= 80:
        return best_entry

    return None


def maybe_route_metadata_query(
    user_query: str,
    *,
    persist_dir: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[MetadataRouteResult]:
    """
    Routing metadata minimal untuk V1.5:
    - intent: page_count (jumlah halaman PDF)
    """
    cfg = cfg or {}
    md_cfg = cfg.get("metadata_routing", {}) or {}
    enabled = bool(md_cfg.get("enabled", False))
    if not enabled:
        return None

    if not is_page_count_query(user_query):
        return None

    filename = str(md_cfg.get("doc_catalog_filename", "doc_catalog.json"))
    catalog, catalog_path = load_doc_catalog(Path(persist_dir), filename=filename)

    doc_ref = extract_doc_ref(user_query)
    entry: Optional[Dict[str, Any]] = None
    resolved_via = "explicit_doc_ref"

    if doc_ref:
        entry = _find_doc_in_catalog(catalog, doc_ref)

    # Fallback: coba resolve alias natural-language dari query
    if not entry:
        for cand in _extract_natural_doc_ref_candidates(user_query):
            cand = cand.strip()
            if not cand:
                continue

            # simpan kandidat natural-language pertama sebagai doc_ref fallback,
            # supaya kalau tidak ketemu di katalog, reason bisa menjadi not_in_catalog
            # alih-alih missing_doc_ref
            if not doc_ref:
                doc_ref = cand

            entry = _find_doc_in_catalog(catalog, cand)
            if entry:
                doc_ref = cand
                resolved_via = "natural_language_alias"
                break

    # Jika tetap tidak ada referensi yang bisa dipetakan
    if not doc_ref and not entry:
        answer = (
            "Jawaban: Untuk menjawab jumlah halaman, sebutkan nama dokumen (filename/doc_id).\n"
            "Bukti:\n"
            f"- doc_catalog: {catalog_path.name} (dibuat saat ingest)\n"
        )
        return MetadataRouteResult(
            handled=True,
            intent="page_count",
            answer_text=answer,
            meta={
                "catalog_path": str(catalog_path),
                "doc_ref": None,
                "found": False,
                "reason": "missing_doc_ref",
            },
        )

    if not entry:
        answer = (
            f'Jawaban: Dokumen "{doc_ref}" tidak ditemukan pada katalog ingest.\n'
            "Bukti:\n"
            f"- doc_catalog: {catalog_path.name}\n"
        )
        return MetadataRouteResult(
            handled=True,
            intent="page_count",
            answer_text=answer,
            meta={
                "catalog_path": str(catalog_path),
                "doc_ref": doc_ref,
                "found": False,
                "reason": "not_in_catalog",
            },
        )

    page_count = entry.get("page_count", None)
    source_file = str(entry.get("source_file", f"{doc_ref or 'unknown'}.pdf"))

    doc_ref_for_key = doc_ref or source_file
    doc_id = str(entry.get("doc_id") or _normalize_doc_key(doc_ref_for_key))

    if page_count is None:
        answer = (
            f'Jawaban: Katalog ingest untuk "{source_file}" tidak memiliki field page_count.\n'
            "Bukti:\n"
            f"- doc_catalog: {catalog_path.name}\n"
        )
        return MetadataRouteResult(
            handled=True,
            intent="page_count",
            answer_text=answer,
            meta={
                "catalog_path": str(catalog_path),
                "doc_ref": doc_ref,
                "doc_id": doc_id,
                "source_file": source_file,
                "found": True,
                "reason": "missing_page_count_field",
            },
        )

    answer = (
        f'Jawaban: Dokumen "{source_file}" memiliki {int(page_count)} halaman.\n'
        "Bukti:\n"
        f"- doc_catalog: {catalog_path.name} | doc_id={doc_id} | page_count={int(page_count)}\n"
    )
    return MetadataRouteResult(
        handled=True,
        intent="page_count",
        answer_text=answer,
        meta={
            "catalog_path": str(catalog_path),
            "doc_ref": doc_ref,
            "doc_id": doc_id,
            "source_file": source_file,
            "page_count": int(page_count),
            "found": True,
            "reason": "ok",
            "resolved_via": resolved_via,
        },
    )
