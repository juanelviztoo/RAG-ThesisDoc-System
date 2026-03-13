from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class MetadataRouteResult:
    handled: bool
    intent: str
    answer_text: str
    meta: Dict[str, Any]


_PAGECOUNT_RE = re.compile(r"\b(jumlah|berapa)\b.*\bhalaman\b|\bpage\s*count\b", re.IGNORECASE)
_QUOTED_FILE_RE = re.compile(r'["“](.+?)["”]')  # ambil isi di antara tanda kutip
_PDFNAME_RE = re.compile(r"([a-z0-9_\-]+\.pdf)\b", re.IGNORECASE)


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


def _find_doc_in_catalog(catalog: Dict[str, Any], doc_ref: str) -> Optional[Dict[str, Any]]:
    """
    Cari doc berdasarkan:
    - key doc_id (stem)
    - cocokkan source_file (tanpa .pdf)
    """
    if not catalog:
        return None
    if not doc_ref:
        return None

    key = _normalize_doc_key(doc_ref)
    # direct key match
    if key in catalog and isinstance(catalog[key], dict):
        return catalog[key]

    # cari lewat source_file
    key_l = key.lower()
    for _, v in catalog.items():
        if not isinstance(v, dict):
            continue
        sf = str(v.get("source_file", "") or "")
        if sf.lower().endswith(".pdf"):
            sf_key = sf[:-4].lower()
        else:
            sf_key = sf.lower()
        if sf_key == key_l:
            return v
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

    # Jika user tidak menyebut dokumen, minta spesifik
    if not doc_ref:
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

    entry = _find_doc_in_catalog(catalog, doc_ref)
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
    source_file = entry.get("source_file", f"{doc_ref}.pdf")
    doc_id = entry.get("doc_id", _normalize_doc_key(doc_ref))

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
        },
    )
