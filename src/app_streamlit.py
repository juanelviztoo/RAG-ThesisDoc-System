from __future__ import annotations

import copy
import html as _html_mod
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.app_ui_render import (
    _prepare_jawaban_markdown,
    build_contexts_with_meta,
    build_ctx_rows,
    build_retrieval_only_answer,
    compute_retrieval_stats,
    dist_to_sim,
    extract_terms_from_query,
    extract_terms_from_text,
    is_global_not_found_answer,
    node_matches_filter,
    parse_answer,
    render_answer_card_header,
    render_bukti_section,
    render_ctx_mapping_header,
    render_ctx_mapping_table,
    render_hybrid_source_score_pills,
    render_pdf_viewer_header,
    render_processing_box,
    render_rerank_before_after_panel,
    render_rerank_summary_box,
    render_source_score_pills,
    render_sources_header,
    render_verification_box,
    render_viewing_banner,
    sim_color_badge,
)
from src.core.config import ConfigPaths, load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager
from src.core.ui_utils import (
    copy_to_clipboard_button,
    download_file_button,
    download_logs_zip_button,
    render_pdf_viewer,
    resolve_pdf_path,
    scroll_to_anchor,
)
from src.rag.fusion_rrf import rrf_fusion
from src.rag.generate import (
    build_generation_meta,
    build_verification_meta,
    contextualize_question,
    generate_answer,
)
from src.rag.metadata_router import (
    load_doc_catalog,
    load_docmeta_sidecar,
    maybe_route_metadata_query,
)
from src.rag.method_detection import (
    METHOD_PATTERNS as _METHOD_PATTERNS,
)
from src.rag.method_detection import (
    build_intent_plan,
    detect_methods_in_contexts,
    detect_methods_in_text,
    has_steps_signal,
    is_anaphora_question,
    is_citation_query,
    is_method_question,
    is_multi_doc_question,
    is_multi_target_question,
    is_steps_question,
    node_supports_method_for_coverage,
    pretty_method_name,
)
from src.rag.method_detection import (
    docid_hit_for_method as _docid_hit_for_method,
)
from src.rag.reranker import (  # ← V3.0c: Cross-Encoder Reranking
    get_reranker_info,
    is_reranker_loaded,
    load_reranker,
    unload_reranker,
)
from src.rag.reranker import (
    rerank as rerank_nodes,
)
from src.rag.retrieve_dense import retrieve_dense
from src.rag.retrieve_sparse import sparse_retrieve_bm25
from src.rag.self_query import (  # ← V3.0b: Metadata Self-Querying
    SelfQueryResult,
    build_self_query,
    check_filter_has_results,
)

# =========================
# Doc-focus lock helpers
# =========================

def _node_doc_id(n: Any) -> str:
    return str(
        getattr(n, "doc_id", "")
        or (n.metadata.get("doc_id") if getattr(n, "metadata", None) else "")
        or "unknown"
    )


def _node_text(n: Any) -> str:
    return str(getattr(n, "text", "") or "")


def build_method_doc_map_from_nodes(nodes: List[Any], methods: List[str]) -> Dict[str, str]:
    """
    Map metode -> doc_id yang paling awal/kuat menyebut metode itu di nodes.
    Reuse _METHOD_PATTERNS yang sudah ada di file ini.
    """
    out: Dict[str, str] = {}
    if not nodes or not methods:
        return out

    for m in methods:
        pats = _METHOD_PATTERNS.get(m, [])
        if not pats:
            continue
        for n in nodes:
            txt = _node_text(n)
            if not txt:
                continue
            for p in pats:
                if re.search(p, txt, flags=re.IGNORECASE):
                    out[m] = _node_doc_id(n)
                    break
            if m in out:
                break
    return out


def apply_doc_focus_filter(
    nodes: List[Any],
    allowed_doc_ids: List[str],
    *,
    min_keep: int = 3,
    strict: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Filter nodes agar hanya berasal dari dokumen tertentu.
    - allowed_doc_ids boleh berisi doc_id ATAU source_file (mis. "...Prototyping.pdf")
    - strict=True: TIDAK fallback ke nodes awal walau hasil sedikit/kosong (hard lock)
    """
    meta: Dict[str, Any] = {
        "enabled": True,
        "strict": bool(strict),
        "doc_ids": [],
        "reason": None,
        "kept": 0,
        "total": len(nodes or []),
    }

    if not nodes:
        meta["reason"] = "no_nodes"
        return [], meta

    # sanitasi + dedupe
    allow_raw = [str(x).strip() for x in (allowed_doc_ids or []) if str(x).strip()]
    allow = list(dict.fromkeys(allow_raw))
    meta["doc_ids"] = allow

    if not allow:
        meta["reason"] = "empty_allowlist"
        return nodes, meta

    allow_l = [a.lower() for a in allow]

    kept: List[Any] = []
    for n in nodes:
        doc_id = str(getattr(n, "doc_id", "") or "").strip()
        doc_l = doc_id.lower()

        md = getattr(n, "metadata", {}) or {}
        sf = str(md.get("source_file", "") or "").strip()
        sf_l = sf.lower()

        hit = False
        for a in allow_l:
            if a.endswith(".pdf"):
                # match ke source_file
                if a == sf_l or a in sf_l:
                    hit = True
                    break
                # juga izinkan match ke doc_id (tanpa .pdf)
                a2 = a[:-4]
                if a2 and (a2 == doc_l or a2 in doc_l):
                    hit = True
                    break
            else:
                # match ke doc_id, dan juga ke source_file jikalau user menulis tanpa .pdf
                if a == doc_l or a in doc_l or a == sf_l or a in sf_l:
                    hit = True
                    break

        if hit:
            kept.append(n)

    meta["kept"] = len(kept)

    # HARD LOCK
    if strict:
        meta["reason"] = "strict_lock"
        return kept, meta

    # soft-lock: jikalau terlalu sedikit, fallback ke nodes awal (perilaku lama)
    if len(kept) < int(min_keep):
        meta["reason"] = "too_few_kept_fallback"
        return nodes, meta

    meta["reason"] = "filtered_ok"
    return kept, meta


# =========================
# Ownership-aware document aggregation (Tahap 2)
# =========================

_OWNER_PREFERRED_BABS = {"BAB_I", "BAB_III", "BAB_IV", "BAB_V"}
_LITERATURE_BABS = {"BAB_II", "DAFTAR_PUSTAKA"}

_OWNER_CUE_PATTERNS: List[str] = [
    r"\bpenelitian\s+ini\s+(?:menggunakan|memakai|menerapkan)\b",
    r"\bdalam\s+penelitian\s+ini\b",
    r"\bpengembangan\s+sistem\s+(?:ini\s+)?(?:menggunakan|memakai|menerapkan)\b",
    r"\bsistem\s+ini\s+(?:dikembangkan|dibangun|menggunakan)\b",
    r"\bmetode\b.{0,60}\bdigunakan\s+dalam\s+penelitian\s+ini\b",
    r"\bmetode\b.{0,60}\bdigunakan\s+karena\b",
    r"\bpenulis\s+(?:menggunakan|menerapkan)\b",
]

_RELATED_WORK_PATTERNS: List[str] = [
    r"\bpenelitian\s+terdahulu\b",
    r"\bstudi\s+terdahulu\b",
    r"\bpenelitian\s+sebelumnya\b",
    r"\bberdasarkan\s+penelitian\s+sebelumnya\b",
    r"\bhasil\s+penelitian\s+sebelumnya\b",
    r"\btinjauan\s+pustaka\b",
    r"\blandasan\s+teori\b",
    r"\bperbedaan\s+skripsi\b",
    r"\bno\s+penulis\b",
    r"\bjudul\b.*\btahun\b",
    r"\bstudi\s+kasus\b",
]


def _node_source_file(n: Any) -> str:
    return str((getattr(n, "metadata", {}) or {}).get("source_file", "") or "")


def _node_bab_label(n: Any) -> str:
    return str(
        getattr(n, "bab_label", None)
        or (getattr(n, "metadata", {}) or {}).get("bab_label", "")
        or "UNKNOWN"
    ).strip()


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _short_snippet(text: str, max_chars: int = 280) -> str:
    t = _normalize_ws(text)
    if len(t) <= int(max_chars):
        return t
    return t[: int(max_chars)].rstrip() + "..."


def _text_mentions_method(text: str, method: str) -> bool:
    pats = _METHOD_PATTERNS.get(method, [])
    hay = _normalize_ws(text).lower()
    return any(re.search(p, hay, flags=re.IGNORECASE) for p in pats)


def _has_owner_cue(text: str) -> bool:
    t = _normalize_ws(text).lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in _OWNER_CUE_PATTERNS)


def _looks_like_related_work_context(text: str, bab_label: str) -> bool:
    t = _normalize_ws(text).lower()
    if str(bab_label or "").upper() in _LITERATURE_BABS:
        return True
    return any(re.search(p, t, flags=re.IGNORECASE) for p in _RELATED_WORK_PATTERNS)


@st.cache_data(show_spinner=False)
def _load_doc_catalog_cached(persist_dir_str: str, filename: str) -> Tuple[Dict[str, Any], str]:
    catalog, catalog_path = load_doc_catalog(Path(persist_dir_str), filename=filename)
    return catalog, str(catalog_path)


@st.cache_data(show_spinner=False)
def _load_docmeta_sidecar_cached(
    persist_dir_str: str,
    filename: str,
) -> Tuple[Dict[str, Any], str]:
    sidecar, sidecar_path = load_docmeta_sidecar(
        Path(persist_dir_str),
        filename=filename,
    )
    return sidecar, str(sidecar_path)


def _get_doc_catalog_entry(catalog: Dict[str, Any], doc_id: str, source_file: str) -> Dict[str, Any]:
    if not isinstance(catalog, dict):
        return {}
    if doc_id in catalog and isinstance(catalog.get(doc_id), dict):
        return dict(catalog[doc_id])

    sf = (source_file or "").strip().lower()
    sf_stem = sf[:-4] if sf.endswith(".pdf") else sf
    for v in catalog.values():
        if not isinstance(v, dict):
            continue
        cand = str(v.get("source_file", "") or "").strip().lower()
        cand_stem = cand[:-4] if cand.endswith(".pdf") else cand
        if sf and (sf == cand or sf_stem == cand_stem):
            return dict(v)
    return {}


def _get_docmeta_sidecar_entry(
    sidecar: Dict[str, Any],
    doc_id: str,
    source_file: str,
) -> Dict[str, Any]:
    if not isinstance(sidecar, dict):
        return {}

    if doc_id in sidecar and isinstance(sidecar.get(doc_id), dict):
        return dict(sidecar[doc_id])

    sf = (source_file or "").strip().lower()
    sf_stem = sf[:-4] if sf.endswith(".pdf") else sf

    for v in sidecar.values():
        if not isinstance(v, dict):
            continue
        cand = str(v.get("source_file", "") or "").strip().lower()
        cand_stem = cand[:-4] if cand.endswith(".pdf") else cand
        if sf and (sf == cand or sf_stem == cand_stem):
            return dict(v)

    return {}


def _resolve_safe_doc_display_meta(
    *,
    catalog_entry: Dict[str, Any],
    sidecar_entry: Dict[str, Any],
    source_file: str,
    doc_id: str,
) -> Dict[str, Any]:
    if sidecar_entry:
        title_display = str(sidecar_entry.get("title_safe", "") or "").strip() or _safe_display_title(
            catalog_entry,
            source_file=source_file,
            doc_id=doc_id,
        )
        author_display = str(sidecar_entry.get("author_safe", "") or "").strip() or str((catalog_entry or {}).get("penulis", "") or "").strip() or "-"
        year_display = str(sidecar_entry.get("year_safe", "") or "").strip() or "-"
        title_source = str(sidecar_entry.get("title_source", "") or "sidecar").strip() or "sidecar"
        title_quality = str(sidecar_entry.get("title_quality", "") or "unknown").strip() or "unknown"
        title_reasons = list(sidecar_entry.get("title_reasons") or [])
        return {
            "title_display": title_display,
            "author_display": author_display,
            "year_display": year_display,
            "title_source": title_source,
            "title_quality": title_quality,
            "title_reasons": title_reasons,
        }

    # fallback legacy-runtime jika sidecar belum tersedia
    title_display = _safe_display_title(
        catalog_entry,
        source_file=source_file,
        doc_id=doc_id,
    )
    author_display = str((catalog_entry or {}).get("penulis", "") or "").strip() or "-"
    year_val = (catalog_entry or {}).get("tahun", None)
    year_display = str(int(year_val)) if isinstance(year_val, int) and year_val > 0 else "-"

    raw_title = str((catalog_entry or {}).get("judul", "") or "").strip()
    title_source = "catalog_raw" if raw_title and title_display == raw_title else "runtime_fallback"
    title_quality = "fallback" if title_source != "catalog_raw" else "unknown"

    return {
        "title_display": title_display,
        "author_display": author_display,
        "year_display": year_display,
        "title_source": title_source,
        "title_quality": title_quality,
        "title_reasons": [],
    }


def _safe_display_title(entry: Dict[str, Any], *, source_file: str, doc_id: str) -> str:
    title = str((entry or {}).get("judul", "") or "").strip()
    if not title:
        return source_file or doc_id

    tl = title.lower()
    research_keywords = [
        "sistem", "rancang", "bangun", "aplikasi", "pengembangan",
        "monitoring", "e-learning", "learning", "ticketing", "informasi",
        "presensi", "face", "recognition", "website", "penerapan", "perancangan",
        "skripsi", "tugas akhir", "berbasis", "web", "mobile",
    ]

    suspicious = False

    if tl.startswith("puji syukur"):
        suspicious = True
    elif len(title) < 20:
        suspicious = True
    elif re.match(r"^\s*\d+\.\s+", title):
        suspicious = True
    elif title.count(",") >= 4:
        suspicious = True
    elif re.search(r"\b(no\s+penulis|hasil\s+penelitian|perbedaan\s+skripsi|penelitian\s+terdahulu|studi\s+terdahulu)\b", tl):
        suspicious = True
    elif ("institut teknologi sumatera" in tl or "jurusan" in tl or "produksi dan industri" in tl) and not any(k in tl for k in research_keywords):
        suspicious = True

    if suspicious:
        sf = str(source_file or "").strip()
        stem = sf[:-4] if sf.lower().endswith(".pdf") else sf
        stem = stem.strip()
        return stem or doc_id

    return title


def _node_has_rerank_payload(n: Any) -> bool:
    return (
        getattr(n, "rerank_score", None) is not None
        or getattr(n, "rerank_rank", None) is not None
        or bool(getattr(n, "is_reserved_anchor", False))
        or bool(getattr(n, "reserved_for_method", None))
    )


def _pick_nodes_by_chunk_ids(
    preferred_nodes: List[Any],
    fallback_nodes: List[Any],
    chunk_ids: List[str],
) -> List[Any]:
    """
    Ambil node berdasarkan urutan chunk_ids, tetapi jika ada beberapa object dengan
    chunk_id yang sama, prioritaskan object yang membawa payload rerank / anchor lebih lengkap.

    Tujuan patch final:
    - final contexts UI/history tidak memakai salinan node "kosong"
    - object kanonik hasil rerank tetap diprioritaskan
    """
    pool: Dict[str, Any] = {}

    def _should_replace(old_node: Any, new_node: Any) -> bool:
        if old_node is None:
            return True

        old_has = _node_has_rerank_payload(old_node)
        new_has = _node_has_rerank_payload(new_node)

        # Jika node baru punya payload rerank lebih kaya, pakai node baru.
        if (not old_has) and new_has:
            return True

        # Jika sama-sama tidak / sama-sama iya, pertahankan node lama
        # agar urutan preferensi caller tetap dihormati.
        return False

    for n in list(preferred_nodes or []) + list(fallback_nodes or []):
        cid = str(getattr(n, "chunk_id", "") or "").strip()
        if not cid:
            continue

        if cid not in pool or _should_replace(pool.get(cid), n):
            pool[cid] = n

    return [pool[cid] for cid in chunk_ids if cid in pool]


def build_owner_aware_doc_aggregation(
    nodes: List[Any],
    *,
    cfg: Dict[str, Any],
    target_method: str,
    max_docs: int = 5,
    max_evidence_chars: int = 280,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    index_cfg = cfg.get("index", {}) or {}
    persist_dir = str(index_cfg.get("persist_dir", "") or "").strip()

    md_cfg = cfg.get("metadata_routing", {}) or {}
    catalog_filename = str(md_cfg.get("doc_catalog_filename", "doc_catalog.json"))

    docmeta_cfg = cfg.get("docmeta", {}) or {}
    sidecar_enabled = bool(docmeta_cfg.get("enabled", True))
    sidecar_filename = str(docmeta_cfg.get("sidecar_filename", "docmeta_sidecar.json") or "docmeta_sidecar.json").strip()

    catalog: Dict[str, Any] = {}
    catalog_path = ""
    sidecar: Dict[str, Any] = {}
    sidecar_path = ""

    if persist_dir:
        catalog, catalog_path = _load_doc_catalog_cached(persist_dir, catalog_filename)
        if sidecar_enabled:
            sidecar, sidecar_path = _load_docmeta_sidecar_cached(persist_dir, sidecar_filename)

    grouped: Dict[str, List[Any]] = {}
    for n in nodes or []:
        did = _node_doc_id(n)
        grouped.setdefault(did, []).append(n)

    summaries: List[Dict[str, Any]] = []
    for doc_id, doc_nodes in grouped.items():
        source_file = _node_source_file(doc_nodes[0])
        docid_hit = bool(_docid_hit_for_method(doc_id, target_method))

        strong_owner_nodes: List[Any] = []
        weak_owner_nodes: List[Any] = []
        mention_only_nodes: List[Any] = []
        neutral_method_nodes: List[Any] = []

        for n in doc_nodes:
            text = _node_text(n)
            bab_label = _node_bab_label(n).upper()
            mentions = _text_mentions_method(text, target_method)
            if not mentions:
                continue

            related = _looks_like_related_work_context(text, bab_label)
            owner_cue = _has_owner_cue(text)

            if owner_cue and (not related):
                strong_owner_nodes.append(n)
            elif (bab_label in _OWNER_PREFERRED_BABS) and (not related):
                weak_owner_nodes.append(n)
            elif related or (bab_label in _LITERATURE_BABS):
                mention_only_nodes.append(n)
            else:
                neutral_method_nodes.append(n)

        owner_score = 0
        owner_score += 4 if docid_hit else 0
        owner_score += 3 * len(strong_owner_nodes)
        owner_score += 1 * len(weak_owner_nodes)
        owner_score += 1 * len(neutral_method_nodes)
        owner_score -= 2 * len(mention_only_nodes)

        accepted_reasons: List[str] = []
        if docid_hit:
            accepted_reasons.append("doc_id_hint")
        if strong_owner_nodes:
            accepted_reasons.append("strong_owner_phrase")
        if weak_owner_nodes:
            accepted_reasons.append("preferred_bab_context")
        if neutral_method_nodes and not mention_only_nodes:
            accepted_reasons.append("direct_method_mention")

        accepted = False
        if docid_hit or strong_owner_nodes:
            accepted = True
        elif weak_owner_nodes and owner_score >= 2:
            accepted = True
        elif neutral_method_nodes and owner_score >= 2 and len(mention_only_nodes) == 0:
            accepted = True

        rep_node = None
        for bucket in (strong_owner_nodes, weak_owner_nodes, neutral_method_nodes, mention_only_nodes, doc_nodes):
            if bucket:
                rep_node = bucket[0]
                break

        secondary_node = None
        for bucket in (strong_owner_nodes[1:], weak_owner_nodes[1:], neutral_method_nodes[1:], doc_nodes[1:]):
            if bucket:
                secondary_node = bucket[0]
                break

        catalog_entry = _get_doc_catalog_entry(catalog, doc_id, source_file)
        sidecar_entry = _get_docmeta_sidecar_entry(sidecar, doc_id, source_file)

        display_meta = _resolve_safe_doc_display_meta(
            catalog_entry=catalog_entry,
            sidecar_entry=sidecar_entry,
            source_file=source_file,
            doc_id=doc_id,
        )

        title_display = display_meta["title_display"]
        author_display = display_meta["author_display"]
        year_display = display_meta["year_display"]
        title_source = display_meta["title_source"]
        title_quality = display_meta["title_quality"]
        title_reasons = display_meta["title_reasons"]

        summaries.append({
            "doc_id": doc_id,
            "source_file": source_file or doc_id,
            "title_display": title_display,
            "author_display": author_display,
            "year_display": year_display,
            "title_source": title_source,
            "title_quality": title_quality,
            "title_reasons": title_reasons,
            "docid_hit": bool(docid_hit),
            "owner_score": int(owner_score),
            "accepted": bool(accepted),
            "accepted_reasons": accepted_reasons,
            "strong_owner_count": len(strong_owner_nodes),
            "weak_owner_count": len(weak_owner_nodes),
            "mention_only_count": len(mention_only_nodes),
            "neutral_method_count": len(neutral_method_nodes),
            "representative_chunk_id": str(getattr(rep_node, "chunk_id", "") or ""),
            "owner_evidence": _short_snippet(_node_text(rep_node) if rep_node else "", max_chars=max_evidence_chars),
            "secondary_evidence": _short_snippet(_node_text(secondary_node) if secondary_node else "", max_chars=max_evidence_chars),
            "representative_bab_label": _node_bab_label(rep_node),
        })

    summaries.sort(key=lambda x: (-int(x.get("owner_score", 0)), str(x.get("doc_id", ""))))
    accepted_docs = [d for d in summaries if d.get("accepted")]
    accepted_docs = accepted_docs[: max(1, int(max_docs))]

    meta: Dict[str, Any] = {
        "enabled": True,
        "executed": True,
        "applied": bool(accepted_docs),
        "target_method": str(target_method),
        "target_method_label": pretty_method_name(target_method),
        "catalog_path": catalog_path or None,
        "candidate_doc_count": int(len(summaries)),
        "accepted_doc_count": int(len(accepted_docs)),
        "representative_chunk_ids": [d.get("representative_chunk_id") for d in accepted_docs if d.get("representative_chunk_id")],
        "candidate_docs": [
            {
                "doc_id": d["doc_id"],
                "source_file": d["source_file"],
                "owner_score": d["owner_score"],
                "docid_hit": d["docid_hit"],
                "strong_owner_count": d["strong_owner_count"],
                "weak_owner_count": d["weak_owner_count"],
                "mention_only_count": d["mention_only_count"],
                "accepted": d["accepted"],
            }
            for d in summaries
        ],
        "accepted_docs": [
            {
                "doc_id": d["doc_id"],
                "source_file": d["source_file"],
                "title_display": d["title_display"],
                "author_display": d["author_display"],
                "year_display": d["year_display"],
                "title_source": d["title_source"],
                "title_quality": d["title_quality"],
                "title_reasons": d["title_reasons"],
                "owner_score": d["owner_score"],
                "accepted_reasons": d["accepted_reasons"],
            }
            for d in accepted_docs
        ],
        "rejected_docs": [
            {
                "doc_id": d["doc_id"],
                "source_file": d["source_file"],
                "owner_score": d["owner_score"],
                "mention_only_count": d["mention_only_count"],
            }
            for d in summaries if not d.get("accepted")
        ],
        "used_summary_contexts": bool(accepted_docs),
        "docmeta_sidecar_path": sidecar_path or None,
    }
    return accepted_docs, meta


def build_doc_aggregation_contexts(
    doc_summaries: List[Dict[str, Any]],
    *,
    target_method: str,
) -> List[str]:
    out: List[str] = []
    method_label = pretty_method_name(target_method)
    for d in doc_summaries or []:
        lines = [
            f"DOC_AGG: {d.get('doc_id', '')} | {d.get('source_file', '')} | owner_method={method_label}",
            f"TITLE: {d.get('title_display', '-')}",
            f"AUTHOR: {d.get('author_display', '-')}",
            f"YEAR: {d.get('year_display', '-')}",
            f"OWNER_SCORE: {d.get('owner_score', 0)}",
            f"OWNERSHIP_BASIS: {', '.join(d.get('accepted_reasons') or []) or '-'}",
            f"OWNER_EVIDENCE: {d.get('owner_evidence', '-') or '-'}",
        ]
        if d.get("secondary_evidence"):
            lines.append(f"ADDITIONAL_EVIDENCE: {d.get('secondary_evidence')}")
        out.append("\n".join(lines).strip())
    return out


# =========================
# Method explanation expansion (Tahap 3)
# =========================

_EXPLANATION_BIAS_PATTERNS: List[str] = [
    r"\bdigunakan\b",
    r"\bdigunakan\s+karena\b",
    r"\balasan\b",
    r"\bmemungkinkan\b",
    r"\bkelebihan\b",
    r"\bkarakteristik\b",
    r"\bbersifat\b",
    r"\bincremental\b",
    r"\bitera(?:si|tif)?\b",
    r"\btahapan?\b",
    r"\blangkah(?:-langkah)?\b",
    r"\bplanning\b",
    r"\bdesign\b",
    r"\bcoding\b",
    r"\btesting\b",
    r"\bcommunication\b",
]


def build_method_explanation_queries(method: str) -> List[str]:
    label = pretty_method_name(method)
    return [
        f"metode {label} digunakan karena",
        f"alasan menggunakan {label}",
        f"karakteristik metode {label}",
        f"{label} dalam penelitian ini",
        f"tahapan metode {label}",
    ]


def _explanation_signal_strength(text: str) -> int:
    t = _normalize_ws(text).lower()
    if not t:
        return 0

    score = 0
    for pat in _EXPLANATION_BIAS_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            score += 1
    return score


def _score_method_explanation_node(
    n: Any,
    *,
    target_method: str,
    target_doc_id: str,
) -> int:
    text = _node_text(n)
    bab_label = _node_bab_label(n).upper()
    score = 0

    if str(getattr(n, "doc_id", "") or "").strip() == str(target_doc_id or "").strip():
        score += 5

    if _node_supports_target_method(n, target_method):
        score += 4

    if _has_owner_cue(text):
        score += 3

    if bab_label in _OWNER_PREFERRED_BABS:
        score += 2

    score += min(_explanation_signal_strength(text), 6)

    if _node_has_strong_steps_signal(n):
        score += 2

    if _looks_like_related_work_context(text, bab_label):
        score -= 5

    if bab_label in _LITERATURE_BABS:
        score -= 3

    return int(score)


def _resolve_method_explanation_target_doc_id(
    *,
    intent_plan: Dict[str, Any],
    doc_focus_meta: Optional[Dict[str, Any]],
    doc_aggregation_meta: Optional[Dict[str, Any]],
    nodes_for_gen: List[Any],
    target_method: str,
) -> str:
    # 1) explicit doc-lock dari query
    dfm = doc_focus_meta or {}
    allowed = [str(x).strip() for x in (dfm.get("allowed_doc_ids") or []) if str(x).strip()]
    if allowed:
        return allowed[0]

    # 2) hasil accepted owner doc dari Tahap 2
    dam = doc_aggregation_meta or {}
    accepted = dam.get("accepted_docs") or []
    if accepted and isinstance(accepted, list) and isinstance(accepted[0], dict):
        doc_id = str(accepted[0].get("doc_id", "") or "").strip()
        if doc_id:
            return doc_id

    # 3) kalau final nodes saat ini hanya 1 dokumen
    doc_ids = list(
        dict.fromkeys(
            [
                str(getattr(n, "doc_id", "") or "").strip()
                for n in (nodes_for_gen or [])
                if str(getattr(n, "doc_id", "") or "").strip()
            ]
        )
    )
    if len(doc_ids) == 1:
        return doc_ids[0]

    # 4) fallback ke method_doc_map jika tersedia
    method_doc_map = (dfm.get("method_doc_map") or {}) if isinstance(dfm, dict) else {}
    if isinstance(method_doc_map, dict):
        cand = str(method_doc_map.get(target_method, "") or "").strip()
        if cand:
            return cand

    return ""


def build_method_explanation_expansion(
    *,
    cfg: Dict[str, Any],
    query: str,
    target_method: str,
    target_doc_id: str,
    base_nodes: List[Any],
    max_queries: int,
    max_added_nodes: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "executed": False,
        "applied": False,
        "target_method": str(target_method),
        "target_method_label": pretty_method_name(target_method),
        "target_doc_id": str(target_doc_id or "") or None,
        "target_doc_source_file": None,
        "expansion_queries": [],
        "added_chunk_ids": [],
        "added_doc_ids": [],
        "selected_explanation_chunk_ids": [],
        "selected_context_count": 0,
        "reason": None,
    }

    if not target_doc_id:
        meta["reason"] = "no_target_doc_id"
        return list(base_nodes or []), meta

    if not target_method:
        meta["reason"] = "no_target_method"
        return list(base_nodes or []), meta

    # Base = node final saat ini, tetapi hanya yang masih berada pada target owner doc
    base = dedupe_nodes_by_chunk_id(list(base_nodes or []))
    owner_base = [
        n for n in base
        if str(getattr(n, "doc_id", "") or "").strip() == target_doc_id
    ]

    queries = build_method_explanation_queries(target_method)[: max(1, int(max_queries))]
    meta["expansion_queries"] = queries
    meta["executed"] = True

    where_filter = {"doc_id": {"$eq": target_doc_id}}
    pool: List[Any] = list(owner_base)

    # Expansion retrieval sengaja dense + doc-locked,
    # supaya explanation tidak kabur lintas dokumen.
    for qx in queries:
        try:
            nodes_extra = retrieve_dense(
                cfg,
                qx,
                diversify=False,
                candidate_k=max(4, int(max_added_nodes)),
                collection="narasi",
                where_filter=where_filter,
            )
        except Exception:
            nodes_extra = []

        if nodes_extra:
            pool.extend(nodes_extra)

    pool = dedupe_nodes_by_chunk_id(pool)
    if pool:
        meta["target_doc_source_file"] = _node_source_file(pool[0]) or None

    scored: List[Tuple[int, Any]] = []
    for n in pool:
        s = _score_method_explanation_node(
            n,
            target_method=target_method,
            target_doc_id=target_doc_id,
        )
        scored.append((s, n))

    scored.sort(key=lambda x: (-(x[0]), float(getattr(x[1], "score", 9999.0))))

    selected: List[Any] = []
    selected_ids: set = set()

    def _append_first(predicate) -> None:
        for s, n in scored:
            cid = str(getattr(n, "chunk_id", "") or "").strip()
            if not cid or cid in selected_ids:
                continue
            if predicate(s, n):
                selected.append(n)
                selected_ids.add(cid)
                return

    # Komposisi ideal:
    # 1) owner/context node
    # 2) reason/characteristic node
    # 3) steps/process node jika ada
    _append_first(lambda s, n: _has_owner_cue(_node_text(n)))
    _append_first(lambda s, n: _explanation_signal_strength(_node_text(n)) >= 2)
    _append_first(lambda s, n: _node_has_strong_steps_signal(n))

    for s, n in scored:
        cid = str(getattr(n, "chunk_id", "") or "").strip()
        if not cid or cid in selected_ids:
            continue
        if s < 2:
            continue
        selected.append(n)
        selected_ids.add(cid)
        if len(selected) >= max(3, int(max_added_nodes)):
            break

    if not selected and owner_base:
        selected = owner_base[:1]

    meta["selected_explanation_chunk_ids"] = [
        str(getattr(n, "chunk_id", "") or "") for n in selected
    ]
    base_ids = {
        str(getattr(n, "chunk_id", "") or "") for n in owner_base
    }
    meta["added_chunk_ids"] = [
        cid for cid in meta["selected_explanation_chunk_ids"]
        if cid and cid not in base_ids
    ]
    meta["added_doc_ids"] = list(
        dict.fromkeys(
            [
                str(getattr(n, "doc_id", "") or "")
                for n in selected
                if str(getattr(n, "doc_id", "") or "")
            ]
        )
    )
    meta["selected_context_count"] = len(selected)
    meta["applied"] = bool(selected)
    meta["reason"] = (
        "owner_doc_explanation_expansion"
        if selected
        else "no_explanation_nodes_selected"
    )

    return selected, meta


# =========================
# Single-target steps handler (Tahap 4.1)
# =========================

_SINGLE_TARGET_STEPS_BIAS_PATTERNS: List[str] = [
    r"\btahapan?\b",
    r"\blangkah(?:-langkah)?\b",
    r"\bmekanisme\b",
    r"\bpenjabaran\b",
    r"\balur\b",
    r"\bprosedur\b",
    # Tahap 4.1:
    # sengaja TIDAK lagi memberi boost untuk flowchart/diagram/gambar
    # karena ini terlalu mudah memancing artefak non-step (DFD, ERD, use case, dst).
]

_SINGLE_TARGET_STEPS_ARTIFACT_PATTERNS: List[str] = [
    r"\bdata\s+flow\s+diagram\b",
    r"\bdfd\b",
    r"\bentity\s+relationship\s+diagram\b",
    r"\berd\b",
    r"\buse\s+case\b",
    r"\bactivity\s+diagram\b",
    r"\bsequence\s+diagram\b",
    r"\bclass\s+diagram\b",
    r"\bdeployment\s+diagram\b",
    r"\bcontext\s+diagram\b",
    r"\bflowmap\b",
    r"\bmockup\b",
    r"\brancangan\s+antarmuka\b",
    r"\bblackbox\b",
    r"\bwhitebox\b",
    r"\blaporan\s+magang\b",
    r"\blogsheet\s+magang\b",
]


def build_single_target_steps_queries(method: str) -> List[str]:
    label = pretty_method_name(method)
    return [
        f"tahapan metode {label} terdiri dari",
        f"langkah-langkah metode {label} yaitu",
        f"prosedur {label} dalam penelitian ini",
        f"alur pengembangan dengan metode {label}",
        f"tahapan implementasi {label}",
        f"penjabaran tahapan {label}",
    ]


def _single_target_steps_signal_strength(text: str) -> int:
    t = _normalize_ws(text).lower()
    if not t:
        return 0

    score = 0
    for pat in _SINGLE_TARGET_STEPS_BIAS_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            score += 1

    score += min(_steps_signal_strength(text), 6)
    return int(score)


def _looks_like_single_target_steps_artifact_text(text: str) -> bool:
    t = _normalize_ws(text).lower()
    if not t:
        return False
    return any(re.search(p, t, flags=re.IGNORECASE) for p in _SINGLE_TARGET_STEPS_ARTIFACT_PATTERNS)


def _filter_valid_single_target_step_items(
    items: List[str],
    *,
    max_items: int,
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    for raw in items or []:
        item = " ".join((raw or "").split()).strip()
        if not item:
            continue

        item_norm = item.lower()

        # Drop artefak non-langkah (DFD/ERD/use case/dll)
        if _looks_like_single_target_steps_artifact_text(item_norm):
            continue

        # Drop judul "gambar x.x ..." / "tabel x.x ..." yang bukan langkah
        if re.search(r"^\s*(gambar|tabel)\s+\d+\.\d+\b", item_norm, flags=re.IGNORECASE):
            continue

        # Drop item yang terlalu pendek / terlalu noise
        if len(item_norm) < 5:
            continue

        sig = re.sub(r"\s+", " ", item_norm)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)

        if len(out) >= int(max_items):
            break

    return out


def _extract_valid_single_target_steps_from_text(
    text: str,
    *,
    max_items: int,
) -> List[str]:
    """
    Reuse extractor utama dari generate_utils via konteks mentah yang sudah dibawa ke UI context,
    tetapi tambahkan filter artefak Tahap 4.1 di layer runtime.
    """
    # NOTE:
    # TIDAK memanggil helper private dari generate_utils di sini.
    # Jadi re-parse ringan secara lokal, konservatif.
    s = _normalize_ws(text)
    if not s:
        return []

    # 1) enumerasi eksplisit
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    enum_items: List[str] = []
    for ln in lines:
        if re.match(r"^\s*(\d+[\.\)]|[-•])\s+", ln):
            item = re.sub(r"^\s*(\d+[\.\)]|[-•])\s+", "", ln).strip()
            if item:
                enum_items.append(item)

    if len(enum_items) >= 2:
        return _filter_valid_single_target_step_items(enum_items, max_items=max_items)

    # 2) connector daftar
    m = re.search(
        r"\b(meliputi|terdiri(\s+dari)?|yaitu|antara\s+lain)\b\s*([^\.]{20,280})",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        tail = (m.group(3) or "").strip()
        parts = [p.strip(" ;:-\n\t") for p in re.split(r",|\bdan\b", tail) if p.strip()]
        return _filter_valid_single_target_step_items(parts, max_items=max_items)

    return []


def _single_target_steps_candidate_doc_ids_from_meta(
    doc_aggregation_meta: Optional[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    dam = doc_aggregation_meta or {}

    for bucket_name in ("accepted_docs", "candidate_docs", "rejected_docs"):
        bucket = dam.get(bucket_name) or []
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            did = str(item.get("doc_id", "") or "").strip()
            if did and did not in out:
                out.append(did)

    return out


def _resolve_single_target_steps_target_doc_id_strict(
    *,
    intent_plan: Dict[str, Any],
    doc_focus_meta: Optional[Dict[str, Any]],
    doc_aggregation_meta: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Tahap 4.1:
    Resolver steps route HARUS lebih ketat dari explanation route.

    Return:
        (target_doc_id, resolve_status)

    resolve_status:
        - resolved_explicit_doc_lock
        - resolved_owner_aggregation
        - ambiguous_owner_docs
        - unresolved_owner_doc
    """
    dfm = doc_focus_meta or {}

    # 1) explicit doc-lock dari query
    allowed = [
        str(x).strip()
        for x in (dfm.get("allowed_doc_ids") or [])
        if str(x).strip()
    ]
    if allowed:
        return allowed[0], "resolved_explicit_doc_lock"

    # 2) accepted owner doc dari Tahap 2
    dam = doc_aggregation_meta or {}
    accepted = dam.get("accepted_docs") or []
    accepted_doc_ids = []
    if isinstance(accepted, list):
        for item in accepted:
            if not isinstance(item, dict):
                continue
            did = str(item.get("doc_id", "") or "").strip()
            if did and did not in accepted_doc_ids:
                accepted_doc_ids.append(did)

    if len(accepted_doc_ids) == 1:
        return accepted_doc_ids[0], "resolved_owner_aggregation"

    if len(accepted_doc_ids) > 1:
        return "", "ambiguous_owner_docs"

    # 3) TIDAK boleh fallback ke single doc_id dari nodes_for_gen / method_doc_map
    # untuk steps route, karena ini menyebabkan false lock ke dokumen yang salah.
    return "", "unresolved_owner_doc"


def _score_single_target_steps_node(
    n: Any,
    *,
    target_method: str,
    target_doc_id: str,
) -> int:
    text = _node_text(n)
    bab_label = _node_bab_label(n).upper()
    score = 0

    if str(getattr(n, "doc_id", "") or "").strip() == str(target_doc_id or "").strip():
        score += 6

    if _node_supports_target_method(n, target_method):
        score += 4

    if _node_has_strong_steps_signal(n):
        score += 7

    if bab_label in {"BAB_III", "BAB_IV", "BAB_V"}:
        score += 3
    elif bab_label in _OWNER_PREFERRED_BABS:
        score += 2

    if _has_owner_cue(text):
        score += 2

    score += min(_single_target_steps_signal_strength(text), 6)

    # Tahap 4.1: penalti ringan untuk chunk yang tampak seperti artefak diagram/desain
    if _looks_like_single_target_steps_artifact_text(text):
        score -= 4

    if _looks_like_related_work_context(text, bab_label):
        score -= 6

    if bab_label in _LITERATURE_BABS:
        score -= 4

    return int(score)


def build_single_target_steps_expansion(
    *,
    cfg: Dict[str, Any],
    query: str,
    target_method: str,
    target_doc_id: str,
    base_nodes: List[Any],
    max_queries: int,
    max_added_nodes: int,
    max_steps: int = 5,
) -> Tuple[List[Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "executed": False,
        "applied": False,
        "target_method": str(target_method or ""),
        "target_method_label": pretty_method_name(target_method),
        "target_doc_id": str(target_doc_id or "") or None,
        "target_doc_source_file": None,
        "expansion_queries": [],
        "added_chunk_ids": [],
        "added_doc_ids": [],
        "selected_chunk_ids": [],
        "selected_context_count": 0,
        "method_support_ctx_nums": [],
        "explicit_steps_ctx_nums": [],
        "extractable_steps_ctx_nums": [],
        "extractable_steps_count": 0,
        "artifact_only_ctx_nums": [],
        "has_method_support": False,
        "has_explicit_steps": False,        # raw explicit steps signal
        "has_extractable_steps": False,     # actionable steps setelah filter artefak
        "ambiguous_candidate_doc_ids": [],
        "status": None,
        "reason": None,
        "max_steps": int(max_steps),
    }

    if not target_doc_id:
        meta["status"] = "unresolved_owner_doc"
        meta["reason"] = "no_target_doc_id"
        return list(base_nodes or []), meta

    if not target_method:
        meta["status"] = "method_not_found"
        meta["reason"] = "no_target_method"
        return list(base_nodes or []), meta

    base = dedupe_nodes_by_chunk_id(list(base_nodes or []))
    owner_base = [
        n for n in base
        if str(getattr(n, "doc_id", "") or "").strip() == target_doc_id
    ]

    queries = build_single_target_steps_queries(target_method)[: max(1, int(max_queries))]
    meta["expansion_queries"] = queries
    meta["executed"] = True

    where_filter = {"doc_id": {"$eq": target_doc_id}}
    pool: List[Any] = list(owner_base)

    for qx in queries:
        try:
            nodes_extra = retrieve_dense(
                cfg,
                qx,
                diversify=False,
                candidate_k=max(4, int(max_added_nodes)),
                collection="narasi",
                where_filter=where_filter,
            )
        except Exception:
            nodes_extra = []

        if nodes_extra:
            pool.extend(nodes_extra)

    pool = dedupe_nodes_by_chunk_id(pool)

    if pool:
        meta["target_doc_source_file"] = _node_source_file(pool[0]) or None

    scored: List[Tuple[int, Any]] = []
    for n in pool:
        s = _score_single_target_steps_node(
            n,
            target_method=target_method,
            target_doc_id=target_doc_id,
        )
        scored.append((s, n))

    scored.sort(key=lambda x: (-(x[0]), float(getattr(x[1], "score", 9999.0))))

    selected: List[Any] = []
    selected_ids: set[str] = set()

    def _append_first(predicate) -> None:
        for s, n in scored:
            cid = str(getattr(n, "chunk_id", "") or "").strip()
            if not cid or cid in selected_ids:
                continue
            if predicate(s, n):
                selected.append(n)
                selected_ids.add(cid)
                return

    # Komposisi ideal:
    # 1) owner-supported steps context
    # 2) steps context lain
    # 3) owner cue context
    _append_first(
        lambda s, n: _node_has_strong_steps_signal(n)
        and _node_bab_label(n).upper() in {"BAB_III", "BAB_IV", "BAB_V"}
    )
    _append_first(lambda s, n: _node_has_strong_steps_signal(n))
    _append_first(
        lambda s, n: _node_supports_target_method(n, target_method)
        and _has_owner_cue(_node_text(n))
    )

    for s, n in scored:
        cid = str(getattr(n, "chunk_id", "") or "").strip()
        if not cid or cid in selected_ids:
            continue
        if s < 4:
            continue
        selected.append(n)
        selected_ids.add(cid)
        if len(selected) >= max(2, int(max_added_nodes)):
            break

    if not selected and owner_base:
        selected = owner_base[: max(1, min(len(owner_base), int(max_added_nodes)))]

    meta["selected_chunk_ids"] = [
        str(getattr(n, "chunk_id", "") or "").strip()
        for n in selected
        if str(getattr(n, "chunk_id", "") or "").strip()
    ]

    base_ids = {
        str(getattr(n, "chunk_id", "") or "").strip()
        for n in owner_base
        if str(getattr(n, "chunk_id", "") or "").strip()
    }
    meta["added_chunk_ids"] = [
        cid for cid in meta["selected_chunk_ids"]
        if cid and cid not in base_ids
    ]
    meta["added_doc_ids"] = list(
        dict.fromkeys(
            [
                str(getattr(n, "doc_id", "") or "").strip()
                for n in selected
                if str(getattr(n, "doc_id", "") or "").strip()
            ]
        )
    )
    meta["selected_context_count"] = len(selected)
    meta["applied"] = bool(selected)

    method_support_ctx_nums: List[int] = []
    explicit_steps_ctx_nums: List[int] = []
    extractable_steps_ctx_nums: List[int] = []
    artifact_only_ctx_nums: List[int] = []

    extractable_items_merged: List[str] = []
    seen_extractable: set[str] = set()

    for idx, n in enumerate(selected, start=1):
        text = _node_text(n)

        if _node_supports_target_method(n, target_method):
            method_support_ctx_nums.append(idx)

        has_raw_steps_signal = _node_has_strong_steps_signal(n)
        if has_raw_steps_signal:
            explicit_steps_ctx_nums.append(idx)

        valid_items = _extract_valid_single_target_steps_from_text(
            text,
            max_items=max_steps,
        )

        if valid_items:
            extractable_steps_ctx_nums.append(idx)

            for item in valid_items:
                sig = re.sub(r"\s+", " ", item.strip()).lower()
                if not sig or sig in seen_extractable:
                    continue
                seen_extractable.add(sig)
                extractable_items_merged.append(item.strip())
                if len(extractable_items_merged) >= int(max_steps):
                    break
        elif has_raw_steps_signal and _looks_like_single_target_steps_artifact_text(text):
            artifact_only_ctx_nums.append(idx)

    meta["method_support_ctx_nums"] = method_support_ctx_nums
    meta["explicit_steps_ctx_nums"] = explicit_steps_ctx_nums
    meta["extractable_steps_ctx_nums"] = extractable_steps_ctx_nums
    meta["extractable_steps_count"] = int(len(extractable_items_merged))
    meta["artifact_only_ctx_nums"] = artifact_only_ctx_nums
    meta["has_method_support"] = bool(method_support_ctx_nums)
    meta["has_explicit_steps"] = bool(explicit_steps_ctx_nums)
    meta["has_extractable_steps"] = bool(extractable_steps_ctx_nums)

    if len(extractable_items_merged) >= 2:
        meta["status"] = "steps_found"
        meta["reason"] = "owner_doc_extractable_steps"
    elif method_support_ctx_nums:
        meta["status"] = "method_confirmed_no_explicit_steps"
        meta["reason"] = "owner_doc_steps_partial"
    else:
        meta["status"] = "method_not_found"
        meta["reason"] = "no_method_support_in_owner_doc"

    return selected, meta


# =========================
# Method comparison formatter meta (Tahap 5)
# =========================

def build_method_comparison_meta(
    nodes: List[Any],
    *,
    target_methods: List[str],
    max_ctx_per_method: int = 2,
) -> Dict[str, Any]:
    clean_targets = [str(x).strip() for x in (target_methods or []) if str(x).strip()]
    clean_targets = list(dict.fromkeys(clean_targets))

    meta: Dict[str, Any] = {
        "enabled": True,
        "executed": False,
        "target_methods": clean_targets,
        "ctx_map_per_method": {},
        "per_method_unique_ctx_map": {},
        "shared_ctx_nums": [],
        "supported_methods": [],
        "weak_methods": [],
        "missing_methods": [],
        "comparison_status": None,
        "balanced": False,
        "selected_context_count": 0,
        "unique_context_count": 0,
        "max_ctx_per_method": int(max_ctx_per_method),
    }

    if len(clean_targets) < 2:
        meta["comparison_status"] = "noop"
        return meta

    strong_by_method: Dict[str, List[int]] = {m: [] for m in clean_targets}
    weak_by_method: Dict[str, List[int]] = {m: [] for m in clean_targets}

    for idx, n in enumerate(nodes or [], start=1):
        text = _node_text(n)
        bab_label = _node_bab_label(n).upper()

        for m in clean_targets:
            if not _text_mentions_method(text, m):
                continue

            if _looks_like_related_work_context(text, bab_label):
                weak_by_method[m].append(idx)
            else:
                strong_by_method[m].append(idx)

    # Tahap 5.1:
    # pilih ctx per metode dengan preferensi:
    # 1) strong unseen
    # 2) strong
    # 3) weak unseen
    # 4) weak

    # Tujuannya agar dua metode tidak terlalu mudah jatuh ke CTX yang sama.
    ctx_map_per_method: Dict[str, List[int]] = {}
    global_used: set[int] = set()

    supported_methods: List[str] = []
    weak_methods: List[str] = []
    missing_methods: List[str] = []

    def _take_ctx_indices(
        pool: List[int],
        chosen: List[int],
        global_used: set[int],
        limit: int,
        *,
        unseen_only: bool,
    ) -> None:
        for idx in pool:
            if len(chosen) >= int(limit):
                return
            if idx in chosen:
                continue
            if unseen_only and idx in global_used:
                continue
            chosen.append(idx)

    for m in clean_targets:
        strong_idxs = list(dict.fromkeys(strong_by_method.get(m, [])))
        weak_idxs = list(dict.fromkeys(weak_by_method.get(m, [])))

        chosen: List[int] = []
        limit = max(1, int(max_ctx_per_method))

        _take_ctx_indices(strong_idxs, chosen, global_used, limit, unseen_only=True)
        _take_ctx_indices(strong_idxs, chosen, global_used, limit, unseen_only=False)
        _take_ctx_indices(weak_idxs, chosen, global_used, limit, unseen_only=True)
        _take_ctx_indices(weak_idxs, chosen, global_used, limit, unseen_only=False)

        ctx_map_per_method[m] = chosen
        for idx in chosen:
            global_used.add(idx)

        if strong_idxs:
            supported_methods.append(m)
        elif weak_idxs:
            weak_methods.append(m)
        else:
            missing_methods.append(m)

    selected_ctx_nums: List[int] = []
    for m in clean_targets:
        selected_ctx_nums.extend(ctx_map_per_method.get(m, []))
    selected_ctx_nums = list(dict.fromkeys(selected_ctx_nums))

    # Hitung shared ctx: ctx yang dipakai >1 metode
    ctx_usage_count: Dict[int, int] = {}
    for m in clean_targets:
        for idx in ctx_map_per_method.get(m, []):
            ctx_usage_count[idx] = ctx_usage_count.get(idx, 0) + 1

    shared_ctx_nums = sorted([idx for idx, cnt in ctx_usage_count.items() if cnt > 1])

    per_method_unique_ctx_map: Dict[str, List[int]] = {}
    for m in clean_targets:
        per_method_unique_ctx_map[m] = [
            idx for idx in ctx_map_per_method.get(m, [])
            if ctx_usage_count.get(idx, 0) == 1
        ]

    unique_context_count = len(selected_ctx_nums)

    meta.update(
        {
            "executed": True,
            "ctx_map_per_method": ctx_map_per_method,
            "per_method_unique_ctx_map": per_method_unique_ctx_map,
            "shared_ctx_nums": shared_ctx_nums,
            "supported_methods": supported_methods,
            "weak_methods": weak_methods,
            "missing_methods": missing_methods,
            "selected_context_count": unique_context_count,
            "unique_context_count": unique_context_count,
        }
    )

    # Tahap 5.1:
    # balanced_comparison HARUS:
    # - dua metode terdukung kuat
    # - tanpa weak/missing
    # - minimal 2 CTX unik secara total
    # - setiap metode punya minimal 1 CTX unik miliknya sendiri
    all_have_unique_ctx = all(
        len(per_method_unique_ctx_map.get(m, [])) >= 1
        for m in clean_targets[:2]
    )

    if (
        len(supported_methods) >= 2
        and not weak_methods
        and not missing_methods
        and unique_context_count >= 2
        and all_have_unique_ctx
    ):
        meta["comparison_status"] = "balanced_comparison"
        meta["balanced"] = True
    elif len(supported_methods) + len(weak_methods) >= 2:
        meta["comparison_status"] = "asymmetric_comparison"
        meta["balanced"] = False
    else:
        meta["comparison_status"] = "insufficient_comparison"
        meta["balanced"] = False

    return meta


# =========================
# Session / Run lifecycle
# =========================

def _new_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    """
    1 session Streamlit = 1 run_id (folder runs/<run_id>/).
    Jika config path berubah, buat run baru.
    _new_run() juga reset state penting agar UI & log tidak tercampur antar sesi.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])
    rm = RunManager(runs_dir=runs_dir, config=cfg).start()
    logger = setup_logger(rm.app_log_path)

    st.session_state["rm"] = rm
    st.session_state["logger"] = logger
    st.session_state["config_path"] = config_path
    st.session_state["turn_id"] = 0
    st.session_state["history"] = []  # list of dicts: {turn_id, query, answer}
    st.session_state["pdf_view"] = None  # {"path": str, "page": int}
    st.session_state["last_nodes"] = None
    st.session_state["last_nodes_before_rerank"] = None   # ← V3.0c: candidate pool pre-rerank
    st.session_state["last_answer_raw"] = None
    st.session_state["last_verification"] = None          # ← V3.0d: post-hoc verification result
    # ← V3.0c: stats eksplisit
    st.session_state["last_retrieval_stats_pre_rerank"] = None
    st.session_state["last_generation_context_stats"] = None
    st.session_state["_scroll_answer"] = False
    st.session_state["last_applied_ui_cfg"] = None
    st.session_state["hl_terms"] = ""
    st.session_state["last_submitted_query"] = ""
    st.session_state["last_turn_id"] = 0
    st.session_state["last_user_query"] = ""
    st.session_state["last_retrieval_query"] = ""
    # doc-focus lock state (lintas turn)
    st.session_state["method_doc_map"] = {}  # method -> doc_id
    st.session_state["last_doc_focus_meta"] = None
    st.session_state["last_history_used"] = None
    st.session_state["last_contextualized"] = None
    st.session_state["turn_data"] = {}       # turn_id -> full render data per turn
    st.session_state["viewing_turn_id"] = None  # None = tampilkan turn terbaru
    st.session_state["last_retrieval_mode"] = "dense" # default saat belum ada query
    st.session_state["last_self_query_result"] = None  # ← V3.0b
    st.session_state["last_reranker_applied"] = False   # ← V3.0c
    st.session_state["last_reranker_info"] = None       # ← V3.0c
    st.session_state["last_rerank_anchor_reservation"] = None  # ← V3.0c
    st.session_state["last_intent_plan"] = None                # ← V3.0d lanjutan: planner result
    st.session_state["last_doc_aggregation_meta"] = None       # ← V3.0d lanjutan: ownership-aware doc aggregation
    st.session_state["last_explanation_expansion_meta"] = None  # ← V3.0d lanjutan: method explanation expansion
    st.session_state["last_single_target_steps_meta"] = None   # ← V3.0d lanjutan: single-target steps handler
    st.session_state["last_method_comparison_meta"] = None    # ← V3.0d lanjutan: comparison formatter
    st.session_state["last_answer_policy_audit_summary"] = None  # ← V3.0d lanjutan: Logging intent_plan & doc_aggregation_meta
    return rm, logger


def get_or_create_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    # Jika belum ada run di session, buat
    if "rm" not in st.session_state or "logger" not in st.session_state:
        return _new_run(cfg, config_path)
    # Jika config path berubah, reset run baru
    if st.session_state.get("config_path") != config_path:
        return _new_run(cfg, config_path)
    return st.session_state["rm"], st.session_state["logger"]


def build_history_window_v2(
    history_pairs: List[Dict[str, Any]],
    *,
    use_memory: bool,
    mem_window: int,
    segment_on_topic_shift: bool,
    anaphora_skip_last_shift: bool,
    current_is_anaphora: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    History policy (robust):
    - Default: segment setelah topic_shift terakhir (include shift) agar topik baru tidak tercampur.
    - Kasus khusus (penting): jika query sekarang ANAFORA dan last history adalah TOPIC_SHIFT,
        maka memakai window SEBELUM shift (pre-shift), karena anafora biasanya merujuk topik sebelum shift.
    """
    meta: Dict[str, Any] = {
        "segment_on_topic_shift": bool(segment_on_topic_shift),
        "anaphora_skip_last_shift": bool(anaphora_skip_last_shift),
        "current_is_anaphora": bool(current_is_anaphora),
        "last_shift_turn_id": None,
        "used_pre_shift": False,
        "window_len": 0,
        "reason": None,
    }

    if not use_memory or mem_window <= 0 or not history_pairs:
        meta["reason"] = "no_memory_or_empty_history"
        return [], meta

    pairs = history_pairs

    # Cari last shift index
    last_shift_idx: Optional[int] = None
    for i in range(len(pairs) - 1, -1, -1):
        df = pairs[i].get("detect_flags", {}) or {}
        if bool(df.get("topic_shift")) or bool(pairs[i].get("topic_shift")):
            last_shift_idx = i
            meta["last_shift_turn_id"] = pairs[i].get("turn_id")
            break

    # Kasus khusus: anafora setelah shift -> pakai pre-shift
    if (
        current_is_anaphora
        and anaphora_skip_last_shift
        and last_shift_idx is not None
        and last_shift_idx == len(pairs) - 1  # shift adalah history paling akhir
    ):
        pre = pairs[:last_shift_idx]  # sebelum shift
        window = pre[-int(mem_window) :] if pre else []
        meta["used_pre_shift"] = True
        meta["reason"] = "anaphora_after_shift_use_pre_shift"
        meta["window_len"] = len(window)
        return window, meta

    # Default: segment setelah last shift (include shift)
    if segment_on_topic_shift and last_shift_idx is not None:
        pairs = pairs[last_shift_idx:]

    window = pairs[-int(mem_window) :]
    meta["reason"] = "default_segment"
    meta["window_len"] = len(window)
    return window, meta


def extract_doc_focus_from_query(q: str) -> List[str]:
    """
    Ambil sinyal doc-focus dari query user, misalnya:
    - ada string berakhiran .pdf
    - ada token doc_id seperti "ITERA_2023_SistemMonitoringTA_Prototyping"
    Return list allowlist (doc_id dan/atau source_file).
    """
    t = (q or "").strip()
    if not t:
        return []

    found: List[str] = []

    # 1) tangkap "....pdf" (baik pakai kutip atau tidak)
    for m in re.findall(r"([A-Za-z0-9_]+\.pdf)", t, flags=re.IGNORECASE):
        if m:
            found.append(m)
            # juga tambahkan doc_id tanpa .pdf
            if m.lower().endswith(".pdf"):
                found.append(m[:-4])

    # 2) tangkap doc_id style dataset: ITERA_2023_...
    for m in re.findall(r"\bITERA_\d{4}_[A-Za-z0-9_]+\b", t):
        if m:
            found.append(m)

    # dedupe + rapikan
    found = [x.strip() for x in found if x.strip()]
    return list(dict.fromkeys(found))


def build_steps_standalone_query(methods: List[str]) -> str:
    """
    Rewrite deterministic untuk pertanyaan tahapan yang anaphora:
    "tahapannya apa saja?" -> "Apa saja tahapan dari metode Prototyping?"
    """
    ms = [m for m in (methods or []) if m]
    if not ms:
        return ""
    labels = ", ".join(pretty_method_name(m) for m in ms)
    return f"Apa saja tahapan dari metode {labels}?"


# =========================
# Compression-followup helpers (T08 fix)
# =========================

_COMPRESSION_FOLLOWUP_PATTERNS = [
    r"\blebih\s+ringkas(?:\s+lagi)?\b",
    r"\blebih\s+singkat(?:\s+lagi)?\b",
    r"\bsingkatnya\b",
    r"\bversi\s+ringkas\b",
    r"\binti(?:nya)?\b",
    r"\bpoin\s+utamanya\b",
    r"\blangkah\s+utama(?:nya)?\b",
    r"\btahap(?:an)?\s+utama(?:nya)?\b",
]


def _history_has_steps_or_method_context(history_window: List[Dict[str, Any]]) -> bool:
    """
    True jika riwayat terdekat masih jelas berada pada konteks:
    - metode tertentu
    - tahapan / langkah / alur / prosedur
    """
    for h in reversed(history_window or []):
        q = str(h.get("query", "") or "")
        ap = str(h.get("answer_preview", "") or "")
        methods = h.get("methods") or []

        if methods:
            return True

        if is_steps_question(q) or is_method_question(q):
            return True

        if re.search(
            r"\b(tahap|tahapan|langkah|alur|prosedur)\w*\b",
            (q + " " + ap).lower(),
        ):
            return True

    return False


def is_compression_followup_question(
    query: str,
    history_window: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Deteksi follow-up kompresi / peringkasan yang secara semantik masih lanjutan,
    walau overlap token literal dengan history bisa sangat rendah.

    Contoh:
    - "Bisa lebih ringkas lagi langkah utamanya?"
    - "Singkatnya bagaimana?"
    - "Inti langkahnya apa?"
    """
    q = " ".join((query or "").lower().split())
    if not q:
        return False

    has_cue = any(
        re.search(p, q, flags=re.IGNORECASE)
        for p in _COMPRESSION_FOLLOWUP_PATTERNS
    )
    if not has_cue:
        return False

    # mode surface-only: dipakai sebelum history_window final dibangun
    if history_window is None:
        return True

    return _history_has_steps_or_method_context(history_window)


def build_steps_summary_standalone_query(methods: List[str]) -> str:
    """
    Rewrite deterministic untuk follow-up kompresi langkah:
    "Bisa lebih ringkas lagi langkah utamanya?"
    -> "Apa saja tahapan utama dari metode Prototyping?"
    """
    ms = [m for m in (methods or []) if m]
    if not ms:
        return ""
    labels = ", ".join(pretty_method_name(m) for m in ms)
    return f"Apa saja tahapan utama dari metode {labels}?"


def infer_target_methods(user_query: str, history_window: List[Dict[str, Any]]) -> List[str]:
    # 1) langsung dari query
    methods = detect_methods_in_text(user_query)
    if methods:
        return methods

    if not history_window:
        return []

    # 2) paling stabil: gunakan field "methods" yang disimpan di history (kalau ada)
    hist_methods: List[str] = []
    for h in history_window:
        ms = h.get("methods")
        if isinstance(ms, list):
            for m in ms:
                if isinstance(m, str) and m and (m not in hist_methods):
                    hist_methods.append(m)

    if hist_methods:
        return hist_methods

    # 3) fallback terakhir: scan query + answer_preview (cara lama)
    if (
        is_anaphora_question(user_query)
        or is_multi_target_question(user_query)
        or is_steps_question(user_query)
    ):
        hist_txt = " ".join(
            f"{h.get('query','')} {h.get('answer_preview','')}" for h in (history_window or [])
        )
        methods = detect_methods_in_text(hist_txt)
        if methods:
            return methods

    return []


def build_method_biased_query(base_query: str, method: str) -> str:
    q = (base_query or "").strip()
    return f"{q} metode {method}".strip()


def method_steps_query(base_query: str, method: str) -> str:
    """
    Query bias khusus untuk cari tahapan suatu metode.
    """
    q = (base_query or "").strip()
    m = (method or "").strip()
    # "tahapan" + metode lebih kuat daripada "... metode X"
    return f"tahapan {m}. {q}".strip()


def dedupe_nodes_by_chunk_id(nodes: List[Any]) -> List[Any]:
    best: Dict[str, Any] = {}
    for n in nodes:
        cid = getattr(n, "chunk_id", None) or (
            n.metadata.get("chunk_id") if getattr(n, "metadata", None) else None
        )
        if not cid:
            continue
        if cid not in best or float(n.score) < float(best[cid].score):
            best[cid] = n
    return sorted(best.values(), key=lambda x: float(x.score))


def select_nodes_with_coverage(
    nodes_sorted: List[Any],
    *,
    top_k: int,
    target_methods: List[str],
    max_per_doc: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    - Untuk setiap metode target, pilih 1 node paling relevan yang benar-benar mendukung metode tsb.
    - Prioritas: doc_id match (stabil) → strict text match → fallback (kecuali PROTOTYPING).
    """

    selected: List[Any] = []
    used = set()
    doc_count = Counter()

    coverage: Dict[str, Any] = {}

    def _pick_for_method(m: str) -> Optional[Any]:
        # Pass 1: doc_id hit + supports
        for n in nodes_sorted:
            if n.chunk_id in used:
                continue
            if _docid_hit_for_method(
                getattr(n, "doc_id", ""), m
            ) and node_supports_method_for_coverage(n, m):
                return n

        # Pass 2: supports (strict text / rule)
        for n in nodes_sorted:
            if n.chunk_id in used:
                continue
            if node_supports_method_for_coverage(n, m):
                return n

        # Pass 3: fallback ringan (untuk non-PROTOTYPING) pakai detect_methods_in_text biasa
        if m != "PROTOTYPING":
            for n in nodes_sorted:
                if n.chunk_id in used:
                    continue
                if m in set(detect_methods_in_text(getattr(n, "text", "") or "")):
                    return n

        return None

    # 1) minimal 1 node per metode
    for m in target_methods:
        pick = _pick_for_method(m)
        if pick:
            selected.append(pick)
            used.add(pick.chunk_id)
            doc_count[pick.doc_id] += 1
            coverage[m] = {
                "found": True,
                "chunk_id": pick.chunk_id,
                "doc_id": pick.doc_id,
                "docid_hit": _docid_hit_for_method(pick.doc_id, m),
            }
        else:
            coverage[m] = {"found": False, "chunk_id": None, "doc_id": None, "docid_hit": False}

    # 2) isi sisa slot dengan best-score, respect max_per_doc
    for n in nodes_sorted:
        if len(selected) >= int(top_k):
            break
        if n.chunk_id in used:
            continue
        if int(max_per_doc) > 0 and doc_count[n.doc_id] >= int(max_per_doc):
            continue
        selected.append(n)
        used.add(n.chunk_id)
        doc_count[n.doc_id] += 1

    meta = {"target_methods": target_methods, "coverage": coverage, "n_selected": len(selected)}
    return selected[: int(top_k)], meta


def _method_surface_patterns(method: str) -> List[str]:
    """
    Regex cue yang lebih ketat untuk mendeteksi representasi metode pada node.

    Tujuan:
    - hindari false positive karena kata lepas seperti "prototype" di node yang
        sebenarnya bukan membahas metode Prototyping sebagai target utama
    - cukup fleksibel untuk menangkap pola umum seperti:
        "metode RAD", "model prototyping", "tahapan metode RAD", dst.
    """
    m = str(method or "").strip().upper()

    if m == "RAD":
        return [
            r"\b(?:metode|model)\s+(?:rapid\s+application\s+development|rad)\b",
            r"\b(?:rapid\s+application\s+development|rad)\b\s+(?:adalah|merupakan|digunakan)\b",
            r"\btahapan?\b[^.\n]{0,80}\b(?:rapid\s+application\s+development|rad)\b",
            r"\b(?:rapid\s+application\s+development|rad)\b[^.\n]{0,80}\btahapan?\b",
            r"\blangkah(?:-langkah)?\b[^.\n]{0,80}\b(?:rapid\s+application\s+development|rad)\b",
        ]

    if m == "PROTOTYPING":
        return [
            r"\b(?:metode|model)\s+prototyp(?:e|ing)\b",
            r"\bprototyp(?:e|ing)\b\s+(?:adalah|merupakan|digunakan)\b",
            r"\btahapan?\b[^.\n]{0,80}\bprototyp(?:e|ing)\b",
            r"\bprototyp(?:e|ing)\b[^.\n]{0,80}\btahapan?\b",
            r"\blangkah(?:-langkah)?\b[^.\n]{0,80}\bprototyp(?:e|ing)\b",
        ]

    if m == "EXTREME_PROGRAMMING":
        return [
            r"\b(?:metode|model)\s+extreme\s+programming\b",
            r"\bextreme\s+programming\b\s+(?:adalah|merupakan|digunakan)\b",
            r"\b(?:metode|model)\s+xp\b",
            r"\bxp\b\s+(?:adalah|merupakan|digunakan)\b",
        ]

    if m == "WATERFALL":
        return [
            r"\b(?:metode|model)\s+waterfall\b",
            r"\bwaterfall\b\s+(?:adalah|merupakan|digunakan)\b",
        ]

    # fallback generik: exact term
    term = re.escape(str(method or "").strip().lower())
    return [rf"\b{term}\b"]


def _has_strong_method_cue(text: str, method: str) -> bool:
    """
    True jika text mengandung cue metode yang cukup kuat.
    """
    t = " ".join(str(text or "").lower().split())
    if not t:
        return False
    for pat in _method_surface_patterns(method):
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def _node_metadata_blob(n: Any) -> str:
    """
    Gabungkan surface metadata yang sering membawa sinyal metode:
    doc_id, source_file, judul, bab_label, stream.
    """
    meta = getattr(n, "metadata", {}) or {}
    parts = [
        str(getattr(n, "doc_id", "") or ""),
        str(meta.get("source_file", "") or ""),
        str(meta.get("judul", "") or ""),
        str(meta.get("bab_label", "") or ""),
        str(meta.get("stream", "") or ""),
    ]
    return " | ".join(parts)


def _steps_signal_strength(text: str) -> int:
    """
    Heuristic strength untuk mendeteksi apakah node benar-benar step-like.

    Bukan sekadar ada kata 'tahap', tetapi lebih menekankan:
    - tahapan / langkah-langkah
    - pola enumerasi 1. 2. 3.
    - frase seperti 'tiga tahapan', 'fase'
    """
    t = " ".join(str(text or "").lower().split())
    if not t:
        return 0

    score = 0

    # strong lexical cues
    if "langkah-langkah" in t or "langkah langkah" in t:
        score += 4
    if "tahapan-tahapan" in t or "tahapan" in t:
        score += 4
    if re.search(r"\btiga\s+tahapan\b", t):
        score += 3
    if re.search(r"\bfase\b", t):
        score += 2

    # enumeration cues
    if re.search(r"(?:^|\s)1\.\s", t):
        score += 2
    if re.search(r"(?:^|\s)2\.\s", t):
        score += 2
    if re.search(r"(?:^|\s)3\.\s", t):
        score += 2

    # weak cues (tidak cukup sendirian)
    if re.search(r"\btahap\b", t):
        score += 1

    return score


def _node_has_strong_steps_signal(n: Any) -> bool:
    """
    True bila node cukup kuat untuk dianggap menjelaskan tahapan/langkah.
    Threshold sengaja > sekadar kata 'tahap' agar tidak terlalu permisif.
    """
    return _steps_signal_strength(_node_text(n)) >= 4


# =========================
# V3.0c: Coverage-preserving rescue after rerank
# =========================

def _node_supports_target_method(n: Any, method: str) -> bool:
    """
    Cek apakah node benar-benar mendukung metode target.

    V2 improvement:
    - doc_id/source_file/judul yang eksplisit mengandung metode → langsung valid
    - body text harus punya strong method cue (bukan sekadar kata lepas)
    - fallback ke matcher coverage lama hanya jika memang tidak ada sinyal kuat lain
    """
    # 1) Sinyal paling kuat: doc_id / metadata surface
    try:
        if _docid_hit_for_method(getattr(n, "doc_id", ""), method):
            return True
    except Exception:
        pass

    try:
        if _has_strong_method_cue(_node_metadata_blob(n), method):
            return True
    except Exception:
        pass

    # 2) Body text dengan cue yang cukup kuat
    try:
        if _has_strong_method_cue(_node_text(n), method):
            return True
    except Exception:
        pass

    # 3) Fallback lama (biarkan paling belakang agar tidak terlalu permisif)
    try:
        if node_supports_method_for_coverage(n, method):
            return True
    except Exception:
        pass

    return False


def _node_adequately_supports_method(
    n: Any,
    method: str,
    *,
    prefer_steps: bool,
) -> bool:
    """
    Adequate representation:
    - normal query : node cukup mendukung metode target
    - steps query  : node harus mendukung metode target DAN punya STRONG steps signal

    Perubahan penting:
    - tidak lagi memakai has_steps_signal() yang terlalu permisif
    - untuk steps/comparison, node yang hanya menyebut metode tanpa struktur tahapan
        tidak dianggap cukup
    """
    if not _node_supports_target_method(n, method):
        return False

    if prefer_steps:
        return _node_has_strong_steps_signal(n)

    return True


def _supported_target_methods(n: Any, target_methods: List[str]) -> List[str]:
    """Daftar metode target yang didukung node ini."""
    out: List[str] = []
    for m in target_methods:
        if _node_supports_target_method(n, m):
            out.append(m)
    return out


def _is_method_adequately_represented(
    nodes: List[Any],
    method: str,
    *,
    prefer_steps: bool,
) -> bool:
    """
    True jika dalam list nodes sudah ada minimal 1 node yang cukup mewakili metode tsb.
    """
    for n in nodes:
        if _node_adequately_supports_method(n, method, prefer_steps=prefer_steps):
            return True
    return False


def _node_docid_supports_method(n: Any, method: str) -> bool:
    """
    True jika doc_id node secara eksplisit mewakili metode target.
    Hal ini lebih ketat daripada sekadar mention di body text.
    """
    try:
        return _docid_hit_for_method(str(getattr(n, "doc_id", "") or ""), method)
    except Exception:
        return False


def _select_best_anchor_node(
    candidate_nodes: List[Any],
    method: str,
    *,
    preferred_doc_id: Optional[str] = None,
    preferred_chunk_id: Optional[str] = None,
    prefer_steps: bool = False,
) -> Optional[Any]:
    """
    Pilih 1 node anchor terbaik untuk metode target dari candidate pool pre-rerank.

    Final patch T02:
    - untuk query steps, kandidat dari doc yang benar-benar merepresentasikan metode
        (via doc_id hit) diprioritaskan di atas dokumen lain yang hanya menyebut metode
    - di dalam pool itu, pilih node yang paling step-rich
    - preferred_doc_id / preferred_chunk_id tetap dipakai sebagai tie-break
    """
    if not candidate_nodes:
        return None

    pref_doc = str(preferred_doc_id or "").strip()
    pref_chunk = str(preferred_chunk_id or "").strip()
    indexed_nodes = list(enumerate(candidate_nodes))

    def _doc_match(n: Any) -> int:
        return 1 if (pref_doc and str(getattr(n, "doc_id", "") or "") == pref_doc) else 0

    def _chunk_match(n: Any) -> int:
        return 1 if (pref_chunk and str(getattr(n, "chunk_id", "") or "") == pref_chunk) else 0

    def _docid_supports(n: Any) -> bool:
        return _node_docid_supports_method(n, method)

    def _supports(n: Any) -> bool:
        return _node_supports_target_method(n, method)

    def _steps_strength(n: Any) -> int:
        return _steps_signal_strength(_node_text(n))

    def _pick_best(pool: List[Tuple[int, Any]]) -> Optional[Any]:
        if not pool:
            return None
        # score priority:
        # 1) stronger steps signal
        # 2) same preferred doc
        # 3) exact preferred chunk
        # 4) earlier rank/order in candidate pool
        best = max(
            pool,
            key=lambda item: (
                _steps_strength(item[1]),     # PALING penting untuk steps query
                _doc_match(item[1]),          # tie-break 1
                _chunk_match(item[1]),        # tie-break 2
                -item[0],                     # candidate pool order lama
            ),
        )
        return best[1]

    if prefer_steps:
        # 1) Kandidat dari doc yang memang "milik" metode + strong steps signal
        pool = [
            (i, n) for i, n in indexed_nodes
            if _docid_supports(n) and _steps_strength(n) >= 4
        ]
        hit = _pick_best(pool)
        if hit is not None:
            return hit

        # 2) Kandidat dari doc yang memang "milik" metode (meski steps signal tidak sekuat threshold)
        pool = [
            (i, n) for i, n in indexed_nodes
            if _docid_supports(n)
        ]
        hit = _pick_best(pool)
        if hit is not None:
            return hit

        # 3) Fallback: kandidat manapun yang support metode + strong steps signal
        pool = [
            (i, n) for i, n in indexed_nodes
            if _supports(n) and _steps_strength(n) >= 4
        ]
        hit = _pick_best(pool)
        if hit is not None:
            return hit

        # 4) Fallback terakhir: kandidat manapun yang support metode
        pool = [
            (i, n) for i, n in indexed_nodes
            if _supports(n)
        ]
        hit = _pick_best(pool)
        if hit is not None:
            return hit

        return None

    # non-steps fallback
    # 1) exact preferred_chunk + support
    for _, n in indexed_nodes:
        if _chunk_match(n) and _supports(n):
            return n

    # 2) preferred_doc + support
    for _, n in indexed_nodes:
        if _doc_match(n) and _supports(n):
            return n

    # 3) Doc ID support
    for _, n in indexed_nodes:
        if _docid_supports(n):
            return n

    # 4) any support
    for _, n in indexed_nodes:
        if _supports(n):
            return n

    # 5) exact preferred chunk as last resort
    for _, n in indexed_nodes:
        if _chunk_match(n):
            return n

    return None


# LEGACY rescue helper — tidak lagi dipakai di main flow V3.0c
def enforce_method_coverage_after_rerank(
    *,
    reranked_nodes: List[Any],
    candidate_nodes: List[Any],
    target_methods: List[str],
    coverage_detail: Optional[Dict[str, Any]] = None,
    final_k: int,
    prefer_steps: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Jaga agar hasil final pasca-rerank tetap merepresentasikan semua metode target.

    Kapan dipakai?
    - comparison / multi-target
    - steps query
    - atau kasus coverage sudah berhasil menemukan metode target pre-rerank,
        tetapi reranker menghilangkan salah satunya dari final Top-N.

    Prinsip:
    - rerank order tetap dihormati semaksimal mungkin
    - namun jika metode target hilang / tidak adequately represented,
        inject 1 anchor node terbaik dari candidate pool pre-rerank
    - jika perlu, ganti node paling aman untuk dibuang (node non-target / duplicate)
    """
    methods = [m for m in target_methods if isinstance(m, str) and m]
    methods = list(dict.fromkeys(methods))

    final_nodes = list(reranked_nodes[: int(final_k)])

    meta: Dict[str, Any] = {
        "triggered": bool(methods),
        "applied": False,
        "prefer_steps": bool(prefer_steps),
        "target_methods": methods,
        "rescued_methods": [],
        "anchor_chunk_ids": {},
        "adequately_represented_after": {},
        "final_n": len(final_nodes),
    }

    if not methods or not candidate_nodes or not final_nodes:
        for m in methods:
            meta["adequately_represented_after"][m] = _is_method_adequately_represented(
                final_nodes,
                m,
                prefer_steps=prefer_steps,
            )
        return final_nodes, meta

    cov_map = coverage_detail if isinstance(coverage_detail, dict) else {}

    # bangun anchor candidate terbaik per metode
    anchors_by_method: Dict[str, Any] = {}
    for m in methods:
        _cov = cov_map.get(m) if isinstance(cov_map, dict) else None
        _pref_doc = None
        _pref_chunk = None
        if isinstance(_cov, dict):
            _pref_doc = str(_cov.get("doc_id") or "").strip() or None
            _pref_chunk = str(_cov.get("chunk_id") or "").strip() or None

        anchor = _select_best_anchor_node(
            candidate_nodes,
            m,
            preferred_doc_id=_pref_doc,
            preferred_chunk_id=_pref_chunk,
            prefer_steps=prefer_steps,
        )
        if anchor is not None:
            anchors_by_method[m] = anchor
            meta["anchor_chunk_ids"][m] = str(getattr(anchor, "chunk_id", "") or "")

    for m in methods:
        # Jika belum adequately represented, metode ini WAJIB diselamatkan.
        # Setelah V2, adequacy checker sudah lebih ketat:
        # - false positive lexical mention harus berkurang
        # - query steps butuh node yang benar-benar step-like
        if _is_method_adequately_represented(
            final_nodes,
            m,
            prefer_steps=prefer_steps,
        ):
            continue

        anchor = anchors_by_method.get(m)
        if anchor is None:
            continue

        anchor_cid = str(getattr(anchor, "chunk_id", "") or "")
        if anchor_cid and any(
            str(getattr(n, "chunk_id", "") or "") == anchor_cid for n in final_nodes
        ):
            # anchor sudah ada di final nodes; kalau masih tidak adequate,
            # berarti memang tidak ada candidate yang lebih baik untuk metode ini.
            continue

        drop_idx: Optional[int] = None

        # cari node yang paling aman untuk diganti:
        # prioritas drop:
        # 1) node yang tidak mendukung target_methods mana pun
        # 2) node yang hanya duplikasi / tidak menjadi satu-satunya representasi adequate
        for idx in range(len(final_nodes) - 1, -1, -1):
            cand = final_nodes[idx]
            supported = _supported_target_methods(cand, methods)

            # kandidat non-target = paling aman dibuang
            if not supported:
                drop_idx = idx
                break

            remaining = [n for j, n in enumerate(final_nodes) if j != idx]

            safe_to_drop = True
            for sm in supported:
                # hanya node yang "adequate" yang perlu diproteksi.
                # kalau node ini tidak adequate (mis. query steps tapi node tidak punya steps signal),
                # maka node tersebut aman untuk diganti dengan anchor yang lebih baik.
                if _node_adequately_supports_method(cand, sm, prefer_steps=prefer_steps):
                    if not _is_method_adequately_represented(
                        remaining,
                        sm,
                        prefer_steps=prefer_steps,
                    ):
                        safe_to_drop = False
                        break

            if safe_to_drop:
                drop_idx = idx
                break

        if drop_idx is None:
            # fallback terakhir: ganti node paling akhir
            drop_idx = len(final_nodes) - 1

        final_nodes[drop_idx] = anchor
        meta["applied"] = True
        meta["rescued_methods"].append(m)

    # dedupe sembari jaga urutan
    deduped: List[Any] = []
    seen_chunk_ids = set()

    for n in final_nodes:
        cid = str(getattr(n, "chunk_id", "") or "")
        if cid and cid in seen_chunk_ids:
            continue
        if cid:
            seen_chunk_ids.add(cid)
        deduped.append(n)

    final_nodes = deduped[: int(final_k)]

    # jika dedupe membuat jumlah final berkurang, isi lagi dari reranked_nodes lalu candidate_nodes
    if len(final_nodes) < int(final_k):
        for pool in (reranked_nodes, candidate_nodes):
            for n in pool:
                cid = str(getattr(n, "chunk_id", "") or "")
                if cid and cid in seen_chunk_ids:
                    continue
                if cid:
                    seen_chunk_ids.add(cid)
                final_nodes.append(n)
                if len(final_nodes) >= int(final_k):
                    break
            if len(final_nodes) >= int(final_k):
                break

    final_nodes = final_nodes[: int(final_k)]
    meta["final_n"] = len(final_nodes)

    for m in methods:
        meta["adequately_represented_after"][m] = _is_method_adequately_represented(
            final_nodes,
            m,
            prefer_steps=prefer_steps,
        )

    return final_nodes, meta


def _dedupe_nodes_keep_order(nodes: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()

    for n in nodes:
        cid = str(getattr(n, "chunk_id", "") or "")
        key = cid if cid else f"__obj_{id(n)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(n)

    return out


def build_final_nodes_with_anchor_reservation(
    *,
    reranked_nodes: List[Any],
    candidate_nodes: List[Any],
    target_methods: List[str],
    coverage_detail: Optional[Dict[str, Any]],
    final_k: int,
    prefer_steps: bool,
    top_n_base: int,
    top_n_effective: int,
    dynamic_top_n: bool,
    complex_query: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Deterministic anchor reservation:
    - tiap metode target reserve 1 anchor terbaik lebih dulu
    - sisa slot diisi dari urutan rerank
    - dedupe & fill dari candidate pool jika perlu

    Hal ini lebih tegas daripada adequacy heuristic.
    """
    methods = [m for m in target_methods if isinstance(m, str) and m]
    methods = list(dict.fromkeys(methods))
    final_k = max(1, int(final_k))

    meta: Dict[str, Any] = {
        "triggered": bool(methods),
        "applied": False,
        "modified": False,
        "strategy": "deterministic_anchor_reservation",
        "prefer_steps": bool(prefer_steps),
        "dynamic_top_n": bool(dynamic_top_n),
        "complex_query": bool(complex_query),
        "top_n_base": int(top_n_base),
        "top_n_effective": int(top_n_effective),
        "target_methods": methods,
        "reserved_methods": [],
        "reserved_anchor_chunk_ids": {},
        "final_n": 0,
    }

    if not methods:
        final_nodes = list(reranked_nodes[:final_k])
        meta["final_n"] = len(final_nodes)
        return final_nodes, meta

    cov_map = coverage_detail if isinstance(coverage_detail, dict) else {}

    reserved_nodes: List[Any] = []
    reserved_methods: List[str] = []
    reserved_anchor_chunk_ids: Dict[str, str] = {}
    seen_reserved = set()

    # 1) Reserve 1 anchor per target method
    for m in methods:
        _cov = cov_map.get(m) if isinstance(cov_map, dict) else None
        _pref_doc = None
        _pref_chunk = None
        if isinstance(_cov, dict):
            _pref_doc = str(_cov.get("doc_id") or "").strip() or None
            _pref_chunk = str(_cov.get("chunk_id") or "").strip() or None

        anchor = _select_best_anchor_node(
            candidate_nodes,
            m,
            preferred_doc_id=_pref_doc,
            preferred_chunk_id=_pref_chunk,
            prefer_steps=prefer_steps,
        )
        if anchor is None:
            continue

        cid = str(getattr(anchor, "chunk_id", "") or "")
        reserved_anchor_chunk_ids[m] = cid

        if cid and cid in seen_reserved:
            continue

        if cid:
            seen_reserved.add(cid)
        reserved_nodes.append(anchor)
        reserved_methods.append(m)

    # 2) Tambahkan remainder dari hasil rerank
    remainder: List[Any] = []
    for n in reranked_nodes:
        cid = str(getattr(n, "chunk_id", "") or "")
        if cid and cid in seen_reserved:
            continue
        remainder.append(n)

    final_nodes = reserved_nodes + remainder
    final_nodes = _dedupe_nodes_keep_order(final_nodes)

    # 3) Jika masih kurang, isi dari candidate pool pre-rerank
    if len(final_nodes) < final_k:
        seen_final = {
            str(getattr(n, "chunk_id", "") or "")
            for n in final_nodes
            if str(getattr(n, "chunk_id", "") or "")
        }
        for n in candidate_nodes:
            cid = str(getattr(n, "chunk_id", "") or "")
            if cid and cid in seen_final:
                continue
            final_nodes.append(n)
            if cid:
                seen_final.add(cid)
            if len(final_nodes) >= final_k:
                break

    original_ids = [
        str(getattr(n, "chunk_id", "") or "")
        for n in list(reranked_nodes[:final_k])
    ]
    final_nodes = final_nodes[:final_k]
    final_ids = [str(getattr(n, "chunk_id", "") or "") for n in final_nodes]

    meta["applied"] = bool(reserved_methods)
    meta["modified"] = final_ids != original_ids
    meta["reserved_methods"] = reserved_methods
    meta["reserved_anchor_chunk_ids"] = reserved_anchor_chunk_ids
    meta["final_n"] = len(final_nodes)

    return final_nodes, meta


def _build_rerank_meta_map(nodes: List[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Ambil map chunk_id -> rerank metadata dari node-node hasil rerank/candidate pool.
    Dipakai untuk menyinkronkan metadata rerank ke final nodes setelah anchor reservation.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for n in nodes or []:
        cid = str(getattr(n, "chunk_id", "") or "")
        if not cid:
            continue

        rs = getattr(n, "rerank_score", None)
        rr = getattr(n, "rerank_rank", None)

        if rs is None and rr is None:
            continue

        out[cid] = {
            "rerank_score": float(rs) if rs is not None else None,
            "rerank_rank": int(rr) if rr is not None else None,
        }

    return out


def _sync_rerank_metadata_to_final_nodes(
    final_nodes: List[Any],
    rerank_meta_map: Dict[str, Dict[str, Any]],
) -> List[Any]:
    """
    Pastikan setiap final node membawa rerank_score / rerank_rank yang benar,
    termasuk node yang masuk lewat anchor reservation.
    """
    for n in final_nodes or []:
        cid = str(getattr(n, "chunk_id", "") or "")
        if not cid:
            continue

        meta = rerank_meta_map.get(cid)
        if not meta:
            continue

        if meta.get("rerank_score") is not None:
            n.rerank_score = meta["rerank_score"]
        if meta.get("rerank_rank") is not None:
            n.rerank_rank = meta["rerank_rank"]

    return final_nodes


def _tag_reserved_anchor_nodes(
    final_nodes: List[Any],
    anchor_meta: Optional[Dict[str, Any]],
) -> List[Any]:
    """
    Tandai final node yang masuk karena anchor reservation, agar UI dapat membedakan:
    - node yang punya rerank_rank/rerank_score
    - node reserved anchor yang memang valid walau rerank metadata = None
    """
    anchor_meta = anchor_meta if isinstance(anchor_meta, dict) else {}
    reserved_map = anchor_meta.get("reserved_anchor_chunk_ids") or {}
    if not isinstance(reserved_map, dict):
        reserved_map = {}

    # chunk_id -> method
    reverse_map: Dict[str, str] = {}
    for method, cid in reserved_map.items():
        _cid = str(cid or "").strip()
        _m = str(method or "").strip()
        if _cid and _m:
            reverse_map[_cid] = _m

    for n in final_nodes or []:
        cid = str(getattr(n, "chunk_id", "") or "")
        reserved_for = reverse_map.get(cid)

        n.is_reserved_anchor = bool(reserved_for)
        n.reserved_for_method = reserved_for

        # Label kecil supaya UI bisa memakai satu field saja jikalau sanggup
        if reserved_for:
            n.rerank_display_label = f"anchor:{reserved_for}"
        else:
            n.rerank_display_label = None

    return final_nodes


def _finalize_nodes_for_answer_rerank_ui(
    *,
    current_nodes: List[Any],
    canonical_preferred_nodes: List[Any],
    canonical_fallback_nodes: List[Any],
    rerank_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
    anchor_meta: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Final hardening untuk node final yang akan dipakai:
    - generator
    - session_state["last_nodes"]
    - turn_data["nodes"]
    - CTX Mapping / After Rerank UI

    Prinsip:
    1) pertahankan urutan current_nodes (karena itu urutan CTX final)
    2) ganti object dengan versi kanonik jika ada
    3) sinkronkan ulang rerank_score / rerank_rank
    4) tag reserved-anchor agar UI jujur
    """
    current_nodes = list(current_nodes or [])
    if not current_nodes:
        return []

    chunk_ids = [
        str(getattr(n, "chunk_id", "") or "").strip()
        for n in current_nodes
        if str(getattr(n, "chunk_id", "") or "").strip()
    ]
    if not chunk_ids:
        return current_nodes

    final_nodes = _pick_nodes_by_chunk_ids(
        canonical_preferred_nodes,
        list(current_nodes or []) + list(canonical_fallback_nodes or []),
        chunk_ids,
    )

    final_nodes = _sync_rerank_metadata_to_final_nodes(
        final_nodes,
        dict(rerank_meta_map or {}),
    )
    final_nodes = _tag_reserved_anchor_nodes(
        final_nodes,
        anchor_meta,
    )

    # Default eksplisit agar UI tidak membawa state lama / stale attrs.
    for n in final_nodes:
        if not hasattr(n, "is_reserved_anchor"):
            n.is_reserved_anchor = False
        if not hasattr(n, "reserved_for_method"):
            n.reserved_for_method = None
        if not hasattr(n, "rerank_display_label"):
            n.rerank_display_label = None

    return final_nodes


# =========================
# Topic Shift Detection
# =========================

_SHIFT_MARKERS = [
    "topik lain",
    "topik berbeda",
    "bahasan lain",
    "ganti topik",
    "sekarang bahas",
    "selanjutnya bahas",
    "ngomongin yang lain",
]

# stopwords ringan untuk deteksi overlap topik (bukan untuk retrieval)
_STOP_TOPIC = {
    "apa",
    "yang",
    "dan",
    "atau",
    "dengan",
    "dalam",
    "pada",
    "untuk",
    "dari",
    "ke",
    "ini",
    "itu",
    "tersebut",
    "nya",
    "jelaskan",
    "sebutkan",
    "menurut",
    "dokumen",
    "skripsi",
    "sistem",
    "metode",
    "model",
    "bab",
    "tabel",
    "gambar",
}


def _content_terms(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {t for t in toks if t not in _STOP_TOPIC}


def detect_topic_shift(
    user_query: str,
    history_window: List[Dict[str, Any]],
    *,
    overlap_threshold: float = 0.15,
    min_terms: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Deteksi topic shift agar memory/contextualize tidak "memaksa nyambung" ke topik lama.
    Heuristik konservatif:
    - Jika ada marker eksplisit -> shift
    - Jika query anafora -> bukan shift
    - Jika overlap term penting query vs history rendah -> shift

    Return:
        (is_shift, meta_debug)
    """
    q = (user_query or "").strip()
    if not q:
        return False, {"reason": "empty_query"}
    if not history_window:
        return False, {"reason": "no_history"}

    t = q.lower()
    if any(m in t for m in _SHIFT_MARKERS):
        return True, {"reason": "explicit_marker", "marker_hit": True}

    # follow-up kompresi / peringkasan: masih topik sama
    if is_compression_followup_question(q, history_window):
        return False, {"reason": "compression_followup"}

    # follow-up anafora biasa: anggap masih topik sama
    if is_anaphora_question(q):
        return False, {"reason": "anaphora"}

    # gabungkan ringkas history (query + answer_preview)
    hist_txt = " ".join(f"{h.get('query','')} {h.get('answer_preview','')}" for h in history_window)

    q_terms = _content_terms(q)
    h_terms = _content_terms(hist_txt)

    if len(q_terms) < int(min_terms) or len(h_terms) < int(min_terms):
        return False, {
            "reason": "too_few_terms",
            "q_terms_n": len(q_terms),
            "h_terms_n": len(h_terms),
        }

    inter = q_terms & h_terms
    overlap_q = len(inter) / max(1, len(q_terms))  # fokus ke query
    is_shift = float(overlap_q) < float(overlap_threshold)

    return bool(is_shift), {
        "reason": "overlap_check",
        "overlap_threshold": float(overlap_threshold),
        "overlap_q": round(float(overlap_q), 6),
        "q_terms_n": int(len(q_terms)),
        "h_terms_n": int(len(h_terms)),
        "inter_n": int(len(inter)),
        "inter_terms_sample": sorted(list(inter))[:10],
    }


def decide_max_bullets(
    *,
    user_query: str,
    contexts: List[str],
    default_max: int = 3,
    expanded_max: int = 5,
) -> Tuple[int, Dict[str, Any]]:
    """
    Default 3, naik ke 5 hanya untuk kondisi tertentu.
    Return: (max_bullets, meta_debug)
    """
    q = (user_query or "").strip()
    methods = detect_methods_in_contexts(contexts)
    multi_methods = len(methods) >= 2

    multi_target = is_multi_target_question(q)
    anaphora = is_anaphora_question(q)

    # naik jika:
    # - pertanyaan multi-target
    # - atau pertanyaan anafora DAN konteks memuat >= 2 metode
    if multi_target or (anaphora and multi_methods):
        max_b = int(expanded_max)
    else:
        max_b = int(default_max)

    meta = {
        "multi_target": bool(multi_target),
        "anaphora": bool(anaphora),
        "methods_detected": methods,
        "multi_methods": bool(multi_methods),
        "max_bullets": int(max_b),
    }
    return max_b, meta


def _make_self_query_log(
    *,
    enabled: bool,
    semantic_query: Optional[str] = None,
    filter_applied: bool = False,
    filter_fields_used: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    extracted_metadata: Optional[Dict[str, Any]] = None,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shape log self_query yang STABIL di semua jalur:
    - normal path
    - metadata_routed
    - self_query disabled
    - disabled_in_eval
    - precheck 0 hasil
    - exception path
    """
    return {
        "enabled": bool(enabled),
        "semantic_query": semantic_query,
        "filter_applied": bool(filter_applied),
        "filter_fields_used": list(filter_fields_used or []),
        "filters": filters,
        "extracted_metadata": dict(extracted_metadata or {}),
        "fallback_reason": fallback_reason,
    }


def _self_query_log_from_result(
    result: Optional[SelfQueryResult],
    *,
    enabled: bool,
    semantic_query: Optional[str] = None,
    filter_applied: Optional[bool] = None,
    fallback_reason: Optional[str] = None,
    clear_filter_payload: bool = False,
) -> Dict[str, Any]:
    """
    Normalisasi SelfQueryResult -> shape log yang konsisten.
    clear_filter_payload=True dipakai saat filter sempat terbentuk,
    tetapi TIDAK jadi diterapkan (mis. pre-check 0 hasil).
    """
    base = _make_self_query_log(enabled=enabled)

    if result is not None:
        base.update(result.to_log_dict())

    base["enabled"] = bool(enabled)

    if semantic_query is not None:
        base["semantic_query"] = semantic_query

    if filter_applied is not None:
        base["filter_applied"] = bool(filter_applied)

    if fallback_reason is not None:
        base["fallback_reason"] = fallback_reason

    if clear_filter_payload and not base["filter_applied"]:
        base["filters"] = None
        base["filter_fields_used"] = []

    return base


def _has_explicit_metadata_constraints(sq_log: Optional[Dict[str, Any]]) -> bool:
    """
    True jika self_query log menunjukkan bahwa user memang meminta metadata eksplisit
    (tahun / prodi / penulis), walaupun filter akhirnya tidak jadi diterapkan.
    """
    if not isinstance(sq_log, dict):
        return False

    ext = sq_log.get("extracted_metadata") or {}
    if not isinstance(ext, dict):
        return False

    tahun = ext.get("tahun")
    prodi = str(ext.get("prodi") or "").strip()
    penulis = str(ext.get("penulis") or "").strip()

    return (tahun is not None) or bool(prodi) or bool(penulis)


def _should_force_metadata_not_found_answer(sq_log: Optional[Dict[str, Any]]) -> bool:
    """
    Guard utama:
    Jika self-query mendeteksi metadata eksplisit, tetapi pre-check mengembalikan 0 hasil,
    maka generator tidak boleh membuat klaim positif dari retrieval fallback umum.
    """
    if not isinstance(sq_log, dict):
        return False

    if str(sq_log.get("fallback_reason") or "") != "precheck_no_results":
        return False

    return _has_explicit_metadata_constraints(sq_log)


def _build_metadata_not_found_answer() -> str:
    """
    Memakai canonical global not-found format agar build_generation_meta()
    dan guardrail generator tetap konsisten.
    """
    return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"


def _single_target_steps_status(meta: Optional[Dict[str, Any]]) -> str:
    return str((meta or {}).get("status", "") or "").strip().lower()


def _should_force_single_target_steps_safe_answer_final(
    intent_plan: Optional[Dict[str, Any]],
    steps_meta: Optional[Dict[str, Any]],
) -> bool:
    """
    Final hard guard di answer layer.
    Dipakai sebagai lapisan pengaman terakhir jika generator masih lolos
    ke jawaban generik padahal route planner + runtime meta sudah jelas.
    """
    primary_intent = str((intent_plan or {}).get("primary_intent", "") or "").strip().lower()
    if primary_intent != "single_target_steps":
        return False

    status = _single_target_steps_status(steps_meta)
    return status in {"ambiguous_owner_docs", "unresolved_owner_doc"}


def _build_single_target_steps_safe_answer_final(
    steps_meta: Optional[Dict[str, Any]],
) -> str:
    meta = dict(steps_meta or {})
    method_label = str(meta.get("target_method_label", "") or meta.get("target_method", "") or "metode").strip() or "metode"
    status = _single_target_steps_status(meta)

    doc_bits = [
        str(x).strip()
        for x in (meta.get("ambiguous_candidate_doc_ids") or [])
        if str(x).strip()
    ]
    doc_text = ", ".join(doc_bits[:3]) if doc_bits else "belum ada dokumen owner tunggal yang terkonfirmasi"

    if status == "ambiguous_owner_docs":
        return (
            f"Jawaban: Ada lebih dari satu dokumen kandidat yang menggunakan metode {method_label}, "
            "sehingga langkah-langkah lengkapnya belum bisa dipastikan untuk satu skripsi target.\n"
            "Bukti:\n"
            f"- Kandidat dokumen owner yang terdeteksi: {doc_text}"
        )

    if status == "unresolved_owner_doc":
        return (
            f"Jawaban: Dokumen skripsi target untuk metode {method_label} belum bisa dipastikan secara tunggal, "
            "sehingga langkah-langkah lengkapnya belum dapat dijabarkan dengan aman.\n"
            "Bukti:\n"
            f"- Hasil disambiguasi owner document belum menghasilkan satu dokumen target yang tegas: {doc_text}"
        )

    return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"


# =========================
# Reranker telemetry gating (T10 polish)
# =========================

def _is_reranker_feature_enabled(cfg: Dict[str, Any]) -> bool:
    """
    True hanya jika config aktif memang punya section reranker.enabled = true.
    Untuk legacy configs (v1_5.yaml / v2.yaml), hasilnya False.
    """
    rk_cfg = cfg.get("reranker") or {}
    return bool(rk_cfg.get("enabled", False))


def _make_disabled_reranker_info(
    cfg: Dict[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    """
    Placeholder info yang netral / bersih untuk jalur:
    - legacy config tanpa reranker
    - metadata router bypass
    - path non-rerank lain yang memang sengaja tidak memakai reranker
    """
    rk_cfg = cfg.get("reranker") or {}
    model_name = rk_cfg.get("model_name")

    return {
        "loaded": False,
        "model_name": model_name if model_name else None,
        "requested_device": None,
        "resolved_device": None,
        "fallback_reason": None,
        "load_failed": False,
        "error_msg": None,
        "unloaded_after_use": False,
        "loaded_after_unload": False,
        "disabled_reason": reason,
    }


def _build_reranker_telemetry(
    *,
    cfg: Dict[str, Any],
    feature_context_enabled: bool,
    reranker_applied: bool,
    reranker_info: Optional[Dict[str, Any]],
    nodes_before_rerank: Optional[List[Any]],
    top_n_base: Optional[int],
    top_n_effective: Optional[int],
    rerank_anchor_reservation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Satu pintu untuk menormalkan telemetry reranker agar:
    - legacy config tidak membawa jejak reranker palsu
    - metadata router tidak membawa stale reranker_info
    - fail-path T06 tetap menyimpan info error dengan jujur
    """
    feature_enabled = _is_reranker_feature_enabled(cfg)

    # 1) Legacy config / reranker memang mati di config
    if not feature_enabled:
        return {
            "applied": False,
            "candidate_count": 0,
            "top_n": None,
            "top_n_base": None,
            "top_n_effective": None,
            "info": _make_disabled_reranker_info(
                cfg,
                reason="disabled_in_config",
            ),
            "anchor_reservation": None,
        }

    # 2) Reranker ada di config, tapi untuk jalur ini sengaja tidak dipakai
    #    (mis. metadata_router bypass).
    if not feature_context_enabled:
        return {
            "applied": False,
            "candidate_count": 0,
            "top_n": None,
            "top_n_base": None,
            "top_n_effective": None,
            "info": _make_disabled_reranker_info(
                cfg,
                reason="bypass_path",
            ),
            "anchor_reservation": None,
        }

    # 3) Jalur reranker memang relevan untuk turn ini.
    #    Jika gagal load / graceful skip, info error tetap dipertahankan.
    info = dict(reranker_info or {})
    if not info:
        info = _make_disabled_reranker_info(cfg, reason="not_attempted")

    return {
        "applied": bool(reranker_applied),
        "candidate_count": len(nodes_before_rerank or []),
        "top_n": int(top_n_effective) if top_n_effective is not None else None,
        "top_n_base": int(top_n_base) if top_n_base is not None else None,
        "top_n_effective": int(top_n_effective) if top_n_effective is not None else None,
        "info": info,
        "anchor_reservation": (
            rerank_anchor_reservation if bool(reranker_applied) else None
        ),
    }


def _deepcopy_log_obj(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    return copy.deepcopy(obj)


def _intent_targets_for_log(intent_plan: Optional[Dict[str, Any]]) -> List[str]:
    ip = intent_plan or {}
    out: List[str] = []

    for key in ("target_methods", "comparison_targets"):
        vals = ip.get(key) or []
        if not isinstance(vals, list):
            continue
        for x in vals:
            s = str(x or "").strip()
            if s and s not in out:
                out.append(s)

    return out


def _accepted_doc_ids_from_doc_aggregation(
    doc_aggregation_meta: Optional[Dict[str, Any]],
) -> List[str]:
    dam = doc_aggregation_meta or {}
    out: List[str] = []

    accepted = dam.get("accepted_docs") or []
    if not isinstance(accepted, list):
        return out

    for item in accepted:
        if not isinstance(item, dict):
            continue
        did = str(item.get("doc_id", "") or "").strip()
        if did and did not in out:
            out.append(did)

    return out


def _build_intent_plan_log(
    intent_plan: Optional[Dict[str, Any]],
    *,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return _deepcopy_log_obj(intent_plan)


def _build_doc_aggregation_log(
    doc_aggregation_meta: Optional[Dict[str, Any]],
    *,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return _deepcopy_log_obj(doc_aggregation_meta)


def _build_answer_policy_audit_summary(
    intent_plan: Optional[Dict[str, Any]],
    doc_aggregation_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ip = intent_plan if isinstance(intent_plan, dict) else {}
    dam = doc_aggregation_meta if isinstance(doc_aggregation_meta, dict) else {}

    return {
        "intent_primary": str(ip.get("primary_intent", "") or "") or None,
        "intent_targets": _intent_targets_for_log(ip),
        "intent_doc_locked": bool(ip.get("doc_locked", False)),
        "intent_needs_doc_aggregation": bool(ip.get("needs_doc_aggregation", False)),
        "intent_needs_explanation_expansion": bool(ip.get("needs_explanation_expansion", False)),
        "intent_needs_steps_expansion": bool(ip.get("needs_steps_expansion", False)),
        "doc_aggregation_executed": bool(dam.get("executed", False)),
        "doc_aggregation_applied": bool(dam.get("applied", False)),
        "doc_aggregation_target_method": str(dam.get("target_method", "") or "") or None,
        "doc_aggregation_candidate_doc_count": int(dam.get("candidate_doc_count", 0) or 0),
        "doc_aggregation_accepted_doc_count": int(dam.get("accepted_doc_count", 0) or 0),
        "doc_aggregation_trigger_reason": str(dam.get("trigger_reason", "") or "") or None,
        "doc_aggregation_accepted_doc_ids": _accepted_doc_ids_from_doc_aggregation(dam),
    }


def main() -> None:
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", page_icon="🔍", layout="wide")
    st.title("🔍 RAG ThesisDoc System - V3.0d")

    load_dotenv()  # baca .env (lokal)

    # Root project untuk resolve file
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> root
    # ambil data_raw dari config kalau ada, fallback "data_raw"
    # (akan set setelah cfg terbaca)
    # DATA_RAW_DIR = Path("data_raw")

    # ===== Sidebar: Controls =====
    st.sidebar.header("Controls & Information")

    # Config path
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/v3.yaml",
        help="Path YAML config yang dipakai untuk run ini (index, retrieval, llm, memory).",
    )
    cfg = load_config(config_path)
    data_raw_dir = Path(cfg.get("paths", {}).get("data_raw_dir", "data_raw"))

    # T10 polish:
    # Jika config lama / legacy dipakai (tanpa reranker aktif),
    # bersihkan state module-level reranker agar info lama tidak "bocor" ke log / sidebar / history viewer.
    if not _is_reranker_feature_enabled(cfg):
        try:
            unload_reranker()
        except Exception:
            pass

    # init/reuse run per session
    rm, logger = get_or_create_run(cfg, config_path)
    logger.info(f"Run active: {rm.run_id}")

    # Guard: pastikan state baru selalu ada (terutama saat code diupdate mid-session)
    if "turn_data" not in st.session_state:
        st.session_state["turn_data"] = {}
    if "viewing_turn_id" not in st.session_state:
        st.session_state["viewing_turn_id"] = None

    # info provider LLM
    provider = os.getenv("LLM_PROVIDER", "(not set)")
    groq_model = os.getenv("GROQ_MODEL", "(not set)")
    openai_model = os.getenv("OPENAI_MODEL", "(not set)")

    # Tombol new session (reset run_id + history)
    if st.sidebar.button("➕ New Session (Run Baru)"):
        rm, logger = _new_run(cfg, config_path)
        st.rerun()

    # =========================
    # Run Information
    # =========================
    st.sidebar.divider()
    st.sidebar.subheader("📰 Run Information")
    st.sidebar.write("Run ID:", rm.run_id)
    _idx_cfg = cfg.get("index", {})
    if "collections" in _idx_cfg:
        _cols_map = _idx_cfg["collections"]
        st.sidebar.write("Collection (Narasi):", _cols_map.get("narasi", "—"))
        st.sidebar.write("Collection (Sitasi):", _cols_map.get("sitasi", "—"))
    else:
        st.sidebar.write("Collection:", _idx_cfg.get("collection_name", "—"))
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])

    st.sidebar.write("LLM Provider:", provider)
    if provider == "groq":
        st.sidebar.write("Model:", groq_model)
    elif provider == "openai":
        st.sidebar.write("Model:", openai_model)

    # ← V3.0c: Tampilkan info reranker model di sidebar
    _rk_sidebar_cfg = cfg.get("reranker") or {}
    if bool(_rk_sidebar_cfg.get("enabled", False)):
        st.sidebar.write("Reranker:", _rk_sidebar_cfg.get("model_name", "—"))

    # --- Runtime status turn terakhir ---
    st.sidebar.divider()
    last_turn_status_slot = st.sidebar.empty()  # diisi setelah processing

    # ===================================================
    # Settings / Config
    # Override controls (UI-level) untuk eksperimen cepat
    # ===================================================
    st.sidebar.divider()
    st.sidebar.subheader("🛠️ Settings (Configuration)")
    # Toggle generator
    use_llm = st.sidebar.checkbox(
        "Use LLM (Generator)",
        value=True,
        help="Jika OFF, sistem hanya menampilkan jawaban retrieval-only (extractive) tanpa LLM.",
    )

    # Basic retrieval + prompt sizing
    top_k_ui = st.sidebar.slider(
        "candidate top_k (before rerank)",
        min_value=3,
        max_value=15,
        value=int(cfg["retrieval"]["top_k"]),
        help="Jumlah kandidat retrieval awal. Jika reranker aktif, kandidat final akan di-rerank lalu dipotong lagi sesuai top_n_rerank.",
    )
    max_chars_ctx_ui = st.sidebar.slider(
        "max_chars_per_ctx (override)",
        min_value=600,
        max_value=3000,
        value=int(cfg.get("llm", {}).get("max_chars_per_ctx", 1800)),
        help="Batas karakter per-chunk konteks yang dikirim ke LLM (mencegah prompt terlalu panjang).",
    )

    # =========================
    # V1.5 Controls: Memory + Contextualization (Mode demo vs eval)
    # =========================
    # demo: memory/contextualization boleh ON untuk pengalaman chat lebih natural
    # eval: dipaksa OFF agar hasil retrieval stabil & mudah dibandingkan antar run
    mem_cfg = cfg.get("memory", {})
    mem_default = bool(mem_cfg.get("enabled", False))
    mem_window_default = int(mem_cfg.get("window", 4))

    with st.sidebar.expander("Memory & Context (V1.5)", expanded=True):
        run_mode = st.selectbox(
            "Mode",
            options=["demo", "eval"],
            help="eval = matikan memory/contextualization agar hasil konsisten untuk evaluasi.",
        )
        eval_mode = run_mode == "eval"
        # Toggle Use Memory + Window + Contextualize (untuk V1.5)
        use_memory = st.checkbox(
            "Use Memory (V1.5)",
            value=mem_default,
            help="Jika ON, pertanyaan lanjutan bisa memanfaatkan riwayat chat (untuk rewrite query).",
        )
        mem_window = st.slider(
            "Memory Window (Turns)",
            min_value=0,
            max_value=10,
            value=mem_window_default,
            help="Berapa turn terakhir yang dipakai untuk contextualization.",
        )
        contextualize_on = st.checkbox(
            "Contextualize Query (Rewrite Sebelum Retrieval)",
            value=True,
            disabled=(not use_memory),
            help="Jika ON, sistem rewrite pertanyaan lanjutan (mis. 'tahapannya?') menjadi standalone sebelum retrieval.",
        )

        # Eval mode: paksa OFF agar konsisten
        if eval_mode:
            use_memory = False
            contextualize_on = False
            mem_window = 0

    # =========================
    # Advanced Retrieval (Diversity)
    # =========================
    with st.sidebar.expander("Advanced Retrieval (Diversity)", expanded=False):
        # NOTE: eval_mode sudah terdefinisi dari expander Memory
        diverse_mode = st.selectbox(
            "Diversity Retrieval (Multi-Doc)",
            options=["Auto", "Off", "On"],
            index=0,
            disabled=eval_mode,
            help="Auto: Aktif jika pertanyaan mengandung 'beberapa dokumen/sumber'.",
        )
        max_per_doc = st.slider(
            "Max Chunks per-Dokumen",
            min_value=1,
            max_value=int(top_k_ui),
            value=min(2, int(top_k_ui)),
            disabled=eval_mode,
            help="Batasi jumlah chunk dari dokumen yang sama agar sources lebih beragam.",
        )
        pool_mult = st.slider(
            "Candidate Pool Multiplier",
            min_value=2,
            max_value=12,
            value=6,
            disabled=eval_mode,
            help="Jumlah kandidat awal = top_k * multiplier (lebih besar = lebih beragam tapi akan lebih lambat).",
        )

        if eval_mode:
            # paksa OFF saat eval agar konsisten
            diverse_mode = "Off"

    # indikator config berubah (render memakai placeholder dan update langsung setelah submit)
    current_ui_cfg = {
        "top_k": int(top_k_ui),
        "max_chars_per_ctx": int(max_chars_ctx_ui),
        "use_llm": bool(use_llm),
        "provider": provider,
        "run_mode": run_mode,
        "use_memory": bool(use_memory),
        "mem_window": int(mem_window),
        "contextualize_on": bool(contextualize_on),
        "diverse_mode": str(diverse_mode),
        "max_per_doc": int(max_per_doc),
        "pool_mult": int(pool_mult),
    }

    # snapshot terakhir yang BENAR-BENAR sudah dipakai untuk menghasilkan hasil terakhir
    if "last_applied_ui_cfg" not in st.session_state:
        st.session_state["last_applied_ui_cfg"] = None  # belum ada query sama sekali

    cfg_indicator_slot = st.sidebar.empty()  # <--- SLOT, akan diisi di bagian bawah setelah submit

    # Key status (tanpa menampilkan key)
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Diagnostics")
    st.sidebar.write("GROQ key set?:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("OPENAI key set?:", bool(os.getenv("OPENAI_API_KEY")))

    # apakah mau config snapshot per baris?
    include_cfg = st.sidebar.checkbox(
        "Log Config Snapshot per-Row",
        value=(os.getenv("LOG_INCLUDE_CONFIG_PER_ROW", "0") == "1"),
        help="Jika ON, setiap baris JSONL menyimpan snapshot config untuk audit (file log jadi lebih besar, namun audit lebih solid).",
    )

    # ===== Main Info Bar =====
    _retrieval_mode_label = str(cfg.get("retrieval", {}).get("mode", "dense")).upper()

    # ← V3.0c: baca status reranker dari config
    _rk_cfg_info      = cfg.get("reranker") or {}
    _rk_enabled_info  = bool(_rk_cfg_info.get("enabled", False))
    _rk_allow_eval_i  = bool(_rk_cfg_info.get("allow_in_eval", True))
    _rk_on_info       = _rk_enabled_info and (run_mode != "eval" or _rk_allow_eval_i)
    _rk_icon          = "🟢" if _rk_on_info else "⚫"

    # ← V3.0c: banner awal diperbarui
    st.success(
        f"Now Running on **V3.0d** — **{_retrieval_mode_label}** Retrieval + "
        f"Memory & Contextualization + Structure-Aware Dual Stream + "
        f"Metadata Self-Querying + Cross-Encoder Reranking + **Post-Hoc Verification**",
        icon="🚥",
    )

    _mem_icon = "🟢" if (use_memory and mem_window > 0) else "⚫"
    _ctx_icon = "🟢" if contextualize_on else "⚫"
    _llm_icon = "🟢" if use_llm else "⚫"
    _mode_badge = "🧪 eval" if eval_mode else "🎯 demo"
    _retrieval_icon = "🔀" if _retrieval_mode_label == "HYBRID" else "🔵"

    # V3.0b: cek apakah self-query aktif dari config
    _sq_cfg_info  = cfg.get("self_query") or {}
    _sq_enabled_info = bool(_sq_cfg_info.get("enabled", False))
    _sq_allow_eval   = bool(_sq_cfg_info.get("allow_in_eval", False))
    _sq_on_info   = _sq_enabled_info and (run_mode != "eval" or _sq_allow_eval)
    _sq_icon      = "🟢" if _sq_on_info else "⚫"

    _info_parts = [
        f"<b>Mode:</b> {_mode_badge}",
        f"{_retrieval_icon} Retrieval: {_retrieval_mode_label}",
        f"{_llm_icon} Generator",
        f"{_mem_icon} Memory ({mem_window} turns)",
        f"{_ctx_icon} Contextualize",
        f"{_sq_icon} Self-Query Filter",   # ← V3.0b
        f"{_rk_icon} Reranker",            # ← V3.0c
    ]
    st.markdown(
        '<div style="background:#1e293b; border:1px solid #334155; padding:0.6rem 1rem; '
        'border-radius:0.6rem; margin-bottom:0.9rem; color:#94a3b8; font-size:0.875rem;">'
        + " &nbsp;|&nbsp; ".join(_info_parts)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ===== History slot (chat-like) =====
    if "history" not in st.session_state:
        st.session_state["history"] = []

    history_slot = st.empty()  # <--- SLOT (akan di-render ulang di bawah setelah submit)

    # ===== Input =====
    user_q = st.text_input("Halo! Ada yang Bisa Saya Bantu? Anda bisa Tanyakan di sini:")

    # Filter sources (UI)
    filter_q = st.text_input(
        "Filter Sources (doc_id / source_file / text) - BOLEH DIISI ATAU KOSONGKAN:", value=""
    )
    show_only_matches = st.checkbox("Show Only Matched Sources", value=True)

    # Tombol submit (pakai variabel agar logika show/hide benar di klik pertama)
    submitted = st.button("🪄 Kirim")

    if submitted and user_q.strip():
        query = user_q.strip()

        # reset viewing mode agar hasil terbaru tampil otomatis setelah submit
        st.session_state["viewing_turn_id"] = None

        # turn id per session
        st.session_state["turn_id"] += 1
        turn_id = int(st.session_state["turn_id"])

        # simpan query terakhir untuk auto-highlight
        st.session_state["last_submitted_query"] = query
        st.session_state["last_turn_id"] = turn_id

        with st.status("⏳ Processing Query... Please Wait a Moment :)", expanded=True) as _proc_status:

            st.write("🔧 Membangun konfigurasi runtime...")
            # Override cfg runtime (tanpa mengubah file yaml)
            cfg_runtime = dict(cfg)
            cfg_runtime["retrieval"] = dict(cfg.get("retrieval", {}))
            cfg_runtime["retrieval"]["top_k"] = int(top_k_ui)
            cfg_runtime["llm"] = dict(cfg.get("llm", {}))
            cfg_runtime["llm"]["max_chars_per_ctx"] = int(max_chars_ctx_ui)

            # ===== Memory window =====
            # ===== DETECT awal (anaphora / multi-target / topic-shift berdasarkan query saja) =====
            detect_anaphora = is_anaphora_question(query)
            detect_multi_target = is_multi_target_question(query)

            # T08 fix:
            # surface detector dipakai lebih awal agar history policy tidak terlalu keras
            detect_compression_followup_surface = is_compression_followup_question(
                query,
                None,
            )

            # ===== History window (policy) =====
            # Ambil window percakapan terakhir (history) untuk membantu rewrite query.
            history_pairs = st.session_state.get("history", [])

            hist_cfg = cfg_runtime.get("history", {}) or {}
            segment_on_shift = bool(hist_cfg.get("segment_on_topic_shift", True))
            anaphora_skip_last_shift = bool(hist_cfg.get("anaphora_skip_last_shift", True))

            history_window, history_policy_meta = build_history_window_v2(
                history_pairs,
                use_memory=use_memory,
                mem_window=int(mem_window),
                segment_on_topic_shift=segment_on_shift,
                anaphora_skip_last_shift=anaphora_skip_last_shift,
                current_is_anaphora=bool(
                    detect_anaphora or detect_compression_followup_surface
                ),
            )

            # Ringkasan memory yang dipakai untuk turn ini (untuk audit log)
            history_used = len(history_window) if (use_memory and mem_window > 0) else 0
            history_turn_ids = [h.get("turn_id") for h in history_window] if history_window else []

            # T08 fix:
            # setelah history_window final terbentuk, evaluasi ulang compression-followup dengan konteks history yang benar-benar dipakai
            detect_compression_followup = is_compression_followup_question(
                query,
                history_window,
            )

            st.write("💭 Mendeteksi konteks & riwayat percakapan...")

            ts_cfg = cfg_runtime.get("topic_shift", {}) or {}
            ts_enabled = bool(ts_cfg.get("enabled", True))
            ts_overlap_threshold = float(ts_cfg.get("overlap_threshold", 0.15))
            ts_min_terms = int(ts_cfg.get("min_terms", 3))

            topic_shift = False
            topic_shift_meta: Dict[str, Any] = {"reason": "not_evaluated"}

            if not ts_enabled:
                topic_shift = False
                topic_shift_meta = {"reason": "disabled"}
            elif not (use_memory and history_window):
                topic_shift = False
                topic_shift_meta = {"reason": "no_memory_or_history"}
            else:
                topic_shift, topic_shift_meta = detect_topic_shift(
                    query,
                    history_window,
                    overlap_threshold=ts_overlap_threshold,
                    min_terms=ts_min_terms,
                )

            detect_flags = {
                "anaphora": bool(detect_anaphora),
                "multi_target": bool(detect_multi_target),
                "topic_shift": bool(topic_shift),
                "compression_followup": bool(detect_compression_followup),
            }

            # ===== V3.0d lanjutan: Intent planner (Tahap 1) =====
            answer_policy_cfg = cfg_runtime.get("answer_policy", {}) or {}
            planner_enabled = bool(answer_policy_cfg.get("planner_enabled", True))
            log_intent_plan = bool(answer_policy_cfg.get("log_intent_plan", True))
            doc_agg_cfg = cfg_runtime.get("doc_aggregation", {}) or {}
            log_doc_aggregation_meta = bool(doc_agg_cfg.get("log_meta", True))

            explicit_doc_focus_for_plan = extract_doc_focus_from_query(query)
            intent_plan: Optional[Dict[str, Any]] = None
            if planner_enabled:
                try:
                    intent_plan = build_intent_plan(
                        query,
                        history_window=history_window,
                        doc_focus=explicit_doc_focus_for_plan,
                    ).to_dict()
                except Exception:
                    intent_plan = None

            # ===== Contextualize / rewrite query =====
            # Contextualization akan menulis ulang pertanyaan user agar eksplisit (mis. coreference "itu", "tahapannya").
            # jika topic_shift True -> skip rewrite dan treat query sebagai fresh query
            retrieval_query = query
            contextualize_attempted = False
            contextualize_skipped = False
            contextualize_mode = "none"

            if use_memory and contextualize_on and history_window and (not topic_shift):
                # T08 fix:
                # treat compression-followup singkat sebagai kelanjutan steps/method, bukan fresh query.

                # CASE 1A: compression follow-up + steps/method context -> deterministic rewrite
                if detect_compression_followup and _history_has_steps_or_method_context(history_window):
                    tm = infer_target_methods(query, history_window)
                    rq2 = build_steps_summary_standalone_query(tm)
                    if rq2:
                        retrieval_query = rq2
                        contextualize_attempted = True
                        contextualize_mode = "heuristic_steps_summary"
                    else:
                        contextualize_attempted = True
                        contextualize_mode = "llm"
                        retrieval_query = contextualize_question(
                            cfg_runtime,
                            question=query,
                            history_pairs=history_window,
                            use_llm=use_llm,
                        )

                # CASE 1B: anaphora + steps -> deterministic rewrite (perilaku lama)
                elif is_steps_question(query) and is_anaphora_question(query):
                    tm = infer_target_methods(query, history_window)
                    rq2 = build_steps_standalone_query(tm)
                    if rq2:
                        retrieval_query = rq2
                        contextualize_attempted = True
                        contextualize_mode = "heuristic_steps"
                    else:
                        contextualize_attempted = True
                        contextualize_mode = "llm"
                        retrieval_query = contextualize_question(
                            cfg_runtime,
                            question=query,
                            history_pairs=history_window,
                            use_llm=use_llm,
                        )

                # CASE 2: default -> LLM/heuristic bawaan contextualize_question()
                else:
                    contextualize_attempted = True
                    contextualize_mode = "llm"
                    retrieval_query = contextualize_question(
                        cfg_runtime,
                        question=query,
                        history_pairs=history_window,
                        use_llm=use_llm,
                    )

            elif use_memory and contextualize_on and history_window and topic_shift:
                contextualize_skipped = True
                contextualize_mode = "skipped_topic_shift"

            contextualize_decision = {
                "enabled": bool(contextualize_on),
                "attempted": bool(contextualize_attempted),
                "skipped": bool(contextualize_skipped),
                "mode": contextualize_mode,
                "skip_reason": (topic_shift_meta.get("reason") if contextualize_skipped else None),
                "topic_shift_enabled": bool(ts_enabled),
                "overlap_threshold": float(ts_overlap_threshold),
                "min_terms": int(ts_min_terms),
            }

            # ===== V3.0b: Self-Querying Metadata Filter (inisialisasi default) =====
            # Diinisialisasi di sini agar field JSONL konsisten di SEMUA jalur
            # retrieval_query = hasil contextualization / rewrite
            # effective_retrieval_query = query FINAL yang benar-benar dikirim ke retrieval
            retrieval_query_before_self_query = retrieval_query
            effective_retrieval_query = retrieval_query_before_self_query

            # reset state agar tidak bocor antar turn
            st.session_state["last_self_query_result"] = None

            _self_query_result: Optional[SelfQueryResult] = None
            _self_query_filter: Optional[Dict[str, Any]] = None
            _self_query_log: Dict[str, Any] = _make_self_query_log(
                enabled=False,
                semantic_query=None,
                fallback_reason="not_executed",
            )

            # ===== V3.0c: Reranker defaults - diinisialisasi sebelum kedua jalur =====
            # ⚠️ Jangan pindah ke dalam if not metadata_routed - harus ada di kedua jalur
            # agar JSONL shape konsisten di metadata_routed path maupun normal path.
            reranker_applied: bool = False
            reranker_info: Dict[str, Any] = get_reranker_info()

            # ← Legacy telemetry (sementara tetap dipertahankan agar backward-compatible)
            rerank_coverage_rescue: Dict[str, Any] = {
                "triggered": False,
                "applied": False,
                "prefer_steps": False,
                "target_methods": [],
                "rescued_methods": [],
                "anchor_chunk_ids": {},
                "adequately_represented_after": {},
                "final_n": 0,
            }

            # ← V3.0c: metadata rescue pasca-rerank (default agar shape log stabil)
            # refinement: deterministic anchor reservation
            rerank_anchor_reservation: Dict[str, Any] = {
                "triggered": False,
                "applied": False,
                "modified": False,
                "strategy": "deterministic_anchor_reservation",
                "prefer_steps": False,
                "dynamic_top_n": False,
                "complex_query": False,
                "top_n_base": None,
                "top_n_effective": None,
                "target_methods": [],
                "reserved_methods": [],
                "reserved_anchor_chunk_ids": {},
                "final_n": 0,
            }

            # Resolve config reranker di sini agar candidate pool pre-rerank
            # bisa dipakai oleh retrieval utama (bukan hanya tercatat di log).
            _reranker_cfg = cfg_runtime.get("reranker") or {}
            _reranker_en = bool(_reranker_cfg.get("enabled", False))
            _rk_allow_eval = bool(_reranker_cfg.get("allow_in_eval", True))
            _reranker_act = _reranker_en and (run_mode != "eval" or _rk_allow_eval)

            # ── Base vs complex top-n (effective top-n baru ditentukan nanti setelah target_methods terlihat)
            _top_n_rerank_base = max(1, int(_reranker_cfg.get("top_n_rerank", 3)))
            _dynamic_top_n = bool(_reranker_cfg.get("dynamic_top_n", True))
            _top_n_rerank_complex = max(
                _top_n_rerank_base,
                int(_reranker_cfg.get("top_n_rerank_complex", 5)),
            )

            # candidate_k harus aman untuk skenario top-n terbesar
            _max_top_n_rerank = _top_n_rerank_complex if _dynamic_top_n else _top_n_rerank_base
            _candidate_k_rerank = max(
                _max_top_n_rerank,
                int(_reranker_cfg.get("candidate_k", top_k_ui)),
            )

            # Jumlah kandidat FINAL sebelum rerank.
            # - Jika reranker aktif: minimal mengikuti candidate_k dari config
            # - Jika reranker nonaktif: kembali ke perilaku lama (top_k_ui)
            _candidate_pool_k = (
                max(int(top_k_ui), _candidate_k_rerank)
                if _reranker_act
                else int(top_k_ui)
            )

            # ===== Metadata Routing (page_count) =====
            md_cfg = cfg_runtime.get("metadata_routing", {}) or {}
            md_enabled = bool(md_cfg.get("enabled", False))
            allow_in_eval = bool(md_cfg.get("allow_in_eval", False))
            if run_mode == "eval" and not allow_in_eval:
                md_enabled = False

            paths = ConfigPaths.from_cfg(cfg_runtime)
            persist_dir = paths.persist_dir

            metadata_routed = False
            metadata_meta: Dict[str, Any] = {"enabled": md_enabled, "handled": False}

            if md_enabled:
                md_res = maybe_route_metadata_query(query, persist_dir=persist_dir, cfg=cfg_runtime)
                if md_res and md_res.handled:
                    metadata_routed = True
                    metadata_meta = {
                        "enabled": True,
                        "handled": True,
                        "intent": md_res.intent,
                        "meta": md_res.meta,
                    }

                    st.write("✏️ Menjawab dari metadata catalog...")

                    # Untuk metadata routing: tidak ada nodes/context
                    nodes: List[Any] = []
                    nodes_before_rerank: List[Any] = []   # ← V3.0c
                    contexts: List[str] = []

                    # ← V3.0c:
                    # retrieval_stats_pre_rerank  = stats candidate pool retrieval
                    # generation_context_stats    = stats context final yang benar-benar masuk ke LLM/generator
                    # Pada metadata routing keduanya identik (kosong), tapi tetap diisi agar shape stabil.
                    retrieval_stats_pre_rerank = compute_retrieval_stats(nodes_before_rerank)
                    generation_context_stats = compute_retrieval_stats(nodes)

                    # Backward-compatible:
                    # pertahankan retrieval_stats lama agar consumer lama tidak rusak.
                    retrieval_stats = generation_context_stats

                    strategy = "metadata_router"
                    ts = datetime.now().isoformat(timespec="seconds")

                    _rk_meta = _build_reranker_telemetry(
                        cfg=cfg_runtime,
                        feature_context_enabled=False,   # metadata router = bypass
                        reranker_applied=False,
                        reranker_info=None,
                        nodes_before_rerank=nodes_before_rerank,
                        top_n_base=None,
                        top_n_effective=None,
                        rerank_anchor_reservation=None,
                    )

                    intent_plan_log = _build_intent_plan_log(
                        intent_plan,
                        enabled=log_intent_plan,
                    )
                    doc_aggregation_log = _build_doc_aggregation_log(
                        None,
                        enabled=log_doc_aggregation_meta,
                    )
                    answer_policy_audit_summary = _build_answer_policy_audit_summary(
                        intent_plan_log,
                        doc_aggregation_log,
                    )

                    retrieval_record = {
                        "run_id": rm.run_id,
                        "turn_id": turn_id,
                        "timestamp": ts,
                        "retrieval_mode": "metadata_router",
                        "llm_provider": provider,
                        "use_llm": False,
                        "top_k": int(top_k_ui),
                        "user_query": query,
                        "retrieval_query": query,
                        "run_mode": run_mode,
                        "use_memory": use_memory,
                        "mem_window": int(mem_window),
                        "history_used": history_used,
                        "history_turn_ids": history_turn_ids,
                        "history_policy": history_policy_meta,
                        "contextualized": False,
                        "detect_flags": detect_flags,
                        "topic_shift_meta": topic_shift_meta,
                        "contextualize_decision": contextualize_decision,
                        "doc_focus": st.session_state.get("last_doc_focus_meta"),
                        "strategy": strategy,
                        "diversity": {
                            "mode": "Off",
                            "enabled": False,
                            "max_per_doc": 0,
                            "pool_mult": 0,
                            "candidate_k": None,
                        },
                        "retrieval_stats": retrieval_stats,  # Backward-compatible: retrieval_stats tetap ada
                        # ← V3.0c: stats eksplisit pre vs post
                        "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                        "generation_context_stats": generation_context_stats,
                        "retrieval_confidence": None,
                        "retrieved_nodes": [],
                        "coverage": None,
                        "metadata_route": metadata_meta,
                        "self_query": _make_self_query_log(  # ← V3.0b
                            enabled=False,
                            semantic_query=None,
                            fallback_reason="metadata_routed",
                        ),
                        # ← V3.0c: reranker tidak jalan di metadata routing
                        "reranker_applied":         _rk_meta["applied"],
                        "reranker_candidate_count": _rk_meta["candidate_count"],
                        "reranker_top_n":           _rk_meta["top_n"],
                        "reranker_top_n_base":      _rk_meta["top_n_base"],
                        "reranker_top_n_effective": _rk_meta["top_n_effective"],
                        "reranker_info":            _rk_meta["info"],
                        "rerank_coverage_rescue":   rerank_coverage_rescue,         # legacy telemetry
                        "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                        "intent_plan": intent_plan_log,
                        "doc_aggregation_meta": doc_aggregation_log,
                        "answer_policy_audit_summary": answer_policy_audit_summary,
                        "explanation_expansion_meta": None,
                        "single_target_steps_meta": None,
                        "method_comparison_meta": None,
                    }
                    rm.log_retrieval(retrieval_record, include_config_snapshot_per_row=include_cfg)

                    answer_raw = md_res.answer_text
                    generator_meta = {
                        "status": "metadata",
                        "format_ok": True,
                        "bukti_ok": True,
                        "citations_present": False,
                        "citations_in_range": True,
                        "grounding_ok": True,
                        "answer_grounding_ok": True,
                    }

                    # ← V3.0d: metadata-routed path = Verification N/A / skipped
                    verification_meta = build_verification_meta(
                        question=query,
                        contexts=[],
                        answer_text=answer_raw,
                        used_ctx=0,
                        used_context_chunk_ids=[],
                        generator_meta=generator_meta,
                        cfg=cfg_runtime,
                        metadata_route_handled=True,
                    )

                    answer_record = {
                        "run_id": rm.run_id,
                        "turn_id": turn_id,
                        "timestamp": ts,
                        "llm_provider": provider,
                        "use_llm": False,
                        "top_k": int(top_k_ui),
                        "user_query": query,
                        "retrieval_query": query,
                        "run_mode": run_mode,
                        "use_memory": use_memory,
                        "mem_window": int(mem_window),
                        "history_used": history_used,
                        "history_turn_ids": history_turn_ids,
                        "history_policy": history_policy_meta,
                        "contextualized": False,
                        "detect_flags": detect_flags,
                        "topic_shift_meta": topic_shift_meta,
                        "contextualize_decision": contextualize_decision,
                        "doc_focus": st.session_state.get("last_doc_focus_meta"),
                        "used_context_chunk_ids": [],
                        "answer": answer_raw,
                        "strategy": strategy,
                        "diversity": {
                            "mode": "Off",
                            "enabled": False,
                            "max_per_doc": 0,
                            "pool_mult": 0,
                            "candidate_k": None,
                        },
                        "retrieval_stats": retrieval_stats,  # Backward-compatible
                        # ← V3.0c: stats eksplisit pre vs post
                        "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                        "generation_context_stats": generation_context_stats,
                        "retrieval_confidence": None,
                        "generator_strategy": "metadata_router",
                        "generator_status": "metadata",
                        "generator_meta": generator_meta,
                        "verification": verification_meta,  # ← V3.0d
                        "intent_plan": intent_plan_log,
                        "doc_aggregation_meta": doc_aggregation_log,
                        "answer_policy_audit_summary": answer_policy_audit_summary,
                        "explanation_expansion_meta": None,
                        "single_target_steps_meta": None,
                        "method_comparison_meta": None,
                        "max_bullets": 1,
                        "bullet_policy": {},
                        "error": None,
                        "coverage": None,
                        "metadata_route": metadata_meta,
                        "self_query": _make_self_query_log(  # ← V3.0b
                            enabled=False,
                            semantic_query=None,
                            fallback_reason="metadata_routed",
                        ),
                        # ← V3.0c /T10 polish
                        "reranker_applied": _rk_meta["applied"],
                        "reranker_model":   (_rk_meta["info"] or {}).get("model_name"),
                        "reranker_top_n":   _rk_meta["top_n"],
                        "reranker_top_n_base": _rk_meta["top_n_base"],
                        "reranker_top_n_effective": _rk_meta["top_n_effective"],
                        "rerank_coverage_rescue": rerank_coverage_rescue,           # legacy telemetry
                        "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                    }
                    rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)

                    # append history (tambahkan detect_flags/topic_shift agar policy bisa bekerja)
                    jawaban, _ = parse_answer(answer_raw)
                    st.session_state["history"].append(
                        {
                            "turn_id": turn_id,
                            "query": query,
                            "answer_preview": (jawaban or answer_raw)[:220]
                            + ("..." if len(jawaban or answer_raw) > 220 else ""),
                            "detect_flags": detect_flags,
                            "topic_shift": bool(detect_flags.get("topic_shift")),
                            "metadata_intent": md_res.intent,
                            "methods": [],  # metadata route tidak relevan untuk chaining metode
                        }
                    )

                    # set last result untuk UI
                    st.session_state["last_answer_raw"] = answer_raw
                    st.session_state["last_verification"] = verification_meta   # ← V3.0d
                    st.session_state["last_intent_plan"] = intent_plan_log
                    st.session_state["last_doc_aggregation_meta"] = doc_aggregation_log
                    st.session_state["last_answer_policy_audit_summary"] = answer_policy_audit_summary
                    st.session_state["last_explanation_expansion_meta"] = None
                    st.session_state["last_single_target_steps_meta"] = None
                    st.session_state["last_method_comparison_meta"] = None
                    st.session_state["last_turn_id"] = turn_id
                    st.session_state["last_nodes"] = []                 # final contexts
                    st.session_state["last_nodes_before_rerank"] = []   # candidate pool pre-rerank
                    st.session_state["last_strategy"] = strategy
                    # ← V3.0c
                    st.session_state["last_retrieval_stats_pre_rerank"] = retrieval_stats_pre_rerank
                    st.session_state["last_generation_context_stats"] = generation_context_stats
                    st.session_state["_scroll_answer"] = True
                    st.session_state["last_user_query"] = query
                    st.session_state["last_retrieval_query"] = query
                    st.session_state["last_applied_ui_cfg"] = current_ui_cfg.copy()
                    st.session_state["last_history_used"] = history_used
                    st.session_state["last_contextualized"] = False
                    st.session_state["last_reranker_applied"] = _rk_meta["applied"]  # ← V3.0c
                    st.session_state["last_reranker_info"] = _rk_meta["info"]  # ← V3.0c
                    st.session_state["last_rerank_anchor_reservation"] = _rk_meta["anchor_reservation"]  # ← V3.0c

                    # simpan full render data untuk turn viewer
                    if "turn_data" not in st.session_state:
                        st.session_state["turn_data"] = {}
                    st.session_state["turn_data"][turn_id] = {
                        # ← V3.0c:
                        # nodes                = final contexts yang benar-benar dipakai
                        # nodes_before_rerank  = candidate pool pre-rerank
                        "nodes": [],
                        "nodes_before_rerank": [],
                        "answer_raw": answer_raw,
                        "user_query": query,
                        "retrieval_query": query,
                        "strategy": strategy,
                        "history_used": history_used,
                        "contextualized": False,
                        "retrieval_mode": "metadata_router",
                        "self_query_filter_applied": False,
                        "reranker_applied": _rk_meta["applied"],  # ← V3.0c
                        # ← V3.0c: siap dipakai nanti di app_ui_render.py
                        "reranker_candidate_count": _rk_meta["candidate_count"],
                        "reranker_top_n": _rk_meta["top_n"],
                        "reranker_top_n_base": _rk_meta["top_n_base"],
                        "reranker_top_n_effective": _rk_meta["top_n_effective"],
                        "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                        "generation_context_stats": generation_context_stats,
                        "rerank_coverage_rescue": rerank_coverage_rescue,           # legacy telemetry
                        "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                        "reranker_info": _rk_meta["info"],
                        "verification": verification_meta,  # ← V3.0d
                        "intent_plan": intent_plan_log,
                        "doc_aggregation_meta": doc_aggregation_log,
                        "answer_policy_audit_summary": answer_policy_audit_summary,
                        "explanation_expansion_meta": None,
                        "single_target_steps_meta": None,
                        "method_comparison_meta": None,
                    }

            if not metadata_routed:
                # =========================
                # Retrieval
                # =========================
                # Pakai retrieval_query (hasil rewrite) agar embedding search lebih tepat.
                # nodes -> dipakai untuk:
                # 1) konteks generator (contexts)
                # 2) CTX mapping + sources UI
                # 3) logging (retrieval.jsonl)
                # decide diversify
                if diverse_mode == "On":
                    diversify = True
                elif diverse_mode == "Off":
                    diversify = False
                else:
                    diversify = is_multi_doc_question(query)  # pakai user query (lebih natural)

                candidate_k = int(_candidate_pool_k) * int(pool_mult) if diversify else None

                # cfg retrieval khusus agar pool kandidat pre-rerank benar-benar bisa > top_k_ui
                # tanpa mengubah cfg_runtime utama yang dipakai untuk logika lain.
                cfg_retrieval = dict(cfg_runtime)
                cfg_retrieval["retrieval"] = dict(cfg_runtime.get("retrieval", {}))
                cfg_retrieval["retrieval"]["top_k"] = int(_candidate_pool_k)

                # mode-aware
                retrieval_mode = str(cfg_runtime["retrieval"]["mode"])

                # ===== V3.0b: Eksekusi Self-Querying Metadata Filter =====
                sq_cfg_rt     = cfg_runtime.get("self_query") or {}
                sq_enabled    = bool(sq_cfg_rt.get("enabled", False))
                sq_allow_eval = bool(sq_cfg_rt.get("allow_in_eval", False))
                _sq_active    = sq_enabled and (run_mode != "eval" or sq_allow_eval)

                if not _sq_active:
                    _disabled_reason = (
                        "disabled_in_eval"
                        if (sq_enabled and run_mode == "eval" and not sq_allow_eval)
                        else "self_query_disabled"
                    )
                    _self_query_log = _make_self_query_log(
                        enabled=False,
                        semantic_query=retrieval_query_before_self_query,
                        fallback_reason=_disabled_reason,
                    )
                else:
                    st.write("🔎 Self-querying metadata filter...")
                    try:
                        # build dari query hasil contextualization / rewrite
                        _self_query_result = build_self_query(cfg_runtime, retrieval_query_before_self_query)
                        st.session_state["last_self_query_result"] = _self_query_result

                        # semantic_query hasil self-query adalah query FINAL untuk retrieval
                        _candidate_semantic_query = (_self_query_result.semantic_query or "").strip()
                        if _candidate_semantic_query:
                            effective_retrieval_query = _candidate_semantic_query

                        # default log dari result
                        _self_query_log = _self_query_log_from_result(
                            _self_query_result,
                            enabled=True,
                            semantic_query=effective_retrieval_query,
                        )

                        if _self_query_result.filter_applied and _self_query_result.filters:
                            _pre_check_ok = check_filter_has_results(
                                _self_query_result.filters,
                                cfg_runtime,
                                "narasi",
                            )

                            if _pre_check_ok:
                                _self_query_filter = _self_query_result.filters
                                _sq_fields = ", ".join(_self_query_result.filter_fields_used)
                                st.write(f"🏷️ Metadata filter aktif: [{_sq_fields}]")

                                _self_query_log = _self_query_log_from_result(
                                    _self_query_result,
                                    enabled=True,
                                    semantic_query=effective_retrieval_query,
                                    filter_applied=True,
                                )
                            else:
                                # graceful fallback: filter sempat terbentuk, tapi TIDAK dipakai
                                st.write("🔄 Metadata filter: 0 hasil → fallback ke retrieval penuh")

                                _self_query_filter = None
                                _self_query_log = _self_query_log_from_result(
                                    _self_query_result,
                                    enabled=True,
                                    semantic_query=effective_retrieval_query,
                                    filter_applied=False,
                                    fallback_reason="precheck_no_results",
                                    clear_filter_payload=True,
                                )

                    except Exception as _sq_exc:
                        # self-query TIDAK BOLEH menjatuhkan pipeline utama
                        logger.warning(f"[V3.0b] Self-query failed (non-critical): {_sq_exc}")
                        st.session_state["last_self_query_result"] = None
                        effective_retrieval_query = retrieval_query_before_self_query
                        _self_query_log = _make_self_query_log(
                            enabled=True,
                            semantic_query=effective_retrieval_query,
                            filter_applied=False,
                            fallback_reason=f"exception:{type(_sq_exc).__name__}",
                        )

                st.write("🔍 Retrieval dokumen dari vector store...")
                if retrieval_mode == "hybrid":
                    # ── Jalur Hybrid: Dense + Sparse BM25 + RRF Fusion ──
                    # Dense: ambil pool lebih besar (bahan RRF), diversify ditangani RRF
                    _hybrid_candidate_k = max(int(_candidate_pool_k) * 3, int(_candidate_pool_k) + 5)
                    dense_nodes_raw = retrieve_dense(
                        cfg_retrieval,
                        effective_retrieval_query,
                        diversify=False,
                        max_per_doc=int(max_per_doc),
                        candidate_k=_hybrid_candidate_k,
                        collection="narasi",
                        where_filter=_self_query_filter,   # ← V3.0b
                    )

                    # Sparse: BM25 + Sastrawi, ambil minimal sebanyak candidate pool pre-rerank
                    sparse_nodes_raw = sparse_retrieve_bm25(
                        cfg_retrieval,
                        effective_retrieval_query,
                        top_k=int(_candidate_pool_k),
                        collection="narasi",
                    )

                    # RRF Fusion — gabungkan ranking dense + sparse menjadi candidate pool pre-rerank
                    _rrf_k = int((cfg_retrieval.get("retrieval") or {}).get("rrf_k", 60))
                    nodes = rrf_fusion(
                        dense_nodes_raw,
                        sparse_nodes_raw,
                        k=_rrf_k,
                        top_k=int(_candidate_pool_k),
                    )

                    # ── V3.0a: Citation routing untuk jalur hybrid - tambah nodes dari sitasi collection ──
                    _has_sitasi = bool(
                        (cfg_runtime.get("index", {}) or {}).get("collections", {}).get("sitasi")
                    )
                    _citation_intent_query = retrieval_query_before_self_query or query
                    if _has_sitasi and is_citation_query(_citation_intent_query):
                        _sitasi_top_k = max(3, int(_candidate_pool_k) // 2)  # proporsional terhadap candidate pool
                        sitasi_dense = retrieve_dense(
                            cfg_retrieval,
                            effective_retrieval_query,
                            diversify=False,
                            max_per_doc=2,
                            candidate_k=_sitasi_top_k * 2,
                            collection="sitasi",
                        )
                        sitasi_sparse = sparse_retrieve_bm25(
                            cfg_retrieval,
                            effective_retrieval_query,
                            top_k=_sitasi_top_k,
                            collection="sitasi",
                        )
                        # RRF fusion untuk sitasi
                        sitasi_fused = rrf_fusion(
                            sitasi_dense,
                            sitasi_sparse,
                            k=_rrf_k,
                            top_k=_sitasi_top_k,
                        )
                        # Merge: dedupe by chunk_id, narasi tetap prioritas slot pertama
                        existing_ids = {n.chunk_id for n in nodes}
                        sitasi_new = [n for n in sitasi_fused if n.chunk_id not in existing_ids]

                        # V3.0b: minimum floor narasi ≥60% dari candidate pool
                        # agar konteks konseptual tetap dominan sebelum rerank
                        _narasi_floor = max(1, int(_candidate_pool_k * 0.6))
                        _sitasi_cap   = int(_candidate_pool_k) - _narasi_floor
                        sitasi_new    = sitasi_new[:_sitasi_cap]   # terapkan cap sitasi

                        narasi_keep   = int(_candidate_pool_k) - len(sitasi_new)
                        nodes         = nodes[:narasi_keep] + sitasi_new

                        # HYBRID: n.score = RRF score → lebih besar = lebih relevan
                        nodes = sorted(
                            nodes[:int(_candidate_pool_k)],
                            key=lambda n: float(n.score),
                            reverse=True,
                        )

                        st.write(f"🧷 Citation routing aktif: +{len(sitasi_new)} chunks dari collection sitasi...")
                else:
                    # ── Jalur Dense Murni (V1.0 / V1.5 behaviour — tidak berubah) ──
                    retrieval_mode = "dense"  # normalisasi agar konsisten di log
                    nodes = retrieve_dense(
                        cfg_retrieval,
                        effective_retrieval_query,
                        diversify=diversify,
                        max_per_doc=int(max_per_doc),
                        candidate_k=candidate_k,
                        collection="narasi",
                        where_filter=_self_query_filter,   # ← V3.0b
                    )

                    # ── V3.0a: Citation routing untuk jalur dense ──
                    _has_sitasi_d = bool(
                        (cfg_runtime.get("index", {}) or {}).get("collections", {}).get("sitasi")
                    )
                    _citation_intent_query_d = retrieval_query_before_self_query or query
                    if _has_sitasi_d and is_citation_query(_citation_intent_query_d):
                        _sitasi_top_k_d = max(3, int(_candidate_pool_k) // 2)
                        sitasi_nodes_d = retrieve_dense(
                            cfg_retrieval,
                            effective_retrieval_query,
                            diversify=False,
                            max_per_doc=2,
                            candidate_k=_sitasi_top_k_d * 2,
                            collection="sitasi",
                        )

                        # Merge: dedupe by chunk_id, narasi tetap prioritas
                        existing_ids_d = {n.chunk_id for n in nodes}
                        sitasi_new_d = [n for n in sitasi_nodes_d if n.chunk_id not in existing_ids_d]

                        # V3.0b: minimum floor narasi ≥60% dari candidate pool
                        _narasi_floor_d = max(1, int(_candidate_pool_k * 0.6))
                        _sitasi_cap_d   = int(_candidate_pool_k) - _narasi_floor_d
                        sitasi_new_d    = sitasi_new_d[:_sitasi_cap_d]  # terapkan cap sitasi

                        narasi_keep_d   = int(_candidate_pool_k) - len(sitasi_new_d)
                        nodes           = nodes[:narasi_keep_d] + sitasi_new_d

                        # DENSE: n.score = Chroma distance → lebih kecil = lebih relevan
                        nodes = sorted(
                            nodes[:int(_candidate_pool_k)],
                            key=lambda n: float(n.score),
                        )

                        st.write(f"🧷 Citation routing aktif: +{len(sitasi_new_d)} chunks dari collection sitasi...")

                # ===== Coverage per-metode (multi-query dense) =====
                cov_cfg = cfg_runtime.get("coverage", {}) or {}
                cov_enabled = bool(cov_cfg.get("enabled", True)) and (run_mode != "eval")
                per_method_k = int(cov_cfg.get("per_method_k", 3))
                min_methods = int(cov_cfg.get("min_methods", 2))

                # Jika topic_shift terdeteksi, JANGAN bawa metode dari history (hindari carry-over)
                if topic_shift:
                    st.session_state["method_doc_map"] = {}  # reset hanya jika benar-benar topic shift

                history_for_methods = [] if topic_shift else history_window
                target_methods = infer_target_methods(query, history_for_methods)

                coverage_meta: Dict[str, Any] = {
                    "enabled": cov_enabled,
                    "performed": False,
                    "target_methods": target_methods,
                }

                if (
                    cov_enabled
                    and len(target_methods) >= min_methods
                    and (detect_anaphora or detect_multi_target or detect_compression_followup)
                ):
                    # ===== steps-aware expansion =====
                    # Jika query adalah steps, cari tambahan PER METODE (bukan hanya yang missing),
                    # karena "metode terdeteksi" ≠ "tahapan tersedia".
                    steps_q = is_steps_question(query)

                    methods_in_base = set(
                        detect_methods_in_text(" ".join([(n.text or "") for n in nodes]))
                    )
                    missing = [m for m in target_methods if m not in methods_in_base]

                    extra_nodes: List[Any] = []

                    # pilih metode yang akan di-query tambahan:
                    # - jika steps_q: semua target_methods (agar tiap metode punya peluang punya tahapan)
                    # - else: hanya missing (perilaku lama)
                    expand_methods = list(target_methods) if steps_q else list(missing)

                    for m in expand_methods:
                        cfg_tmp = dict(cfg_runtime)
                        cfg_tmp["retrieval"] = dict(cfg_runtime.get("retrieval", {}))
                        cfg_tmp["retrieval"]["top_k"] = int(per_method_k)

                        if steps_q:
                            q_m = method_steps_query(retrieval_query, m)
                        else:
                            q_m = build_method_biased_query(retrieval_query, m)

                        extra_nodes.extend(
                            retrieve_dense(
                                cfg_tmp,
                                q_m,
                                diversify=False,
                                max_per_doc=int(max_per_doc),
                                candidate_k=None,
                                collection="narasi",
                            )
                        )

                    merged = dedupe_nodes_by_chunk_id(nodes + extra_nodes)

                    # select nodes yang menjamin coverage per metode
                    nodes, cov_detail = select_nodes_with_coverage(
                        merged,
                        top_k=int(_candidate_pool_k),
                        target_methods=target_methods,
                        max_per_doc=int(max_per_doc) if diversify else 0,
                    )

                    coverage_meta = {
                        "enabled": True,
                        "performed": True,
                        "target_methods": target_methods,
                        "missing_before": missing,
                        "detail": cov_detail,
                        "steps_query": bool(steps_q),
                        "expanded_methods": expand_methods,
                    }

                st.write("📊 Coverage & doc-focus analysis...")
                # =========================
                # Doc-focus lock (anaphora + steps)
                # =========================
                # 1) explicit doc-focus di query (mis. menyebut "...pdf" atau "ITERA_2023_..."):
                #    -> HARUS hard lock dan tidak boleh fallback ke dokumen lain
                # 2) jikalau user follow-up "tahapannya?" (anafóra), jangan sampai retrieval "loncat" ke dokumen lain.
                #    -> saya kunci nodes ke doc_id yang pernah diasosiasikan dengan metode (dari turn sebelumnya).
                # pastikan doc_focus_meta selalu ada (biar tidak error dan log konsisten)
                doc_focus_meta: Dict[str, Any] = {
                    "enabled": False,
                    "doc_ids": [],
                    "reason": None,
                    "kept": 0,
                    "total": len(nodes or []),
                    "strict": False,
                    "mode": None,
                    # tambahan agar generator-lock bisa dipakai
                    "locked": False,
                    "allowed_doc_ids": [],
                    "focus_methods": [],
                    "method_doc_map": {},
                    "method_bias_query": None,
                    "method_bias_added": 0,
                }

                steps_q_now = bool(
                    is_steps_question(query) or detect_compression_followup
                )

                # --- (A) PRIORITAS 1: explicit doc-focus dari query user ---
                explicit_doc_focus = extract_doc_focus_from_query(query)  # list[str]
                if explicit_doc_focus:
                    # (1) HARD LOCK nodes ke dokumen yang disebut
                    nodes, dfm = apply_doc_focus_filter(
                        nodes,
                        list(explicit_doc_focus),
                        min_keep=0,
                        strict=True,  # HARD LOCK
                    )
                    doc_focus_meta.update(dfm)
                    doc_focus_meta["enabled"] = True
                    doc_focus_meta["mode"] = "explicit_query"

                    # tandai untuk dipakai generator lock
                    doc_focus_meta["locked"] = True
                    doc_focus_meta["allowed_doc_ids"] = list(explicit_doc_focus)

                    # (2) Jika pertanyaan METODE dan doc-focus explicit,
                    # lakukan 1 pass retrieval tambahan yang "method-biased",
                    # lalu hard-lock hasilnya ke dokumen yang sama, merge, dan ambil top_k terbaik.
                    if is_method_question(query):
                        focus_methods = detect_methods_in_text(query) or list(target_methods or [])
                        doc_focus_meta["focus_methods"] = list(focus_methods)

                        # query bias: memancing paragraf yang menyebut "metode yang digunakan"
                        method_hint = (
                            " ".join(pretty_method_name(m) for m in focus_methods)
                            if focus_methods
                            else ""
                        )
                        q_method_bias = (
                            f"metode pengembangan yang digunakan {method_hint} "
                            f"menggunakan metode {method_hint} "
                            f"metode yang digunakan"
                        ).strip()

                        cfg_m = dict(cfg_runtime)
                        cfg_m["retrieval"] = dict(cfg_runtime.get("retrieval", {}))

                        # ambil lebih banyak kandidat agar peluang kena paragraf "metode yang digunakan" lebih tinggi
                        cfg_m["retrieval"]["top_k"] = max(int(top_k_ui), 8)

                        extra_m = retrieve_dense(
                            cfg_m,
                            q_method_bias,
                            diversify=False,
                            max_per_doc=int(max_per_doc),
                            candidate_k=None,
                            collection="narasi",
                        )

                        # hard-lock extra ke doc yang sama
                        extra_m, _ = apply_doc_focus_filter(
                            extra_m,
                            list(explicit_doc_focus),
                            min_keep=0,
                            strict=True,
                        )

                        # merge + dedupe + ambil top_k terbaik (distance kecil = lebih relevan)
                        if extra_m:
                            nodes = dedupe_nodes_by_chunk_id(nodes + extra_m)
                            nodes = nodes[: int(top_k_ui)]

                        doc_focus_meta["method_bias_query"] = q_method_bias
                        doc_focus_meta["method_bias_added"] = int(len(extra_m or []))

                    else:
                        doc_focus_meta["focus_methods"] = []
                        doc_focus_meta["method_bias_query"] = None
                        doc_focus_meta["method_bias_added"] = 0

                # --- (B) PRIORITAS 2: anaphora+steps doc-focus dari method_doc_map (turn sebelumnya) ---
                elif (detect_anaphora or detect_compression_followup) and steps_q_now and (not topic_shift):
                    if "method_doc_map" not in st.session_state:
                        st.session_state["method_doc_map"] = {}

                    md_map = st.session_state.get("method_doc_map") or {}

                    # metode target dari history_window (infer_target_methods sudah dipakai)
                    tm_focus = list(target_methods or [])
                    allowed_doc_ids: List[str] = []
                    for m in tm_focus:
                        d = md_map.get(m)
                        if isinstance(d, str) and d.strip():
                            allowed_doc_ids.append(d.strip())

                    # dedupe (biar bersih)
                    allowed_doc_ids = list(dict.fromkeys(allowed_doc_ids))

                    if allowed_doc_ids:
                        nodes, dfm = apply_doc_focus_filter(
                            nodes,
                            allowed_doc_ids,
                            min_keep=0,
                            strict=True,  # HARD LOCK agar tidak bocor ke dokumen lain
                        )
                        doc_focus_meta.update(dfm)
                        doc_focus_meta["enabled"] = True
                        doc_focus_meta["mode"] = (
                            "compression_steps_followup"
                            if detect_compression_followup and not detect_anaphora
                            else "anaphora_steps"
                        )

                # simpan meta doc-focus untuk audit/debug UI
                st.session_state["last_doc_focus_meta"] = doc_focus_meta

                # =========================
                # V3.0c refinement: dynamic effective top-n
                # =========================
                _complex_rerank_query = bool(
                    detect_multi_target
                    or steps_q_now
                    or len(list(dict.fromkeys(target_methods or []))) >= 2
                )

                _top_n_rerank_effective = int(_top_n_rerank_base)
                if _reranker_act and _dynamic_top_n and _complex_rerank_query:
                    _top_n_rerank_effective = int(_top_n_rerank_complex)

                _top_n_rerank_effective = max(1, int(_top_n_rerank_effective))

                if _top_n_rerank_effective != _top_n_rerank_base:
                    st.write(
                        f"📈 Dynamic top-n aktif: query kompleks → "
                        f"Top-{_top_n_rerank_effective} final contexts..."
                    )

                # =========================
                # Update method_doc_map
                # (WAJIB menggunakan `nodes` pre-rerank, bukan nodes_for_gen)
                # =========================
                if "method_doc_map" not in st.session_state:
                    st.session_state["method_doc_map"] = {}

                if not is_steps_question(query):
                    if target_methods:
                        md_new: Dict[str, str] = {}

                        try:
                            if (
                                isinstance(coverage_meta, dict)
                                and coverage_meta.get("performed")
                                and isinstance(coverage_meta.get("detail"), dict)
                            ):
                                cov_map2 = coverage_meta["detail"].get("coverage") or {}
                                if isinstance(cov_map2, dict):
                                    for m, v in cov_map2.items():
                                        if isinstance(v, dict) and v.get("doc_id"):
                                            md_new[str(m)] = str(v["doc_id"])
                        except Exception:
                            pass

                        if not md_new:
                            md_new = build_method_doc_map_from_nodes(nodes, list(target_methods))

                        if md_new:
                            st.session_state["method_doc_map"].update(md_new)

                # =========================
                # V3.0c: Snapshot candidate pool pre-rerank
                # =========================
                # Simpan candidate pool di titik ini:
                # - retrieval
                # - citation routing
                # - coverage expansion
                # - doc-focus
                #
                # Setelah titik ini, reranker mengubah subset final untuk generator,
                # tetapi candidate pool asli tetap disimpan untuk audit & UI before-vs-after.
                nodes_before_rerank: List[Any] = list(nodes)
                # ← Final patch T02: simpan hasil rerank mentah sebelum anchor reservation
                nodes_reranked_pre_anchor: List[Any] = []

                # =========================
                # V3.0c: Cross-Encoder Reranking (Top-N)
                # =========================
                # Titik insersi SETELAH method_doc_map update (yang butuh nodes penuh/pre-rerank)
                # dan SEBELUM build_contexts_with_meta (yang harus pakai nodes_for_gen).
                #
                # ⚠️ PERINGATAN INTEGRASI - JANGAN SORT ULANG nodes_for_gen SETELAH BLOK INI:
                # n.score menyimpan skor retrieval lama (score_rrf untuk hybrid, distance untuk dense).
                # Relevansi final pasca-rerank ada di URUTAN nodes_for_gen dan rerank_rank,
                # BUKAN di n.score. Sort berdasarkan n.score setelah ini akan merusak urutan.
                nodes_for_gen: List[Any] = list(nodes)   # default: tanpa reranking

                # Lookup map: chunk_id → {rerank_score, rerank_rank}
                # Dipakai untuk enrichment logging retrieved_nodes (nodes pre-rerank penuh)
                _rerank_score_map: Dict[str, Any] = {}

                if _reranker_act and nodes:
                    st.write(f"⚡ Cross-Encoder Reranking ({len(nodes)} kandidat → Top-{_top_n_rerank_effective})...")

                    _reranker_info_runtime: Dict[str, Any] = {}
                    _reranker_info_post_unload: Dict[str, Any] = {}

                    try:
                        load_reranker(cfg_runtime)
                        _reranker_info_runtime = get_reranker_info()

                        if is_reranker_loaded():
                            nodes_for_gen = rerank_nodes(
                                # Pakai effective_retrieval_query (query final pasca-self-query)
                                # agar Cross-Encoder menilai relevansi terhadap query yang sama
                                # dengan yang dipakai saat retrieval.
                                effective_retrieval_query,
                                nodes,
                                top_n_rerank=_top_n_rerank_effective,
                            )
                            reranker_applied = True
                            nodes_reranked_pre_anchor = list(nodes_for_gen)

                            # Bangun lookup map untuk enrichment log retrieved_nodes
                            _rerank_score_map = {
                                str(getattr(n, "chunk_id", "") or ""): {
                                    "rerank_score": getattr(n, "rerank_score", None),
                                    "rerank_rank": getattr(n, "rerank_rank", None),
                                }
                                for n in nodes_reranked_pre_anchor
                                if str(getattr(n, "chunk_id", "") or "")
                            }
                            st.write(
                                f"✅ Reranking selesai: Top-{len(nodes_for_gen)} "
                                f"(device={_reranker_info_runtime.get('resolved_device', '?')}) dikirim ke LLM..."
                            )
                        else:
                            # Model gagal dimuat → graceful skip
                            nodes_for_gen = list(nodes[:_top_n_rerank_effective])
                            nodes_reranked_pre_anchor = list(nodes_for_gen)
                            reranker_applied = False
                            _rk_err = _reranker_info_runtime.get("error_msg", "unknown error")
                            st.warning(
                                f"⚠️ Reranker tidak berhasil dimuat ({_rk_err}) → "
                                f"graceful skip: Top-{_top_n_rerank_effective} dari retrieval langsung dipakai."
                            )

                    except Exception as _re_exc:
                        logger.warning(
                            f"[V3.0c] Reranker error (non-critical, graceful skip): {_re_exc}"
                        )
                        nodes_for_gen = list(nodes[:_top_n_rerank_effective])
                        nodes_reranked_pre_anchor = list(nodes_for_gen)
                        reranker_applied = False

                    finally:
                        # ⚠️ WAJIB: unload SEBELUM LLM dimuat untuk membebaskan VRAM
                        try:
                            unload_reranker()
                        except Exception:
                            pass
                        _reranker_info_post_unload = get_reranker_info()

                    # Simpan snapshot info runtime (model/device/fallback yang benar-benar dipakai),
                    # lalu tambahkan status cleanup setelah unload untuk audit.
                    reranker_info = dict(_reranker_info_runtime or _reranker_info_post_unload or {})
                    reranker_info["unloaded_after_use"] = True
                    reranker_info["loaded_after_unload"] = bool(
                        _reranker_info_post_unload.get("loaded", False)
                    )

                else:
                    # Reranker disabled atau dimatikan di eval
                    reranker_info = get_reranker_info()
                    if not _reranker_en:
                        logger.debug("[V3.0c] Reranker disabled in config → skip")
                    elif run_mode == "eval" and not _rk_allow_eval:
                        logger.debug("[V3.0c] Reranker disabled in eval mode → skip")

                # T10 polish:
                # Normalisasi telemetry reranker agar legacy config tidak membawa
                # fingerprint palsu, tetapi T06 fail-path tetap jujur.
                _rk_meta = _build_reranker_telemetry(
                    cfg=cfg_runtime,
                    feature_context_enabled=bool(_reranker_act),
                    reranker_applied=bool(reranker_applied),
                    reranker_info=reranker_info,
                    nodes_before_rerank=nodes_before_rerank,
                    top_n_base=(_top_n_rerank_base if _reranker_act else None),
                    top_n_effective=(_top_n_rerank_effective if _reranker_act else None),
                    rerank_anchor_reservation=rerank_anchor_reservation,
                )

                # =========================
                # V3.0c: Coverage-preserving rescue after rerank
                # refinement - deterministic anchor reservation
                # =========================
                # Tujuan:
                # - untuk query comparison / multi-target / steps,
                #   hasil final pasca-rerank tidak boleh kehilangan representasi metode target
                # - reranker tetap aktif, tetapi jika salah satu metode hilang,
                #   inject 1 anchor node terbaik dari candidate pool pre-rerank
                #
                # Contoh kasus yang diperbaiki:
                # pre-rerank berhasil menemukan RAD + PROTOTYPING,
                # tetapi final top-3 setelah rerank menjadi RAD + RAD + EXTREME_PROGRAMMING.
                # Pada kondisi ini, PROTOTYPING harus "diselamatkan" kembali ke final contexts.
                _coverage_detail_map: Dict[str, Any] = {}
                if (
                    isinstance(coverage_meta, dict)
                    and isinstance(coverage_meta.get("detail"), dict)
                ):
                    _coverage_detail_map = coverage_meta["detail"].get("coverage") or {}

                _anchor_target_methods = list(dict.fromkeys(target_methods or []))
                _reserve_method_anchors = bool(
                    (cfg_runtime.get("reranker") or {}).get("reserve_method_anchors", True)
                )

                _anchor_should_run = bool(
                    reranker_applied
                    and _reserve_method_anchors
                    and _anchor_target_methods
                    and (
                        detect_multi_target
                        or steps_q_now
                        or len(_anchor_target_methods) >= 2
                    )
                )

                if _anchor_should_run:
                    nodes_for_gen, rerank_anchor_reservation = build_final_nodes_with_anchor_reservation(
                        reranked_nodes=nodes_for_gen,
                        candidate_nodes=nodes_before_rerank,
                        target_methods=_anchor_target_methods,
                        coverage_detail=_coverage_detail_map,
                        final_k=int(_top_n_rerank_effective),
                        prefer_steps=bool(steps_q_now),
                        top_n_base=int(_top_n_rerank_base),
                        top_n_effective=int(_top_n_rerank_effective),
                        dynamic_top_n=bool(_dynamic_top_n),
                        complex_query=bool(_complex_rerank_query),
                    )

                    logger.info(
                        "[V3.0c][anchor-reservation] reserved=%s anchors=%s modified=%s final_n=%s",
                        rerank_anchor_reservation.get("reserved_methods"),
                        rerank_anchor_reservation.get("reserved_anchor_chunk_ids"),
                        rerank_anchor_reservation.get("modified"),
                        rerank_anchor_reservation.get("final_n"),
                    )

                    if rerank_anchor_reservation.get("applied"):
                        _reserved = ", ".join(rerank_anchor_reservation.get("reserved_methods") or [])
                        _msg = (
                            "🛟 Anchor reservation aktif: "
                            f"metode target [{_reserved}] dijaga tetap hadir di final contexts."
                        )
                        if rerank_anchor_reservation.get("modified"):
                            _msg += " Final set dimodifikasi setelah rerank..."
                        st.write(_msg)

                # =========================
                # V3.0c final patch: sync rerank metadata ke final nodes
                # =========================
                # Tujuan:
                # - CTX Mapping final tidak menampilkan None untuk rerank_score/rerank_rank
                # - Sources pills rerank tetap muncul
                # - panel After Rerank konsisten dengan retrieval.jsonl
                _rerank_meta_map = _build_rerank_meta_map(nodes_reranked_pre_anchor)
                nodes_for_gen = _sync_rerank_metadata_to_final_nodes(
                    nodes_for_gen,
                    _rerank_meta_map,
                )

                # ← UX patch: tandai node reserved anchor agar UI tidak menganggap
                # rerank_score/rerank_rank=None sebagai bug.
                nodes_for_gen = _tag_reserved_anchor_nodes(
                    nodes_for_gen,
                    rerank_anchor_reservation,
                )

                # =========================
                # Contexts (dari nodes_for_gen - apa yang benar-benar dikirim ke LLM)
                # =========================
                # ← V3.0c: pakai nodes_for_gen (post-rerank), bukan nodes_before_rerank
                # Ownership-aware document aggregation (Tahap 2)
                # Ownership-aware document aggregation (Tahap 2)
                doc_agg_cfg = cfg_runtime.get("doc_aggregation", {}) or {}
                doc_agg_enabled = bool(doc_agg_cfg.get("enabled", True))
                doc_agg_allow_explanation = bool(doc_agg_cfg.get("allow_for_method_explanation", True))
                doc_agg_max_docs = int(doc_agg_cfg.get("max_docs_for_generation", 5))
                doc_agg_max_evidence_chars = int(doc_agg_cfg.get("max_evidence_chars", 280))

                doc_aggregation_meta: Dict[str, Any] = {
                    "enabled": bool(doc_agg_enabled),
                    "executed": False,
                    "applied": False,
                    "target_method": None,
                    "target_method_label": None,
                    "catalog_path": None,
                    "candidate_doc_count": 0,
                    "accepted_doc_count": 0,
                    "candidate_docs": [],
                    "accepted_docs": [],
                    "rejected_docs": [],
                    "representative_chunk_ids": [],
                    "used_summary_contexts": False,
                    "trigger_reason": None,
                }

                contexts = build_contexts_with_meta(nodes_for_gen)
                nodes_for_answer = list(nodes_for_gen)

                _agg_targets = list(
                    dict.fromkeys(
                        (intent_plan or {}).get("target_methods") or target_methods or []
                    )
                )
                _agg_primary_intent = str(
                    (intent_plan or {}).get("primary_intent", "") or ""
                ).strip().lower()
                _agg_needs_runtime = bool((intent_plan or {}).get("needs_doc_aggregation"))

                if (
                    _agg_primary_intent == "method_explanation"
                    and doc_agg_allow_explanation
                    and len(_agg_targets) == 1
                ):
                    _candidate_doc_ids = {
                        str(getattr(n, "doc_id", "") or "")
                        for n in (nodes_before_rerank or [])
                        if str(getattr(n, "doc_id", "") or "").strip()
                    }
                    if len(_candidate_doc_ids) > 1:
                        _agg_needs_runtime = True
                        doc_aggregation_meta["trigger_reason"] = "method_explanation_doc_disambiguation"

                if doc_agg_enabled and _agg_needs_runtime and len(_agg_targets) == 1:
                    _agg_target = str(_agg_targets[0]).strip()
                    doc_summaries, doc_aggregation_meta = build_owner_aware_doc_aggregation(
                        nodes_before_rerank or nodes_for_gen,
                        cfg=cfg_runtime,
                        target_method=_agg_target,
                        max_docs=doc_agg_max_docs,
                        max_evidence_chars=doc_agg_max_evidence_chars,
                    )
                    if not doc_aggregation_meta.get("trigger_reason"):
                        doc_aggregation_meta["trigger_reason"] = (
                            "intent_needs_doc_aggregation"
                            if (intent_plan or {}).get("needs_doc_aggregation")
                            else "method_explanation_doc_disambiguation"
                        )

                    rep_chunk_ids = list(doc_aggregation_meta.get("representative_chunk_ids") or [])
                    rep_nodes = _pick_nodes_by_chunk_ids(
                        nodes_for_gen,
                        nodes_before_rerank,
                        rep_chunk_ids,
                    )

                    if doc_summaries:
                        contexts = build_doc_aggregation_contexts(
                            doc_summaries,
                            target_method=_agg_target,
                        )
                        if rep_nodes:
                            nodes_for_answer = rep_nodes

                # Method explanation route + explanation-biased expansion (Tahap 3)
                method_expl_cfg = cfg_runtime.get("method_explanation", {}) or {}
                expl_enabled = bool(method_expl_cfg.get("expansion_enabled", True))
                expl_max_queries = int(method_expl_cfg.get("max_expansion_queries", 5))
                expl_max_added_nodes = int(method_expl_cfg.get("max_added_nodes", 4))

                explanation_expansion_meta: Dict[str, Any] = {
                    "enabled": bool(expl_enabled),
                    "executed": False,
                    "applied": False,
                    "target_method": None,
                    "target_method_label": None,
                    "target_doc_id": None,
                    "target_doc_source_file": None,
                    "expansion_queries": [],
                    "added_chunk_ids": [],
                    "added_doc_ids": [],
                    "selected_explanation_chunk_ids": [],
                    "selected_context_count": 0,
                    "reason": None,
                }

                if (
                    expl_enabled
                    and str((intent_plan or {}).get("primary_intent", "") or "").strip().lower() == "method_explanation"
                    and bool((intent_plan or {}).get("needs_explanation_expansion", False))
                    and len(_agg_targets) == 1
                ):
                    _expl_target = str(_agg_targets[0]).strip()
                    _target_doc_id = _resolve_method_explanation_target_doc_id(
                        intent_plan=dict(intent_plan or {}),
                        doc_focus_meta=st.session_state.get("last_doc_focus_meta") or {},
                        doc_aggregation_meta=doc_aggregation_meta,
                        nodes_for_gen=nodes_for_answer or nodes_for_gen,
                        target_method=_expl_target,
                    )

                    selected_expl_nodes, explanation_expansion_meta = build_method_explanation_expansion(
                        cfg=cfg_runtime,
                        query=query,
                        target_method=_expl_target,
                        target_doc_id=_target_doc_id,
                        base_nodes=nodes_for_answer or nodes_for_gen,
                        max_queries=expl_max_queries,
                        max_added_nodes=expl_max_added_nodes,
                    )

                    if selected_expl_nodes:
                        nodes_for_answer = selected_expl_nodes
                        contexts = build_contexts_with_meta(nodes_for_answer)

                # Single-target steps handler + steps-biased expansion (Tahap 4)
                steps_cfg = cfg_runtime.get("single_target_steps", {}) or {}
                steps_enabled = bool(steps_cfg.get("expansion_enabled", True))
                steps_max_queries = int(steps_cfg.get("max_expansion_queries", 6))
                steps_max_added_nodes = int(steps_cfg.get("max_added_nodes", 4))
                steps_max_items = int(steps_cfg.get("max_steps", 5))

                single_target_steps_meta: Dict[str, Any] = {
                    "enabled": bool(steps_enabled),
                    "executed": False,
                    "applied": False,
                    "target_method": None,
                    "target_method_label": None,
                    "target_doc_id": None,
                    "target_doc_source_file": None,
                    "expansion_queries": [],
                    "added_chunk_ids": [],
                    "added_doc_ids": [],
                    "selected_chunk_ids": [],
                    "selected_context_count": 0,
                    "method_support_ctx_nums": [],
                    "explicit_steps_ctx_nums": [],
                    "extractable_steps_ctx_nums": [],
                    "extractable_steps_count": 0,
                    "artifact_only_ctx_nums": [],
                    "has_method_support": False,
                    "has_explicit_steps": False,
                    "has_extractable_steps": False,
                    "ambiguous_candidate_doc_ids": [],
                    "status": None,
                    "reason": None,
                    "max_steps": int(steps_max_items),
                }

                if (
                    steps_enabled
                    and str((intent_plan or {}).get("primary_intent", "") or "").strip().lower() == "single_target_steps"
                    and len(_agg_targets) == 1
                ):
                    _steps_target = str(_agg_targets[0]).strip()
                    _doc_focus_meta = st.session_state.get("last_doc_focus_meta") or {}

                    _target_doc_id, _resolve_status = _resolve_single_target_steps_target_doc_id_strict(
                        intent_plan=dict(intent_plan or {}),
                        doc_focus_meta=_doc_focus_meta,
                        doc_aggregation_meta=doc_aggregation_meta,
                    )

                    if _resolve_status == "ambiguous_owner_docs":
                        single_target_steps_meta.update(
                            {
                                "executed": True,
                                "target_method": _steps_target,
                                "target_method_label": pretty_method_name(_steps_target),
                                "status": "ambiguous_owner_docs",
                                "reason": "multiple_owner_docs_after_aggregation",
                                "ambiguous_candidate_doc_ids": _single_target_steps_candidate_doc_ids_from_meta(
                                    doc_aggregation_meta
                                ),
                            }
                        )

                    elif _resolve_status == "unresolved_owner_doc":
                        single_target_steps_meta.update(
                            {
                                "executed": True,
                                "target_method": _steps_target,
                                "target_method_label": pretty_method_name(_steps_target),
                                "status": "unresolved_owner_doc",
                                "reason": "no_single_owner_doc_resolved",
                                "ambiguous_candidate_doc_ids": _single_target_steps_candidate_doc_ids_from_meta(
                                    doc_aggregation_meta
                                ),
                            }
                        )

                    else:
                        selected_step_nodes, single_target_steps_meta = build_single_target_steps_expansion(
                            cfg=cfg_runtime,
                            query=query,
                            target_method=_steps_target,
                            target_doc_id=_target_doc_id,
                            base_nodes=nodes_for_answer or nodes_for_gen,
                            max_queries=steps_max_queries,
                            max_added_nodes=steps_max_added_nodes,
                            max_steps=steps_max_items,
                        )

                        # Penting:
                        # walau status akhirnya partial, contexts tetap harus diganti ke selected_step_nodes
                        # agar CTX index di meta sinkron dengan contexts final untuk generator.
                        if selected_step_nodes:
                            nodes_for_answer = selected_step_nodes
                            contexts = build_contexts_with_meta(nodes_for_answer)

                # Method comparison formatter meta (Tahap 5)
                cmp_cfg = cfg_runtime.get("comparison_formatter", {}) or {}
                cmp_enabled = bool(cmp_cfg.get("enabled", True))
                cmp_max_ctx_per_method = int(cmp_cfg.get("max_ctx_per_method", 2))

                method_comparison_meta: Dict[str, Any] = {
                    "enabled": bool(cmp_enabled),
                    "executed": False,
                    "target_methods": [],
                    "ctx_map_per_method": {},
                    "supported_methods": [],
                    "weak_methods": [],
                    "missing_methods": [],
                    "comparison_status": None,
                    "balanced": False,
                    "selected_context_count": 0,
                    "max_ctx_per_method": int(cmp_max_ctx_per_method),
                }

                if (
                    cmp_enabled
                    and str((intent_plan or {}).get("primary_intent", "") or "").strip().lower() == "method_comparison"
                    and len(_agg_targets) >= 2
                ):
                    method_comparison_meta = build_method_comparison_meta(
                        nodes_for_answer or nodes_for_gen,
                        target_methods=list(_agg_targets),
                        max_ctx_per_method=cmp_max_ctx_per_method,
                    )

                # ─────────────────────────────────────────────────────────────
                # Final patch: sinkronisasi metadata rerank ke final contexts/UI
                # ─────────────────────────────────────────────────────────────
                # nodes_for_answer pada titik ini bisa sudah diganti oleh:
                # - doc aggregation representative nodes
                # - explanation expansion
                # - single-target steps expansion
                #
                # Karena beberapa jalur tersebut dapat membawa salinan node baru
                # (chunk_id sama, object berbeda), maka saya lakukan final pass
                # untuk:
                # 1) rehydrate object final dari pool kanonik
                # 2) sync ulang rerank_score/rerank_rank
                # 3) tag reserved anchor untuk UI audit
                _rerank_meta_map_for_final_ui: Dict[str, Dict[str, Any]] = {
                    str(k): dict(v)
                    for k, v in (_rerank_score_map or {}).items()
                    if isinstance(v, dict)
                }

                nodes_for_answer = _finalize_nodes_for_answer_rerank_ui(
                    current_nodes=nodes_for_answer or nodes_for_gen,
                    canonical_preferred_nodes=nodes_reranked_pre_anchor or nodes_for_gen,
                    canonical_fallback_nodes=nodes_before_rerank or [],
                    rerank_meta_map=_rerank_meta_map_for_final_ui,
                    anchor_meta=rerank_anchor_reservation,
                )

                # contexts HARUS dibangun ulang dari node final yang sudah disinkronkan
                contexts = build_contexts_with_meta(nodes_for_answer)

                # Tentukan max bullet Bukti dinamis (3 default, bisa naik jadi 5)
                max_bullets, bullets_meta = decide_max_bullets(
                    user_query=query,  # pakai user query asli
                    contexts=contexts,
                    default_max=3,
                    expanded_max=5,
                )

                # ← V3.0c:
                # Bedakan secara eksplisit:
                # - retrieval_stats_pre_rerank = kualitas candidate pool retrieval
                # - generation_context_stats   = kualitas final contexts yang benar-benar masuk ke LLM
                retrieval_stats_pre_rerank = compute_retrieval_stats(nodes_before_rerank)
                generation_context_stats = compute_retrieval_stats(nodes_for_answer)

                # Backward-compatible:
                # pertahankan retrieval_stats lama mengacu ke final contexts yang dipakai generator, agar consumer lama tidak rusak.
                retrieval_stats = generation_context_stats

                strategy = retrieval_mode  # "dense" atau "hybrid"
                if retrieval_query_before_self_query != query:
                    strategy = f"{strategy}+rewrite"
                if _self_query_filter is not None:
                    strategy = f"{strategy}+selfquery"   # ← V3.0b
                # +diverse marker hanya relevan untuk jalur dense murni
                if diversify and retrieval_mode != "hybrid":
                    strategy = f"{strategy}+diverse"
                if reranker_applied:
                    strategy = f"{strategy}+rerank"   # ← V3.0c

                # timestamp per turn
                ts = datetime.now().isoformat(timespec="seconds")

                # log retrieval via RunManager
                retrieval_record = {
                    "run_id": rm.run_id,
                    "turn_id": turn_id,
                    "timestamp": ts,
                    "retrieval_mode": cfg_runtime["retrieval"]["mode"],
                    "llm_provider": provider,
                    "use_llm": use_llm,
                    "top_k": int(top_k_ui),
                    "user_query": query,
                    "retrieval_query": effective_retrieval_query,
                    "run_mode": run_mode,
                    "use_memory": use_memory,
                    "mem_window": int(mem_window),
                    "history_used": history_used,
                    "history_turn_ids": history_turn_ids,
                    "contextualized": (retrieval_query_before_self_query != query),
                    "detect_flags": detect_flags,
                    "topic_shift_meta": topic_shift_meta,
                    "history_policy": history_policy_meta,
                    "coverage": coverage_meta,
                    "metadata_route": metadata_meta,
                    "contextualize_decision": contextualize_decision,
                    "doc_focus": st.session_state.get("last_doc_focus_meta"),
                    "strategy": strategy,
                    "diversity": {
                        "mode": diverse_mode,
                        "enabled": bool(diversify),
                        "max_per_doc": int(max_per_doc),
                        "pool_mult": int(pool_mult),
                        "candidate_k": int(candidate_k) if candidate_k is not None else None,
                    },
                    "retrieval_stats": retrieval_stats,  # Backward-compatible: tetap ada
                    # ← V3.0c: stats eksplisit pre vs post
                    "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                    "generation_context_stats": generation_context_stats,
                    "retrieval_confidence": retrieval_stats.get("sim_top1"),
                    "self_query": _self_query_log,   # ← V3.0b
                    # ← V3.0c: reranker metadata di level record
                    "reranker_applied":         _rk_meta["applied"],
                    "reranker_candidate_count": _rk_meta["candidate_count"],  # jumlah node sebelum rerank
                    "reranker_top_n":           _rk_meta["top_n"],  # effective final top-n
                    "reranker_top_n_base":      _rk_meta["top_n_base"],
                    "reranker_top_n_effective": _rk_meta["top_n_effective"],
                    "reranker_info":            _rk_meta["info"],
                    "rerank_coverage_rescue":   rerank_coverage_rescue,         # legacy telemetry
                    "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                    # ← V3.0c: retrieved_nodes = SEMUA kandidat pre-rerank (bukan hanya nodes_for_gen)
                    # sehingga log memperlihatkan: "10 node di-retrieve, 3 lolos rerank, 7 tidak"
                    # rerank_score/rerank_rank diisi hanya untuk node yang masuk Top-N (dari _rerank_score_map)
                    "retrieved_nodes": [
                        {
                            "doc_id": n.doc_id,
                            "chunk_id": n.chunk_id,
                            # NOTE: score dari Chroma umumnya adalah distance (lebih kecil = lebih relevan)
                            # Dense: distance + similarity (None untuk jalur hybrid)
                            "distance": n.score if retrieval_mode != "hybrid" else None,
                            "similarity": dist_to_sim(n.score) if retrieval_mode != "hybrid" else None,
                            # Hybrid: RRF score + komponen dense/sparse (None untuk jalur dense)
                            "score_rrf":    float(n.score) if retrieval_mode == "hybrid" else None,
                            "score_dense":  n.score_dense,
                            "score_sparse": n.score_sparse,
                            "rank_dense":   n.rank_dense,
                            "rank_sparse":  n.rank_sparse,
                            # ← V3.0c: terisi jika node masuk Top-N rerank; None jika tidak
                            "rerank_score": _rerank_score_map.get(n.chunk_id, {}).get("rerank_score"),
                            "rerank_rank":  _rerank_score_map.get(n.chunk_id, {}).get("rerank_rank"),
                            "stream":      n.stream,       # (V3.0a; None untuk V1/V2)
                            "bab_label":   n.bab_label,    # (V3.0a; None untuk V1/V2)
                            "metadata": n.metadata,
                            "text": n.text,  # akan di-trim otomatis oleh RunManager.log_retrieval
                        }
                        for n in nodes # iterasi nodes PRE-RERANK (penuh)
                    ],
                }
                rm.log_retrieval(retrieval_record, include_config_snapshot_per_row=include_cfg)

                # =========================
                # Generation
                # =========================
                # Kalau generator ON: jawab berdasarkan contexts top-k.
                # Kalau OFF: mode retrieval-only (extractive) agar UI tetap konsisten (Jawaban + Bukti).
                # Retrieval memakai retrieval_query (hasil rewrite) -> untuk mendapatkan konteks yang tepat.
                # Tetapi LLM harus menjawab pertanyaan USER (query asli) agar tidak mismatch.
                user_query = query  # selalu menjawab berdasarkan user_query
                # ← V3.0c: used_ctx dari nodes_for_gen (yang benar-benar dikirim ke LLM)
                used_ctx = min(len(contexts), min(int(top_k_ui), 8))  # generate.py memakai hard cap 8
                generator_error = None

                # ===== Generation policy (per-metode) =====
                gen_cfg = cfg_runtime.get("generation", {}) or {}
                per_method_bullets = int(gen_cfg.get("per_method_bullets", 2))
                max_total_bullets = int(gen_cfg.get("max_total_bullets", 6))
                force_method_list = bool(gen_cfg.get("force_per_method_for_method_list", True))

                # target_methods untuk generator:
                # - prioritas: dari coverage detail (paling reliable)
                # - fallback: target_methods dari infer_target_methods
                final_intent_plan = dict(intent_plan or {})
                _dfm_for_plan = st.session_state.get("last_doc_focus_meta") or {}
                if isinstance(_dfm_for_plan, dict) and _dfm_for_plan.get("locked"):
                    final_intent_plan["doc_locked"] = True
                    final_intent_plan["allowed_doc_ids"] = list(
                        dict.fromkeys(
                            [
                                str(x).strip()
                                for x in (
                                    _dfm_for_plan.get("allowed_doc_ids")
                                    or _dfm_for_plan.get("doc_ids")
                                    or []
                                )
                                if str(x).strip()
                            ]
                        )
                    )

                gen_targets: List[str] = []
                missing_methods: List[str] = []
                cov_map: Dict[str, Any] = {}  # supaya selalu terdefinisi

                if (
                    isinstance(coverage_meta, dict)
                    and coverage_meta.get("performed")
                    and isinstance(coverage_meta.get("detail"), dict)
                ):
                    gen_targets = list(coverage_meta["detail"].get("target_methods") or [])
                    cov_map = coverage_meta["detail"].get("coverage") or {}
                    if isinstance(cov_map, dict):
                        missing_methods = [
                            m
                            for m, v in cov_map.items()
                            if isinstance(v, dict) and (not v.get("found"))
                        ]
                else:
                    gen_targets = list(target_methods or [])

                if (not gen_targets) and isinstance(final_intent_plan, dict):
                    tm_plan = final_intent_plan.get("target_methods") or []
                    if isinstance(tm_plan, list):
                        gen_targets = [m for m in tm_plan if isinstance(m, str) and m]

                # ===== Doc-focus lock: kunci target generator ke metode yang dilock =====
                dfm = st.session_state.get("last_doc_focus_meta") or {}
                if isinstance(dfm, dict) and dfm.get("locked"):
                    focus_methods = dfm.get("focus_methods") or []
                    method_doc_map = dfm.get("method_doc_map") or {}

                    if isinstance(focus_methods, list) and focus_methods:
                        # override gen_targets agar generator tidak melebar ke dokumen lain
                        gen_targets = [m for m in focus_methods if isinstance(m, str) and m]
                        # siapkan cov_map minimal untuk steps-aware missing (dipakai di bawah)
                        cov_map = {
                            m: {"found": True, "doc_id": str(method_doc_map.get(m, "") or "")}
                            for m in gen_targets
                        }
                        # reset missing_methods baseline; nanti dihitung lagi oleh blok steps-aware
                        missing_methods = []

                # kapan force per-metode?
                q_low = (query or "").lower()
                is_steps_q = bool(
                    re.search(r"\b(tahap|tahapan)\w*\b|\b(langkah|alur|prosedur)\w*\b", q_low)
                )
                is_method_list_q = bool(
                    re.search(r"\bmetode\b.*\bapa\s+saja\b|\bmetode\s+apa\s+saja\b", q_low)
                )

                if isinstance(final_intent_plan, dict):
                    if bool(final_intent_plan.get("is_steps_query")):
                        is_steps_q = True

                # ===== fallback gen_targets untuk pertanyaan METODE (single-target) =====
                # kasus: "metode pengembangan apa yang digunakan?" -> detect_multi_target False -> infer_target_methods bisa kosong
                # supaya generator tidak jatuh ke jawaban template "Informasi relevan ditemukan..."
                if is_method_question(query) and (not gen_targets):
                    ctx_methods = detect_methods_in_contexts(contexts)
                    if ctx_methods:
                        # kalau bukan "metode apa saja" (list), ambil 1 metode dominan (top doc) agar tidak melebar
                        if (not is_method_list_q) and len(ctx_methods) > 1 and nodes_for_gen:
                            top_doc = str(getattr(nodes_for_gen[0], "doc_id", "") or "").lower()
                            picked = None
                            for m in ctx_methods:
                                if _docid_hit_for_method(top_doc, m):
                                    picked = m
                                    break
                            gen_targets = [picked or ctx_methods[0]]
                        else:
                            gen_targets = ctx_methods[:3]  # batasi agar tidak terlalu padat

                # ===== steps-aware missing_methods (lebih ketat & presisi) =====
                # Untuk query tahapan: sebuah metode dianggap "available" hanya jika konteks dari DOC metode tsb
                # memuat sinyal tahapan yang benar-benar eksplisit (has_steps_signal = True).
                if is_steps_q and gen_targets:
                    # doc_id per metode dari coverage detail → paling stabil
                    method_doc: Dict[str, str] = {}
                    if isinstance(cov_map, dict):
                        for m, v in cov_map.items():
                            if isinstance(v, dict) and v.get("doc_id"):
                                method_doc[m] = str(v["doc_id"])

                    # fallback tambahan: jikalau coverage tidak punya doc_id (atau coverage tidak performed),
                    # ambil dari method_doc_map lintas turn (doc-focus lock)
                    session_md = st.session_state.get("method_doc_map") or {}
                    for m in gen_targets:
                        if m not in method_doc and session_md.get(m):
                            method_doc[m] = str(session_md.get(m))

                    def _ctx_for_doc_id(doc_id: str) -> List[str]:
                        if not doc_id:
                            return []
                        # match tepat pada header pertama: "DOC: <doc_id> |"
                        pat = re.compile(rf"(?im)^DOC:\s*{re.escape(doc_id)}\s*\|")
                        return [c for c in (contexts or []) if pat.search(c or "")]

                    steps_missing: List[str] = []

                    for m in gen_targets:
                        doc_id = (method_doc.get(m, "") or "").strip()
                        ctx_for_m: List[str] = _ctx_for_doc_id(doc_id)

                        # fallback: kalau doc_id tidak ada (jarang), memakai deteksi metode dari teks konteks
                        if (not ctx_for_m) and (not doc_id):
                            for c in contexts or []:
                                try:
                                    if m in set(detect_methods_in_text(c or "")):
                                        ctx_for_m.append(c)
                                except Exception:
                                    pass

                        single_target_steps_route = (
                            str((final_intent_plan or {}).get("primary_intent", "") or "").strip().lower()
                            == "single_target_steps"
                        )

                        # Untuk single-target steps:
                        # - TIDAK boleh dianggap missing hanya karena langkah eksplisit belum muncul.
                        # - baru dianggap missing jika memang tidak ada konteks yang mendukung metode tsb.
                        if not ctx_for_m:
                            steps_missing.append(m)
                            continue

                        # Untuk multi-target steps:
                        # - tidak ada konteks yang jelas untuk metode tsb, ATAU
                        # - ada konteks tapi tidak ada sinyal tahapan eksplisit
                        if (not single_target_steps_route) and (not any(has_steps_signal(x) for x in ctx_for_m)):
                            steps_missing.append(m)

                    # gabungkan: missing dari coverage + missing dari steps-signal (unique, urut stabil)
                    missing_methods = list(dict.fromkeys((missing_methods or []) + steps_missing))

                force_per_method = False
                if len(gen_targets) >= 2:
                    # steps query: selalu force (menghindari salah config/regex edge)
                    if is_steps_q:
                        force_per_method = True
                    # method-list query: pakai config flag (opsional)
                    elif is_method_list_q and force_method_list:
                        force_per_method = True
                    # multi-target eksplisit: force
                    elif detect_multi_target:
                        force_per_method = True

                # override max_bullets jika per-metode aktif
                if force_per_method and len(gen_targets) >= 2:
                    max_bullets = min(
                        int(max_total_bullets), int(per_method_bullets) * len(gen_targets)
                    )

                bullets_meta = dict(bullets_meta or {})
                bullets_meta["max_bullets"] = int(max_bullets)
                bullets_meta["answer_policy_route"] = (final_intent_plan or {}).get("primary_intent")
                # enrich bullets_meta untuk logging
                bullets_meta.update(
                    {
                        "force_per_method": bool(force_per_method),
                        "target_methods_for_gen": gen_targets,
                        "per_method_bullets": int(per_method_bullets),
                        "max_total_bullets": int(max_total_bullets),
                        "missing_methods": missing_methods,
                        "is_steps_q": bool(is_steps_q),
                        "is_method_list_q": bool(is_method_list_q),
                    }
                )

                if isinstance(single_target_steps_meta, dict) and single_target_steps_meta.get("executed"):
                    bullets_meta["single_target_steps_status"] = single_target_steps_meta.get("status")

                generator_strategy_name = "llm" if use_llm else "retrieval_only"
                _force_metadata_not_found = _should_force_metadata_not_found_answer(_self_query_log)
                st.write("🎰 Generating jawaban dengan LLM..." if use_llm else "📄 Mode retrieval-only...")

                if _force_metadata_not_found:
                    st.write("⛔ Exact metadata match tidak ditemukan; jawaban dipaksa not-found untuk mencegah false positive.")
                    answer_raw = _build_metadata_not_found_answer()
                    generator_strategy_name = "self_query_guard"
                elif use_llm:
                    try:
                        answer_raw = generate_answer(
                            cfg_runtime,
                            user_query,
                            contexts,
                            max_bullets=max_bullets,
                            target_methods=gen_targets,
                            missing_methods=missing_methods,
                            force_per_method=force_per_method,
                            per_method_bullets=per_method_bullets,
                            max_total_bullets=max_total_bullets,
                            intent_plan=final_intent_plan,
                            single_target_steps_meta=single_target_steps_meta,
                            method_comparison_meta=method_comparison_meta,
                        )
                    except Exception as ex:
                        logger.exception("LLM call failed")
                        generator_error = {"type": type(ex).__name__, "message": str(ex)}
                        st.warning(f"LLM error: {type(ex).__name__}: {ex}")
                        answer_raw = "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"
                else:
                    answer_raw = build_retrieval_only_answer(nodes_for_answer, max_points=3)

                # Tahap 4.3 (final hardening):
                # Jika route planner + runtime meta sudah menyatakan single_target_steps
                # dalam status ambiguity/unresolved, answer layer HARUS memaksa safe answer.
                # ini adalah lapisan pengaman terakhir, analog dengan self_query_guard.
                if _should_force_single_target_steps_safe_answer_final(
                    final_intent_plan,
                    single_target_steps_meta,
                ):
                    answer_raw = _build_single_target_steps_safe_answer_final(
                        single_target_steps_meta
                    )
                    generator_strategy_name = "single_target_steps_guard"

                # meta generator (dibuat setelah answer_raw final)
                generator_meta = build_generation_meta(
                    question=user_query,
                    contexts=contexts,
                    answer_text=answer_raw,
                    used_ctx=used_ctx,
                    max_bullets=max_bullets,
                )

                # ← V3.0d: post-hoc verification dibangun di answer layer
                # dan bekerja terhadap final contexts + answer final.
                verification_meta = build_verification_meta(
                    question=query,
                    contexts=contexts,
                    answer_text=answer_raw,
                    used_ctx=used_ctx,
                    used_context_chunk_ids=[n.chunk_id for n in nodes_for_answer],
                    generator_meta=generator_meta,
                    cfg=cfg_runtime,
                    metadata_route_handled=bool(metadata_meta.get("handled")),
                )

                intent_plan_log = final_intent_plan if log_intent_plan else None
                doc_aggregation_log = (
                    copy.deepcopy(doc_aggregation_meta)
                    if log_doc_aggregation_meta
                    else None
                )
                answer_policy_audit_summary = _build_answer_policy_audit_summary(
                    intent_plan_log,
                    doc_aggregation_log,
                )

                generator_status = generator_meta.get("status", "unknown")
                jawaban, bukti = parse_answer(answer_raw)

                # log answers via RunManager
                answer_record = {
                    "run_id": rm.run_id,
                    "turn_id": turn_id,
                    "timestamp": ts,
                    "llm_provider": provider,
                    "use_llm": use_llm,
                    "top_k": int(top_k_ui),
                    "user_query": query,
                    "retrieval_query": effective_retrieval_query,
                    "run_mode": run_mode,
                    "use_memory": use_memory,
                    "mem_window": int(mem_window),
                    "history_used": history_used,
                    "history_turn_ids": history_turn_ids,
                    "contextualized": bool(retrieval_query_before_self_query != query),
                    "detect_flags": detect_flags,
                    "topic_shift_meta": topic_shift_meta,
                    "history_policy": history_policy_meta,
                    "coverage": coverage_meta,
                    "metadata_route": metadata_meta,
                    "contextualize_decision": contextualize_decision,
                    "doc_focus": st.session_state.get("last_doc_focus_meta"),
                    "intent_plan": intent_plan_log,
                    "answer_policy_route": (final_intent_plan or {}).get("primary_intent"),
                    "doc_aggregation_meta": doc_aggregation_log,
                    "answer_policy_audit_summary": answer_policy_audit_summary,
                    "explanation_expansion_meta": explanation_expansion_meta,
                    "single_target_steps_meta": single_target_steps_meta,
                    "method_comparison_meta": method_comparison_meta,
                    # ← V3.0c: chunk IDs yang benar-benar masuk ke LLM (nodes_for_gen)
                    "used_context_chunk_ids": [n.chunk_id for n in nodes_for_answer],
                    "answer": answer_raw,
                    "strategy": strategy,
                    "diversity": {
                        "mode": diverse_mode,
                        "enabled": bool(diversify),
                        "max_per_doc": int(max_per_doc),
                        "pool_mult": int(pool_mult),
                        "candidate_k": int(candidate_k) if candidate_k is not None else None,
                    },
                    "retrieval_stats": retrieval_stats,  # Backward-compatible
                    # ← V3.0c: stats eksplisit pre vs post
                    "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                    "generation_context_stats": generation_context_stats,
                    "retrieval_confidence": retrieval_stats.get("sim_top1"),
                    "generator_strategy": generator_strategy_name,
                    "generator_status": generator_status,
                    "generator_meta": generator_meta,
                    "verification": verification_meta,  # ← V3.0d
                    "max_bullets": int(max_bullets),
                    "bullet_policy": bullets_meta,  # berisi methods_detected, anaphora, multi_target, dll
                    "error": generator_error,
                    "self_query": _self_query_log,   # ← V3.0b
                    # ← V3.0c
                    "reranker_applied": _rk_meta["applied"],
                    "reranker_model":   (_rk_meta["info"] or {}).get("model_name"),
                    "reranker_top_n":   _rk_meta["top_n"],
                    "reranker_top_n_base": _rk_meta["top_n_base"],
                    "reranker_top_n_effective": _rk_meta["top_n_effective"],
                    "rerank_coverage_rescue": rerank_coverage_rescue,           # legacy telemetry
                    "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                }
                rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)

                # --- simpan metode ke history agar rewrite steps stabil ---
                # PENTING: JANGAN gunakan bullets_meta.methods_detected sebagai fallback.
                # methods_detected berasal dari chunk yang di-retrieve (bukan dari jawaban nyata),
                # sehingga bisa menyebabkan method carry-over ke turn berikutnya secara keliru.
                # Contoh: query off-topic (Coldplay) → chunk ETicketing/RAD di-retrieve →
                # methods_detected = ["RAD", "PROTOTYPING", "XP"] → tersimpan di history →
                # anaphora follow-up mengira topiknya tentang metode pengembangan.
                turn_methods: List[str] = []
                try:
                    # 1) paling reliable: metode yang dipakai generator (gen_targets)
                    if isinstance(gen_targets, list) and gen_targets:
                        turn_methods = [m for m in gen_targets if isinstance(m, str) and m]

                    # 2) fallback: HANYA target_methods_for_gen (bukan methods_detected!)
                    if not turn_methods and isinstance(bullets_meta, dict):
                        md = bullets_meta.get("target_methods_for_gen") or []
                        if isinstance(md, list):
                            turn_methods = [m for m in md if isinstance(m, str) and m]

                    # 3) fallback terakhir: scan jawaban raw
                    # Hanya jika jawaban bukan "tidak ditemukan" (mencegah carry-over kosong)
                    if not turn_methods:
                        if answer_raw and "tidak ditemukan pada dokumen" not in answer_raw.lower():
                            turn_methods = detect_methods_in_text(answer_raw)

                except Exception:
                    turn_methods = []

                # dedupe + batasi
                turn_methods = list(dict.fromkeys(turn_methods))[:5]

                st.session_state["history"].append(
                    {
                        "turn_id": turn_id,
                        "query": query,
                        "answer_preview": (jawaban or answer_raw)[:220]
                        + ("..." if len(jawaban or answer_raw) > 220 else ""),
                        "detect_flags": detect_flags,
                        "topic_shift": bool(detect_flags.get("topic_shift")),
                        "metadata_intent": None,
                        "methods": turn_methods,
                    }
                )

                # persist last result (bersifat tetap hingga query baru dijalankan)
                st.session_state["last_answer_raw"] = answer_raw  # menyimpan last answer dari copy
                st.session_state["last_verification"] = verification_meta  # ← V3.0d
                st.session_state["last_intent_plan"] = intent_plan_log
                st.session_state["last_doc_aggregation_meta"] = doc_aggregation_log
                st.session_state["last_answer_policy_audit_summary"] = answer_policy_audit_summary
                st.session_state["last_explanation_expansion_meta"] = explanation_expansion_meta
                st.session_state["last_single_target_steps_meta"] = single_target_steps_meta
                st.session_state["last_method_comparison_meta"] = method_comparison_meta
                st.session_state["last_turn_id"] = turn_id
                # ← V3.0c:
                # last_nodes                = final contexts yang benar-benar dipakai LLM
                # last_nodes_before_rerank  = candidate pool pre-rerank untuk UI before-vs-after
                st.session_state["last_nodes"] = nodes_for_answer
                st.session_state["last_nodes_before_rerank"] = nodes_before_rerank
                st.session_state["last_strategy"] = strategy
                st.session_state["last_retrieval_mode"] = retrieval_mode 
                # ← V3.0c: simpan stats eksplisit untuk current/latest render
                st.session_state["last_retrieval_stats_pre_rerank"] = retrieval_stats_pre_rerank
                st.session_state["last_generation_context_stats"] = generation_context_stats
                st.session_state["_scroll_answer"] = True
                st.session_state["last_user_query"] = query
                st.session_state["last_retrieval_query"] = effective_retrieval_query
                # tandai bahwa konfigurasi UI saat ini sudah dipakai untuk hasil terakhir
                st.session_state["last_applied_ui_cfg"] = current_ui_cfg.copy()
                st.session_state["last_history_used"] = history_used
                st.session_state["last_contextualized"] = bool(retrieval_query != query)
                st.session_state["last_reranker_applied"] = _rk_meta["applied"]  # ← V3.0c
                st.session_state["last_reranker_info"] = _rk_meta["info"]        # ← V3.0c
                st.session_state["last_rerank_anchor_reservation"] = _rk_meta["anchor_reservation"]

                # simpan full render data untuk turn viewer
                if "turn_data" not in st.session_state:
                    st.session_state["turn_data"] = {}
                st.session_state["turn_data"][turn_id] = {
                    # ← V3.0c:
                    # nodes                = final contexts yang benar-benar dipakai LLM
                    # nodes_before_rerank  = candidate pool pre-rerank
                    "nodes": nodes_for_answer,
                    "nodes_before_rerank": nodes_before_rerank,
                    "answer_raw": answer_raw,
                    "user_query": query,
                    "retrieval_query": effective_retrieval_query,
                    "strategy": strategy,
                    "history_used": history_used,
                    "contextualized": bool(retrieval_query_before_self_query != query),
                    "retrieval_mode": retrieval_mode,
                    "self_query_filter_applied": bool(_self_query_filter is not None),  # ← V3.0b
                    "reranker_applied": _rk_meta["applied"],  # ← V3.0c
                    # ← V3.0c: disiapkan untuk app_ui_render.py
                    "reranker_candidate_count": _rk_meta["candidate_count"],
                    "reranker_top_n": _rk_meta["top_n"],
                    "reranker_top_n_base": _rk_meta["top_n_base"],
                    "reranker_top_n_effective": _rk_meta["top_n_effective"],
                    "retrieval_stats_pre_rerank": retrieval_stats_pre_rerank,
                    "generation_context_stats": generation_context_stats,
                    "rerank_coverage_rescue": rerank_coverage_rescue,           # legacy telemetry
                    "rerank_anchor_reservation": _rk_meta["anchor_reservation"],
                    "reranker_info": _rk_meta["info"],
                    "verification": verification_meta,  # ← V3.0d
                    "intent_plan": intent_plan_log,
                    "doc_aggregation_meta": doc_aggregation_log,
                    "answer_policy_audit_summary": answer_policy_audit_summary,
                    "explanation_expansion_meta": explanation_expansion_meta,
                    "single_target_steps_meta": single_target_steps_meta,
                    "method_comparison_meta": method_comparison_meta,
                }

            _proc_status.update(label="💡 Finished!", state="complete", expanded=False)

    # ===== Render Last Turn Status slot (setelah processing, nilainya sudah final) =====
    _turn_count  = int(st.session_state.get("turn_id", 0))
    _sid         = st.session_state.get("viewing_turn_id")
    _tds         = st.session_state.get("turn_data", {})
    _is_hist_sb  = (_sid is not None and _sid in _tds)

    if _is_hist_sb:
        # sidebar reflect turn yang sedang dilihat
        _sb_td       = _tds[_sid]
        _sb_turn_str = str(_sid)
        _sb_mem      = _sb_td.get("history_used")
        _sb_ctx      = _sb_td.get("contextualized")
        _sb_mem_str  = f"{_sb_mem} turns" if _sb_mem is not None else "—"
        _sb_ctx_str  = ("☑️" if _sb_ctx else "❌") if _sb_ctx is not None else "—"
    else:
        _sb_turn_str = str(_turn_count) if _turn_count > 0 else "—"
        _sb_mem      = st.session_state.get("last_history_used")
        _sb_ctx      = st.session_state.get("last_contextualized")
        _sb_mem_str  = f"{_sb_mem} turns" if _sb_mem is not None else "—"
        _sb_ctx_str  = ("☑️" if _sb_ctx else "❌") if _sb_ctx is not None else "—"

    with last_turn_status_slot.container():
        st.subheader("🚨 Last Turn Status")
        st.write("Turns (Session):", _sb_turn_str)
        st.write("Memory Used:", _sb_mem_str)
        st.write("Contextualized:", _sb_ctx_str)
        # ← V3.0c: tampilkan status reranker di turn status
        _sb_rk = (
            _tds[_sid].get("reranker_applied")
            if _is_hist_sb
            else st.session_state.get("last_reranker_applied", False)
        )
        if _sb_rk is not None:
            st.write("Reranker:", "☑️ Applied" if _sb_rk else "❌ Not Applied")
        # Baris tambahan: hanya muncul saat user sedang melihat turn historis
        if _is_hist_sb:
            st.markdown(
                f'<div style="background:rgba(37,99,235,0.08);border-left:3px solid #3b82f6;'
                f'border-radius:0 0.3rem 0.3rem 0;padding:0.3rem 0.6rem;margin-top:0.35rem;">'
                f'<span style="color:#93c5fd;font-size:0.8rem;">👁️ Viewing:</span>'
                f'<span style="color:#60a5fa;font-weight:700;font-size:0.8rem;margin-left:0.4rem;">'
                f'Chat-Turn {_sid}</span></div>',
                unsafe_allow_html=True,
            )

    # ===== Downloads (Logs) - Extend Sidebar =====
    with st.sidebar.expander("Downloads (Run Logs)", expanded=False):
        manifest_path = rm.run_path / "manifest.json"
        download_file_button(
            "📥 Download manifest.json",
            manifest_path,
            "application/json",
            disabled_caption="Manifest selalu ada.",
        )
        download_file_button(
            "📥 Download retrieval.jsonl",
            rm.retrieval_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "📥 Download answers.jsonl",
            rm.answers_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "📥 Download app.log",
            rm.app_log_path,
            "text/plain",
            disabled_caption="Log aplikasi untuk debug.",
        )
        download_logs_zip_button(
            "📥 Download logs.zip (all)",
            files=[manifest_path, rm.retrieval_log_path, rm.answers_log_path, rm.app_log_path],
            zip_name=f"{rm.run_id}_logs.zip",
        )

    # ===== Compute has_result sekali =====
    has_result = (
        st.session_state.get("last_answer_raw") is not None
        and int(st.session_state.get("last_turn_id", 0)) > 0
    )

    # ===== Render indikator config (setelah submit) =====
    last_applied = st.session_state.get("last_applied_ui_cfg")
    config_dirty = (last_applied is not None) and (last_applied != current_ui_cfg)

    if not has_result:
        # belum ada query sama sekali → pesan netral (tidak membingungkan)
        cfg_indicator_slot.info("ℹ️ Belum ada Query. **Atur Konfigurasi** lalu Tekan **Kirim**.")
    else:
        if config_dirty:
            cfg_indicator_slot.warning(
                "⚠️ **Konfigurasi Diubah**! Tekan **Kirim** untuk Menerapkan Perubahan ke Hasil Berikutnya."
            )
        else:
            # hijau (success) supaya jelas “sudah diterapkan”
            cfg_indicator_slot.success("❇️ **Konfigurasi Aktif** sudah Diterapkan pada Hasil Terakhir.")

    # ===== Render history area (setelah submit) =====
    _viewing_id_now = st.session_state.get("viewing_turn_id")
    _last_turn_id_now = int(st.session_state.get("last_turn_id", 0))
    with history_slot.container():
        with st.expander("📚 Riwayat Tanya-Jawab (Session)", expanded=False):
            if not st.session_state["history"]:
                st.caption("Belum ada Pertanyaan pada Sesi ini!")
            else:
                # Sentinel: ID unik agar CSS :has() hanya menarget expander history ini,
                # tidak bocor ke expander lain (Sources, Downloads, dll.)
                st.markdown(
                    '<div id="hist-scope" style="display:none;"></div>',
                    unsafe_allow_html=True,
                )
                # CSS: style tombol history (card)
                st.markdown(
                    """
                    <style>
                    /* ── Scrollable area ── */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"] {
                        max-height: 360px !important;
                        overflow-y: auto !important;
                        overflow-x: hidden !important;
                        padding-right: 0.3rem !important;
                        /* Fade-out bawah: sinyal visual ada konten tersembunyi */
                        -webkit-mask-image: linear-gradient(
                            to bottom, black 0%, black 80%, transparent 100%
                        ) !important;
                        mask-image: linear-gradient(
                            to bottom, black 0%, black 80%, transparent 100%
                        ) !important;
                    }
                    /* Scrollbar */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar {
                        width: 5px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-track {
                        background: #0d1117 !important;
                        border-radius: 999px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb {
                        background: #1e3a52 !important;
                        border-radius: 999px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb:hover {
                        background: #3b82f6 !important;
                    }
                    /* Firefox */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"] {
                        scrollbar-width: thin !important;
                        scrollbar-color: #1e3a52 #0d1117 !important;
                    }

                    /* ── Inactive turn: card button ── */
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button {
                        text-align: left !important;
                        white-space: pre-wrap !important;
                        height: auto !important;
                        min-height: 4.8rem !important;
                        padding: 0.75rem 1.05rem !important;
                        line-height: 1.6 !important;
                        background: #0d1117 !important;
                        border: 1px solid #1e2d40 !important;
                        border-radius: 0.55rem !important;
                        color: #cbd5e1 !important;
                        font-size: 0.875rem !important;
                        width: 100% !important;
                        margin-bottom: 0.65rem !important;
                        display: flex !important;
                        align-items: flex-start !important;
                        justify-content: flex-start !important;
                    }
                    /* Wildcard: semua child element (p, div, span) ikut rata kiri */
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button *,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button p,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button div,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button span {
                        text-align: left !important;
                        width: 100% !important;
                        margin: 0 !important;
                        display: block !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button:hover {
                        background: #0f1e2e !important;
                        border-color: #3b82f6 !important;
                        color: #f1f5f9 !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button:hover * {
                        color: #f1f5f9 !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div.stButton > button:focus:not(:active) {
                        border-color: #60a5fa !important;
                        box-shadow: 0 0 0 2px #1d4ed833 !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                for item in st.session_state["history"][-10:]:
                    _tid       = item["turn_id"]
                    _is_active = (_viewing_id_now == _tid)
                    _q_raw     = item.get("query", "")
                    _a_raw     = item.get("answer_preview", "")
                    _is_latest = (_tid == _last_turn_id_now and _viewing_id_now is None)

                    if _is_active:
                        # ── Turn aktif: vibrant border, dark fill ──
                        _q_esc = _html_mod.escape(_q_raw)
                        _a_esc = _html_mod.escape(_a_raw)
                        _badge_label = "✦ Terbaru · Sedang Dilihat" if _is_latest else "👁️ Sedang Dilihat"
                        st.markdown(
                            f'<div style="'
                            f"border:1px solid #2563eb;"
                            f"border-left:4px solid #3b82f6;"
                            f"border-radius:0.55rem;"
                            f"background:rgba(37,99,235,0.07);"
                            f"padding:0.75rem 1.05rem;"
                            f"margin-bottom:0.65rem;"
                            f'">'
                            # ── Header row: Turn N + badge ──
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;margin-bottom:0.35rem;">'
                            f'<span style="color:#60a5fa;font-weight:700;font-size:0.8rem;'
                            f'letter-spacing:0.04em;text-transform:uppercase;">'
                            f'Chat-Turn {_tid}</span>'
                            f'<span style="background:rgba(59,130,246,0.15);color:#93c5fd;'
                            f'border:1px solid #1d4ed8;padding:1px 10px;border-radius:999px;'
                            f'font-size:0.73rem;font-weight:600;">'
                            f'{_badge_label}</span>'
                            f'</div>'
                            # ── Divider ──
                            f'<div style="border-top:1px solid #1e3a5f;'
                            f'margin-bottom:0.45rem;"></div>'
                            # ── Q row ──
                            f'<div style="color:#e2e8f0;font-size:0.875rem;'
                            f'line-height:1.6;margin-bottom:0.3rem;text-align:left;">'
                            f'<span style="color:#60a5fa;font-weight:700;">Q{_tid}:</span>'
                            f'&nbsp;{_q_esc}'
                            f'</div>'
                            # ── A row ──
                            f'<div style="color:#94a3b8;font-size:0.83rem;'
                            f'line-height:1.55;text-align:left;">'
                            f'<span style="color:#7dd3fc;font-weight:700;">A:</span>'
                            f'&nbsp;{_a_esc}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    else:
                        # ── Turn tidak aktif: button card left-aligned ──
                        _latest_marker = "  ✦" if _is_latest else ""
                        _q_disp = _q_raw[:100] + ("…" if len(_q_raw) > 100 else "")
                        _a_disp = _a_raw[:160] + ("…" if len(_a_raw) > 160 else "")
                        # Newline dipakai agar button Streamlit tampil multi-baris (pre-wrap)
                        _divider_line = "─" * 38
                        _btn_label = (
                            f"Chat-Turn {_tid}{_latest_marker}\n"
                            f"{_divider_line}\n"
                            f"Q{_tid}: {_q_disp}\n"
                            f"A:  {_a_disp}"
                        )
                        if st.button(
                            _btn_label,
                            key=f"view_turn_{_tid}",
                            use_container_width=True,
                        ):
                            st.session_state["viewing_turn_id"] = _tid
                            st.rerun()

    # ===== Render results (JIKA, sudah adanya query tersubmit DAN user belum/hendak mengirimkan query baru) =====
    if has_result:
        # --- Turn Viewer: tentukan sumber data yang akan di-render ---
        _viewing_id = st.session_state.get("viewing_turn_id")
        _turn_data_store = st.session_state.get("turn_data", {})
        _is_viewing_history = (
            _viewing_id is not None and _viewing_id in _turn_data_store
        )

        # CATATAN V3.0c:
        # - nodes                = contexts final pasca-rerank (yang benar-benar dipakai LLM/UI utama)
        # - nodes_before_rerank  = candidate pool pre-rerank (untuk audit & future UI before-vs-after)
        # - retrieval_stats_pre_rerank_render / generation_context_stats_render
        #   sudah disiapkan di sini agar app_ui_render.py nanti bisa menampilkan ringkasan impact reranker tanpa perlu menghitung ulang dari nol.
        if _is_viewing_history:
            _td = _turn_data_store[_viewing_id]
            # final contexts yang benar-benar dipakai LLM
            nodes: List[Any] = _td.get("nodes") or []
            # candidate pool pre-rerank (untuk future UI before-vs-after)
            nodes_before_rerank: List[Any] = _td.get("nodes_before_rerank") or []
            answer_raw: str = _td.get("answer_raw", "")
            turn_id = int(_viewing_id or 0)
            uq = _td.get("user_query", "")
            rq = _td.get("retrieval_query", "")
            last_strategy = _td.get("strategy", "")
            retrieval_mode_render = str(_td.get("retrieval_mode", "dense"))
            # ← V3.0c: stats eksplisit, disiapkan untuk render future enhancement
            retrieval_stats_pre_rerank_render = _td.get("retrieval_stats_pre_rerank")
            generation_context_stats_render = _td.get("generation_context_stats")
        else:
            # final contexts yang benar-benar dipakai LLM
            nodes = st.session_state.get("last_nodes") or []
            # candidate pool pre-rerank (untuk future UI before-vs-after)
            nodes_before_rerank = st.session_state.get("last_nodes_before_rerank") or []
            answer_raw = st.session_state.get("last_answer_raw", "")
            turn_id = st.session_state.get("last_turn_id", 0)
            uq = st.session_state.get("last_user_query", "")
            rq = st.session_state.get("last_retrieval_query", "")
            last_strategy = st.session_state.get("last_strategy", "")
            retrieval_mode_render = str(st.session_state.get("last_retrieval_mode", "dense"))
            # ← V3.0c: stats eksplisit, disiapkan untuk render future enhancement
            retrieval_stats_pre_rerank_render = st.session_state.get("last_retrieval_stats_pre_rerank")
            generation_context_stats_render = st.session_state.get("last_generation_context_stats")

        jawaban, bukti = parse_answer(answer_raw)
        global_not_found = is_global_not_found_answer(answer_raw)

        # Anchor target
        st.markdown('<div id="answer-anchor"></div>', unsafe_allow_html=True)

        # Banner + tombol kembali jika sedang viewing historis
        if _is_viewing_history:
            render_viewing_banner(turn_id)
            if st.button("↩️ Kembali ke Hasil Terbaru", key="result_back_latest"):
                st.session_state["viewing_turn_id"] = None
                st.rerun()

        render_processing_box(uq, rq)

        # ← V3.0c: ringkasan before-vs-after rerank (deskriptif, bukan evaluatif)
        _reranker_applied_render = (
            _td.get("reranker_applied")
            if _is_viewing_history
            else st.session_state.get("last_reranker_applied", False)
        )
        _reranker_candidate_count_render = (
            _td.get("reranker_candidate_count")
            if _is_viewing_history
            else 0
        )
        _reranker_top_n_render = (
            _td.get("reranker_top_n")
            if _is_viewing_history
            else None
        )

        if not _is_viewing_history:
            if bool(st.session_state.get("last_reranker_applied", False)):
                _reranker_candidate_count_render = len(nodes_before_rerank) if nodes_before_rerank else 0
                _reranker_top_n_render = len(nodes) if nodes else None

        render_rerank_summary_box(
            reranker_applied=bool(_reranker_applied_render),
            candidate_count=_reranker_candidate_count_render,
            top_n=_reranker_top_n_render,
            retrieval_stats_pre_rerank=retrieval_stats_pre_rerank_render,
            generation_context_stats=generation_context_stats_render,
        )

        # ← V3.0d Phase 2A + 2C:
        # render verification box di answer layer.
        # - latest turn  : ambil dari session_state["last_verification"]
        # - history replay: ambil dari turn_data[turn_id]["verification"]
        _verification_render = (
            (_td.get("verification") if isinstance(_td, dict) else None)
            if _is_viewing_history
            else st.session_state.get("last_verification")
        )

        render_verification_box(_verification_render)

        # Answer section — styled card
        h1, h2 = st.columns([0.88, 0.12])
        with h1:
            render_answer_card_header()
        with h2:
            copy_to_clipboard_button(answer_raw, label="📑", key=f"copy_{rm.run_id}_{turn_id}")

        with st.container(border=True):
            if jawaban:
                # Format untuk render Markdown yang rapi (terutama multi-metode + numbered list)
                st.markdown(_prepare_jawaban_markdown(jawaban))
            else:
                st.write(answer_raw)

        # Evidence section — styled card
        render_bukti_section(bukti)

        # ← V3.0c UI enrichment: before vs after rerank (debug / audit)
        _anchor_meta_render = (
            _td.get("rerank_anchor_reservation")
            if _is_viewing_history
            else st.session_state.get("last_rerank_anchor_reservation")
        )

        _rk_cfg_render = cfg.get("reranker") or {}
        _show_before_after_panel = bool(_rk_cfg_render.get("show_before_after_panel", True))
        _before_after_max_pre_rows = int(_rk_cfg_render.get("before_after_ui_max_pre_rows", 6))

        if _show_before_after_panel:
            render_rerank_before_after_panel(
                reranker_applied=bool(_reranker_applied_render),
                nodes_before_rerank=nodes_before_rerank,
                nodes_after_rerank=nodes,
                retrieval_mode=retrieval_mode_render,
                anchor_meta=_anchor_meta_render,
                max_pre_rows=_before_after_max_pre_rows,
            )

        # Auto-scroll after submit
        if st.session_state.get("_scroll_answer"):
            scroll_to_anchor("answer-anchor")
            st.session_state["_scroll_answer"] = False

        # CTX mapping + filter
        render_ctx_mapping_header()

        if last_strategy == "metadata_router":
            st.info(
                "Pertanyaan ini dijawab dari **doc_catalog (metadata routing)**, sehingga tidak ada CTX chunk yang ditampilkan."
            )
            highlight_terms = []  # tidak dipakai
        else:
            # ===== jika GLOBAL not_found, jangan tampilkan CTX mapping (default) =====
            if global_not_found:
                st.warning(
                    "⚠️ Jawaban berstatus **Tidak ditemukan pada dokumen**, sehingga **CTX Mapping tidak ditampilkan** "
                    "agar tidak misleading. Aktifkan mode debug jika ingin melihat sources yang sempat ter-retrieve."
                )
                show_debug = st.checkbox(
                    "Show CTX Mapping (Debug)",
                    value=False,
                    key=f"dbg_ctx_{rm.run_id}_{turn_id}",
                )
                if not show_debug:
                    highlight_terms = []
                else:
                    ctx_rows = build_ctx_rows(nodes, retrieval_mode_render)
                    render_ctx_mapping_table(ctx_rows)

                    # Highlight terms (manual override). Jika kosong -> auto dari query terakhir
                    hl_manual = st.text_input(
                        "Highlight Keyword di PDF (opsional, pisahkan dengan koma). Kosongkan = Auto dari Query.",
                        key="hl_terms",
                    )
                    manual_terms = [t.strip() for t in (hl_manual or "").split(",") if t.strip()]

                    last_query = st.session_state.get("last_submitted_query", "")
                    query_terms = extract_terms_from_query(last_query)
                    ctx1_text = nodes[0].text if nodes else ""
                    ctx_terms = extract_terms_from_text(ctx1_text, max_terms=6)
                    combined = []
                    seen = set()
                    for t in query_terms + ctx_terms:
                        tl = t.lower().strip()
                        if not tl or tl in seen:
                            continue
                        seen.add(tl)
                        combined.append(t)

                    highlight_terms = manual_terms if manual_terms else combined[:8]

            else:
                ctx_rows = build_ctx_rows(nodes, retrieval_mode_render)
                render_ctx_mapping_table(ctx_rows)

                # Highlight terms (manual override). Jika kosong -> auto dari query terakhir
                hl_manual = st.text_input(
                    "Highlight Keyword di PDF (opsional, pisahkan dengan koma). Kosongkan = Auto dari Query.",
                    key="hl_terms",
                )
                manual_terms = [t.strip() for t in (hl_manual or "").split(",") if t.strip()]

                last_query = st.session_state.get("last_submitted_query", "")
                query_terms = extract_terms_from_query(last_query)
                ctx1_text = nodes[0].text if nodes else ""
                ctx_terms = extract_terms_from_text(ctx1_text, max_terms=6)
                combined = []
                seen = set()
                for t in query_terms + ctx_terms:
                    tl = t.lower().strip()
                    if not tl or tl in seen:
                        continue
                    seen.add(tl)
                    combined.append(t)

                highlight_terms = manual_terms if manual_terms else combined[:8]

        # Sources + PDF tools + filter (UI)
        render_sources_header()

        if last_strategy == "metadata_router":
            st.info(
                "Tidak ada Top-k Sources karena tidak dilakukan retrieval (jawaban diambil dari doc_catalog)."
            )
        else:
            # ===== jika GLOBAL not_found, jangan tampilkan Sources (default) =====
            if global_not_found:
                st.warning(
                    "⚠️ Jawaban berstatus **Tidak ditemukan pada dokumen**, sehingga **Sources (Top-k) tidak ditampilkan** "
                    "agar tidak misleading. Aktifkan mode debug jika ingin melihat sources yang sempat ter-retrieve."
                )
                show_debug_src = st.checkbox(
                    "Show Sources (Top-k) (Debug)",
                    value=False,
                    key=f"dbg_src_{rm.run_id}_{turn_id}",
                )
                if not show_debug_src:
                    indexed_nodes = []
                else:
                    indexed_nodes = list(enumerate(nodes, start=1))
            else:
                indexed_nodes = list(enumerate(nodes, start=1))

            if filter_q.strip():
                filtered = [(i, n) for i, n in indexed_nodes if node_matches_filter(n, filter_q)]
                if show_only_matches:
                    indexed_nodes = filtered
                st.caption(f"Filter match: {len(filtered)} dari {len(nodes)} sources.")

            for i, n in indexed_nodes:
                source_file = n.metadata.get("source_file", "")
                page = int(n.metadata.get("page", 1) or 1)

                preview = (n.text or "").strip().replace("\n", " ")
                preview = preview[:260] + ("..." if len(preview) > 260 else "")

                # ← V3.0c: siapkan string rerank untuk title expander
                _rk_str = (
                    f" | rk={n.rerank_score:.4f} [#{n.rerank_rank}]"
                    if n.rerank_score is not None else ""
                )

                # ── Hitung skor tampilan (mode-aware) ──
                dist = 0.0  # inisialisasi untuk type-checker
                sim  = 0.0
                if retrieval_mode_render == "hybrid":
                    _rrf_sc    = float(n.score)
                    _d_dist    = float(n.score_dense)  if n.score_dense  is not None else None
                    _d_sim     = dist_to_sim(_d_dist)  if _d_dist         is not None else None
                    _s_bm25    = float(n.score_sparse) if n.score_sparse is not None else None
                    # Badge berbasis RRF Score - lebih intuitif (fair untuk semua jenis node)
                    if _rrf_sc >= 0.025:
                        _badge = "🟢"
                    elif _rrf_sc >= 0.015:
                        _badge = "🟡"
                    else:
                        _badge = "🔴"

                    _d_str     = f"sim = {_d_sim:.3f}" if _d_sim   is not None else "dense = —"
                    _s_str     = f"bm25 = {_s_bm25:.2f}" if _s_bm25 is not None else "sparse = —"
                    # ← V3.0c: tambahkan _rk_str ke title hybrid
                    title = (
                        f"{_badge} {n.doc_id} | rrf = {_rrf_sc:.5f} | {_d_str} | {_s_str}"
                        f"{_rk_str} | {source_file} p.{page}"
                    )
                else:
                    _rrf_sc = None
                    _d_dist = None
                    _d_sim  = None
                    _s_bm25 = None
                    dist    = float(n.score)
                    sim     = dist_to_sim(dist)
                    _badge  = sim_color_badge(sim)
                    # ← V3.0c: tambahkan _rk_str ke title dense
                    title   = (
                        f"{_badge} {n.doc_id} | sim = {sim:.4f} | dist = {dist:.4f}"
                        f"{_rk_str} | {source_file} p.{page}"
                    )
                with st.expander(title):
                    # ── Score pills (mode-aware) - delegasi ke app_ui_render ──
                    if retrieval_mode_render == "hybrid":
                        assert _rrf_sc is not None 
                        render_hybrid_source_score_pills(
                            rrf_score=_rrf_sc,
                            score_dense=_d_dist,
                            score_sparse=_s_bm25,
                            rank_dense=n.rank_dense,
                            rank_sparse=n.rank_sparse,
                            ctx_label=f"CTX {i}",
                            stream=n.stream,
                            rerank_score=n.rerank_score,   # ← V3.0c
                            rerank_rank=n.rerank_rank,     # ← V3.0c
                            is_reserved_anchor=bool(getattr(n, "is_reserved_anchor", False)),
                            reserved_for_method=getattr(n, "reserved_for_method", None),
                        )
                    else:
                        render_source_score_pills(
                            sim,
                            dist,
                            ctx_label=f"CTX {i}",
                            stream=n.stream,
                            rerank_score=n.rerank_score,   # ← V3.0c
                            rerank_rank=n.rerank_rank,     # ← V3.0c
                            is_reserved_anchor=bool(getattr(n, "is_reserved_anchor", False)),
                            reserved_for_method=getattr(n, "reserved_for_method", None),
                        )

                    st.caption(f"Preview: {preview}")

                    # Resolve PDF path
                    pdf_path = resolve_pdf_path(
                        project_root=PROJECT_ROOT,
                        data_raw_dir=data_raw_dir,
                        metadata=n.metadata,
                    )

                    cols = st.columns([0.20, 0.20, 0.60], gap="small")
                    with cols[0]:
                        if pdf_path and st.button(
                            f"👀 View PDF (p.{page})", key=f"viewpdf_{rm.run_id}_{turn_id}_{i}"
                        ):
                            st.session_state["pdf_view"] = {
                                "path": str(pdf_path),
                                "page": int(page),
                                "terms": highlight_terms,
                            }
                            st.rerun()
                        if not pdf_path:
                            st.button("View PDF", disabled=True)

                    with cols[1]:
                        if pdf_path:
                            download_file_button(
                                "📥 Download PDF",
                                pdf_path,
                                "application/pdf",
                                file_name=pdf_path.name,
                                key=f"dlpdf_{rm.run_id}_{turn_id}_{i}",
                            )
                        else:
                            st.button("📥 Download PDF", disabled=True)

                    with cols[2]:
                        if pdf_path:
                            st.caption(f"PDF: {pdf_path.name} | page={page}")
                        else:
                            st.caption("PDF path tidak ditemukan dari metadata.")

                    st.write(n.text)

    # PDF Viewer section
    pdf_view = st.session_state.get("pdf_view")
    if pdf_view and isinstance(pdf_view, dict) and pdf_view.get("path"):
        pv_path = Path(str(pdf_view["path"]))
        pv_page = int(pdf_view.get("page", 1) or 1)
        render_pdf_viewer_header(pv_path.name, pv_page)

        if st.button("Close Viewer"):
            st.session_state["pdf_view"] = None
            st.rerun()

        if pv_path.exists():
            terms = pdf_view.get("terms") if isinstance(pdf_view, dict) else None
            render_pdf_viewer(pv_path, page=pv_page, height=720, highlight_terms=terms)
        else:
            st.warning("File PDF tidak ditemukan di path yang tersimpan.")

    if not has_result:
        # UI awal bersih
        st.caption(
            "Silahkan bertanya lalu klik **Kirim** untuk mendapatkan Jawaban, Bukti, CTX Mapping, dan Sources."
        )


if __name__ == "__main__":
    main()
