from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb

# ============================================================================
# IO helpers
# ============================================================================

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================================
# Normalization helpers
# ============================================================================

_NOT_FOUND_GLOBAL_RE = re.compile(
    r"(?is)^\s*Jawaban\s*:\s*Tidak\s+ditemukan\s+pada\s+dokumen\.\s*Bukti\s*:\s*-\s*$"
)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _row_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (
        str(row.get("run_id", "") or "").strip(),
        _safe_int(row.get("turn_id", 0), 0),
    )


def _normalize_chunk_ids(values: Any) -> List[str]:
    """
    Normalisasi list chunk_id:
    - terima list / tuple / None
    - strip string kosong
    - dedupe, preserve order
    """
    if not isinstance(values, (list, tuple)):
        return []

    out: List[str] = []
    seen = set()

    for x in values:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out


def _load_manifest_config(run_path: Path) -> Dict[str, Any]:
    """
    Ambil snapshot config dari manifest.json milik run ini.
    RunManager memang menyimpan config snapshot sekali per-run di manifest.json,
    sehingga evaluator harus memakai config run asli, bukan config aktif saat ini.
    """
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        return {}

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cfg = data.get("config") or {}
    return cfg if isinstance(cfg, dict) else {}


def _resolve_persist_dir_from_cfg(cfg: Dict[str, Any]) -> str:
    index_cfg = cfg.get("index") or {}
    persist_dir = str(index_cfg.get("persist_dir", "") or "").strip()
    return persist_dir


def _resolve_collection_names_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve collection names yang mungkin dipakai evaluator:
    - V3.x dual collection  -> narasi + sitasi
    - V1/V2 single collection -> collection_name tunggal
    """
    index_cfg = cfg.get("index") or {}

    collections = index_cfg.get("collections")
    if isinstance(collections, dict):
        out: List[str] = []
        for key in ("narasi", "sitasi"):
            name = str(collections.get(key, "") or "").strip()
            if name and name not in out:
                out.append(name)
        return out

    single = str(index_cfg.get("collection_name", "") or "").strip()
    return [single] if single else []


def _same_question_surface(a: str, b: str) -> bool:
    """
    Bandingkan dua pertanyaan secara ringan:
    - trim
    - rapikan whitespace
    - case-sensitive sengaja dipertahankan agar konservatif
    """
    aa = " ".join((a or "").split()).strip()
    bb = " ".join((b or "").split()).strip()
    return bool(aa) and bool(bb) and aa == bb


def _pick_question_bundle(answer_row: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Return:
        - question_user   : pertanyaan asli user
        - question_eval   : pertanyaan untuk evaluator (prefer retrieval_query)
        - question_source : penjelasan asal question_eval

    Rule 4A.2:
    - Jika retrieval_query identik dengan user_query, label sumber = "user_query"
        agar audit tidak misleading.
    """
    q_user = str(answer_row.get("user_query", "") or "").strip()

    # backward-compat untuk schema lama
    q_legacy = str(answer_row.get("question", "") or "").strip()
    if not q_user and q_legacy:
        q_user = q_legacy

    q_retrieval = str(answer_row.get("retrieval_query", "") or "").strip()

    if q_retrieval:
        if q_user and _same_question_surface(q_retrieval, q_user):
            return q_user, q_retrieval, "user_query"
        if (not q_user) and q_legacy and _same_question_surface(q_retrieval, q_legacy):
            return q_legacy, q_retrieval, "question"
        return q_user or q_retrieval, q_retrieval, "retrieval_query"

    if q_user:
        return q_user, q_user, "user_query"
    if q_legacy:
        return q_legacy, q_legacy, "question"
    return "", "", "empty"


def _build_retrieval_text_map(retrieval_row: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map chunk_id -> text preview dari retrieval.jsonl.
    fallback jika full chunk dari index tidak berhasil diambil.
    """
    if not isinstance(retrieval_row, dict):
        return {}

    nodes = retrieval_row.get("retrieved_nodes", [])
    if not isinstance(nodes, list):
        return {}

    out: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        cid = str(node.get("chunk_id", "") or "").strip()
        txt = str(node.get("text", "") or "").strip()
        if cid and txt and cid not in out:
            out[cid] = txt
    return out


def _fetch_contexts_from_index(
    *,
    chunk_ids: Sequence[str],
    run_cfg: Dict[str, Any],
) -> Dict[str, str]:
    """
    Ambil full chunk text langsung dari ChromaDB berdasarkan used_context_chunk_ids.

    Prinsip:
    - index/chroma = sumber utama evaluator
    - retrieval.jsonl = fallback preview-only
    """
    out: Dict[str, str] = {}
    if not chunk_ids:
        return out

    persist_dir = _resolve_persist_dir_from_cfg(run_cfg)
    collection_names = _resolve_collection_names_from_cfg(run_cfg)

    if not persist_dir or not collection_names:
        return out

    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
    except Exception:
        return out

    for col_name in collection_names:
        try:
            col = client.get_collection(name=col_name)
        except Exception:
            # collection memang bisa tidak ada (mis. sitasi pada config single-collection)
            continue

        try:
            result = col.get(
                ids=list(chunk_ids),
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        ids = result.get("ids") or []
        docs = result.get("documents") or []

        for i, cid in enumerate(ids):
            cid_str = str(cid or "").strip()
            if not cid_str or cid_str in out:
                continue
            if i >= len(docs):
                continue

            txt = str(docs[i] or "").strip()
            if txt:
                out[cid_str] = txt

    return out


def _metadata_route_handled(row: Optional[Dict[str, Any]]) -> bool:
    row = row if isinstance(row, dict) else {}
    md = row.get("metadata_route") or {}
    if not isinstance(md, dict):
        return False
    return bool(md.get("handled", False))


def _is_global_not_found_answer(answer_text: str) -> bool:
    t = (answer_text or "").strip()
    if not t:
        return False

    # Normalisasi ringan agar bentuk "Bukti:\n-" tetap dikenali
    t = re.sub(r"(?im)^\s*Bukti\s*:\s*\n\s*-\s*$", "Bukti: -", t).strip()
    return bool(_NOT_FOUND_GLOBAL_RE.match(t))


# ============================================================================
# Retrieval join helpers
# ============================================================================

def _build_retrieval_index(retrieval_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Index retrieval row by (run_id, turn_id).

    Jika ada duplikasi key, row terakhir menang.
    Hal ini aman karena satu turn idealnya memang hanya punya satu retrieval record.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in retrieval_rows:
        if not isinstance(row, dict):
            continue
        out[_row_key(row)] = row
    return out


def _legacy_contexts_from_answer(answer_row: Dict[str, Any]) -> List[str]:
    """
    Backward-compatible:
    beberapa schema lama menyimpan contexts langsung di answers.jsonl.
    """
    raw = answer_row.get("contexts", [])
    if not isinstance(raw, list):
        return []

    out: List[str] = []
    for x in raw:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def _reconstruct_final_contexts(
    answer_row: Dict[str, Any],
    retrieval_row: Optional[Dict[str, Any]],
    run_cfg: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Rekonstruksi final contexts yang benar-benar dipakai generator.

    Prioritas sumber:
    1) full chunk dari Chroma/index
    2) preview text dari retrieval.jsonl
    3) legacy contexts dari answers.jsonl
    """
    used_context_chunk_ids = _normalize_chunk_ids(
        answer_row.get("used_context_chunk_ids")
        or answer_row.get("source_chunk_ids")
        or []
    )

    fallback_contexts = _legacy_contexts_from_answer(answer_row)

    # Tidak ada final chunk ids → fallback schema lama
    if not used_context_chunk_ids:
        if fallback_contexts:
            return fallback_contexts, [], [], "answers_fallback"
        return [], [], [], "none"

    # Sumber utama: full chunk dari index
    full_index_map = _fetch_contexts_from_index(
        chunk_ids=used_context_chunk_ids,
        run_cfg=run_cfg,
    )

    # Sumber fallback: retrieval log preview
    retrieval_map = _build_retrieval_text_map(retrieval_row)

    contexts: List[str] = []
    missing_chunk_ids: List[str] = []

    source_flags: List[str] = []

    for cid in used_context_chunk_ids:
        txt = full_index_map.get(cid)
        if txt:
            contexts.append(txt)
            if "index_full" not in source_flags:
                source_flags.append("index_full")
            continue

        txt = retrieval_map.get(cid)
        if txt:
            contexts.append(txt)
            if "retrieval_join" not in source_flags:
                source_flags.append("retrieval_join")
            continue

        missing_chunk_ids.append(cid)

    if contexts:
        if source_flags == ["index_full"]:
            context_source = "index_full"
        elif source_flags == ["retrieval_join"]:
            context_source = "retrieval_join"
        elif "index_full" in source_flags and "retrieval_join" in source_flags:
            context_source = "index_plus_retrieval_fallback"
        else:
            context_source = "+".join(source_flags) if source_flags else "unknown"

        return contexts, used_context_chunk_ids, missing_chunk_ids, context_source

    # Fallback terakhir: schema lama
    if fallback_contexts:
        return fallback_contexts, used_context_chunk_ids, missing_chunk_ids, "answers_fallback"

    return [], used_context_chunk_ids, missing_chunk_ids, "none"


# ============================================================================
# Row builders
# ============================================================================

def _build_audit_row(
    *,
    answer_row: Dict[str, Any],
    retrieval_row: Optional[Dict[str, Any]],
    question_user: str,
    question_eval: str,
    question_source: str,
    answer_text: str,
    contexts: List[str],
    used_context_chunk_ids: List[str],
    missing_chunk_ids: List[str],
    context_source: str,
    include_in_eval: bool,
    skip_reason: Optional[str],
) -> Dict[str, Any]:
    verification = answer_row.get("verification") or {}
    if not isinstance(verification, dict):
        verification = {}

    generator_meta = answer_row.get("generator_meta") or {}
    if not isinstance(generator_meta, dict):
        generator_meta = {}

    metadata_route = answer_row.get("metadata_route") or {}
    if not isinstance(metadata_route, dict):
        metadata_route = {}

    retrieval_mode = None
    if isinstance(retrieval_row, dict):
        retrieval_mode = retrieval_row.get("retrieval_mode")

    contexts_have_trimmed_preview = any("...<trimmed>" in c for c in contexts)

    return {
        "run_id": answer_row.get("run_id"),
        "turn_id": answer_row.get("turn_id"),
        "question": question_eval,
        "question_user": question_user,
        "question_eval": question_eval,
        "question_source": question_source,
        "answer": answer_text,
        "contexts": contexts,
        "context_count": len(contexts),
        "used_context_chunk_ids": used_context_chunk_ids,
        "missing_context_chunk_ids": missing_chunk_ids,
        "context_source": context_source,
        "contexts_have_trimmed_preview": contexts_have_trimmed_preview,
        "include_in_eval": bool(include_in_eval),
        "skip_reason": skip_reason,
        "strategy": answer_row.get("strategy"),
        "retrieval_mode": retrieval_mode,
        "generator_status": answer_row.get("generator_status") or generator_meta.get("status"),
        "verification_label": verification.get("label"),
        "verification_score": verification.get("score"),
        "metadata_route_handled": bool(metadata_route.get("handled", False)),
        "retrieval_record_found": isinstance(retrieval_row, dict),
    }


def _build_eval_row(
    *,
    answer_row: Dict[str, Any],
    retrieval_row: Optional[Dict[str, Any]],
    question_user: str,
    question_eval: str,
    question_source: str,
    answer_text: str,
    contexts: List[str],
    used_context_chunk_ids: List[str],
) -> Dict[str, Any]:
    """
    Row utama untuk evaluator RAGAS + field tambahan audit.
    """
    verification = answer_row.get("verification") or {}
    if not isinstance(verification, dict):
        verification = {}

    generator_meta = answer_row.get("generator_meta") or {}
    if not isinstance(generator_meta, dict):
        generator_meta = {}

    retrieval_mode = None
    if isinstance(retrieval_row, dict):
        retrieval_mode = retrieval_row.get("retrieval_mode")

    return {
        # field utama evaluator
        # "question" sengaja memakai question_eval agar evaluator melihat
        # bentuk pertanyaan yang paling fair / paling standalone.
        "question": question_eval,
        "answer": answer_text,
        "contexts": contexts,

        # field audit tambahan
        "question_user": question_user,
        "question_eval": question_eval,
        "question_source": question_source,
        "run_id": answer_row.get("run_id"),
        "turn_id": answer_row.get("turn_id"),
        "strategy": answer_row.get("strategy"),
        "retrieval_mode": retrieval_mode,
        "generator_status": answer_row.get("generator_status") or generator_meta.get("status"),
        "verification_label": verification.get("label"),
        "verification_score": verification.get("score"),
        "used_context_chunk_ids": used_context_chunk_ids,
        "context_count": len(contexts),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument(
        "--include_metadata_routes",
        action="store_true",
        help="Sertakan row metadata routing ke eval_dataset.json (default: skip).",
    )
    ap.add_argument(
        "--include_not_found",
        action="store_true",
        help="Sertakan row canonical global not_found ke eval_dataset.json (default: skip).",
    )
    args = ap.parse_args()

    run_path = Path(args.runs_dir) / args.run_id
    answers_path = run_path / "answers.jsonl"
    retrieval_path = run_path / "retrieval.jsonl"
    run_cfg = _load_manifest_config(run_path)

    if not answers_path.exists():
        raise FileNotFoundError(f"answers.jsonl tidak ditemukan: {answers_path}")

    answer_rows = list(read_jsonl(answers_path))
    retrieval_rows = list(read_jsonl(retrieval_path)) if retrieval_path.exists() else []
    retrieval_index = _build_retrieval_index(retrieval_rows)

    eval_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for answer_row in answer_rows:
        if not isinstance(answer_row, dict):
            continue

        key = _row_key(answer_row)
        retrieval_row = retrieval_index.get(key)

        question_user, question_eval, question_source = _pick_question_bundle(answer_row)
        answer_text = str(answer_row.get("answer", "") or "").strip()

        contexts, used_context_chunk_ids, missing_chunk_ids, context_source = _reconstruct_final_contexts(
            answer_row,
            retrieval_row,
            run_cfg,
        )

        metadata_route_handled = (
            _metadata_route_handled(answer_row)
            or _metadata_route_handled(retrieval_row)
        )
        global_not_found = _is_global_not_found_answer(answer_text)

        include_in_eval = True
        skip_reason: Optional[str] = None

        if metadata_route_handled and not args.include_metadata_routes:
            include_in_eval = False
            skip_reason = "metadata_route_skipped"
        elif global_not_found and not args.include_not_found:
            include_in_eval = False
            skip_reason = "global_not_found_skipped"
        elif not question_eval:
            include_in_eval = False
            skip_reason = "empty_question"
        elif not answer_text:
            include_in_eval = False
            skip_reason = "empty_answer"
        elif not contexts:
            include_in_eval = False
            skip_reason = "no_final_contexts"

        audit_row = _build_audit_row(
            answer_row=answer_row,
            retrieval_row=retrieval_row,
            question_user=question_user,
            question_eval=question_eval,
            question_source=question_source,
            answer_text=answer_text,
            contexts=contexts,
            used_context_chunk_ids=used_context_chunk_ids,
            missing_chunk_ids=missing_chunk_ids,
            context_source=context_source,
            include_in_eval=include_in_eval,
            skip_reason=skip_reason,
        )
        audit_rows.append(audit_row)

        if include_in_eval:
            eval_rows.append(
                _build_eval_row(
                    answer_row=answer_row,
                    retrieval_row=retrieval_row,
                    question_user=question_user,
                    question_eval=question_eval,
                    question_source=question_source,
                    answer_text=answer_text,
                    contexts=contexts,
                    used_context_chunk_ids=used_context_chunk_ids,
                )
            )

    eval_path = run_path / "eval_dataset.json"
    audit_path = run_path / "ragas_rows.jsonl"
    audit_pretty_path = run_path / "ragas_rows.pretty.json"

    eval_path.write_text(
        json.dumps(eval_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(audit_path, audit_rows)
    audit_pretty_path.write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved eval dataset : {eval_path}")
    print(f"Saved audit rows   : {audit_path}")
    print(f"Saved audit pretty : {audit_pretty_path}")
    print(f"Total answers      : {len(answer_rows)}")
    print(f"Eval rows included : {len(eval_rows)}")
    print(f"Rows skipped       : {len(audit_rows) - len(eval_rows)}")


if __name__ == "__main__":
    main()