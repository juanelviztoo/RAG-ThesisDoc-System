from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.schemas import RetrievedNode

# ===== module-level caches (per process) =====
_EMB_CACHE: Dict[Tuple[str, str, bool], HuggingFaceEmbeddings] = {}
_VS_CACHE: Dict[Tuple[str, str, str, str, bool], Chroma] = {}


def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_embeddings(cfg: Dict[str, Any]) -> HuggingFaceEmbeddings:
    model_name = _cfg_get(
        cfg,
        ["embeddings", "model_name"],
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    device = _cfg_get(cfg, ["embeddings", "device"], "cpu")
    normalize = bool(_cfg_get(cfg, ["embeddings", "normalize"], True))

    key = (str(model_name), str(device), normalize)
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]

    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )
    _EMB_CACHE[key] = emb
    return emb


def _get_vectorstore(cfg: Dict[str, Any]) -> Chroma:
    persist_dir = str(cfg["index"]["persist_dir"])
    collection_name = str(cfg["index"]["collection_name"])

    # embeddings cache key params
    model_name = _cfg_get(
        cfg,
        ["embeddings", "model_name"],
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    device = _cfg_get(cfg, ["embeddings", "device"], "cpu")
    normalize = bool(_cfg_get(cfg, ["embeddings", "normalize"], True))

    vs_key = (persist_dir, collection_name, str(model_name), str(device), normalize)
    if vs_key in _VS_CACHE:
        return _VS_CACHE[vs_key]

    embeddings = _build_embeddings(cfg)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    _VS_CACHE[vs_key] = vs
    return vs


def clear_retrieval_cache() -> None:
    """Opsional: panggil kalau ingin reset cache saat dev/debug."""
    _EMB_CACHE.clear()
    _VS_CACHE.clear()


def _doc_key(n: RetrievedNode) -> str:
    """Kunci dokumen untuk diversifikasi (doc_id -> fallback source_file)."""
    d = (n.doc_id or "").strip()
    if d:
        return d
    sf = (n.metadata or {}).get("source_file")
    return str(sf or "unknown").strip() or "unknown"


def _diversify_by_doc(
    nodes: List[RetrievedNode],
    *,
    top_k: int,
    max_per_doc: int,
) -> List[RetrievedNode]:
    """
    Pilih top_k nodes yang lebih beragam antar dokumen.

    Strategi:
    - Pass 1: ambil 1 per doc (sebar dulu)
    - Pass 2: tambah hingga max_per_doc per doc
    - Pass 3: kalau masih kurang, isi sisa (relax cap) agar tetap dapat top_k
    """
    if not nodes:
        return []

    top_k = max(1, int(top_k))
    max_per_doc = max(1, int(max_per_doc))

    picked: List[RetrievedNode] = []
    picked_ids: set[str] = set()
    per_doc: Counter[str] = Counter()

    # Pass 1: 1 per doc
    for n in nodes:
        if len(picked) >= top_k:
            break
        cid = str(n.chunk_id)
        dk = _doc_key(n)
        if dk in per_doc:
            continue
        if cid in picked_ids:
            continue
        picked.append(n)
        picked_ids.add(cid)
        per_doc[dk] += 1

    # Pass 2: hingga max_per_doc per doc
    for n in nodes:
        if len(picked) >= top_k:
            break
        cid = str(n.chunk_id)
        dk = _doc_key(n)
        if cid in picked_ids:
            continue
        if per_doc[dk] >= max_per_doc:
            continue
        picked.append(n)
        picked_ids.add(cid)
        per_doc[dk] += 1

    # Pass 3: relax cap bila belum cukup
    if len(picked) < top_k:
        for n in nodes:
            if len(picked) >= top_k:
                break
            cid = str(n.chunk_id)
            if cid in picked_ids:
                continue
            picked.append(n)
            picked_ids.add(cid)

    return picked[:top_k]


def retrieve_dense(
    cfg: Dict[str, Any],
    query: str,
    *,
    diversify: bool = False,
    max_per_doc: int = 2,
    candidate_k: Optional[int] = None,
) -> List[RetrievedNode]:
    """
    Dense retrieval dari Chroma: similarity_search_with_score().
    Return list RetrievedNode (untuk logging & UI sources).

    diversify=True:
        - ambil kandidat lebih banyak (candidate_k)
        - pilih top_k yang lebih beragam antar dokumen (max_per_doc per dokumen)
    """
    top_k = int(cfg["retrieval"]["top_k"])
    vs = _get_vectorstore(cfg)

    # Unifikasi cap ke satu konstanta
    _MAX_CANDIDATE_K = 400 

    if diversify:
        if candidate_k is None:
            # default: pool lebih besar supaya punya “bahan” untuk sebar antar dokumen
            candidate_k = min(max(top_k * 6, top_k), _MAX_CANDIDATE_K)
        else:
            candidate_k = max(int(candidate_k), top_k)
            candidate_k = min(candidate_k, _MAX_CANDIDATE_K)

        results: List[Tuple[Any, float]] = vs.similarity_search_with_score(query, k=candidate_k)
    else:
        results = vs.similarity_search_with_score(query, k=top_k)

    nodes: List[RetrievedNode] = []
    for doc, score in results:
        md = dict(doc.metadata or {})
        nodes.append(
            RetrievedNode(
                doc_id=str(md.get("doc_id", md.get("source_file", "unknown"))),
                chunk_id=str(md.get("chunk_id", "unknown")),
                score=float(score),
                text=str(doc.page_content),
                metadata=md,
            )
        )

    if not diversify:
        return nodes[:top_k]

    return _diversify_by_doc(nodes, top_k=top_k, max_per_doc=max_per_doc)
