from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


def retrieve_dense(cfg: Dict[str, Any], query: str) -> List[RetrievedNode]:
    """
    Dense retrieval dari Chroma: similarity_search_with_score().
    Return list RetrievedNode (untuk logging & UI sources).
    """
    top_k = int(cfg["retrieval"]["top_k"])
    vs = _get_vectorstore(cfg)

    results: List[Tuple[Any, float]] = vs.similarity_search_with_score(query, k=top_k)

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
    return nodes
