from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.schemas import RetrievedNode


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

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def retrieve_dense(cfg: Dict[str, Any], query: str) -> List[RetrievedNode]:
    """
    Dense retrieval dari Chroma: similarity_search_with_score().
    Return list RetrievedNode (untuk logging & UI sources).
    """
    persist_dir = cfg["index"]["persist_dir"]
    collection_name = cfg["index"]["collection_name"]
    top_k = int(cfg["retrieval"]["top_k"])

    embeddings = _build_embeddings(cfg)

    vs = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

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
