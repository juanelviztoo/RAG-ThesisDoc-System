from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.core.schemas import RetrievedNode
from src.rag.retrieve_dense import _get_vectorstore  # reuse cache yang sudah ada

# ─────────────────────────────────────────
# Sastrawi — optional dependency
# ─────────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory as _StemmerFactory

    _STEMMER = _StemmerFactory().create_stemmer()
    _SASTRAWI_AVAILABLE = True
except ImportError:
    _STEMMER = None
    _SASTRAWI_AVAILABLE = False


# ─────────────────────────────────────────
# Module-level BM25 cache
# key: (persist_dir, collection_name, use_sastrawi_str)
# value: (BM25Okapi, List[metadata_dict], List[text_str])
# ─────────────────────────────────────────
_BM25_CACHE: Dict[
    Tuple[str, str, str],
    Tuple[BM25Okapi, List[Dict[str, Any]], List[str]],
] = {}


# ─────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────
def _tokenize(text: str, *, use_sastrawi: bool) -> List[str]:
    """
    Lowercase → strip non-word chars → split → optional Sastrawi stem.
    PENTING: stemming HANYA dipakai di jalur sparse ini, BUKAN di dense.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]

    if use_sastrawi and _SASTRAWI_AVAILABLE and _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


# ─────────────────────────────────────────
# Build & cache BM25 index
# ─────────────────────────────────────────
def _build_bm25_index(
    cfg: Dict[str, Any],
    *,
    use_sastrawi: bool,
) -> Tuple[BM25Okapi, List[Dict[str, Any]], List[str]]:
    """
    Load semua chunk dari ChromaDB lalu bangun BM25Okapi index.
    Corpus selalu sinkron dengan vector index - tidak ada state terpisah.

    Return: (bm25_index, enriched_metadatas, texts)
    """
    vs = _get_vectorstore(cfg)
    collection = vs._collection  # akses underlying chromadb.Collection

    # "ids" TIDAK dimasukkan ke include - selalu dikembalikan otomatis oleh ChromaDB
    result = collection.get(include=["documents", "metadatas"])

    texts: List[str] = result.get("documents") or []
    raw_metadatas: List[Any] = result.get("metadatas") or []   # List[Metadata] → List[Any]
    chroma_ids: List[str] = result.get("ids") or []            # selalu ada, tidak perlu di-include

    if not texts:
        # Corpus kosong — kembalikan index dummy agar tidak crash
        bm25 = BM25Okapi([[""]])
        return bm25, [], []

    # Tokenize corpus (dengan/tanpa stemming)
    tokenized_corpus = [_tokenize(t, use_sastrawi=use_sastrawi) for t in texts]

    k1 = float((cfg.get("bm25") or {}).get("k1", 1.5))
    b  = float((cfg.get("bm25") or {}).get("b", 0.75))
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Enrich metadata dengan chroma_id sebagai fallback chunk_id identifier
    enriched: List[Dict[str, Any]] = []
    for i, md in enumerate(raw_metadatas):
        m: Dict[str, Any] = dict(md)          # cast Metadata → Dict[str, Any]
        if i < len(chroma_ids) and "chunk_id" not in m:
            m["chunk_id"] = chroma_ids[i]
        enriched.append(m)

    return bm25, enriched, texts


def _get_bm25_index(
    cfg: Dict[str, Any],
    *,
    use_sastrawi: bool,
) -> Tuple[BM25Okapi, List[Dict[str, Any]], List[str]]:
    """Kembalikan BM25 index dari cache, bangun jika belum ada."""
    persist_dir = str(cfg["index"]["persist_dir"])
    collection_name = str(cfg["index"]["collection_name"])
    # use_sastrawi masuk ke cache key karena tokenisasi corpus berbeda
    cache_key = (persist_dir, collection_name, str(use_sastrawi))

    if cache_key not in _BM25_CACHE:
        _BM25_CACHE[cache_key] = _build_bm25_index(cfg, use_sastrawi=use_sastrawi)

    return _BM25_CACHE[cache_key]


def clear_sparse_cache() -> None:
    """
    Reset BM25 cache.
    Wajib dipanggil setelah re-ingest agar index tidak stale.
    """
    _BM25_CACHE.clear()


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────
def sparse_retrieve_bm25(
    cfg: Dict[str, Any],
    query: str,
    *,
    top_k: Optional[int] = None,
) -> List[RetrievedNode]:
    """
    BM25 sparse retrieval + optional Sastrawi stemming.

    Aturan wajib:
        - Sastrawi stemming HANYA di jalur ini.
        - Dense retrieval TIDAK menggunakan Sastrawi, tanpa pengecualian.

    Return: List[RetrievedNode] diurutkan descending by BM25 score,
            dengan score_sparse terisi (score = score_sparse = raw BM25 score).
    """
    use_sastrawi = bool((cfg.get("bm25") or {}).get("use_sastrawi", False))

    if use_sastrawi and not _SASTRAWI_AVAILABLE:
        warnings.warn(
            "use_sastrawi=True tapi library Sastrawi tidak terinstall. "
            "Fallback ke tokenisasi biasa (tanpa stemming). "
            "Install dengan: pip install PySastrawi",
            RuntimeWarning,
            stacklevel=2,
        )

    if top_k is None:
        top_k = int(cfg["retrieval"]["top_k"])

    bm25, metadatas, texts = _get_bm25_index(cfg, use_sastrawi=use_sastrawi)

    # Corpus kosong — early return
    if not texts:
        return []

    query_tokens = _tokenize(query, use_sastrawi=use_sastrawi)
    raw_scores = bm25.get_scores(query_tokens)  # ndarray, len == corpus size

    # Top-k indices by BM25 score descending
    top_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:top_k]

    nodes: List[RetrievedNode] = []
    for idx in top_indices:
        raw_score = float(raw_scores[idx])
        md = metadatas[idx]
        text = texts[idx]

        chunk_id = str(md.get("chunk_id", f"sparse_{idx}"))
        doc_id = str(md.get("doc_id", md.get("source_file", "unknown")))

        nodes.append(
            RetrievedNode(
                chunk_id=chunk_id,
                doc_id=doc_id,
                score=raw_score,          # score utama = BM25 raw
                score_sparse=raw_score,   # expose untuk RRF & UI Sources
                text=text,
                metadata=md,
            )
        )

    return nodes
