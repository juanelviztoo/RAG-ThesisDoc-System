"""
src/rag/retrieve_sparse.py
==========================
V3.0a — BM25 Sparse Retrieval dengan Dual Collection Support

Perubahan dari V2.0:
    - Import _resolve_collection_name dari retrieve_dense (bukan _get_vectorstore langsung)
        → agar cache key BM25 konsisten dengan collection yang benar (narasi/sitasi)
    - Tambah parameter `collection` di sparse_retrieve_bm25() dan _get_bm25_index()
        → default "narasi" (backward-compatible, V1/V2 tidak perlu ubah caller)
    - RetrievedNode sekarang membawa field stream dan bab_label (dari metadata chunk)
    - Cache key kini memakai nama collection yang sudah di-resolve (bukan field config mentah)

ATURAN WAJIB (tidak boleh dilanggar di versi manapun):
    - Sastrawi stemming HANYA di _tokenize() dalam modul ini.
    - retrieve_dense.py dan modul lain TIDAK boleh memanggil Sastrawi.
    - Constraint ini berlaku tanpa pengecualian.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.core.schemas import RetrievedNode
from src.rag.retrieve_dense import (
    _get_vectorstore,  # reuse Chroma + embedding cache yang sudah ada
    _resolve_collection_name,  # V3.0a: resolve nama collection dari config
)

# ═════════════════════════════════════════════════════════════════════════════
# Sastrawi — optional dependency
# ═════════════════════════════════════════════════════════════════════════════
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory as _StemmerFactory
    _STEMMER = _StemmerFactory().create_stemmer()
    _SASTRAWI_AVAILABLE = True
except ImportError:
    _STEMMER = None
    _SASTRAWI_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# Module-level BM25 cache
#
# key  : (persist_dir, resolved_collection_name, use_sastrawi_str)
# value: (BM25Okapi, List[metadata_dict], List[text_str])
#
# Cache key memakai resolved_collection_name (bukan field config mentah)
# agar cache per-collection terpisah dengan benar di V3.0a.
# ═════════════════════════════════════════════════════════════════════════════
_BM25_CACHE: Dict[
    Tuple[str, str, str],
    Tuple[BM25Okapi, List[Dict[str, Any]], List[str]],
] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str, *, use_sastrawi: bool) -> List[str]:
    """
    Lowercase → strip non-word chars → split → optional Sastrawi stem.
    ATURAN KERAS:
        Sastrawi stemming HANYA dipanggil dari fungsi ini.
        Tidak ada modul lain yang boleh mengimpor atau memanggil Sastrawi secara langsung.
        Jalur dense retrieval TIDAK di-stemming - menjaga konteks semantik embedding.
    """
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]

    if use_sastrawi and _SASTRAWI_AVAILABLE and _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


# ═════════════════════════════════════════════════════════════════════════════
# Build & cache BM25 index
# ═════════════════════════════════════════════════════════════════════════════

def _build_bm25_index(
    cfg: Dict[str, Any],
    *,
    use_sastrawi: bool,
    collection: str = "narasi",
) -> Tuple[BM25Okapi, List[Dict[str, Any]], List[str]]:
    """
    Load semua chunk dari ChromaDB collection yang ditentukan,
    lalu bangun BM25Okapi index dari corpus tersebut.

    Corpus selalu sinkron dengan vector index — tidak ada state terpisah.
    BM25 corpus narasi ≠ BM25 corpus sitasi; keduanya di-cache terpisah.

    Args:
        cfg          : config dict
        use_sastrawi : aktifkan Sastrawi stemming untuk tokenisasi corpus
        collection   : "narasi" | "sitasi" — menentukan ChromaDB collection sumber

    Returns:
        (BM25Okapi, enriched_metadatas, texts)
    """
    vs         = _get_vectorstore(cfg, collection)
    chroma_col = vs._collection   # underlying chromadb.Collection

    # "ids" TIDAK dimasukkan ke include - selalu dikembalikan otomatis oleh ChromaDB
    result = chroma_col.get(include=["documents", "metadatas"])

    texts:         List[str] = result.get("documents") or []
    raw_metadatas: List[Any] = result.get("metadatas")  or []  # List[Metadata] → List[Any]
    chroma_ids:    List[str] = result.get("ids")         or [] # selalu ada, tidak perlu di-include

    if not texts:
        # Corpus kosong (collection belum di-ingest) → kembalikan index dummy agar tidak crash
        bm25 = BM25Okapi([[""]])
        return bm25, [], []

    # ── Tokenize corpus (dengan/tanpa stemming) ───────────────────────────────
    tokenized_corpus = [_tokenize(t, use_sastrawi=use_sastrawi) for t in texts]

    # BM25 hyperparameters (dapat di-override dari config)
    k1  = float((cfg.get("bm25") or {}).get("k1", 1.5))
    b   = float((cfg.get("bm25") or {}).get("b",  0.75))
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # ── Enrich metadata ───────────────────────────────────────────────────────
    # Gunakan chroma_id sebagai fallback chunk_id jika metadata tidak punya field tersebut.
    # Hal ini menjaga konsistensi chunk_id untuk audit log.
    enriched: List[Dict[str, Any]] = []
    for i, md in enumerate(raw_metadatas):
        m: Dict[str, Any] = dict(md)   # cast Metadata → Dict[str, Any]
        if i < len(chroma_ids) and "chunk_id" not in m:
            m["chunk_id"] = chroma_ids[i]
        enriched.append(m)

    return bm25, enriched, texts


def _get_bm25_index(
    cfg: Dict[str, Any],
    *,
    use_sastrawi: bool,
    collection: str = "narasi",
) -> Tuple[BM25Okapi, List[Dict[str, Any]], List[str]]:
    """
    Kembalikan BM25 index dari cache; bangun jika belum ada.

    Cache key: (persist_dir, resolved_collection_name, use_sastrawi_str)
        - resolved_collection_name: nama collection ChromaDB yang sebenarnya
        (bukan string "narasi"/"sitasi") - ini memastikan cache key unik
        bahkan jika dua config berbeda memetakan ke collection yang sama.
        - use_sastrawi masuk ke key karena tokenisasi corpus berbeda
        (dengan/tanpa stemming menghasilkan index yang tidak bisa dipakai silang).
    """
    persist_dir             = str(cfg["index"]["persist_dir"])
    resolved_collection     = _resolve_collection_name(cfg, collection)
    cache_key               = (persist_dir, resolved_collection, str(use_sastrawi))

    if cache_key not in _BM25_CACHE:
        _BM25_CACHE[cache_key] = _build_bm25_index(
            cfg, use_sastrawi=use_sastrawi, collection=collection
        )

    return _BM25_CACHE[cache_key]


def clear_sparse_cache() -> None:
    """
    Reset seluruh BM25 cache.

    WAJIB dipanggil setelah re-ingest agar BM25 index tidak stale.
    ingest.py V3.0a memanggil ini secara otomatis di akhir proses ingest.
    Bisa juga dipanggil manual saat dev/debug.
    """
    _BM25_CACHE.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def sparse_retrieve_bm25(
    cfg: Dict[str, Any],
    query: str,
    *,
    top_k:      Optional[int] = None,
    collection: str           = "narasi",
) -> List[RetrievedNode]:
    """
    BM25 sparse retrieval + optional Sastrawi stemming.

    V3.0a: Parameter `collection` mengontrol dari corpus mana BM25 dijalankan.
        - "narasi" : corpus BAB I–V (default) → untuk query umum & hybrid
        - "sitasi" : corpus Daftar Pustaka     → untuk citation discovery
    V1/V2 backward-compat: `collection` diabaikan secara efektif karena
        `_resolve_collection_name()` akan mengembalikan `collection_name` tunggal.

    ATURAN WAJIB:
        - Sastrawi stemming HANYA di _tokenize() dalam modul ini.
        - Dense retrieval TIDAK menggunakan Sastrawi, tanpa pengecualian.

    Args:
        cfg        : config dict dari load_config()
        query      : query string (sebelum tokenisasi)
        top_k      : jumlah hasil (default: cfg["retrieval"]["top_k"])
        collection : "narasi" | "sitasi" - corpus BM25 yang dipakai

    Returns:
        List[RetrievedNode] diurutkan descending by BM25 score.
        Field score dan score_sparse berisi raw BM25 score.
        Field stream dan bab_label diisi dari metadata chunk (V3.0a);
        None untuk chunk dari V1/V2 yang tidak punya field tersebut.
    """
    use_sastrawi = bool((cfg.get("bm25") or {}).get("use_sastrawi", False))

    # Peringatan jika Sastrawi diminta tapi tidak terinstall
    if use_sastrawi and not _SASTRAWI_AVAILABLE:
        warnings.warn(
            "use_sastrawi=True tapi library PySastrawi tidak terinstall. "
            "Fallback ke tokenisasi biasa (tanpa stemming). "
            "Install dengan: pip install PySastrawi",
            RuntimeWarning,
            stacklevel=2,
        )

    if top_k is None:
        top_k = int(cfg["retrieval"]["top_k"])

    bm25, metadatas, texts = _get_bm25_index(
        cfg, use_sastrawi=use_sastrawi, collection=collection
    )

    # Corpus kosong → early return (jangan crash)
    if not texts:
        return []

    # ── Scoring ───────────────────────────────────────────────────────────────
    query_tokens = _tokenize(query, use_sastrawi=use_sastrawi)
    raw_scores   = bm25.get_scores(query_tokens)   # ndarray, len == corpus size

    # Top-k indices by BM25 score descending
    top_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:top_k]

    # ── Build RetrievedNode ───────────────────────────────────────────────────
    nodes: List[RetrievedNode] = []
    for idx in top_indices:
        raw_score = float(raw_scores[idx])
        md        = metadatas[idx]
        text      = texts[idx]

        chunk_id = str(md.get("chunk_id", f"sparse_{idx}"))
        doc_id   = str(md.get("doc_id", md.get("source_file", "unknown")))

        # ── V3.0a: stream + bab_label dari metadata chunk ────────────────────
        # Untuk chunk dari V3.0a ingest: metadata sudah punya field ini.
        # Untuk chunk dari V1/V2: tidak ada → None (backward-compat).
        # Fallback stream dari collection jika metadata tidak menyertakannya.
        stream_val    = md.get("stream")    or None
        bab_label_val = md.get("bab_label") or None
        if stream_val is None and collection in ("narasi", "sitasi"):
            stream_val = collection

        nodes.append(
            RetrievedNode(
                chunk_id     = chunk_id,
                doc_id       = doc_id,
                score        = raw_score,    # score utama = BM25 raw score
                score_sparse = raw_score,    # expose untuk RRF fusion & UI Sources
                text         = text,
                metadata     = md,
                stream       = stream_val,
                bab_label    = bab_label_val,
            )
        )

    return nodes
