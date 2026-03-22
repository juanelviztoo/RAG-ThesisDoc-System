from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.schemas import RetrievedNode

# ═════════════════════════════════════════════════════════════════════════════
# Module-level caches (per-process)
# ═════════════════════════════════════════════════════════════════════════════
_EMB_CACHE: Dict[Tuple[str, str, bool], HuggingFaceEmbeddings] = {}
_VS_CACHE: Dict[Tuple[str, str, str, str, bool], Chroma] = {}

# Unifikasi cap ke satu konstanta
_MAX_CANDIDATE_K = 400


# ═════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_collection_name(cfg: Dict[str, Any], collection: str = "narasi") -> str:
    """
    Resolve nama collection ChromaDB dari config.
    Mendukung dua format config secara backward-compatible:

    V3.0a (dual collection):
        index:
            collections:
            narasi: "skripsi_structured_narasi"
            sitasi: "skripsi_structured_sitasi"
        → _resolve_collection_name(cfg, "narasi") → "skripsi_structured_narasi"
        → _resolve_collection_name(cfg, "sitasi") → "skripsi_structured_sitasi"

    V1.0 / V2.0 (single collection):
        index:
            collection_name: "skripsi_naive"
        → _resolve_collection_name(cfg, "narasi") → "skripsi_naive"
            (collection parameter diabaikan - hanya ada satu collection)

    Args:
        cfg        : config dict dari load_config()
        collection : "narasi" | "sitasi"  (diabaikan untuk V1/V2 config)

    Returns:
        str - nama collection ChromaDB yang akan diquery
    """
    index_cfg = cfg.get("index", {})

    # V3.0a: ada sub-dict "collections"
    if "collections" in index_cfg:
        collections_map = index_cfg["collections"]
        if collection in collections_map:
            return str(collections_map[collection])
        # fallback: ambil "narasi" jika key tidak dikenal
        return str(collections_map.get("narasi", "skripsi_structured_narasi"))

    # V1/V2: single collection_name
    return str(index_cfg.get("collection_name", "skripsi_naive"))


# ═════════════════════════════════════════════════════════════════════════════
# Embedding + vectorstore builder (dengan caching)
# ═════════════════════════════════════════════════════════════════════════════

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


def _get_vectorstore(cfg: Dict[str, Any], collection: str = "narasi") -> Chroma:
    """
    Mendapatkan Chroma vectorstore untuk collection yang diminta.
    Cache berbasis (persist_dir, collection_name, model, device, normalize).

    Args:
        cfg        : config dict
        collection : "narasi" | "sitasi"  (untuk V3.0a)
                    diabaikan untuk V1/V2 config (selalu pakai collection_name tunggal)
    """
    persist_dir     = str(cfg["index"]["persist_dir"])
    collection_name = _resolve_collection_name(cfg, collection)

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
    """Reset embedding + vectorstore cache (berguna saat dev/debug atau re-ingest)."""
    _EMB_CACHE.clear()
    _VS_CACHE.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Diversity helper
# ═════════════════════════════════════════════════════════════════════════════

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
    - Pass 3: kalau masih kurang, isi sisa (relax cap) agar tetap dapat top_k node
    """
    if not nodes:
        return []

    top_k      = max(1, int(top_k))
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


# ═════════════════════════════════════════════════════════════════════════════
# V3.0a field extractor helper
# ═════════════════════════════════════════════════════════════════════════════

def _extract_v3_fields(
    md: Dict[str, Any],
    collection: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Ekstrak field V3.0a (stream, bab_label) dari metadata chunk.

    Untuk chunk dari V3.0a: metadata sudah menyertakan "stream" dan "bab_label"
    yang di-set oleh chunking.py (via _build_chunk_meta).

    Untuk chunk dari V1/V2: metadata tidak punya field ini → kembalikan
    (None, None) agar backward-compatible (field optional di RetrievedNode).

    Args:
        md         : metadata dict dari Chroma document
        collection : "narasi" | "sitasi" — sebagai fallback jika metadata
                    tidak punya "stream" (chunk dari V1/V2 yang di-ingest ulang tanpa field baru)
    """
    stream    = md.get("stream")    or None   # bisa "" → None
    bab_label = md.get("bab_label") or None   # bisa "" → None

    # Fallback: jika stream tidak ada di metadata tapi collection diketahui
    # (kemungkinan ingest V3.0a tapi metadata belum ter-update)
    if stream is None and collection in ("narasi", "sitasi"):
        stream = collection

    return stream, bab_label


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def retrieve_dense(
    cfg: Dict[str, Any],
    query: str,
    *,
    diversify:   bool          = False,
    max_per_doc: int           = 2,
    candidate_k: Optional[int] = None,
    collection:  str           = "narasi",
) -> List[RetrievedNode]:
    """
    Dense retrieval dari ChromaDB via similarity_search_with_score().

    diversify=True:
        - ambil kandidat lebih banyak (candidate_k)
        - pilih top_k yang lebih beragam antar dokumen (max_per_doc per dokumen)

    V3.0a: Parameter `collection` mengontrol dari collection mana query dijalankan.
        - "narasi"  → skripsi_structured_narasi  (default, untuk query umum)
        - "sitasi"  → skripsi_structured_sitasi  (untuk query referensi/citation)
    V1/V2 backward-compat: `collection` diabaikan, selalu pakai collection_name tunggal.

    Args:
        cfg         : config dict dari load_config()
        query       : query string
        diversify   : aktifkan diversity retrieval (max_per_doc per dokumen)
        max_per_doc : batas chunk per dokumen saat diversify=True (default 2)
        candidate_k : ukuran pool kandidat untuk diversify
                      (default: min(top_k * 6, _MAX_CANDIDATE_K))
        collection  : "narasi" | "sitasi" — collection ChromaDB yang diquery
                    (V3.0a only; diabaikan untuk V1/V2 single-collection config)

    Returns:
        List[RetrievedNode] - diurutkan descending by score (lower distance = better).
        Field stream dan bab_label terisi untuk chunk V3.0a; None untuk V1/V2.
    """
    top_k = int(cfg["retrieval"]["top_k"])
    vs    = _get_vectorstore(cfg, collection)

    # ── Hitung k untuk similarity_search ─────────────────────────────────────
    if diversify:
        if candidate_k is None:
            # default: pool lebih besar supaya punya “bahan” untuk sebar antar dokumen
            _ck = min(max(top_k * 6, top_k), _MAX_CANDIDATE_K)
        else:
            _ck = min(max(int(candidate_k), top_k), _MAX_CANDIDATE_K)
        results: List[Tuple[Any, float]] = vs.similarity_search_with_score(query, k=_ck)
    else:
        results = vs.similarity_search_with_score(query, k=top_k)

    # ── Bangun RetrievedNode ──────────────────────────────────────────────────
    nodes: List[RetrievedNode] = []
    for doc, score in results:
        md = dict(doc.metadata or {})
        stream, bab_label = _extract_v3_fields(md, collection)
        nodes.append(
            RetrievedNode(
                doc_id      = str(md.get("doc_id", md.get("source_file", "unknown"))),
                chunk_id    = str(md.get("chunk_id", "unknown")),
                score       = float(score),
                score_dense = float(score), # expose ke jalur hybrid & UI
                text        = str(doc.page_content),
                metadata    = md,
                stream      = stream,
                bab_label   = bab_label,
            )
        )

    if not diversify:
        return nodes[:top_k]

    return _diversify_by_doc(nodes, top_k=top_k, max_per_doc=max_per_doc)
