from __future__ import annotations

from typing import Dict, List

from src.core.schemas import RetrievedNode


def rrf_fusion(
    dense_nodes: List[RetrievedNode],
    sparse_nodes: List[RetrievedNode],
    *,
    k: int = 60,
    top_k: int = 10,
) -> List[RetrievedNode]:
    """
    Reciprocal Rank Fusion (RRF) — menggabungkan ranking dense + sparse.

    Formula: score_rrf(d) = Σ  1 / (k + rank_i(d))
            untuk setiap retriever i yang mengandung dokumen d.

    Kontrak:
        - Input harus sudah terurut descending by relevance masing-masing retriever.
        - Input TIDAK dimodifikasi (semua objek output adalah RetrievedNode baru).
        - Output terurut descending by RRF score.
        - Semua field skor terisi: score (RRF), score_dense, score_sparse,
            rank_dense, rank_sparse.
        - Node yang hanya muncul di satu retriever tetap masuk output
            (rank di retriever lain = None, kontribusi RRF = hanya dari satu sisi).
    """
    # Akumulasi RRF scores
    rrf_scores: Dict[str, float] = {}

    # Pool node: dense prioritas sebagai "base" (teks + metadata lebih reliable
    # karena berasal dari ChromaDB langsung). Sparse sebagai fallback.
    node_pool: Dict[str, RetrievedNode] = {}

    # Simpan score & rank asli per chunk_id dari masing-masing retriever
    d_score: Dict[str, float] = {}
    s_score: Dict[str, float] = {}
    d_rank: Dict[str, int] = {}
    s_rank: Dict[str, int] = {}

    # ── Proses jalur dense ──
    for rank, node in enumerate(dense_nodes, start=1):
        cid = node.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

        # score_dense: gunakan field score_dense jikalau sudah diisi,
        # fallback ke score (distance Chroma)
        d_score[cid] = (
            node.score_dense if node.score_dense is not None else node.score
        )
        d_rank[cid] = rank

        if cid not in node_pool:
            node_pool[cid] = node  # dense diprioritaskan

    # ── Proses jalur sparse ──
    for rank, node in enumerate(sparse_nodes, start=1):
        cid = node.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

        s_score[cid] = (
            node.score_sparse if node.score_sparse is not None else node.score
        )
        s_rank[cid] = rank

        if cid not in node_pool:
            node_pool[cid] = node  # sparse sebagai fallback

    # ── Sort by RRF score descending ──
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )

    # ── Bangun output — RetrievedNode baru per chunk (tidak mutasi input) ──
    result: List[RetrievedNode] = []
    for cid in sorted_ids[:top_k]:
        base = node_pool[cid]
        fused = RetrievedNode(
            chunk_id=base.chunk_id,
            doc_id=base.doc_id,
            score=rrf_scores[cid],          # RRF score final (untuk sorting & display)
            score_dense=d_score.get(cid),   # None jikalau tidak muncul di dense
            score_sparse=s_score.get(cid),  # None jikalau tidak muncul di sparse
            rank_dense=d_rank.get(cid),     # None jikalau tidak muncul di dense
            rank_sparse=s_rank.get(cid),    # None jikalau tidak muncul di sparse
            text=base.text,
            metadata=base.metadata,
        )
        result.append(fused)

    return result
