from __future__ import annotations

from typing import Dict, List


def rrf_fusion(
    dense: List[Dict],
    sparse: List[Dict],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) gabungkan ranking dense + sparse.
    Input list harus sudah terurut descending by relevance masing-masing retriever.
    """
    scores: Dict[str, float] = {}
    merged: Dict[str, Dict] = {}

    def add(list_nodes: List[Dict], key_rank: str):
        for rank, n in enumerate(list_nodes, start=1):
            chunk_id = n["chunk_id"]
            scores.setdefault(chunk_id, 0.0)
            scores[chunk_id] += 1.0 / (k + rank)
            merged.setdefault(chunk_id, n)
            merged[chunk_id][key_rank] = rank

    add(dense, "rank_dense")
    add(sparse, "rank_sparse")

    out = []
    for chunk_id, base in merged.items():
        item = dict(base)
        item["score"] = float(scores[chunk_id])
        out.append(item)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]