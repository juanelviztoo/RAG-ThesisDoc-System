from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedNode:
    chunk_id:     str
    doc_id:       str
    score:        float
    text:         str
    metadata:     Dict[str, Any]
    score_dense:  Optional[float] = None
    score_sparse: Optional[float] = None
    rank_dense:   Optional[int]   = None
    rank_sparse:  Optional[int]   = None
    # ── V3.0a ─────────────────────────────────────────────────────────────────
    # stream    : collection asal chunk ("narasi" | "sitasi")
    #             None untuk chunk dari pipeline V1/V2 (backward-compat)
    # bab_label : label section ("BAB_I" | "BAB_II" | ... | "DAFTAR_PUSTAKA" | None)
    #             Diisi dari metadata["bab_label"] saat retrieve, None untuk V1/V2
    stream:    Optional[str] = None
    bab_label: Optional[str] = None
    # ── V3.0c ─────────────────────────────────────────────────────────────────
    # rerank_score : skor dari Cross-Encoder (makin tinggi = makin relevan)
    #                None jika reranking tidak diterapkan / graceful skip
    # rerank_rank  : posisi setelah reranking (1 = paling relevan)
    #                None jika reranking tidak diterapkan / graceful skip
    rerank_score: Optional[float] = None
    rerank_rank:  Optional[int]   = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # biar log tidak kebesaran, text bisa dipotong di layer logging
        return d


@dataclass
class RetrievalLogRecord:
    run_id: str
    turn_id: int
    timestamp: str
    mode: str  # dense | hybrid | metadata_router
    query_original: str
    query_rewritten: Optional[str]
    retrieved_nodes: List[Dict[str, Any]]
    top_k: int
    config_hash: str
    config_snapshot: Optional[Dict[str, Any]] = None  # optional (kalau mau disisipkan per baris)


@dataclass
class AnswerLogRecord:
    run_id: str
    turn_id: int
    timestamp: str
    question: str
    answer: str
    contexts: List[str]
    source_chunk_ids: List[str]
    config_hash: str
    config_snapshot: Optional[Dict[str, Any]] = None
