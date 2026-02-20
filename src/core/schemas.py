from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedNode:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    score_dense: Optional[float] = None
    score_sparse: Optional[float] = None
    rank_dense: Optional[int] = None
    rank_sparse: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # biar log tidak kebesaran, text bisa dipotong di layer logging
        return d


@dataclass
class RetrievalLogRecord:
    run_id: str
    turn_id: int
    timestamp: str
    mode: str  # dense | hybrid
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
