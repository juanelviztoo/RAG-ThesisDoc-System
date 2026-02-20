from __future__ import annotations

from typing import Any, Dict, List


def dense_retrieve(cfg: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Return list of RetrievedNode (as dict):
    {chunk_id, doc_id, score, text, metadata, ...}
    """
    # TODO: implement using Chroma similarity_search_with_score
    raise NotImplementedError
