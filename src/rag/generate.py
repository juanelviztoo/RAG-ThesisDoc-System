"""
generate.py
===========
Public API untuk RAG generation. Thin re-export wrapper.
Semua implementasi ada di generate_utils.py.

External code yang import dari sini TIDAK berubah:
    from src.rag.generate import generate_answer, build_generation_meta, contextualize_question
"""
from src.rag.generate_utils import (
    build_generation_meta,
    contextualize_question,
    generate_answer,
)

__all__ = ["build_generation_meta", "contextualize_question", "generate_answer"]