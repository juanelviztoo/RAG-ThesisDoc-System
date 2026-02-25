from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _get_cfg(cfg: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_int(x: Any, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _build_llm(cfg: Dict[str, Any]):
    """
    Build ChatModel berdasarkan provider.
    Provider bisa dari env LLM_PROVIDER atau cfg['llm']['provider'].
    """
    load_dotenv()  # membaca .env (jika ada) - aman walau dipanggil berkali-kali

    provider_raw = os.getenv("LLM_PROVIDER") or _get_cfg(cfg, ["llm", "provider"], "groq") or "groq"
    provider = str(provider_raw).strip().lower()

    temperature = _as_float(_get_cfg(cfg, ["llm", "temperature"], 0.2), 0.2)
    max_tokens = _as_int(_get_cfg(cfg, ["llm", "max_tokens"], 512), 512)
    timeout = _as_int(_get_cfg(cfg, ["llm", "timeout"], 60), 60)
    max_retries = _as_int(_get_cfg(cfg, ["llm", "max_retries"], 2), 2)

    # model fallback (kalau env model tidak ada)
    model_fallback = _get_cfg(cfg, ["llm", "model"], None)

    if provider == "groq":
        # LangChain Groq: install langchain-groq, env GROQ_API_KEY
        from langchain_groq import ChatGroq  # type: ignore  # package external

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL") or model_fallback or "llama-3.1-8b-instant"
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY belum di-set. Isi di file .env (lokal) sebelum memanggil LLM."
            )

        # Tidak pass api_key agar type checker tidak error (LangChain baca dari env)
        return ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if provider == "openai":
        # LangChain OpenAI: install langchain-openai, env OPENAI_API_KEY
        from langchain_openai import ChatOpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL") or model_fallback or "gpt-4o-mini"
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY belum di-set. Isi di file .env (lokal) sebelum memanggil LLM."
            )

        # ChatOpenAI supports api_key=... (env yang handle)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            model_kwargs={"max_tokens": max_tokens},
        )

    if provider == "ollama":
        # LangChain Ollama: install langchain-ollama, Ollama server harus jalan
        from langchain_ollama import ChatOllama  # type: ignore

        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = os.getenv("OLLAMA_MODEL") or model_fallback or "llama3"

        # Ollama (biasanya) memakai num_predict untuk batas output tokens
        return ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            # model_kwargs={"num_predict": max_tokens},
        )

    raise ValueError(f"Provider tidak dikenal: {provider}. Gunakan: groq | openai | ollama")


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " ...<trimmed>"


def _build_context_block(
    contexts: List[str], max_ctx: int, max_chars_per_ctx: int
) -> Tuple[str, int]:
    """
    Return:
      - ctx_block: string berisi [CTX i] ... yang sudah di-trim
      - used_ctx: jumlah CTX yang dipakai
    """
    ctxs = contexts[:max_ctx]
    parts: List[str] = []
    for i, c in enumerate(ctxs, start=1):
        c2 = _trim_text(c, max_chars_per_ctx)
        parts.append(f"[CTX {i}]\n{c2}")
    return ("\n\n".join(parts) if parts else "(Tidak ada konteks.)"), len(ctxs)


_CITATION_RE = re.compile(r"\[CTX\s*\d+\]", re.IGNORECASE)


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def generate_answer(cfg: Dict[str, Any], question: str, contexts: List[str]) -> str:
    """
    Generate jawaban berbasis konteks (RAG) dengan aturan:
    - Gunakan hanya informasi di konteks
    - Jika tidak cukup: jawab "Tidak ditemukan pada dokumen."
    - Wajib menyertakan sitasi [CTX n] untuk klaim faktual
    Untuk V1-V2, contexts masih berupa list teks chunk.
    """
    llm = _build_llm(cfg)

    # Kontrol konteks
    max_ctx = _as_int(_get_cfg(cfg, ["retrieval", "top_k"], 8), 8)
    max_ctx = min(max_ctx, 8)  # hard cap agar prompt tidak terlalu berat
    max_chars_per_ctx = _as_int(_get_cfg(cfg, ["llm", "max_chars_per_ctx"], 1800), 1800)

    ctx_block, used_ctx = _build_context_block(
        contexts, max_ctx=max_ctx, max_chars_per_ctx=max_chars_per_ctx
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Kamu adalah asisten QA untuk dokumen akademik.\n"
                "Peran: menjawab pertanyaan berdasarkan potongan dokumen (konteks) yang diberikan.\n\n"
                "Aturan ketat:\n"
                "1) Jawab dalam Bahasa Indonesia, formal, ringkas, dan jelas.\n"
                "2) Gunakan HANYA informasi dari konteks. Jangan menambah asumsi.\n"
                "3) Jika konteks tidak cukup untuk menjawab, tulis persis: 'Tidak ditemukan pada dokumen.'\n"
                "4) Setiap klaim faktual WAJIB disertai sitasi [CTX n].\n"
                "5) Jangan menyebut hal di luar konteks (mis. nama penulis, tahun, atau metode lain) jika tidak ada di CTX.\n"
                "6) Jangan menyebut [CTX] yang tidak ada (hanya 1..N sesuai konteks).",
            ),
            (
                "human",
                "Pertanyaan:\n{question}\n\n"
                "Konteks:\n{context}\n\n"
                "Tugas:\n"
                "- Jawab pertanyaan secara langsung.\n"
                "- Ambil bukti hanya dari konteks.\n\n"
                "Format jawaban WAJIB (ikuti persis):\n"
                "<jawaban satu paragraf ringkas namun informatif secara keseluruhan>\n"
                "Bukti:\n"
                "- <poin bukti 1> [CTX n]\n"
                "- <poin bukti 2> [CTX n]\n"
                "- <poin bukti 3> [CTX n]\n\n"
                "Ketentuan Bukti:\n"
                "- Maksimal 3 poin.\n"
                "- Tiap poin berisi fakta spesifik (mis. nama metode/alat uji/tahapan).\n"
                "- Jika tidak ditemukan, tulis:\n"
                "Jawaban: Tidak ditemukan pada dokumen.\n"
                "Bukti: -",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"question": question.strip(), "context": ctx_block}).strip()

    # Guardrail: jika model tidak memberi sitasi padahal ada konteks, paksa fallback yang aman
    if (
        used_ctx > 0
        and out
        and "Tidak ditemukan pada dokumen" not in out
        and not _has_citation(out)
    ):
        # fallback aman agar tidak halu tanpa sitasi
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    return out
