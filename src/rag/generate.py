from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _get_cfg(cfg: Dict[str, Any], path: List[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_llm(cfg: Dict[str, Any]):
    """
    Build ChatModel berdasarkan provider.
    Provider bisa dari env LLM_PROVIDER atau cfg['llm']['provider'].
    """
    load_dotenv()  # membaca .env (jika ada) - aman walau dipanggil berkali-kali

    provider = (
        (os.getenv("LLM_PROVIDER") or _get_cfg(cfg, ["llm", "provider"], "groq")).strip().lower()
    )
    temperature = float(_get_cfg(cfg, ["llm", "temperature"], 0.2))
    max_tokens = _get_cfg(cfg, ["llm", "max_tokens"], 512)
    timeout = _get_cfg(cfg, ["llm", "timeout"], 60)
    max_retries = int(_get_cfg(cfg, ["llm", "max_retries"], 2))

    # model fallback (kalau env model tidak ada)
    model_fallback = _get_cfg(cfg, ["llm", "model"], None)

    if provider == "groq":
        # LangChain Groq: install langchain-groq, env GROQ_API_KEY
        from langchain_groq import ChatGroq  # type: ignore  # package external

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL") or model_fallback or "llama3-8b-8192"
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY belum di-set. Isi di file .env (lokal) sebelum memanggil LLM."
            )

        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
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

        # ChatOpenAI supports api_key=... directly (or env)
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    if provider == "ollama":
        # LangChain Ollama: install langchain-ollama, Ollama server harus jalan
        from langchain_ollama import ChatOllama  # type: ignore

        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = os.getenv("OLLAMA_MODEL") or model_fallback or "llama3"

        return ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    raise ValueError(f"Provider tidak dikenal: {provider}. Gunakan: groq | openai | ollama")


def generate_answer(cfg: Dict[str, Any], question: str, contexts: List[str]) -> str:
    """
    Generate jawaban berbasis konteks (RAG).
    Untuk V1-V2, contexts masih berupa list teks chunk.
    """
    llm = _build_llm(cfg)

    # Batasi konteks biar prompt tidak “meledak”
    max_ctx = 8
    contexts = contexts[:max_ctx]

    ctx_block = (
        "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts)])
        if contexts
        else "(Tidak ada konteks.)"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Kamu adalah asisten QA untuk dokumen akademik. "
                "Jawab dalam Bahasa Indonesia, ringkas tapi jelas. "
                "Gunakan HANYA informasi dari konteks. "
                "Jika tidak cukup, katakan 'Tidak ditemukan pada dokumen'.",
            ),
            (
                "human",
                "Pertanyaan:\n{question}\n\nKonteks:\n{context}\n\n"
                "Instruksi:\n"
                "- Beri jawaban final.\n"
                "- Jika mengutip fakta penting, sebutkan CTX mana yang mendukung (mis. [CTX 2]).",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": ctx_block})
