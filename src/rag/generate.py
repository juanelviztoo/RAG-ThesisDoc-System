from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Sequence, Tuple

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


# Terima [CTX 2], [CTX2], (CTX 2), CTX 2
_CITATION_RE = re.compile(r"(\[CTX\s*\d+\]|\(CTX\s*\d+\)|\bCTX\s*\d+\b)", re.IGNORECASE)


def _normalize_citations(text: str) -> str:
    """Ubah variasi sitasi jadi format standar [CTX n]."""
    t = text or ""
    # (CTX 2) -> [CTX 2]
    t = re.sub(r"\(CTX\s*(\d+)\)", r"[CTX \1]", t, flags=re.IGNORECASE)
    # CTX 2 -> [CTX 2] (jaga agar tidak dobel)
    t = re.sub(r"(?<!\[)\bCTX\s*(\d+)\b(?!\])", r"[CTX \1]", t, flags=re.IGNORECASE)
    return t


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


_BUKTI_SPLIT_RE = re.compile(r"\bBukti\s*:\s*", re.IGNORECASE)


def _enforce_qa_format(out: str) -> str:
    """
    Pastikan output selalu punya:
      Jawaban: ...
      Bukti: ...
    Agar parse_answer() dan UI stabil walau LLM kadang bandel.
    """
    t = (out or "").strip()
    if not t:
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # kalau sudah "Tidak ditemukan", pastikan format lengkap
    if "tidak ditemukan pada dokumen" in t.lower():
        # jika user cuma nulis satu kalimat tanpa bukti
        if "bukti" not in t.lower():
            return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"
        # kalau sudah ada bukti, biarkan lewat bawah
        # (tetap akan dinormalisasi)

    has_jawaban = re.search(r"^\s*Jawaban\s*:", t, flags=re.IGNORECASE | re.MULTILINE) is not None
    has_bukti = re.search(r"\bBukti\s*:", t, flags=re.IGNORECASE) is not None

    # kasus LLM: tidak ada "Jawaban:" tapi ada "Bukti:"
    if (not has_jawaban) and has_bukti:
        parts = _BUKTI_SPLIT_RE.split(t, maxsplit=1)
        head = (parts[0] or "").strip()
        tail = (parts[1] or "").strip() if len(parts) > 1 else "-"
        # pastikan tail minimal ada sesuatu
        if not tail:
            tail = "-"
        # kalau tail tidak diawali '-' dan bukan '-', biarkan (parse_answer akan handle)
        return f"Jawaban: {head}\nBukti:\n{tail}"

    # kasus LLM: tidak ada "Jawaban:" dan tidak ada "Bukti:" -> bungkus saja
    if (not has_jawaban) and (not has_bukti):
        return f"Jawaban: {t}\nBukti: -"

    # kasus LLM: ada Jawaban tapi tidak ada Bukti -> tambahkan Bukti placeholder
    if has_jawaban and (not has_bukti):
        return t.rstrip() + "\n\nBukti: -"

    return t


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
                "3) Jika konteks tidak cukup untuk menjawab, tulis persis:\n"
                "   Jawaban: Tidak ditemukan pada dokumen.\n"
                "   Bukti: -\n"
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
                "Jawaban: <jawaban satu paragraf ringkas namun informatif secara keseluruhan>\n"
                "Bukti:\n"
                "- <poin bukti 1> [CTX n]\n"
                "- <poin bukti 2> [CTX n]\n"
                "- <poin bukti 3> [CTX n]\n\n"
                "Ketentuan Bukti:\n"
                "- Maksimal 3 poin.\n"
                "- Tiap poin berisi fakta spesifik (mis. nama metode/alat uji/tahapan).\n"
                "- Jika tidak ada dukungan konteks, tulis persis:\n"
                "Jawaban: Tidak ditemukan pada dokumen.\n"
                "Bukti: -\n",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"question": question.strip(), "context": ctx_block}).strip()

    out = _normalize_citations(out).strip()
    out = _enforce_qa_format(out)

    # Kalau jawabannya tidak "Tidak ditemukan", tapi sitasi tidak terdeteksi,
    # lakukan 1x repair attempt (lebih baik daripada langsung fallback).
    if (
        used_ctx > 0
        and out
        and "Tidak ditemukan pada dokumen" not in out
        and not _has_citation(out)
    ):
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Kamu bertugas memperbaiki format jawaban agar sesuai aturan sitasi.\n"
                    "Aturan:\n"
                    "1) Jangan mengubah makna jawaban.\n"
                    "2) Tambahkan sitasi [CTX n] pada setiap klaim faktual, terutama di bagian Bukti.\n"
                    "3) Output harus mengikuti format:\n"
                    "   Jawaban: ...\n"
                    "   Bukti:\n"
                    "   - ... [CTX n]\n"
                    "4) Jika memang tidak ada dukungan dari konteks, tulis persis:\n"
                    "   Jawaban: Tidak ditemukan pada dokumen.\n"
                    "   Bukti: -",
                ),
                (
                    "human",
                    "Pertanyaan:\n{question}\n\n"
                    "Konteks:\n{context}\n\n"
                    "Jawaban sebelumnya (perlu diperbaiki sitasinya):\n{answer}\n\n"
                    "Perbaiki sekarang:",
                ),
            ]
        )
        fixed = (repair_prompt | llm | StrOutputParser()).invoke(
            {"question": question.strip(), "context": ctx_block, "answer": out}
        )
        fixed = _normalize_citations((fixed or "").strip())

        if fixed:
            out = fixed

    # Guardrail terakhir: jika model tidak memberi sitasi padahal ada konteks, paksa fallback yang aman
    if (
        used_ctx > 0
        and out
        and "Tidak ditemukan pada dokumen" not in out
        and not _has_citation(out)
    ):
        # fallback aman agar tidak halu tanpa sitasi
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    return out


def _format_history_pairs(history_pairs: Sequence[Dict[str, Any]], max_chars: int = 1200) -> str:
    """
    history_pairs: list item seperti {"turn_id":..., "query":..., "answer_preview":...}
    saya ringkas agar prompt rewrite tidak meledak.
    """
    lines: list[str] = []
    for it in history_pairs:
        q = str(it.get("query", "")).strip()
        a = str(it.get("answer_preview", "")).strip()
        if not q:
            continue
        if len(a) > 240:
            a = a[:240] + "..."
        lines.append(f"User: {q}\nAssistant: {a}")
    text = "\n\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text or "(no history)"


def contextualize_question(
    cfg: Dict[str, Any],
    question: str,
    history_pairs: Sequence[Dict[str, Any]],
    *,
    use_llm: bool = True,
) -> str:
    """
    Rewrite pertanyaan menjadi pertanyaan mandiri (standalone) berdasarkan riwayat percakapan.
    Dipakai sebelum retrieval ke Chroma.

    - Jika use_llm=True: coba pakai LLM (lebih akurat).
    - Jika gagal / use_llm=False: fallback heuristic sederhana.
    """
    q = (question or "").strip()
    if not q:
        return ""

    if not history_pairs:
        return q

    # fallback heuristic (tanpa LLM)
    def _heuristic() -> str:
        last_q = str(history_pairs[-1].get("query", "")).strip()
        if not last_q:
            return q
        # gabungkan ringkas agar embedding tetap relevan
        return f"{q}\n\n(Konteks sebelumnya: {last_q})"

    if not use_llm:
        return _heuristic()

    try:
        llm = _build_llm(cfg)
    except Exception:
        return _heuristic()

    history_txt = _format_history_pairs(history_pairs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Kamu adalah asisten rewriting query untuk retrieval dokumen.\n"
                "Tugas: ubah pertanyaan terbaru menjadi pertanyaan mandiri (standalone) "
                "dengan memasukkan konteks penting dari riwayat.\n"
                "Aturan:\n"
                "1) Output HANYA 1 kalimat pertanyaan (tanpa penjelasan).\n"
                "2) Bahasa Indonesia.\n"
                "3) Jangan menambah fakta baru.\n",
            ),
            (
                "human",
                "RIWAYAT:\n{history}\n\n"
                "PERTANYAAN TERBARU:\n{question}\n\n"
                "Tuliskan versi pertanyaan mandiri (standalone):",
            ),
        ]
    )

    out = (prompt | llm | StrOutputParser()).invoke({"history": history_txt, "question": q})
    out = (out or "").strip().strip('"').strip("'")

    return out if out else _heuristic()
