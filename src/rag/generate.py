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


_CTX_NUM_RE = re.compile(r"\[CTX\s*(\d+)\]", re.IGNORECASE)


def _extract_ctx_nums(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CTX_NUM_RE.finditer(text or "")]


def _is_valid_answer_format(out: str) -> bool:
    t = (out or "").strip()
    if not t:
        return False
    has_j = re.search(r"^\s*Jawaban\s*:", t, flags=re.IGNORECASE | re.MULTILINE) is not None
    has_b = re.search(r"^\s*Bukti\s*:", t, flags=re.IGNORECASE | re.MULTILINE) is not None
    return has_j and has_b


def _citations_in_range(out: str, used_ctx: int) -> bool:
    nums = _extract_ctx_nums(out)
    if used_ctx <= 0:
        # kalau tidak ada konteks, seharusnya tidak ada sitasi
        return len(nums) == 0
    return all(1 <= n <= used_ctx for n in nums)


def _bukti_has_bullets_or_dash(out: str) -> bool:
    t = (out or "").strip()
    if "Bukti:" not in t:
        return False
    _, tail = t.split("Bukti:", 1)
    tail = tail.strip()
    if tail == "-":
        return True
    # minimal ada satu bullet
    return any(ln.strip().startswith("-") for ln in tail.splitlines() if ln.strip())


# ========= Output cleaning (anti echo "Pertanyaan:") =========
_JAWABAN_START_RE = re.compile(r"(?im)^\s*Jawaban\s*:\s*")
_PERTANYAAN_LINE_RE = re.compile(r"(?im)^\s*Pertanyaan\s*:\s*.*(?:\n|$)")


def _strip_question_preamble(text: str) -> str:
    """
    Jika LLM menulis:
        Pertanyaan: ...
        Jawaban: ...
        Bukti: ...
    maka buang bagian 'Pertanyaan:' agar output selalu mulai dari 'Jawaban:'.
    """
    t = (text or "").strip()
    if not t:
        return t

    # Kalau ada "Jawaban:" di tengah, potong semua sebelum itu.
    m = _JAWABAN_START_RE.search(t)
    if m and m.start() > 0:
        t = t[m.start() :].lstrip()

    # Jaga-jaga: hapus baris "Pertanyaan:" yang masih tersisa
    t = _PERTANYAAN_LINE_RE.sub("", t).strip()
    return t


_BUKTI_SPLIT_RE = re.compile(r"\bBukti\s*:\s*", re.IGNORECASE)


def _enforce_qa_format(out: str) -> str:
    """
    Pastikan output selalu punya:
        Jawaban: ...
        Bukti: ...
    Agar parse_answer() dan UI stabil walau LLM kadang bandel.
    """
    # penting: bersihkan preamble "Pertanyaan:" dulu
    t = _strip_question_preamble(out)

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


def _ensure_bukti_bullets(out: str) -> str:
    """
    Memastikan setelah 'Bukti:' isinya berupa daftar bullet.
    Kalau LLM mengisi bukti sebagai teks biasa, saya ubah jadi minimal 1 bullet.
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return t

    head, tail = t.split("Bukti:", 1)
    head = head.strip()
    tail = tail.strip()

    # kalau Bukti: - sudah benar
    if tail == "-" or tail.startswith("-"):
        return f"{head}\nBukti:\n{tail}" if not tail.startswith("-") else f"{head}\nBukti:\n{tail}"

    # kalau bukti berupa beberapa baris tapi tanpa dash
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return f"{head}\nBukti: -"

    # ubah setiap baris menjadi bullet
    bullets = "\n".join([f"- {ln.lstrip('-').strip()}" for ln in lines])
    return f"{head}\nBukti:\n{bullets}"


# ========= Bukti post-processing: dedupe + cap + grounding check =========

_CTX_INLINE_RE = re.compile(r"\[CTX\s*(\d+)\]", re.IGNORECASE)


def _extract_bukti_lines(out: str) -> List[str]:
    """
    Ambil list bullet dari bagian Bukti.
    Output: list string TANPA leading '- '.
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return []

    _, tail = t.split("Bukti:", 1)
    tail = (tail or "").strip()
    if not tail or tail == "-":
        return []

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    bullets: List[str] = []
    for ln in lines:
        if ln == "-":
            continue
        if ln.startswith("-"):
            bullets.append(ln[1:].strip())
        else:
            # toleransi kalau ada yang tidak diawali '-'
            bullets.append(ln.strip())
    return bullets


def _rebuild_with_bukti(out_head: str, bullets: List[str]) -> str:
    """
    Bangun ulang output dengan Bukti yang rapi.
    """
    head = (out_head or "").strip()
    if not bullets:
        return f"{head}\nBukti: -"

    blk = "\n".join([f"- {b.strip()}" for b in bullets if b.strip()])
    if not blk.strip():
        return f"{head}\nBukti: -"
    return f"{head}\nBukti:\n{blk}"


def _cap_and_dedupe_bukti(out: str, max_points: int = 3) -> str:
    """
    HARD ENFORCE:
    - dedupe bullet (berdasarkan teks tanpa [CTX n])
    - cap maksimal max_points
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return t

    head, _ = t.split("Bukti:", 1)
    head = head.strip()

    bullets = _extract_bukti_lines(t)
    if not bullets:
        return f"{head}\nBukti: -"

    def _sig(x: str) -> str:
        # signature untuk dedupe: hapus sitasi, lowercase, rapikan spasi
        x2 = re.sub(r"\[CTX\s*\d+\]", "", x, flags=re.IGNORECASE)
        x2 = " ".join(x2.split()).strip().lower()
        return x2

    seen = set()
    uniq: List[str] = []
    for b in bullets:
        s = _sig(b)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(b)

    capped = uniq[:max_points]
    return _rebuild_with_bukti(head, capped)


def _keywords_for_grounding(text: str, max_terms: int = 10) -> List[str]:
    """
    Keyword untuk cek grounding.
    - Ambil token >= 3 huruf agar RAD/UAT/UX kebaca
    - Buang stopword + kata generik
    """
    s = re.sub(r"\[CTX\s*\d+\]", "", (text or ""), flags=re.IGNORECASE)
    toks = re.findall(r"[a-zA-Z]{3,}", s.lower())

    stop = set(_STOP_QA) | {
        "jawaban",
        "bukti",
        "pertanyaan",
        "konteks",
        "digunakan",
        "dipakai",
        "menggunakan",
        "penelitian",
        "dokumen",
        "beberapa",
        "lainnya",
        "lain",
        "metode",
        "model",
        "sistem",
    }

    out: List[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _bullet_grounded(bullet: str, contexts: List[str], used_ctx: int) -> bool:
    """
    Bullet dianggap grounded jika:
    - memiliki >=1 sitasi CTX valid
    - dan minimal 1 keyword penting pada bullet muncul di salah satu CTX yang dirujuk
    """
    if not bullet or used_ctx <= 0 or not contexts:
        return False

    ctx_nums = [int(x) for x in _CTX_INLINE_RE.findall(bullet or "")]
    ctx_nums = [n for n in ctx_nums if 1 <= n <= used_ctx]
    if not ctx_nums:
        return False

    kws = _keywords_for_grounding(bullet)
    if not kws:
        # kalau bullet terlalu generik tanpa keyword, anggap tidak aman
        return False

    for n in ctx_nums:
        hay = (contexts[n - 1] or "").lower()
        if any(k in hay for k in kws):
            return True

    return False


def _bukti_grounding_ok(out: str, contexts: List[str], used_ctx: int) -> bool:
    """
    Semua bullet Bukti harus grounded.
    """
    bullets = _extract_bukti_lines(out)
    if not bullets:
        # kalau tidak ada bukti (Bukti: -), dianggap tidak perlu grounding
        return True

    for b in bullets:
        if not _bullet_grounded(b, contexts, used_ctx):
            return False
    return True


_STOP_QA = {
    "apa",
    "yang",
    "dan",
    "atau",
    "dengan",
    "dalam",
    "pada",
    "untuk",
    "dari",
    "ke",
    "ini",
    "itu",
    "adalah",
    "menurut",
    "jelaskan",
    "sebutkan",
    "minimal",
    "jika",
    "ada",
    "metode",
    "sistem",
    "dokumen",
    "tahap",
    "tahapan",
    "langkah",
    "prosedur",
}


def _keywords_for_support(text: str, max_terms: int = 8) -> list[str]:
    toks = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in _STOP_QA or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _looks_supported(question: str, contexts: list[str]) -> bool:
    """
    Heuristik ringan: kalau ada overlap keyword penting query di konteks,
    tidak mudah bilang 'Tidak ditemukan'.
    """
    qs = _keywords_for_support(question)
    if not qs or not contexts:
        return False
    hay = (" ".join(contexts)).lower()
    return any(t in hay for t in qs)


def _extractive_fallback_from_contexts(contexts: list[str], used_ctx: int) -> str:
    """
    Fallback aman: tidak membuat fakta baru, hanya menampilkan bukti ringkas dengan sitasi.
    """
    if used_ctx <= 0 or not contexts:
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    bullets: list[str] = []
    for i, c in enumerate(contexts[: min(3, used_ctx)], start=1):
        sn = " ".join((c or "").split())
        sn = sn[:240] + ("..." if len(sn) > 240 else "")
        bullets.append(f"- {sn} [CTX {i}]")

    return (
        "Jawaban: Informasi relevan ditemukan pada potongan dokumen berikut.\n"
        "Bukti:\n" + "\n".join(bullets)
    )


def generate_answer(cfg: Dict[str, Any], question: str, contexts: List[str]) -> str:
    """
    Generate jawaban berbasis konteks (RAG) dengan aturan:
    - Gunakan hanya informasi di konteks
    - Jika tidak cukup: jawab "Tidak ditemukan pada dokumen."
    - Wajib menyertakan sitasi [CTX n] untuk klaim faktual
    - HARD ENFORCE: Bukti maksimal 3 poin (cap + dedupe)
    - Anti klaim tidak presisi: validasi grounding bullet Bukti terhadap CTX
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
                "6) Jangan menyebut [CTX] yang tidak ada (hanya 1..N sesuai konteks)."
                "7) Jangan menuliskan ulang label 'Pertanyaan:' pada output. Output harus dimulai dari 'Jawaban:'."
                "8) Bagian Bukti maksimal 3 bullet.",
            ),
            (
                "human",
                "Input user (jangan ditulis ulang sebagai 'Pertanyaan:' di output):\n{question}\n\n"
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
                "- Tiap poin berisi fakta spesifik dan benar-benar didukung CTX (mis. nama metode/alat uji/tahapan).\n"
                "- Jika tidak ada dukungan konteks, tulis persis:\n"
                "Jawaban: Tidak ditemukan pada dokumen.\n"
                "Bukti: -\n",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"question": question.strip(), "context": ctx_block}).strip()

    # ======= Normalisasi awal =======
    # 0) bersihkan kemungkinan preamble "Pertanyaan:"
    out = _strip_question_preamble(out)

    # 1) Normalisasi sitasi + paksa format + paksa bukti jadi bullet (max. 3)
    out = _normalize_citations(out).strip()
    out = _enforce_qa_format(out)
    out = _ensure_bukti_bullets(out)
    out = _cap_and_dedupe_bukti(out, max_points=3)

    # Anti false-negative: kalau model bilang "Tidak ditemukan" padahal konteks terlihat relevan,
    # coba 1x "force answer" (tetap hanya dari konteks).
    if (
        used_ctx > 0
        and ("Tidak ditemukan pada dokumen" in out)
        and _looks_supported(question, contexts)
    ):
        force_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Konteks yang diberikan mengandung informasi relevan.\n"
                    "Tugasmu: jawab menggunakan konteks tersebut.\n"
                    "Aturan:\n"
                    "1) Output WAJIB format:\n"
                    "   Jawaban: ...\n"
                    "   Bukti:\n"
                    "   - ... [CTX n]\n"
                    "2) Maksimal 3 bullet.\n"
                    f"3) CTX valid hanya 1..{used_ctx}.\n"
                    "4) Jangan menambah fakta baru.\n"
                    "5) Jangan menjawab 'Tidak ditemukan' jika ada bukti di konteks.\n"
                    "6) Jangan menulis ulang 'Pertanyaan:' pada output.\n",
                ),
                (
                    "human",
                    "Pertanyaan:\n{question}\n\n" "Konteks:\n{context}\n\n" "Jawab sekarang:",
                ),
            ]
        )
        forced = (force_prompt | llm | StrOutputParser()).invoke(
            {"question": question.strip(), "context": ctx_block}
        )
        forced = _strip_question_preamble((forced or "").strip())
        forced = _normalize_citations(forced)
        forced = _enforce_qa_format(forced)
        forced = _ensure_bukti_bullets(forced)
        forced = _cap_and_dedupe_bukti(forced, max_points=3)
        if forced:
            out = forced

    # ======= Repair format / citation-range =======
    # 2) Validasi format dasar
    need_repair_format = not _is_valid_answer_format(out) or (not _bukti_has_bullets_or_dash(out))

    # 3) Validasi range sitasi (kalau jawaban bukan "Tidak ditemukan")
    need_repair_range = False
    if used_ctx > 0 and ("Tidak ditemukan pada dokumen" not in out):
        if not _citations_in_range(out, used_ctx):
            need_repair_range = True

    # 4) Jika format atau sitasi range bermasalah → 1x repair attempt
    if need_repair_format or need_repair_range:
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Perbaiki jawaban agar sesuai format & aturan sitasi.\n"
                    "Aturan WAJIB:\n"
                    "1) Output harus mengikuti format:\n"
                    "   Jawaban: <1 paragraf>\n"
                    "   Bukti:\n"
                    "   - ... [CTX n]\n"
                    "   - ... [CTX n]\n"
                    "2) Maksimal 3 bullet di Bukti.\n"
                    f"3) CTX yang tersedia hanya 1 sampai {used_ctx}.\n"
                    "4) Jangan menambah fakta baru. Jangan mengubah makna.\n"
                    "5) Jangan menulis ulang 'Pertanyaan:' di output.\n"
                    "6) Jika tidak ada dukungan konteks, tulis persis:\n"
                    "   Jawaban: Tidak ditemukan pada dokumen.\n"
                    "   Bukti: -",
                ),
                (
                    "human",
                    "Pertanyaan:\n{question}\n\n"
                    "Konteks:\n{context}\n\n"
                    "Jawaban sebelumnya (perlu diperbaiki):\n{answer}\n\n"
                    "Perbaiki sekarang:",
                ),
            ]
        )
        fixed = (repair_prompt | llm | StrOutputParser()).invoke(
            {"question": question.strip(), "context": ctx_block, "answer": out}
        )
        fixed = _strip_question_preamble((fixed or "").strip())
        fixed = _normalize_citations(fixed)
        fixed = _enforce_qa_format(fixed)
        fixed = _ensure_bukti_bullets(fixed)
        fixed = _cap_and_dedupe_bukti(fixed, max_points=3)
        if fixed:
            out = fixed

    # ======= Grounding check untuk anti klaim tidak presisi =======
    # Hanya berlaku jika bukan "Tidak ditemukan"
    if used_ctx > 0 and ("Tidak ditemukan pada dokumen" not in out):
        grounding_ok = _bukti_grounding_ok(out, contexts, used_ctx)

        if not grounding_ok:
            # 1x grounding repair attempt: paksa bukti benar-benar "mengutip/meringkas langsung" CTX
            grounding_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Perbaiki jawaban agar setiap poin Bukti benar-benar DIDUKUNG oleh CTX yang dirujuk.\n"
                        "Aturan:\n"
                        "1) Output WAJIB mulai dari 'Jawaban:' (tanpa 'Pertanyaan:').\n"
                        "2) Bukti maksimal 3 bullet.\n"
                        "3) Setiap bullet harus merangkum atau mengutip singkat isi CTX yang sama (bukan asumsi).\n"
                        "4) Jangan menyebut metode/hal lain jika tidak ada di CTX yang dirujuk.\n"
                        f"5) CTX valid hanya 1..{used_ctx}.\n",
                    ),
                    (
                        "human",
                        "Pertanyaan:\n{question}\n\n"
                        "Konteks:\n{context}\n\n"
                        "Jawaban sebelumnya (ada indikasi bukti kurang grounded):\n{answer}\n\n"
                        "Perbaiki sekarang:",
                    ),
                ]
            )
            grounded = (grounding_prompt | llm | StrOutputParser()).invoke(
                {"question": question.strip(), "context": ctx_block, "answer": out}
            )
            grounded = _strip_question_preamble((grounded or "").strip())
            grounded = _normalize_citations(grounded)
            grounded = _enforce_qa_format(grounded)
            grounded = _ensure_bukti_bullets(grounded)
            grounded = _cap_and_dedupe_bukti(grounded, max_points=3)

            if grounded and _bukti_grounding_ok(grounded, contexts, used_ctx):
                out = grounded
            else:
                # fallback aman: bukti ekstraktif (snippet CTX)
                return _extractive_fallback_from_contexts(contexts, used_ctx)

    # ======= Guardrails terakhir =======
    out = _cap_and_dedupe_bukti(out, max_points=3)

    # 5) Guardrail: kalau jawab bukan "Tidak ditemukan" tapi tidak ada sitasi → fallback aman
    if (
        used_ctx > 0
        and out
        and ("Tidak ditemukan pada dokumen" not in out)
        and (not _has_citation(out))
    ):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts):
            return _extractive_fallback_from_contexts(contexts, used_ctx)
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # 6) Guardrail: kalau sitasi keluar dari range → fallback aman
    if (
        used_ctx > 0
        and out
        and ("Tidak ditemukan pada dokumen" not in out)
        and (not _citations_in_range(out, used_ctx))
    ):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts):
            return _extractive_fallback_from_contexts(contexts, used_ctx)
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # 7) Guardrail: format tidak valid → fallback aman
    if not _is_valid_answer_format(out) or (not _bukti_has_bullets_or_dash(out)):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts):
            return _extractive_fallback_from_contexts(contexts, used_ctx)
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    return out


def build_generation_meta(
    *,
    question: str,
    contexts: List[str],
    answer_text: str,
    used_ctx: int,
) -> Dict[str, Any]:
    """
    Metadata ringan untuk logging/audit.
    Tidak mengubah jawaban, hanya menganalisis:
    - format output (Jawaban/Bukti)
    - keberadaan sitasi dan range CTX
    - status akhir (ok / not_found / invalid_format / invalid_citation / error / empty)
    - sinyal relevansi (heuristik overlap keyword)

    Tambahan:
    - bukti_count: jumlah bullet Bukti yang terbaca
    - grounding_ok: semua bullet Bukti grounded ke CTX yang dirujuk
    """
    t = (answer_text or "").strip()
    supported = _looks_supported(question, contexts) if used_ctx > 0 else False

    bullets = _extract_bukti_lines(t)
    bukti_count = len(bullets) if bullets else 0

    # default meta
    meta: Dict[str, Any] = {
        "used_ctx": int(used_ctx),
        "supported_heuristic": bool(supported),
        "format_ok": False,
        "bukti_ok": False,
        "bukti_count": int(bukti_count),
        "citations_present": False,
        "citations_in_range": False,
        "grounding_ok": True,  # default True jika tidak ada bukti / Bukti: -
        "ctx_nums": [],
        "status": "empty" if not t else "ok",
    }

    if not t:
        return meta

    # status not_found
    if "tidak ditemukan pada dokumen" in t.lower():
        meta["status"] = "not_found"
        # tetap isi validasi agar bisa dianalisis
        meta["format_ok"] = _is_valid_answer_format(t)
        meta["bukti_ok"] = _bukti_has_bullets_or_dash(t)
        meta["citations_present"] = _has_citation(t)
        meta["ctx_nums"] = _extract_ctx_nums(t)
        meta["citations_in_range"] = _citations_in_range(t, used_ctx)
        meta["grounding_ok"] = True
        return meta

    # validasi normal
    meta["format_ok"] = _is_valid_answer_format(t)
    meta["bukti_ok"] = _bukti_has_bullets_or_dash(t)
    meta["citations_present"] = _has_citation(t)
    meta["ctx_nums"] = _extract_ctx_nums(t)
    meta["citations_in_range"] = _citations_in_range(t, used_ctx)

    # grounding check (hanya kalau ada bullet & ada konteks)
    if used_ctx > 0 and bullets:
        meta["grounding_ok"] = _bukti_grounding_ok(t, contexts, used_ctx)
    else:
        meta["grounding_ok"] = True

    # turunkan status jika ada masalah
    if not meta["format_ok"] or not meta["bukti_ok"]:
        meta["status"] = "invalid_format"
    elif used_ctx > 0 and (not meta["citations_present"]):
        meta["status"] = "missing_citation"
    elif used_ctx > 0 and (not meta["citations_in_range"]):
        meta["status"] = "invalid_citation_range"
    elif used_ctx > 0 and (not meta["grounding_ok"]):
        meta["status"] = "weak_grounding"
    else:
        meta["status"] = "ok"

    return meta


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
    - Tambahan V1.5+: quality gate, kalo rewrite terlalu umum -> fallback heuristic
    """
    q = (question or "").strip()
    if not q:
        return ""

    if not history_pairs:
        return q

    # fallback heuristic (tanpa LLM)
    def _heuristic() -> str:
        last = history_pairs[-1] if history_pairs else {}
        last_q = str(last.get("query", "")).strip()
        last_a = str(last.get("answer_preview", "")).strip()

        focus = (last_a or last_q).strip()
        if len(focus) > 260:
            focus = focus[:260] + "..."

        if not focus:
            return q

        # gabungkan ringkas agar embedding tetap relevan
        return f"{q}\n\n(Konteks sebelumnya: {focus})"

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
                "3) Jangan menambah fakta baru.\n"
                "4) Jangan menghasilkan pertanyaan yang terlalu umum (mis. hanya 'Apa tahapannya?').\n",
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

    # ===== Quality gate: kalau rewrite terlalu lemah/umum, fallback heuristic =====
    def _focus_terms(txt: str, max_terms: int = 8) -> list[str]:
        toks = re.findall(r"[a-zA-Z]{3,}", (txt or "").lower())
        stop = set(_STOP_QA) | {"jawaban", "bukti", "pertanyaan", "konteks", "sebelumnya"}
        out_terms: list[str] = []
        seen: set[str] = set()
        for t in toks:
            if t in stop or t in seen:
                continue
            seen.add(t)
            out_terms.append(t)
            if len(out_terms) >= max_terms:
                break
        return out_terms

    def _is_weak_rewrite(rewrite: str) -> bool:
        r = (rewrite or "").strip()
        if not r:
            return True

        # terlalu pendek / terlalu generik
        if len(r.split()) < 4:
            return True

        # masih mengandung kata ganti yang biasanya butuh referen
        if re.search(r"\b(itu|tersebut|ini|nya)\b", r.lower()):
            return True

        # harus membawa minimal 1 kata fokus dari turn sebelumnya (query/answer_preview)
        last = history_pairs[-1] if history_pairs else {}
        last_q = str(last.get("query", "")).strip()
        last_a = str(last.get("answer_preview", "")).strip()
        focus = _focus_terms(f"{last_q} {last_a}")

        if focus and (not any(t in r.lower() for t in focus)):
            return True

        return False

    if _is_weak_rewrite(out):
        return _heuristic()

    return out
