"""
generate_utils.py
=================
Semua implementasi helper untuk RAG generation.
generate.py adalah thin re-export wrapper dari modul ini.

Diimpor di generate.py:
    from src.rag.generate_utils import generate_answer, build_generation_meta, contextualize_question
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.rag.method_detection import (
    METHOD_LABELS as _METHOD_LABEL,
)
from src.rag.method_detection import (
    METHOD_PATTERNS as _METHOD_MATCH_PATTERNS,
)
from src.rag.method_detection import (
    has_steps_signal as _has_steps_signal_local,
)
from src.rag.method_detection import (
    pretty_method_name as _nice_method,
)


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

        # NOTE:
        # Pylance sering “false positive” karena signature ChatOpenAI beda antar versi.
        # Pakai kwargs dict agar linting aman + tetap kompatibel lintas versi.
        kwargs_new: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
            "model_kwargs": {"max_tokens": max_tokens},
            # api_key sengaja tidak dipass; langchain_openai bisa membaca dari env
        }

        try:
            return ChatOpenAI(**kwargs_new)
        except TypeError:
            # fallback untuk versi yang pakai nama parameter lama
            kwargs_old: Dict[str, Any] = {
                "model_name": model,
                "temperature": temperature,
                "request_timeout": timeout,
                "max_retries": max_retries,
                "max_tokens": max_tokens,
                # api_key tetap dari env
            }
            try:
                return ChatOpenAI(**kwargs_old)
            except TypeError:
                # fallback terakhir: biarkan default dari env/library
                return ChatOpenAI()

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


# ===== Citation helpers =====
# Terima [CTX 2], [CTX2], (CTX 2), CTX 2
_CITATION_RE = re.compile(r"(\[CTX\s*\d+\]|\(CTX\s*\d+\)|\bCTX\s*\d+\b)", re.IGNORECASE)
_CTX_NUM_RE = re.compile(r"\[CTX\s*(\d+)\]", re.IGNORECASE)


def _normalize_citations(text: str) -> str:
    """Ubah variasi sitasi jadi format standar [CTX n] + bersihkan noise [angka] non-CTX."""
    t = text or ""
    # (CTX 2) -> [CTX 2]
    t = re.sub(r"\(CTX\s*(\d+)\)", r"[CTX \1]", t, flags=re.IGNORECASE)
    # CTX 2 -> [CTX 2] (jaga agar tidak dobel)
    t = re.sub(r"(?<!\[)\bCTX\s*(\d+)\b(?!\])", r"[CTX \1]", t, flags=re.IGNORECASE)

    # Hapus referensi bracket angka yang BUKAN CTX, contoh: [34], [36]
    # Tetap biarkan [CTX 1] aman.
    t = re.sub(r"\[(?!\s*CTX\b)\s*\d{1,4}\s*\]", "", t, flags=re.IGNORECASE)

    # rapikan spasi bekas penghapusan
    t = re.sub(r"\s{2,}", " ", t).replace(" ]", "]").replace("[ ", "[").strip()
    return t


def _has_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def _extract_ctx_nums(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CTX_NUM_RE.finditer(text or "")]


def _citations_in_range(out: str, used_ctx: int) -> bool:
    nums = _extract_ctx_nums(out)
    if used_ctx <= 0:
        # kalau tidak ada konteks, seharusnya tidak ada sitasi
        return len(nums) == 0
    return all(1 <= n <= used_ctx for n in nums)


def _is_valid_answer_format(out: str) -> bool:
    t = (out or "").strip()
    if not t:
        return False
    has_j = re.search(r"^\s*Jawaban\s*:", t, flags=re.IGNORECASE | re.MULTILINE) is not None
    has_b = re.search(r"^\s*Bukti\s*:", t, flags=re.IGNORECASE | re.MULTILINE) is not None
    return has_j and has_b


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

    # Fix: strip "JAWABAN: RAD:" / "JAWABAN RAD:" → "RAD:"
    # Case-SENSITIVE (tidak memakai flag i) agar tidak menyentuh "Jawaban:" yang valid
    # Menangkap dua varian: dengan colon setelah JAWABAN ("JAWABAN: RAD:") atau tanpa ("JAWABAN RAD:")
    t = re.sub(r"(?m)^JAWABAN\s*:?\s*([A-Z][A-Za-z_ ]+?)\s*:", r"\1:", t).strip()

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
        return f"{head}\nBukti:\n{tail}"

    # kalau bukti berupa beberapa baris tapi tanpa dash
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return f"{head}\nBukti: -"

    # ubah setiap baris menjadi bullet
    bullets = "\n".join([f"- {ln.lstrip('-').strip()}" for ln in lines])
    return f"{head}\nBukti:\n{bullets}"


# =========================
# Global-not-found detector
# =========================
# Tujuan: bedakan
# - GLOBAL not found: Jawaban: Tidak ditemukan pada dokumen. + Bukti: -
# - PARTIAL not found: mis. "RAD: Tidak ditemukan..." tapi metode lain ada isi
_NOT_FOUND_GLOBAL_RE = re.compile(
    r"(?is)^\s*Jawaban\s*:\s*Tidak\s+ditemukan\s+pada\s+dokumen\.\s*" r"Bukti\s*:\s*-\s*$"
)


def _is_global_not_found(out: str) -> bool:
    """
    True hanya jika output benar-benar GLOBAL not found.
    BUKAN hanya karena ada substring 'Tidak ditemukan pada dokumen' di salah satu baris.
    """
    t = (out or "").strip()
    if not t:
        return True

    # rapikan agar pola stabil
    t = _strip_question_preamble(t)
    t = _normalize_citations(t).strip()
    t = _enforce_qa_format(t).strip()
    t = _ensure_bukti_bullets(t).strip()

    # normalisasi bentuk "Bukti:\n-" menjadi "Bukti: -" agar regex stabil
    t = re.sub(r"(?im)^\s*Bukti\s*:\s*\n\s*-\s*$", "Bukti: -", t).strip()

    return bool(_NOT_FOUND_GLOBAL_RE.match(t))


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


def _method_line_content(out: str, method: str) -> str:
    """
    Ambil isi line metode di bagian Jawaban:
        RAD: <isi>
    return "" jika tidak ada.
    """
    t = (out or "").strip()
    if not t:
        return ""
    head = _jawaban_head_only(t)
    # buang label Jawaban:
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).strip()

    lab = _nice_method(method)
    m = re.search(rf"(?im)^\s*{re.escape(lab)}\s*:\s*(.*)$", head_body)
    return (m.group(1) or "").strip() if m else ""


def _method_has_real_content(out: str, method: str) -> bool:
    """
    True kalau line metode ada dan isinya bukan kosong, bukan '-' dan bukan 'Tidak ditemukan...'
    """
    c = _method_line_content(out, method)
    if not c:
        return False
    c_low = c.lower().strip()
    if c_low in {"-", "—"}:
        return False
    if "tidak ditemukan pada dokumen" in c_low:
        return False
    return True


def _prune_missing_methods_against_answer(
    out: str, target_methods: List[str], missing_methods: List[str]
) -> List[str]:
    """
    Anti-kontradiksi:
    - Jika sebuah metode sudah punya konten nyata di Jawaban, jangan izinkan masuk missing_methods.
    """
    missing_set = {m for m in (missing_methods or []) if m}
    for m in target_methods or []:
        if not m:
            continue
        if m in missing_set and _method_has_real_content(out, m):
            missing_set.remove(m)
    return [m for m in (missing_methods or []) if m in missing_set]


def _find_ctx_idx_for_method(method: str, contexts: List[str], used_ctx: int) -> Optional[int]:
    pats = _METHOD_MATCH_PATTERNS.get(method, [])
    if not pats:
        return None
    for i, c in enumerate((contexts or [])[:used_ctx], start=1):
        hay = c or ""
        for p in pats:
            if re.search(p, hay, flags=re.IGNORECASE):
                return i
    return None


def _quote_snippet(text: str, patterns: List[str], max_len: int = 220) -> str:
    """
    Ambil potongan kalimat terdekat yang mengandung pattern.
    Kalau tidak ketemu, fallback ke prefix text.
    """
    s = " ".join((text or "").split()).strip()
    if not s:
        return ""

    low = s.lower()
    hit_pos = None

    for p in patterns or []:
        m = re.search(p, low, flags=re.IGNORECASE)
        if m:
            hit_pos = m.start()
            break

    if hit_pos is None:
        return s[:max_len] + ("..." if len(s) > max_len else "")

    start = max(0, hit_pos - 80)
    end = min(len(s), hit_pos + 140)
    sn = s[start:end].strip()

    if len(sn) > max_len:
        sn = sn[:max_len] + "..."
    return sn


def _move_inline_cited_bullets_to_bukti(out: str) -> str:
    """
    Jika Bukti kosong / '-' tapi di bagian Jawaban terdapat bullet '- ... [CTX n]',
    maka pindahkan bullet tersebut ke Bukti agar format stabil.
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return t

    head, tail = t.split("Bukti:", 1)
    tail_body = (tail or "").strip()

    # Hanya lakukan jika Bukti kosong/placeholder
    if tail_body and tail_body != "-":
        return t

    head_lines = head.splitlines()
    moved: List[str] = []
    kept: List[str] = []

    for ln in head_lines:
        ln2 = ln.strip()
        if ln2.startswith("-") and re.search(r"\[CTX\s*\d+\]", ln2, flags=re.IGNORECASE):
            moved.append(ln2)
        else:
            kept.append(ln)

    if not moved:
        return t

    new_head = "\n".join(kept).rstrip()
    new_bukti = "\n".join(moved).strip()
    if not new_bukti:
        new_bukti = "-"

    return new_head + "\nBukti:\n" + new_bukti


def _clean_inline_method_bullets_in_jawaban(out: str) -> str:
    """
    Bersihkan bullet "bukti" yang nyasar di bagian Jawaban (sebelum 'Bukti:').

    Kasus yang sering bikin berantakan pada 'TAHAP + multi-metode':
    - Jawaban berisi baris seperti:
        - (RAD) ...
        - (Extreme Programming) ...
        yang sebenarnya "bukti" tapi suka nyasar di Jawaban, sering tanpa [CTX n].
    Solusi:
    - Jika bullet punya [CTX n] -> pindahkan ke Bukti (append).
    - Jika bullet TIDAK punya [CTX n] dan bentuknya seperti bukti (prefix '(...)' / DOC: / dll) -> HAPUS dari Jawaban.
        (Karena tidak bisa "menebak" [CTX] yang benar.)
    """
    t = (out or "").strip()
    if not t:
        return t

    # harus sudah punya "Bukti:" (generate_answer selalu enforce ini)
    if "Bukti:" not in t:
        return t

    # split case-insensitive pakai regex yang sudah ada
    parts = _BUKTI_SPLIT_RE.split(t, maxsplit=1)
    head = (parts[0] or "").rstrip()
    tail = (parts[1] or "").strip() if len(parts) > 1 else ""

    head_lines = head.splitlines()
    kept_head: list[str] = []
    moved_to_bukti: list[str] = []

    for ln in head_lines:
        s = (ln or "").strip()

        # kandidat "bukti nyasar" hanya jikalau dia bullet
        if s.startswith("-"):
            has_ctx = bool(re.search(r"\[CTX\s*\d+\]", s, flags=re.IGNORECASE))
            # bullet yang sangat khas "bukti" (bukan list langkah biasa)
            looks_like_evidence = bool(
                re.match(r"^\-\s*\(\s*[A-Za-z_ ]+\s*\)", s)  # - (RAD) ...
                or re.search(r"\bDOC\s*:", s, flags=re.IGNORECASE)  # DOC:
                or re.search(r"\[CTX\s*\d+\]", s, flags=re.IGNORECASE)  # ada CTX
            )

            if looks_like_evidence:
                if has_ctx:
                    # pindahkan ke Bukti
                    moved_to_bukti.append(s)
                # kalau tidak ada CTX -> dibuang saja (anti-ngarang)
                continue

        kept_head.append(ln)

    # kalau tidak ada yang diubah, return original
    if (not moved_to_bukti) and (kept_head == head_lines):
        return t

    # rapikan bukti existing
    tail_lines: list[str] = []
    if tail and tail != "-":
        tail_lines = [x.strip() for x in tail.splitlines() if x.strip()]
    # kalau Bukti: - dan ada moved bullet -> hilangkan '-' placeholder
    if tail == "-" and moved_to_bukti:
        tail_lines = []

    # append moved bullet (dedupe nanti ditangani _cap_and_dedupe_bukti)
    combined = tail_lines + moved_to_bukti
    if not combined:
        combined = ["-"]

    new_head = "\n".join(kept_head).rstrip()
    new_tail = "\n".join(combined).strip()
    return f"{new_head}\nBukti:\n{new_tail}"


def _deterministic_method_answer(
    question: str,
    targets: List[str],
    contexts: List[str],
    used_ctx: int,
    *,
    max_bullets: int,
) -> str:
    """
    Jawaban deterministik untuk pertanyaan tipe METODE (biar stabil).
    Hanya memakai metode yang benar-benar ketemu di konteks.
    """
    if used_ctx <= 0 or not contexts:
        return ""

    found: List[Tuple[str, int]] = []
    for m in targets or []:
        if not m:
            continue
        idx = _find_ctx_idx_for_method(m, contexts, used_ctx)
        if idx:
            found.append((m, idx))

    if not found:
        return ""

    # dedupe berdasarkan CTX index
    seen = set()
    uniq: List[Tuple[str, int]] = []
    for m, idx in found:
        if idx in seen:
            continue
        seen.add(idx)
        uniq.append((m, idx))

    uniq = uniq[: max(1, min(int(max_bullets), 8))]

    names = ", ".join(_nice_method(m) for m, _ in uniq)
    ql = (question or "").lower()
    is_list = bool(re.search(r"\bapa\s+saja\b", ql))

    if len(uniq) == 1 and (not is_list):
        jaw = f"Jawaban: Metode pengembangan yang digunakan adalah {names}."
    else:
        jaw = f"Jawaban: Metode pengembangan yang teridentifikasi pada konteks adalah {names}."

    bullets: List[str] = []
    for m, idx in uniq:
        raw = contexts[idx - 1] or ""
        pats = _METHOD_MATCH_PATTERNS.get(m, [])
        sn = _quote_snippet(raw, pats, max_len=220)
        bullets.append(f"- ({_nice_method(m)}) {sn} [CTX {idx}]")

    return jaw + "\nBukti:\n" + "\n".join(bullets)


def _strip_doc_header_line(x: str) -> str:
    # hapus "DOC: ...\n" agar ekstraksi tahapan lebih bersih
    return re.sub(r"(?im)^DOC:\s.*\n", "", (x or "")).strip()


def _looks_like_tahapan(item: str) -> bool:
    """
    Poin 3: Heuristik untuk membedakan tahapan/langkah eksplisit
    dari karakteristik/kelebihan/deskripsi umum suatu metode.

    - True  jika mengandung stage-keyword  → jelas tahapan
    - False jika mengandung characteristic-keyword tanpa stage-keyword → bukan tahapan
    - Default True (konservatif, biarkan lewat jika ambigu)
    """
    s = " ".join((item or "").split()).strip().lower()
    if not s or len(s) < 4:
        return False

    # Kata kunci yang khas muncul dalam nama/deskripsi tahapan proses pengembangan
    _STAGE_KWS = [
        "analisis", "perancangan", "implementasi", "pengujian", "evaluasi",
        "planning", "design", "coding", "testing", "deployment",
        "identifikasi", "pengumpulan", "penerapan", "pemodelan",
        "perencanaan", "pengkodean", "pemeliharaan", "penyebaran",
        "interaksi", "alur data", "prototype", "user interface",
        "kebutuhan", "requirement", "spesifikasi", "review", "feedback",
        "iterasi", "pengembangan sistem", "basis data", "antarmuka",
    ]
    if any(k in s for k in _STAGE_KWS):
        return True

    # Kata kunci yang khas muncul pada deskripsi karakteristik/kelebihan (bukan tahapan)
    _CHARACTERISTIC_KWS = [
        "singkat", "cepat", "fleksibel", "adaptif", "incremental",
        "iteratif", "efisien", "mudah", "sederhana", "ringkas",
        "bersifat", "merupakan ciri", "kelebihan", "keunggulan",
    ]
    if any(k in s for k in _CHARACTERISTIC_KWS):
        return False

    # Default: konservatif → anggap valid
    return True


def _extract_steps_from_text(text: str, max_items: int = 5) -> List[str]:
    """
    Ekstrak tahapan dari sebuah chunk secara heuristik aman (tanpa mengarang):
    - enumerasi: 1., 2), -, •
    - atau frasa pengantar daftar: 'meliputi/terdiri dari/yaitu/antara lain' lalu split
    """
    s = _strip_doc_header_line(text or "")
    if not s:
        return []

    # 1) enumerasi eksplisit
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    enum: List[str] = []
    for ln in lines:
        if re.match(r"^(\d+[\.\)]|[-•])\s+", ln):
            ln2 = re.sub(r"^(\d+[\.\)]|[-•])\s+", "", ln).strip()
            if ln2:
                enum.append(ln2)
    if len(enum) >= 2:
        return enum[:max_items]

    # 2) daftar setelah connector (lebih konservatif)
    m = re.search(
        r"\b(meliputi|terdiri(\s+dari)?|yaitu|antara\s+lain)\b\s*([^\.]{20,280})",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        tail = (m.group(3) or "").strip()
        parts = [p.strip(" ;:-\n\t") for p in re.split(r",|\bdan\b", tail) if p.strip()]
        cleaned: List[str] = []
        for p in parts:
            p2 = " ".join(p.split()).strip()
            # buang item terlalu pendek (agar tidak noise)
            if len(p2) >= 5:
                cleaned.append(p2)
            if len(cleaned) >= max_items:
                break
        if len(cleaned) >= 2:
            # validasi, hanya kembalikan item yang benar-benar tahapan
            # (bukan karakteristik/kelebihan seperti "waktu singkat", "adaptasi cepat")
            validated = [p for p in cleaned if _looks_like_tahapan(p)]
            if len(validated) >= 2:
                return validated[:max_items]
            # Kurang dari 2 item valid → anggap tidak ada tahapan eksplisit di CTX ini
            return []

    return []


def _candidate_ctx_indices_for_method_steps(
    method: str,
    contexts: List[str],
    used_ctx: int,
    *,
    max_candidates: int = 3,
) -> List[int]:
    """
    Ambil beberapa CTX kandidat untuk satu metode.
    Prioritas:
    1) method-hit + steps-signal
    2) method-hit saja
    """
    pats = _METHOD_MATCH_PATTERNS.get(method, [])
    if not pats:
        return []

    ranked: List[Tuple[int, int]] = []
    for i, c in enumerate((contexts or [])[:used_ctx], start=1):
        cc = c or ""
        if not any(re.search(p, cc, flags=re.IGNORECASE) for p in pats):
            continue

        priority = 0 if _has_steps_signal_local(cc) else 1
        ranked.append((priority, i))

    ranked.sort(key=lambda x: (x[0], x[1]))
    return [idx for _, idx in ranked[: max(1, int(max_candidates))]]


def _extract_steps_from_ctx_candidates(
    contexts: List[str],
    ctx_indices: List[int],
    *,
    max_items: int = 6,
) -> List[str]:
    """
    Gabungkan tahapan unik dari beberapa CTX kandidat.
    Tetap konservatif: hanya memakai hasil _extract_steps_from_text().
    """
    merged: List[str] = []
    seen: set[str] = set()

    for idx in ctx_indices or []:
        if not (1 <= idx <= len(contexts)):
            continue

        items = _extract_steps_from_text(contexts[idx - 1] or "", max_items=max_items)
        for item in items:
            norm = re.sub(r"\s+", " ", (item or "").strip()).lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(item.strip())

            if len(merged) >= int(max_items):
                return merged

    return merged


def _find_best_ctx_for_method_steps(
    method: str, contexts: List[str], used_ctx: int
) -> Optional[int]:
    """
    Pilih CTX terbaik untuk tahapan metode:
    prefer: (metode match) + (ada sinyal tahapan).
    """
    candidates = _candidate_ctx_indices_for_method_steps(
        method,
        contexts,
        used_ctx,
        max_candidates=1,
    )
    return candidates[0] if candidates else None


def _partial_steps_note(*, is_comparison: bool = False) -> str:
    base = (
        "Konteks final mengonfirmasi metode ini dibahas, tetapi daftar tahapan eksplisitnya "
        "tidak tampil pada potongan yang tersedia."
    )
    if is_comparison:
        return base + " Perbedaannya dengan metode lain hanya bisa dijelaskan secara parsial."
    return base


def _deterministic_steps_answer(
    *,
    question: str,
    targets: List[str],
    contexts: List[str],
    used_ctx: int,
    max_bullets: int,
    max_items_per_method: int = 5,
) -> str:
    """
    Jawaban deterministik untuk pertanyaan TAHAP + multi-metode:
    - stabil (tidak bergantung pada format LLM)
    - tidak mengarang (hanya ringkas isi chunk)
    """
    if used_ctx <= 0 or not contexts or len(targets) < 2:
        return ""

    max_items_per_method = int(max(2, min(max_items_per_method, 6)))
    max_bullets = int(max(1, min(max_bullets, 8)))

    jaw_lines: List[str] = ["Jawaban:"]
    bukti_lines: List[str] = []

    is_comparison = bool(
        re.search(r"\b(perbedaan|beda|bandingkan|dibandingkan|versus|vs)\b", (question or "").lower())
    )

    for m in targets:
        lab = _nice_method(m)
        ctx_candidates = _candidate_ctx_indices_for_method_steps(
            m, contexts, used_ctx, max_candidates=3
        )
        idx = ctx_candidates[0] if ctx_candidates else None

        if not idx:
            jaw_lines.append(f"{lab}: Tidak ditemukan pada dokumen.")
            continue

        steps = _extract_steps_from_ctx_candidates(
            contexts,
            ctx_candidates,
            max_items=max_items_per_method,
        )

        if len(steps) >= 2:
            joined = "; ".join(steps[:max_items_per_method])
            jaw_lines.append(f"{lab}: {joined} [CTX {idx}]")
            bukti_lines.append(f"- ({lab}) Tahapan yang disebutkan: {joined} [CTX {idx}]")
        else:
            note = (
                "Konteks final mengonfirmasi metode ini, tetapi daftar tahapan eksplisitnya tidak tampil "
                "pada potongan yang tersedia."
            )
            if is_comparison:
                note += " Perbedaannya dengan metode lain hanya bisa dijelaskan secara parsial."
            jaw_lines.append(f"{lab}: {note}")
            bukti_lines.append(f"- ({lab}) {note} [CTX {idx}]")

    # cap bukti
    bukti_lines = bukti_lines[:max_bullets]
    if not bukti_lines:
        return "\n".join(jaw_lines).strip() + "\nBukti: -"

    return "\n".join(jaw_lines).strip() + "\nBukti:\n" + "\n".join(bukti_lines)


def _structured_steps_answer_llm(
    *,
    llm: Any,
    question: str,
    targets: List[str],
    contexts: List[str],
    used_ctx: int,
    max_bullets: int,
) -> str:
    """
    Khusus intent TAHAP + multi-metode saat deterministic_steps_answer=False.
    Strategi:
    - Ekstrak langkah/tahapan per metode dari CTX (tanpa mengarang)
    - LLM hanya memoles bahasa + bridging + format (tanpa melihat konteks mentah)
    - Bukti dibangun deterministik agar:
        (a) tidak sama persis dengan Jawaban
        (b) ada bukti per metode yang memang punya dukungan
    """
    if used_ctx <= 0 or not contexts or len(targets) < 2:
        return ""

    max_bullets = int(max(1, min(max_bullets, 8)))

    # apakah user meminta "jelaskan/uraikan" -> beri sedikit elaborasi (tanpa menambah fakta baru)
    ql = (question or "").lower()
    wants_explain = bool(re.search(r"\b(jelaskan|uraikan|paparkan)\b", ql))

    is_comparison = bool(
        re.search(r"\b(perbedaan|beda|bandingkan|dibandingkan|versus|vs)\b", ql)
    )

    # -------- 1) Ekstrak fakta aman per metode --------
    facts: List[Dict[str, Any]] = []
    for m in targets:
        lab = _nice_method(m)
        ctx_candidates = _candidate_ctx_indices_for_method_steps(
            m, contexts, used_ctx, max_candidates=3
        )
        idx = ctx_candidates[0] if ctx_candidates else None

        if not idx:
            facts.append(
                {"method": lab, "ctx": None, "steps": [], "note": "Tidak ditemukan pada dokumen."}
            )
            continue

        steps = _extract_steps_from_ctx_candidates(
            contexts,
            ctx_candidates,
            max_items=6,
        )

        if len(steps) >= 2:
            facts.append({"method": lab, "ctx": idx, "steps": steps[:6], "note": ""})
        else:
            # Metode disebut, tapi urutan langkah tidak eksplisit pada potongan konteks
            facts.append(
                {
                    "method": lab,
                    "ctx": idx,
                    "steps": [],
                    "note": _partial_steps_note(is_comparison=is_comparison),
                }
            )

    # jikalau semua metode ctx None -> benar-benar tidak ada dukungan
    if not any(f.get("ctx") for f in facts):
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # -------- 2) Siapkan ringkasan fakta untuk LLM (tanpa konteks mentah) --------
    # Format ini sengaja “kering” agar LLM tidak bisa menyalin teks dokumen.
    fact_lines: List[str] = []
    for f in facts:
        method = f["method"]
        ctx = f.get("ctx")
        steps = f.get("steps") or []
        note = (f.get("note") or "").strip()

        fact_lines.append(f"METODE: {method}")
        fact_lines.append(f"CTX: {ctx if ctx is not None else '-'}")
        if steps:
            fact_lines.append("TAHAPAN (persis dari CTX, jangan ditambah):")
            for s in steps:
                fact_lines.append(f"- {s}")
        else:
            fact_lines.append(f"CATATAN (persis): {note}")
        fact_lines.append("")  # spacer

    facts_block = "\n".join(fact_lines).strip()

    style_note = (
        'Tambahkan 1 kalimat ringkas setelah daftar tahapan untuk menjawab nuansa "jelaskan" '
        "(tanpa menambah fakta baru)."
        if wants_explain
        else "Cukup bridging + daftar tahapan."
    )

    human_msg = (
        "Pertanyaan user:\n{question}\n\n"
        "FAKTA (satu-satunya sumber yang boleh dipakai):\n{facts}\n\n"
        "Catatan gaya:\n- " + style_note + "\n\n"
        "Tulis bagian JAWABAN sekarang:"
    )

    # -------- 3) LLM: susun Jawaban yang lebih manusiawi & rapi --------
    # Output LLM DIHANYAKAN bagian Jawaban (tanpa Bukti) agar tidak duplikasi.
    style_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Kamu menyusun bagian JAWABAN untuk RAG dokumen akademik.\n"
                "PENTING: kamu hanya boleh memakai FAKTA yang diberikan (bukan menebak dari luar).\n\n"
                "Aturan format (wajib):\n"
                "1) Output TIDAK boleh mengandung 'Bukti:' sama sekali.\n"
                "2) Tulis TERPISAH per metode sesuai urutan kemunculan pada FAKTA.\n"
                "3) Format per metode:\n"
                "   <METODE>:\n"
                "   <kalimat bridging singkat>\n"
                "   1. <tahap 1>\n"
                "   2. <tahap 2>\n"
                "   (dst, tiap item di baris baru)\n"
                "   [CTX n]\n"
                "   (kosongkan 1 baris antar metode)\n"
                "4) Jika sebuah metode tidak punya daftar tahapan eksplisit tetapi masih punya CTX pendukung,\n"
                "   tulis secara jujur bahwa metode tersebut terkonfirmasi dibahas, namun daftar tahap eksplisit\n"
                "   tidak tampil pada potongan konteks yang tersedia.\n"
                "   Jangan mengganti kondisi ini menjadi 'Tidak ditemukan pada dokumen.' bila CTX pendukung ada.\n"
                "   Format:\n"
                "   <METODE>:\n"
                "   <bridging singkat> + <CATATAN>\n"
                "   [CTX n] (jika CTX ada) atau tanpa sitasi jika CTX '-'\n"
                "5) Jangan menyalin teks mentah dokumen, jangan menulis 'DOC:', nomor subbab, atau angka halaman.\n"
                "6) Bridging harus ramah dibaca.\n"
                "7) Gunakan penomoran '1.', '2.', '3.' (bukan '1)').\n",
            ),
            ("human", human_msg),
        ]
    )

    jawaban_body = (style_prompt | llm | StrOutputParser()).invoke(
        {"question": (question or "").strip(), "facts": facts_block}
    )
    jawaban_body = (jawaban_body or "").strip()

    # Guard kecil: jikalau LLM masih memasukkan "Bukti:"
    jawaban_body = re.split(r"(?im)^\s*Bukti\s*:\s*", jawaban_body)[0].rstrip()

    # -------- Override paksa untuk metode yang TIDAK punya tahapan --------
    # LLM terkadang mengabaikan constraint CATATAN dan mengarang tahapan.
    # Solusi: setelah LLM generate, scan line-by-line dan replace section metode
    # tanpa tahapan dengan "Tidak ditemukan pada dokumen." secara deterministik.
    facts_labels = [f["method"] for f in facts]

    def _normalize_method_headers(text: str, method_labels: List[str]) -> str:
        """
        Pre-processor: pastikan setiap header metode berada di awal baris baru.
        LLM terkadang menulis header metode di akhir baris sebelumnya, contoh:
        "...4. Pengujian dan implementasi Prototyping:\nPrototyping adalah..."
        Fungsi ini menyisipkan '\n' sebelum header yang tidak berada di awal baris.
        """
        if not method_labels or not text:
            return text
        for label in method_labels:
            escaped = re.escape(label)
            # Match "<label>:" yang didahului karakter selain newline (= inline/mid-line)
            # Sisipkan newline SEBELUM label tsb
            text = re.sub(
                rf"(?m)(?<=[^\n])(\s*)({escaped}\s*:)",
                r"\n\2",
                text,
                flags=re.IGNORECASE,
            )
        return text

    def _override_no_steps_sections(body: str) -> str:
        """
        Untuk setiap metode dalam facts yang steps=[], paksa isi section-nya
        menjadi "Tidak ditemukan pada dokumen." - menghapus apapun yang LLM karang.
        """
        lines_in = body.splitlines()
        lines_out: List[str] = []
        skip_until_next_header = False
        override_text: Optional[str] = None

        # Regex: header metode manapun dari facts (case-insensitive)
        all_labels_re = re.compile(
            r"(?i)^(" + "|".join(re.escape(m) for m in facts_labels) + r")\s*:\s*(.*)?$"
        )

        for ln in lines_in:
            stripped = ln.strip()
            m_hdr = all_labels_re.match(stripped)

            if m_hdr:
                # header metode
                cur_label = m_hdr.group(1).strip()
                # Cari apakah metode ini punya tahapan
                fact_match = next(
                    (f for f in facts if f["method"].lower() == cur_label.lower()), None
                )
                has_steps = bool(fact_match and (fact_match.get("steps") or []))

                if not has_steps:
                    # Override: tulis "Tidak ditemukan" dan skip konten berikutnya
                    ctx_val = fact_match.get("ctx") if fact_match else None
                    note = (fact_match.get("note") or "").strip() if fact_match else ""

                    if note and ctx_val:
                        override_text = f"{cur_label}: {note}"
                    else:
                        override_text = f"{cur_label}: Tidak ditemukan pada dokumen."

                    lines_out.append(override_text)
                    skip_until_next_header = True
                else:
                    # Metode punya tahapan → tulis header asli, stop skip
                    lines_out.append(ln)
                    skip_until_next_header = False

            elif skip_until_next_header:
                # Dalam section metode yang di-override → skip konten yang dikarang LLM
                # Kecuali blank line (biarkan untuk spacing)
                if not stripped:
                    lines_out.append(ln)
                # else: buang baris konten yang dikarang LLM (hallucination)

            else:
                lines_out.append(ln)

        return "\n".join(lines_out)

    # Pre-processor: pisahkan header metode yang muncul inline ke baris baru
    # WAJIB sebelum _override_no_steps_sections agar line-by-line matching akurat
    jawaban_body = _normalize_method_headers(jawaban_body, facts_labels)
    jawaban_body = _override_no_steps_sections(jawaban_body)
    jawaban_body = jawaban_body.strip()

    # -------- 4) Bukti: buat deterministik per metode (tidak sama dengan jawaban) --------
    bukti: List[str] = []
    for f in facts:
        method = f["method"]
        ctx = f.get("ctx")
        steps = f.get("steps") or []
        note = (f.get("note") or "").strip()

        if ctx is None:
            continue

        if steps:
            # ringkas (beda format dengan Jawaban)
            sn = "; ".join(steps[:4])
            bukti.append(f"- ({method}) Tahapan yang disebutkan: {sn} [CTX {int(ctx)}]")
        else:
            bukti.append(f"- ({method}) {note} [CTX {int(ctx)}]")

    bukti = bukti[:max_bullets]
    bukti_block = "\n".join(bukti) if bukti else "-"

    out = "Jawaban:\n" + jawaban_body.strip() + "\n\nBukti:\n" + bukti_block

    # normalisasi standar supaya konsisten
    out = _strip_question_preamble(out)
    out = _normalize_citations(out)
    out = _enforce_qa_format(out)
    out = _ensure_bukti_bullets(out)
    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
    return out


def _humanize_multimethod_steps_answer(
    out: str,
    *,
    question: str,
    target_methods: List[str],
    contexts: List[str],
    used_ctx: int,
    max_items_per_method: int = 5,
) -> str:
    """
    Humanize khusus TAHAP + multi-metode:
    - Jikalau Jawaban terlihat copy-paste Bukti / hanya daftar dipisah ';',
        --> ubah menjadi format bernomor per metode.
    - Tidak menambah fakta: hanya memformat ulang isi yang sudah ada,
        --> atau mengekstrak tahapan secara aman dari CTX yang dirujuk.
    - Bukti dibiarkan apa adanya (supaya evidence tetap kuat).
    """
    t = (out or "").strip()
    if not t or ("Bukti:" not in t) or (not target_methods) or used_ctx <= 0:
        return out

    head, tail = _split_head_tail(t)
    if not head.strip():
        return out

    # deteksi cepat: Jawaban terlalu mirip Bukti / banyak ';'
    bukti_bullets = _extract_bukti_lines(t)
    hb = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).strip()
    hb_slim = re.sub(r"\[CTX\s*\d+\]", "", hb, flags=re.IGNORECASE)
    hb_slim = " ".join(hb_slim.split()).lower()

    bb_slim = re.sub(r"\[CTX\s*\d+\]", "", " ".join(bukti_bullets or []), flags=re.IGNORECASE)
    bb_slim = " ".join(bb_slim.split()).lower()

    # buang label metode dari hb_slim agar deteksi substring lebih adil
    for m in target_methods:
        lab = _nice_method(m).lower() + ":"
        hb_slim = hb_slim.replace(lab, "")

    needs = False
    if len(hb_slim) >= 50 and hb_slim and bb_slim and (hb_slim in bb_slim):
        needs = True
    if hb.count(";") >= 2:
        needs = True

    if not needs:
        return out

    # pastikan head memiliki "Jawaban:"
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).rstrip("\n")
    lines = head_body.splitlines()

    labels = [_nice_method(m) for m in target_methods]
    hdr_re = re.compile(
        r"(?im)^\s*(?:-\s*)?("
        + "|".join(re.escape(lb) for lb in sorted(labels, key=len, reverse=True))
        + r")\s*:\s*(.*)\s*$"
    )

    prefix: List[str] = []
    sections: Dict[str, List[str]] = {}
    cur_label: Optional[str] = None

    for ln in lines:
        m = hdr_re.match(ln or "")
        if m:
            cur_label = (m.group(1) or "").strip() or None
            if cur_label is not None:
                sections[cur_label] = [ln]
            continue
        if cur_label is None:
            prefix.append(ln)
        else:
            sections.setdefault(cur_label, []).append(ln)

    # helper: cari CTX utama dari konten (atau dari bukti)
    def _pick_ctx_for_method(method_key: str, content: str) -> Optional[int]:
        nums = [n for n in _extract_ctx_nums(content) if 1 <= n <= used_ctx]
        if nums:
            return nums[0]

        # fallback dari bukti yang terdeteksi metodenya
        for b in bukti_bullets or []:
            mm = _bullet_detect_method(b, target_methods)
            if mm == method_key:
                nums2 = [n for n in _extract_ctx_nums(b) if 1 <= n <= used_ctx]
                if nums2:
                    return nums2[0]
        return None

    # helper: ekstrak item langkah dari teks
    def _steps_from_content(content_no_ctx: str, ctx_primary: Optional[int]) -> List[str]:
        # 1) split ';' jika cukup
        parts = [p.strip(" \t-•;:,") for p in (content_no_ctx or "").split(";") if p.strip()]
        parts = [" ".join(p.split()).strip() for p in parts if len(p.strip()) >= 4]
        if len(parts) >= 2:
            return parts[:max_items_per_method]

        # 2) fallback: ekstrak dari CTX asli (lebih aman & biasanya lebih lengkap)
        if ctx_primary and (1 <= ctx_primary <= used_ctx):
            return _extract_steps_from_text(
                contexts[ctx_primary - 1] or "", max_items=max_items_per_method
            )

        return []

    want_explain = bool(re.search(r"\b(jelaskan|uraikan|paparkan)\b", (question or "").lower()))

    rebuilt: List[str] = []
    if " ".join([x.strip() for x in prefix if x.strip()]).strip():
        rebuilt.append("\n".join(prefix).strip())

    for mkey in target_methods:
        lab = _nice_method(mkey)
        sec = sections.get(lab, [])
        inline = ""
        extra = []

        if sec:
            m0 = hdr_re.match(sec[0] or "")
            inline = (m0.group(2) or "").strip() if m0 else ""
            extra = [x.strip() for x in sec[1:] if (x or "").strip()]

        content = " ".join([inline] + extra).strip()
        content_low = content.lower()

        has_partial_note = bool(
            "daftar tahapan eksplisit" in content_low
            or "konteks final mengonfirmasi metode ini" in content_low
            or "perbedaannya dengan metode lain hanya bisa dijelaskan secara parsial" in content_low
        )

        # not-found tetap
        if not content:
            rebuilt.append(f"{lab}: Tidak ditemukan pada dokumen.")
            continue

        if "tidak ditemukan pada dokumen" in content_low:
            rebuilt.append(f"{lab}: Tidak ditemukan pada dokumen.")
            continue

        ctx_primary = _pick_ctx_for_method(mkey, content)

        # jikalau CTX ada namun tidak ada sinyal tahapan, jangan memakai "kelebihan" sebagai tahapan
        if ctx_primary and (1 <= ctx_primary <= used_ctx):
            if not _has_steps_signal_local(contexts[ctx_primary - 1] or ""):
                # masih diperbolehkan jikalau konten sendiri jelas-jelas berisi daftar langkah (';' >=2)
                if content.count(";") < 2 and (
                    not re.search(
                        r"\b(planning|design|coding|testing|implementasi|pengujian)\b", content_low
                    )
                ):
                    if has_partial_note:
                        rebuilt.append(f"{lab}: {content.strip()}")
                    else:
                        rebuilt.append(f"{lab}: Tidak ditemukan pada dokumen.")
                    continue

        # bersihkan CTX dari konten (untuk formatting)
        content_no_ctx = re.sub(r"\[CTX\s*\d+\]", "", content, flags=re.IGNORECASE).strip()
        content_no_ctx = " ".join(content_no_ctx.split()).strip(" ;,.-")

        steps = _steps_from_content(content_no_ctx, ctx_primary)

        if len(steps) < 2:
            if has_partial_note:
                rebuilt.append(f"{lab}: {content.strip()}")
            else:
                rebuilt.append(f"{lab}: Tidak ditemukan pada dokumen.")
            continue

        # bangun section yang lebih manusiawi
        if want_explain:
            header = f"{lab}: Tahapan yang disebutkan pada konteks adalah sebagai berikut"
        else:
            header = f"{lab}: Tahapan yang disebutkan pada konteks meliputi"

        if ctx_primary:
            header = f"{header} [CTX {ctx_primary}]"
        else:
            header = f"{header}."

        sec_lines = [header]
        for i, it in enumerate(steps[:max_items_per_method], start=1):
            sec_lines.append(f"{i}) {it}")

        rebuilt.append("\n".join(sec_lines))

    new_body = "\n\n".join([x.strip() for x in rebuilt if x.strip()]).strip()
    new_head = "Jawaban:\n" + new_body if new_body else "Jawaban: Tidak ditemukan pada dokumen."
    return (new_head.strip() + "\n" + tail.strip()).strip()


def _bullet_detect_method(bullet: str, targets: List[str]) -> Optional[str]:
    """
    Deteksi metode sebuah bullet.
    Prefer jika bullet pakai prefix "(RAD)" dsb.
    """
    b = (bullet or "").strip()

    # Prefix (RAD) ... / (Prototyping) ...
    m = re.match(r"^\(?\s*([A-Za-z_ ]+?)\s*\)?\s*[:\-]", b)
    if m:
        head = m.group(1).strip().lower()
        for t in targets:
            if _nice_method(t).lower() == head:
                return t
            # toleransi
            if t.lower() in head.replace(" ", "_"):
                return t

    # Fallback: regex match di isi bullet
    bl = b.lower()
    for t in targets:
        pats = _METHOD_MATCH_PATTERNS.get(t, [])
        for p in pats:
            if re.search(p, bl, flags=re.IGNORECASE):
                return t
    return None


def _cap_bukti_per_method(
    out: str,
    *,
    target_methods: List[str],
    per_method_bullets: int,
    max_total_bullets: int,
) -> str:
    """
    Cap bukti:
    - maksimal per_method_bullets untuk tiap metode target
    - maksimal max_total_bullets total
    Bullet yang tidak terdeteksi metodenya masuk bucket OTHER (dipakai hanya jika slot masih ada).
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return t

    head, _ = t.split("Bukti:", 1)
    head = head.strip()

    bullets = _extract_bukti_lines(t)
    if not bullets:
        return f"{head}\nBukti: -"

    per_method_bullets = int(max(1, min(per_method_bullets, 3)))
    max_total_bullets = int(max(1, min(max_total_bullets, 8)))

    buckets: Dict[str, List[str]] = {m: [] for m in target_methods}
    other: List[str] = []

    for b in bullets:
        m = _bullet_detect_method(b, target_methods)
        if m and m in buckets:
            buckets[m].append(b)
        else:
            other.append(b)

    picked: List[str] = []

    # ambil per metode dulu (urut sesuai target_methods)
    for m in target_methods:
        picked.extend(buckets[m][:per_method_bullets])

    # isi sisa dari OTHER jika masih ada slot
    if len(picked) < max_total_bullets:
        remain = max_total_bullets - len(picked)
        picked.extend(other[:remain])

    picked = picked[:max_total_bullets]
    return _rebuild_with_bukti(head, picked)


# =========================
# Last Micro-fix helpers
# =========================

_METHOD_HEADER_ALIASES: Dict[str, str] = {
    # alias header yang kadang "nyasar" dari model
    "Scrum": "Agile",  # kalau target-nya AGILE
    "XP": "Extreme Programming",  # kalau target-nya EXTREME_PROGRAMMING
}


def _split_head_tail(out: str) -> Tuple[str, str]:
    """
    Split output menjadi head (Jawaban...) dan tail (mulai dari 'Bukti:').
    tail SELALU mengandung 'Bukti:' (fallback jika tidak ada).
    """
    t = (out or "").strip()
    if not t:
        return "", "Bukti: -"

    m = re.search(r"(?im)^\s*Bukti\s*:\s*", t)
    if m:
        head = t[: m.start()].rstrip()
        tail = t[m.start() :].lstrip()
        return head, tail

    return t.rstrip(), "Bukti: -"


def _prune_or_rename_non_target_method_sections(out: str, *, target_methods: List[str]) -> str:
    """
    - Hapus section metode "nyasar" (mis. Scrum:) ketika bukan target.
    - Jika itu alias metode target (Scrum -> Agile, XP -> Extreme Programming), rename header-nya.
    """
    t = (out or "").strip()
    if not t or not target_methods:
        return out

    head, tail = _split_head_tail(t)

    # Pastikan head punya "Jawaban:"
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    # Ambil body setelah "Jawaban:"
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).rstrip("\n")
    body_lines = head_body.splitlines()

    allowed_labels = {_nice_method(m) for m in (target_methods or []) if m}
    # label metode yang "dianggap metode" untuk pruning (bukan semua colon!)
    known_labels = (
        set(_METHOD_LABEL.values())
        | set(_METHOD_HEADER_ALIASES.keys())
        | set(_METHOD_HEADER_ALIASES.values())
    )

    # Regex header metode khusus known_labels saja
    hdr_re = re.compile(
        r"(?im)^\s*("
        + "|".join(re.escape(x) for x in sorted(known_labels, key=len, reverse=True))
        + r")\s*:\s*(.*)$"
    )

    prefix: List[str] = []
    blocks: List[Tuple[str, List[str]]] = []
    cur_label: Optional[str] = None
    cur_lines: List[str] = []

    def _flush():
        nonlocal cur_label, cur_lines
        if cur_label is None:
            if cur_lines:
                prefix.extend(cur_lines)
        else:
            blocks.append((cur_label, cur_lines))
        cur_label, cur_lines = None, []

    for ln in body_lines:
        m = hdr_re.match(ln or "")
        if m:
            _flush()
            cur_label = (m.group(1) or "").strip()
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    _flush()

    kept_lines: List[str] = []
    # prefix (non-metode) saya keep apa adanya
    kept_lines.extend(prefix)

    for label, lines in blocks:
        label_norm = label
        if label in _METHOD_HEADER_ALIASES:
            label_norm = _METHOD_HEADER_ALIASES[label]

        # Keep jika label_norm termasuk allowed
        if label_norm in allowed_labels:
            if label != label_norm:
                # rename header line
                lines2 = []
                # ganti hanya baris header pertama
                m0 = hdr_re.match(lines[0] or "")
                if m0:
                    rest = (m0.group(2) or "").strip()
                    if rest:
                        lines2.append(f"{label_norm}: {rest}")
                    else:
                        lines2.append(f"{label_norm}:")
                    lines2.extend(lines[1:])
                    lines = lines2
            kept_lines.extend(lines)
            continue

        # Drop jika label adalah known method label tapi bukan target
        if label_norm in known_labels:
            continue

        # fallback (harusnya jarang): keep
        kept_lines.extend(lines)

    new_body = "\n".join([x.rstrip() for x in kept_lines]).strip()
    new_head = "Jawaban:\n" + new_body if new_body else "Jawaban: Tidak ditemukan pada dokumen."
    return (new_head.strip() + "\n" + tail.strip()).strip()


def _ensure_bukti_from_head_if_missing(
    out: str,
    *,
    target_methods: List[str],
    max_points: int,
) -> str:
    """
    Jika 'Bukti: -' tapi di Jawaban ada sitasi [CTX n],
    maka bangun Bukti dari baris-baris Jawaban yang mengandung [CTX n].

    - Untuk multi-metode, otomatis prefix bullet dengan '(METODE)' berdasarkan section tempat baris itu berada.
    """
    t = (out or "").strip()
    if not t or "Bukti:" not in t:
        return out

    # sudah ada bukti? jangan ganggu
    if _extract_bukti_lines(t):
        return out

    # kalau tidak ada sitasi di head, tidak dapat membangun bukti
    if not _has_citation(t):
        return out

    head, tail = _split_head_tail(t)
    tail_body = re.sub(r"(?im)^\s*Bukti\s*:\s*", "", tail).strip()
    if tail_body and tail_body != "-":
        return out

    # pastikan head memiliki "Jawaban:"
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).rstrip("\n")
    lines = head_body.splitlines()

    allowed_labels = {_nice_method(m) for m in (target_methods or []) if m}
    hdr_re = (
        re.compile(
            r"(?im)^\s*("
            + "|".join(re.escape(x) for x in sorted(allowed_labels, key=len, reverse=True))
            + r")\s*:\s*(.*)$"
        )
        if allowed_labels
        else None
    )

    bullets: List[str] = []
    cur_label: Optional[str] = None

    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue

        if hdr_re:
            mh = hdr_re.match(s)
            if mh:
                cur_label = (mh.group(1) or "").strip()
                # jikalau header line juga memiliki [CTX], dia juga bisa jadi bukti
                if _CTX_NUM_RE.search(s):
                    pass
                else:
                    continue

        if not _CTX_NUM_RE.search(s):
            continue

        # bersihkan numbering/bullet awal agar bukti lebih enak dibaca
        s2 = re.sub(r"^\s*[\-\•]\s*", "", s)
        s2 = re.sub(r"^\s*\d+[\.\)]\s*", "", s2).strip()

        # prefix metode jikalau multi-metode dan user tahu sedang berada di section metode
        if cur_label and allowed_labels:
            if not re.match(r"^\(\s*[A-Za-z_ ]+\s*\)", s2):
                s2 = f"({cur_label}) {s2}"

        bullets.append(s2)

    # dedupe + cap
    def _sig(x: str) -> str:
        x2 = re.sub(r"\[CTX\s*\d+\]", "", x, flags=re.IGNORECASE)
        x2 = " ".join(x2.split()).strip().lower()
        return x2

    uniq: List[str] = []
    seen = set()
    for b in bullets:
        sg = _sig(b)
        if not sg or sg in seen:
            continue
        seen.add(sg)
        uniq.append(b)

    uniq = uniq[: int(max(1, min(max_points, 8)))]
    if not uniq:
        return out

    rebuilt = _rebuild_with_bukti(head.strip(), uniq)
    return rebuilt.strip()


def _strip_ctx_from_jawaban(out: str) -> str:
    """
    Hapus semua token [CTX n] dari bagian Jawaban (head).
    Sitasi [CTX n] hanya boleh ada di bagian Bukti, tidak di Jawaban.

    Dipanggil SETELAH _ensure_bukti_from_head_if_missing() agar
    info sitasi tidak hilang sebelum sempat dipindahkan ke Bukti.
    """
    t = (out or "").strip()
    if not t:
        return t

    head, tail = _split_head_tail(t)

    # Hapus [CTX n] beserta whitespace di sebelah kiri (mis. " [CTX 1]" → "")
    head_clean = re.sub(r"\s*\[CTX\s*\d+\]", "", head, flags=re.IGNORECASE)
    # Rapikan spasi ganda yang mungkin tertinggal
    head_clean = re.sub(r"[ \t]{2,}", " ", head_clean)
    # Rapikan trailing whitespace per baris (misal "teks  \n" → "teks\n")
    head_clean = re.sub(r"[ \t]+\n", "\n", head_clean)
    # Rapikan baris kosong beruntun (max 2)
    head_clean = re.sub(r"\n{3,}", "\n\n", head_clean).strip()

    return head_clean + "\n" + tail.strip()


def _fix_empty_numbered_lines(out: str) -> str:
    """
    Hapus baris yang hanya berisi nomor tanpa konten, dan pola nomor beruntun inline tanpa isi.
    Mencegah "...incremental. 1. 2." muncul di output.
    Hanya berlaku di bagian Jawaban (head), Bukti tidak disentuh.
    """
    t = (out or "").strip()
    if not t:
        return t

    head, tail = _split_head_tail(t)

    # Hapus baris yang HANYA berisi nomor list: "1." "2)" "3." dst
    head_clean = re.sub(r"(?m)^\s*\d+[\.\)]\s*$", "", head)
    # Hapus pola nomor beruntun inline tanpa konten: "...teks. 1. 2." di akhir baris/kalimat
    head_clean = re.sub(r"(\s+\d+[\.\)]\s*){2,}(?=\s|$)", " ", head_clean)
    # Rapikan whitespace
    head_clean = re.sub(r"[ \t]{2,}", " ", head_clean)
    head_clean = re.sub(r"[ \t]+\n", "\n", head_clean)
    head_clean = re.sub(r"\n{3,}", "\n\n", head_clean).strip()

    return head_clean + "\n" + tail.strip()


def _output_sanity_check(
    out: str,
    *,
    target_methods: Optional[List[str]] = None,
    used_ctx: int = 0,
) -> Dict[str, Any]:
    """
    Sanity check output multi-metode.
    Return dict berisi hasil check (a)-(d) untuk logging/debug dan conditional gate.

    Check:
    (a) jawaban_no_ctx     : Jawaban tidak mengandung [CTX n]
    (b) bukti_has_citation : Semua bullet Bukti memiliki sitasi [CTX n]
    (c) no_duplicate_methods: Tidak ada metode yang tercetak dobel di Jawaban
    (d) steps_on_newline   : Tidak ada nomor list inline "1. X 2. Y" dalam satu baris
    """
    t = (out or "").strip()
    if not t:
        return {"all_pass": False, "reason": "empty_output"}

    jawaban_head = _jawaban_head_only(t)
    bukti_lines = _extract_bukti_lines(t)

    # (a) Jawaban tidak mengandung [CTX]
    jawaban_no_ctx = not _has_citation(jawaban_head)

    # (b) Semua bullet Bukti punya sitasi (True jika tidak ada bukti sama sekali)
    bukti_has_citation = (
        all(_has_citation(b) for b in bukti_lines)
        if bukti_lines
        else True
    )

    # (c) Tidak ada metode dobel di Jawaban
    no_duplicate_methods = True
    if target_methods:
        for m in (target_methods or []):
            if not m:
                continue
            lab = _nice_method(m)
            count = len(re.findall(rf"(?im)^\s*{re.escape(lab)}\s*:", jawaban_head))
            if count > 1:
                no_duplicate_methods = False
                break

    # (d) Tidak ada pola nomor list inline ("1. teks1 2. teks2" di satu baris)
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", jawaban_head).strip()
    inline_steps_hit = re.search(
        r"\d+[\.\)]\s+\S.{5,}\s+\d+[\.\)]\s+\S", head_body
    )
    steps_on_newline = not bool(inline_steps_hit)

    all_pass = all([jawaban_no_ctx, bukti_has_citation, no_duplicate_methods, steps_on_newline])

    return {
        "all_pass": bool(all_pass),
        "jawaban_no_ctx": bool(jawaban_no_ctx),
        "bukti_has_citation": bool(bukti_has_citation),
        "no_duplicate_methods": bool(no_duplicate_methods),
        "steps_on_newline": bool(steps_on_newline),
    }


def _repair_steps_multimethod_head_using_bukti(
    out: str,
    *,
    target_methods: List[str],
    max_items_per_method: int = 3,
) -> str:
    """
    Khusus q_intent == 'TAHAP' + multi-metode:
    Jika suatu metode di Jawaban:
    - kosong / tidak bermakna / hanya berisi 'Tidak ditemukan...'
    tapi di Bukti ada bullet yang jelas untuk metode tsb,
    maka isi section Jawaban metode tsb dengan ringkasan konservatif dari bullet Bukti.

    Hal ini mencegah kontradiksi: Jawaban bilang 'Tidak ditemukan', tapi Bukti punya tahapan.
    """
    t = (out or "").strip()
    if not t or (not target_methods) or ("Bukti:" not in t):
        return out

    head, tail = _split_head_tail(t)

    # pastikan head punya "Jawaban:"
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    # rapikan dulu: buang metode non-target agar parsing section stabil
    tmp = (head.strip() + "\n" + tail.strip()).strip()
    tmp = _prune_or_rename_non_target_method_sections(tmp, target_methods=target_methods)
    head, tail = _split_head_tail(tmp)

    # mapping bukti -> method
    bukti_bullets = _extract_bukti_lines(tmp)
    by_method: Dict[str, List[str]] = {m: [] for m in target_methods}
    for b in bukti_bullets:
        mm = _bullet_detect_method(b, target_methods)
        if mm and mm in by_method:
            by_method[mm].append(b)

    # parse head sections by target labels
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).rstrip("\n")
    lines = head_body.splitlines()

    labels = [_nice_method(m) for m in target_methods]
    hdr_re = re.compile(
        r"(?im)^\s*("
        + "|".join(re.escape(lb) for lb in sorted(labels, key=len, reverse=True))
        + r")\s*:\s*(.*)\s*$"
    )

    # build sections dict
    prefix: List[str] = []
    sections: Dict[str, List[str]] = {}  # label -> lines (termasuk header line)
    cur_label: Optional[str] = None
    cur_lines: List[str] = []

    def _flush():
        nonlocal cur_label, cur_lines
        if cur_label is None:
            if cur_lines:
                prefix.extend(cur_lines)
        else:
            sections[cur_label] = cur_lines[:]  # overwrite (latest)
        cur_label, cur_lines = None, []

    for ln in lines:
        m = hdr_re.match(ln or "")
        if m:
            _flush()
            cur_label = (m.group(1) or "").strip()
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    _flush()

    def _section_meaningful(sec_lines: List[str]) -> bool:
        if not sec_lines:
            return False
        # cek inline after colon di header
        mh = hdr_re.match(sec_lines[0] or "")
        inline = (mh.group(2) or "").strip() if mh else ""
        if inline and ("tidak ditemukan pada dokumen" not in inline.lower()):
            return True

        # cek isi section
        for x in sec_lines[1:]:
            xs = (x or "").strip()
            if not xs:
                continue
            # jika semua isinya "tidak ditemukan", anggap tidak meaningful
            if "tidak ditemukan pada dokumen" in xs.lower():
                continue
            return True
        return False

    rebuilt_sections: List[str] = []

    for m in target_methods:
        lab = _nice_method(m)
        sec = sections.get(lab)
        if not isinstance(sec, list):
            sec = []  # hard safety

        has_bukti = bool(by_method.get(m))
        has_real = _section_meaningful(sec) if sec else False

        if has_real:
            rebuilt_sections.append("\n".join([str(ln).rstrip() for ln in sec]).rstrip())
            continue

        if has_bukti:
            # isi section dari Bukti (konservatif, tidak mengarang)
            new_lines = [f"{lab}:"]
            for bb in by_method[m][: int(max(1, min(max_items_per_method, 5)))]:
                # buang prefix "(METODE)" di awal bullet biar tidak dobel di head
                bb2 = re.sub(r"^\(\s*[A-Za-z_ ]+\s*\)\s*", "", (bb or "").strip())
                new_lines.append(f"- {bb2}")
            rebuilt_sections.append("\n".join(new_lines).rstrip())
            continue

        rebuilt_sections.append(f"{lab}: Tidak ditemukan pada dokumen.")

    # susun ulang head
    prefix_txt = "\n".join([x.rstrip() for x in prefix]).strip()
    body_parts: List[str] = []
    if prefix_txt:
        body_parts.append(prefix_txt)
    body_parts.append("\n\n".join([p for p in rebuilt_sections if p.strip()]))

    new_body = "\n".join(body_parts).strip()
    new_head = "Jawaban:\n" + new_body if new_body else "Jawaban: Tidak ditemukan pada dokumen."

    return (new_head.strip() + "\n" + tail.strip()).strip()


def _jawaban_head_only(out: str) -> str:
    t = (out or "").strip()
    if not t:
        return ""
    m = re.search(r"(?im)^\s*Bukti\s*:\s*", t)
    if m:
        return t[: m.start()].strip()
    return t


def _has_method_header_line(out: str, label: str) -> bool:
    head = _jawaban_head_only(out)
    if not head:
        return False
    return re.search(rf"(?im)^\s*{re.escape(label)}\s*:", head) is not None


def _all_method_headers_present(out: str, target_methods: List[str]) -> bool:
    for m in target_methods:
        label = _nice_method(m)
        if not _has_method_header_line(out, label):
            return False
    return True


def _ensure_method_headers_and_missing_lines(
    out: str,
    *,
    target_methods: List[str],
    missing_methods: List[str],
) -> str:
    """
    Enforce struktur Jawaban per-metode:
    - Pastikan semua metode punya baris "<Label>: ...".
    - Untuk missing_methods: paksa jadi "<Label>: Tidak ditemukan pada dokumen."
    - Jika masih ada metode yang tidak muncul, append "<Label>: Tidak ditemukan pada dokumen."
    """
    t = (out or "").strip()
    if not t:
        return t

    # Split head/tail (Bukti)
    m = re.search(r"(?im)^\s*Bukti\s*:\s*", t)
    if m:
        head = t[: m.start()].rstrip()
        tail = t[m.start() :].lstrip()
    else:
        head = t.rstrip()
        tail = "Bukti: -"

    # Pastikan head punya "Jawaban:"
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    # Normalisasi header metode yang kadang terbentuk sebagai bullet:
    # "- Prototyping:" -> "Prototyping:"
    for mth in target_methods or []:
        lab = _nice_method(mth)
        head = re.sub(
            rf"(?im)^\s*-\s*{re.escape(lab)}\s*:\s*",
            f"{lab}: ",
            head,
        )

    # 1) Replace isi metode yang missing -> "Tidak ditemukan..."
    for mm in missing_methods or []:
        lab = _nice_method(mm)
        pat = re.compile(rf"(?im)^\s*{re.escape(lab)}\s*:\s*.*$", re.MULTILINE)
        if pat.search(head):
            head = pat.sub(f"{lab}: Tidak ditemukan pada dokumen.", head, count=1)

    # 2) Append metode yang belum muncul sama sekali
    append_lines: List[str] = []
    for mth in target_methods:
        lab = _nice_method(mth)
        if not re.search(rf"(?im)^\s*{re.escape(lab)}\s*:", head):
            append_lines.append(f"{lab}: Tidak ditemukan pada dokumen.")

    if append_lines:
        head = head.strip() + "\n" + "\n".join(append_lines)

    return head.strip() + "\n" + tail.strip()


def _fill_empty_sections_as_not_found_for_steps(
    out: str,
    *,
    target_methods: List[str],
    not_found_text: str = "Tidak ditemukan pada dokumen.",
) -> str:
    """
    Khusus mode multi-metode (biasanya TAHAP):
    Jika suatu metode sudah ada header "<METODE>:" tetapi section-nya kosong (tidak ada konten sama sekali
    sampai header metode berikutnya), maka ubah menjadi:
        "<METODE>: Tidak ditemukan pada dokumen."
    Hal ini mencegah output seperti:
        Prototyping:
        Extreme Programming:
    """
    t = (out or "").strip()
    if not t or (not target_methods):
        return out

    head = _jawaban_head_only(t)
    tail = t[len(head) :].lstrip()  # sisa setelah Jawaban block (biasanya mulai dari Bukti:)

    # Pastikan bekerja pada body jawaban tanpa label "Jawaban:"
    body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).rstrip("\n")

    lines = body.splitlines()

    # deteksi header metode: "<Label>: ..."
    labels = [_nice_method(m) for m in target_methods]
    header_pat = re.compile(
        r"(?im)^\s*(?:-\s*)?(" + "|".join(re.escape(lb) for lb in labels) + r")\s*:\s*(.*)\s*$"
    )

    # kumpulkan posisi header
    headers: List[Tuple[int, str, str]] = []  # (idx, label, inline_after_colon)
    for i, ln in enumerate(lines):
        m = header_pat.match(ln)
        if m:
            headers.append((i, m.group(1), (m.group(2) or "").strip()))

    if not headers:
        return out  # bukan format multi-metode

    # helper cek apakah section punya konten bermakna
    def _section_has_content(section_lines: List[str], inline: str) -> bool:
        if inline and inline != "-":
            return True
        for x in section_lines:
            xs = (x or "").strip()
            if not xs:
                continue
            if xs == "-":
                continue
            # bullet / enumerasi / teks normal dianggap konten
            return True
        return False

    # perbaiki section kosong
    for k, (start_i, label, inline_txt) in enumerate(headers):
        end_i = headers[k + 1][0] if (k + 1) < len(headers) else len(lines)
        # section adalah baris-baris setelah header sampai sebelum header berikutnya
        section = lines[start_i + 1 : end_i]

        if not _section_has_content(section, inline_txt):
            # ganti baris header menjadi "<Label>: Tidak ditemukan..."
            lines[start_i] = f"{label}: {not_found_text}"

            # bersihkan baris kosong beruntun tepat setelahnya (opsional, biar rapi)
            # (hapus hanya blank lines di awal section, jangan sentuh section lain)
            j = start_i + 1
            while j < end_i and j < len(lines) and (not (lines[j] or "").strip()):
                lines[j] = ""  # biarkan kosong; nanti join tetap rapi
                j += 1 

    new_body = "\n".join(lines).strip()
    new_head = "Jawaban:\n" + new_body

    # gabungkan kembali dengan tail (Bukti dst)
    merged = (new_head + ("\n" + tail if tail else "")).strip()
    return merged


def _ensure_multimethod_steps_output_not_blank(
    out: str,
    *,
    target_methods: List[str],
    missing_methods: List[str],
) -> str:
    """
    Pipeline kecil untuk kasus TAHAP + multi-metode:
    1) Pastikan semua header ada + missing_methods dipasang "Tidak ditemukan"
    2) Isi section kosong menjadi "Tidak ditemukan" (anti blank)
    """
    t = (out or "").strip()
    if not t or not target_methods:
        return out

    # 1) pastikan header + missing line
    t = _ensure_method_headers_and_missing_lines(
        t, target_methods=target_methods, missing_methods=missing_methods
    )

    # 2) isi section kosong (termasuk yang missing_methods tidak terdeteksi)
    t = _fill_empty_sections_as_not_found_for_steps(t, target_methods=target_methods)

    return t


def _guess_method_for_free_text(text: str, target_methods: List[str]) -> Optional[str]:
    """
    Tebak metode paling mungkin untuk free-text (tanpa header).
    Tujuan: jika model menulis "Tiga tahapan ... Planning/Design/Implementation"
    tapi tidak menulis "RAD:", saya bisa mengatribusikan ke RAD.
    """
    tl = (text or "").lower()

    # 1) Match pattern resmi yang sudah dipunyai sebelumnya
    for m in target_methods:
        pats = _METHOD_MATCH_PATTERNS.get(m, [])
        for p in pats:
            if re.search(p, tl, flags=re.IGNORECASE):
                return m

    # 2) Heuristik tahapan RAD
    if "rad" in tl or "rapid application development" in tl:
        if "RAD" in target_methods:
            return "RAD"
    if re.search(r"\b(planning|perencanaan)\b", tl) and re.search(r"\b(design|perancangan)\b", tl):
        if "RAD" in target_methods:
            return "RAD"
    if re.search(r"\bimplementation|implementasi\b", tl) and ("RAD" in target_methods):
        return "RAD"

    # 3) Heuristik prototyping
    if re.search(r"\bprototyp", tl) or "prototipe" in tl:
        if "PROTOTYPING" in target_methods:
            return "PROTOTYPING"

    # 4) Heuristik XP / extreme programming
    if "extreme programming" in tl or re.search(r"\bxp\b", tl):
        if "EXTREME_PROGRAMMING" in target_methods:
            return "EXTREME_PROGRAMMING"
    if re.search(r"pair programming|continuous integration|refactor|test[- ]driven", tl):
        if "EXTREME_PROGRAMMING" in target_methods:
            return "EXTREME_PROGRAMMING"

    return target_methods[0] if target_methods else None


def _normalize_multimethod_head(out: str, *, target_methods: List[str]) -> str:
    """
    Jika mode multi-metode aktif tapi model menulis jawaban tahapan sebagai free-text
    (tanpa 'RAD:' / 'Prototyping:'), ubah jadi format per-metode agar tidak dikontradiksi
    oleh deterministic enforce.
    """
    t = (out or "").strip()
    if not t:
        return t

    # Split head/tail (Bukti)
    m = re.search(r"(?im)^\s*Bukti\s*:\s*", t)
    if m:
        head = t[: m.start()].rstrip()
        tail = t[m.start() :].lstrip()
    else:
        head = t.rstrip()
        tail = "Bukti: -"

    # Pastikan ada 'Jawaban:' di head
    if re.search(r"(?im)^\s*Jawaban\s*:", head) is None:
        head = "Jawaban:\n" + head.strip()

    # Kalau semua header metode sudah ada, tidak perlu normalisasi
    if _all_method_headers_present(head, target_methods):
        return head.strip() + "\n" + tail.strip()

    # Ambil isi head tanpa label "Jawaban:"
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).strip()
    if not head_body:
        return head.strip() + "\n" + tail.strip()

    # Deteksi apakah ada baris metode (header line) yang sudah ada sebagian
    lines = [ln.rstrip() for ln in head_body.splitlines() if ln.strip()]
    method_lines_idx = []
    for i, ln in enumerate(lines):
        # line "<Label>: ..."
        if re.match(r"^\s*[A-Za-z_ ].+?:\s*.*$", ln) and (
            ln.lower().strip().startswith("jawaban") is False
        ):
            method_lines_idx.append(i)

    # Jika tidak ada method header sama sekali -> atribusikan seluruh free-text ke 1 metode yang paling mungkin
    if not method_lines_idx:
        guessed = _guess_method_for_free_text(head_body, target_methods) or (
            target_methods[0] if target_methods else None
        )
        if guessed:
            lab = _nice_method(guessed)
            # Pertahankan formatting multiline (enumerasi 1.,2.,3. tetap)
            new_head = "Jawaban:\n" + f"{lab}: {head_body}"
            return new_head.strip() + "\n" + tail.strip()
        return head.strip() + "\n" + tail.strip()

    # Jika ada method header sebagian + ada free-text sebelum header pertama:
    first_idx = method_lines_idx[0]
    prefix_lines = lines[:first_idx]
    rest_lines = lines[first_idx:]

    free_text = "\n".join(prefix_lines).strip()
    if free_text:
        guessed = _guess_method_for_free_text(free_text, target_methods)
        if guessed:
            lab = _nice_method(guessed)

            # Jika sudah ada line untuk metode guessed, append free_text ke line tersebut
            found = False
            for j, ln in enumerate(rest_lines):
                if re.match(rf"(?im)^\s*{re.escape(lab)}\s*:", ln):
                    rest_lines[j] = ln + " " + free_text
                    found = True
                    break
            if not found:
                # kalau belum ada header, sisipkan header baru paling atas
                rest_lines.insert(0, f"{lab}: {free_text}")

        # drop free-text dari prefix (karena sudah dipindah)
        prefix_lines = []

    new_body = "\n".join(rest_lines).strip()
    new_head = "Jawaban:\n" + new_body
    return new_head.strip() + "\n" + tail.strip()


# ===== Grounding checks =====
_STOP_QA = {
    # kata tanya / kata sangat umum
    "apa",
    "apakah",
    "siapa",
    "kapan",
    "dimana",
    "di",
    "mana",
    "bagaimana",
    "mengapa",
    "kenapa",
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
    # kata generik domain dokumen (sering membuat false-positive support)
    "penulis",
    "penelitian",
    "skripsi",
    "tugas",
    "akhir",
    "bab",
    "halaman",
    "gambar",
    "tabel",
    "lampiran",
    "dokumen",
    "paper",
    "artikel",
    "jurnal",
    # kata generik RAG
    "konteks",
    "jawaban",
    "bukti",
    # kata generik metodologi (biar support tidak “terpancing” kata-kata ini saja)
    "metode",
    "model",
    "sistem",
    "pengembangan",
    "proses",
    "tahap",
    "tahapan",
    "langkah",
    "alur",
    "prosedur",
    # kata kerja umum
    "memiliki",
    "punya",
    "menggunakan",
    "digunakan",
    "dipakai",
}


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

    # jika bullet diawali "(METODE)" dan metode itu muncul di CTX yang dirujuk, anggap grounded
    m = re.match(r"^\(\s*([A-Za-z_ ]+)\s*\)", bullet.strip())
    method_prefix = m.group(1).strip().lower() if m else ""

    kws = _keywords_for_grounding(bullet)
    for n in ctx_nums:
        hay = (contexts[n - 1] or "").lower()
        if method_prefix and method_prefix in hay:
            return True
        if kws and any(k in hay for k in kws):
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


def _extract_jawaban_text(out: str) -> str:
    """
    Ambil teks jawaban (tanpa label) dari output yang sudah terformat:
    Jawaban: ...
    Bukti: ...
    """
    t = (out or "").strip()
    if not t:
        return ""

    # potong sebelum Bukti:
    if "Bukti:" in t:
        head = t.split("Bukti:", 1)[0].strip()
    else:
        head = t.strip()

    # buang label Jawaban:
    head = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head).strip()
    return head


def _jawaban_grounding_ok(out: str, contexts: List[str]) -> bool:
    """
    Cek grounding untuk isi Jawaban (bukan Bukti).
    Konservatif: minta minimal 1-2 keyword penting dari jawaban muncul di konteks gabungan.
    Hal ini membantu mengurangi hallucination seperti 'workshop desain RAD' bila tidak ada di CTX.
    """
    jaw = _extract_jawaban_text(out)
    if not jaw:
        return True

    kws = _keywords_for_grounding(jaw, max_terms=12)
    if not kws:
        # jawaban terlalu generik, biarkan lewat
        return True

    hay = (" ".join(contexts or [])).lower()
    hits = sum(1 for k in kws if k in hay)

    # threshold konservatif:
    # - jika keyword sedikit, minimal 1
    # - jika keyword banyak, minimal 2
    need = 1 if len(kws) <= 4 else 2
    return hits >= need


def _keywords_for_support(text: str, max_terms: int = 8) -> list[str]:
    # keyword support: ambil token >=4 huruf, buang stopword
    toks = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    stop = set(_STOP_QA) | {
        "jawaban",
        "bukti",
        "pertanyaan",
        "konteks",
        "sebelumnya",
        "penelitian",
        "penulis",
        "dokumen",
        "sistem",
        "metode",
        "model",
    }

    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _looks_supported(question: str, contexts: list[str]) -> bool:
    """
    Heuristik ringan: kalau ada overlap keyword penting query di konteks,
    mencegah model terlalu sering berkata 'Tidak ditemukan' saat konteks jelas mengandung jawaban.

    Micro-fix:
    - lebih ketat: membutuhkan hit pada keyword yang “spesifik”, bukan hanya kata generik seperti 'penulis'
    - jika keyword spesifik tidak ada di konteks -> return False
    """
    qs = _keywords_for_support(question)
    if not qs or not contexts:
        return False

    hay = (" ".join(contexts)).lower()

    hits = [t for t in qs if t in hay]
    if not hits:
        return False

    # jikalau semua hits ternyata generik -> jangan dianggap supported
    generic_hits = {
        "penulis",
        "penelitian",
        "skripsi",
        "dokumen",
        "sistem",
        "metode",
        "model",
        "bab",
        "gambar",
        "tabel",
        "lampiran",
        "halaman",
    }
    if all(h in generic_hits for h in hits):
        return False

    return True


def _extractive_fallback_from_contexts(
    contexts: list[str],
    used_ctx: int,
    *,
    intent: str = "UMUM",
    targets: Optional[List[str]] = None,
    per_method_bullets: int = 1,
    max_total_bullets: int = 6,
) -> str:
    """
    Fallback aman (extractive):
    - Default: jawaban umum + bukti ringkas
    - Khusus intent TAHAP + multi-metode: buat Jawaban per-metode + Bukti per-metode (tanpa LLM)
    """

    if used_ctx <= 0 or not contexts:
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # ---------- helper internal ----------
    stage_terms = [
        "planning",
        "perencanaan",
        "design",
        "perancangan",
        "implementation",
        "implementasi",
        "coding",
        "pengkodean",
        "testing",
        "pengujian",
        "fase",
        "tahap",
        "tahapan",
        "langkah",
        "prosedur",
    ]

    def _strip_doc_header(x: str) -> str:
        # hapus "DOC: ...\n" supaya snippet lebih bersih
        return re.sub(r"(?im)^DOC:\s.*\n", "", (x or "")).strip()

    def _has_steps_signal_local(x: str) -> bool:
        t = (x or "").lower()
        # sinyal eksplisit
        if re.search(r"\b(tahap|tahapan|langkah|fase|prosedur)\w*\b", t):
            return True
        # enumerasi
        if re.search(r"(?m)^\s*(\d+[\.\)]|[-•])\s+", x or ""):
            return True
        # stage terms minimal 2
        hits = sum(1 for s in stage_terms if re.search(rf"\b{re.escape(s)}\b", t))
        return hits >= 2

    def _best_ctx_for_method(m: str) -> Optional[int]:
        pats = _METHOD_MATCH_PATTERNS.get(m, [])
        if not pats:
            return None

        # pass 1: method hit + steps signal
        for i, c in enumerate(contexts[:used_ctx], start=1):
            cc = c or ""
            if any(re.search(p, cc, flags=re.IGNORECASE) for p in pats) and _has_steps_signal_local(
                cc
            ):
                return i

        # pass 2: method hit saja
        for i, c in enumerate(contexts[:used_ctx], start=1):
            cc = c or ""
            if any(re.search(p, cc, flags=re.IGNORECASE) for p in pats):
                return i

        return None

    # ---------- MULTI-METHOD STEPS FALLBACK ----------
    ts = [m for m in (targets or []) if m]
    if intent == "TAHAP" and len(ts) >= 2:
        per_method_bullets = int(max(1, min(per_method_bullets, 2)))
        max_total_bullets = int(max(1, min(max_total_bullets, 8)))

        jawaban_lines: List[str] = ["Jawaban:"]
        bukti_lines: List[str] = []

        for m in ts:
            lab = _nice_method(m)
            ctx_candidates = _candidate_ctx_indices_for_method_steps(
                m, contexts, used_ctx, max_candidates=3
            )
            idx = ctx_candidates[0] if ctx_candidates else None

            if not idx:
                jawaban_lines.append(f"{lab}: Tidak ditemukan pada dokumen.")
                continue

            steps = _extract_steps_from_ctx_candidates(
                contexts,
                ctx_candidates,
                max_items=6,
            )

            if len(steps) >= 2:
                joined = "; ".join(steps[:6])
                # Jawaban per-metode tetap harus ada sitasi karena memuat klaim
                jawaban_lines.append(f"{lab}: {joined} [CTX {idx}]")
                # Bukti per-metode (cap total di akhir)
                bukti_lines.append(f"- ({lab}) Tahapan yang disebutkan: {joined} [CTX {idx}]")
            else:
                note = _partial_steps_note(is_comparison=False)
                jawaban_lines.append(f"{lab}: {note}")
                bukti_lines.append(f"- ({lab}) {note} [CTX {idx}]")

        # cap bukti total
        bukti_lines = bukti_lines[:max_total_bullets]

        def _fb_cleanup(s: str) -> str:
            s = _fix_empty_numbered_lines(s)
            s = _strip_ctx_from_jawaban(s)
            return s

        if not bukti_lines:
            return _fb_cleanup("\n".join(jawaban_lines).strip() + "\nBukti: -")

        return _fb_cleanup(
            "\n".join(jawaban_lines).strip() + "\nBukti:\n" + "\n".join(bukti_lines)
        )

    # ---------- DEFAULT FALLBACK (UMUM / single target) ----------
    bullets: list[str] = []
    for i, c in enumerate(contexts[: min(3, used_ctx)], start=1):
        sn = " ".join((_strip_doc_header(c) or "").split()).strip()
        sn = sn[:240] + ("..." if len(sn) > 240 else "")
        bullets.append(f"- {sn} [CTX {i}]")

    return (
        "Jawaban: Informasi relevan ditemukan pada potongan dokumen berikut.\n"
        "Bukti:\n" + "\n".join(bullets)
    )


def generate_answer(
    cfg: Dict[str, Any],
    question: str,
    contexts: List[str],
    *,
    max_bullets: int = 3,
    target_methods: Optional[List[str]] = None,
    missing_methods: Optional[List[str]] = None,
    force_per_method: bool = False,
    per_method_bullets: int = 2,
    max_total_bullets: int = 6,
) -> str:
    """
    Generate jawaban berbasis konteks (RAG) dengan aturan:
    - Gunakan hanya informasi di konteks
    - Jika tidak cukup: jawab "Tidak ditemukan pada dokumen."
    - Wajib menyertakan sitasi [CTX n] untuk klaim faktual
    - HARD ENFORCE: Bukti maksimal max_bullets (cap + dedupe)
    - Anti klaim tidak presisi: validasi grounding bullet Bukti terhadap CTX
    - Konteks bisa berisi metadata lintas dokumen (mis. "DOC: ... | file p.X")
            -> dorong jawaban menjadi eksplisit lintas dokumen (hindari "penelitian ini" generik)
    - Bukti default maksimal 3 poin, bisa naik (mis. 5 / max_bullets) untuk multi-target tertentu
            -> enforced via cap+dedupe, dan prompt mengikuti max_bullets.
    - Jika multi-metode (target_methods>=2) dan force_per_method=True, output wajib per metode.
    - Bukti dibatasi per metode (per_method_bullets) dan total (max_total_bullets).
    - Jika metode target tidak didukung konteks: tulis "<METODE>: Tidak ditemukan pada dokumen."
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

    # intent hint agar LLM tidak menganggap "tahapan" sebagai "metode"
    ql = (question or "").lower()
    is_steps = bool(re.search(r"\b(tahap|tahapan)\w*\b|\b(langkah|alur|prosedur)\w*\b", ql))
    is_method_list = bool(re.search(r"\bmetode\b.*\bapa\s+saja\b|\bmetode\s+apa\s+saja\b", ql))

    if is_steps:
        q_intent = "TAHAP"
    elif re.search(r"\b(metode|model)\b", ql):
        q_intent = "METODE"
    else:
        q_intent = "UMUM"

    targets = [m for m in (target_methods or []) if m]
    multi_method_mode = bool(force_per_method and len(targets) >= 2)

    support_relaxed = bool(used_ctx > 0 and q_intent in {"TAHAP", "METODE"} and len(targets) > 0)

    # ===== fallback targets untuk deterministic =====
    # Kalau app tidak mengirim target_methods (targets kosong), fallback ke daftar metode yang dikenal.
    det_targets = list(targets) if targets else list(_METHOD_MATCH_PATTERNS.keys())

    # auto force per-method untuk pertanyaan "metode apa saja" jika multi target tersedia
    if (not multi_method_mode) and (len(targets) >= 2) and is_method_list:
        multi_method_mode = True
    # auto-force juga untuk steps/tahapan
    if (not multi_method_mode) and (len(targets) >= 2) and is_steps:
        multi_method_mode = True

    # max bullets safety
    per_method_bullets = int(max(1, min(per_method_bullets, 3)))
    max_total_bullets = int(max(1, min(max_total_bullets, 8)))
    max_bullets = int(max(1, min(max_bullets, 8)))

    # kalau multi-method mode, pakai kebijakan total yang eksplisit
    if multi_method_mode:
        max_bullets = min(max_total_bullets, per_method_bullets * max(1, len(targets)))

    # ===== Stabilizer: deterministic untuk pertanyaan METODE =====
    # Supaya pertanyaan seperti "metode pengembangan apa yang digunakan?"
    # tidak jatuh ke jawaban template "Informasi relevan ditemukan..."
    # Semua jalur harus tetap melewati repair/humanize/final cleanup bersama.
    out = ""

    det_on = _get_cfg(cfg, ["generation", "deterministic_method_answer"], True)
    if bool(det_on) and used_ctx > 0 and q_intent == "METODE":
        det = _deterministic_method_answer(
            question=question,
            targets=det_targets,
            contexts=contexts,
            used_ctx=used_ctx,
            max_bullets=max_bullets,
        )
        if det:
            out = det

    # ===== Stabilizer: deterministic untuk pertanyaan TAHAP + multi-metode =====
    det_steps_on = _get_cfg(cfg, ["generation", "deterministic_steps_answer"], False)

    if (not out) and used_ctx > 0 and q_intent == "TAHAP" and len(targets) >= 2:
        # (A) Jikalau user sengaja mengaktifkan mode deterministik (pure extractive)
        if bool(det_steps_on):
            dets = _deterministic_steps_answer(
                question=question,
                targets=targets,
                contexts=contexts,
                used_ctx=used_ctx,
                max_bullets=max_bullets,
                max_items_per_method=5,
            )
            if dets:
                out = dets

        if not out:
            # (B) Default (dan rekomendasi): structured steps + LLM polish (tanpa konteks mentah)
            structured = _structured_steps_answer_llm(
                llm=llm,
                question=question,
                targets=targets,
                contexts=contexts,
                used_ctx=used_ctx,
                max_bullets=max_bullets,
            )
            if structured:
                # cleanup sebelum early return
                out = structured

    targets_str = ", ".join([_nice_method(m) for m in targets]) if targets else "-"
    missing = [m for m in (missing_methods or []) if m]
    missing_clean = missing
    missing_str = ", ".join([_nice_method(m) for m in missing]) if missing else "-"

    # ===== Prompting =====
    if multi_method_mode and (q_intent == "TAHAP"):
        format_hint = (
            "Format jawaban WAJIB (ikuti persis):\n"
            "Jawaban:\n"
            f"{_nice_method(targets[0])}: <kalimat ringkas>\n"
            "1) <tahap 1>\n"
            "2) <tahap 2>\n"
            "(dst jika ada)\n"
            f"{_nice_method(targets[1])}: <kalimat ringkas>\n"
            "1) <tahap 1>\n"
            "2) <tahap 2>\n"
            "(dst)\n"
            "Bukti:\n"
            f"- ({_nice_method(targets[0])}) <kutip/para-phrase paling relevan> [CTX n]\n"
            f"- ({_nice_method(targets[1])}) <kutip/para-phrase paling relevan> [CTX n]\n"
            "...\n"
        )
    elif multi_method_mode:
        format_hint = (
            "Format jawaban WAJIB (ikuti persis):\n"
            "Jawaban:\n"
            f"{_nice_method(targets[0])}: <isi>\n"
            "(lanjutkan per metode target sesuai urutan)\n"
            "Bukti:\n"
            f"- ({_nice_method(targets[0])}) <fakta spesifik> [CTX n]\n"
            f"- ({_nice_method(targets[0])}) <fakta spesifik> [CTX n]\n"
            f"- ({_nice_method(targets[1])}) <fakta spesifik> [CTX n]\n"
            "...\n"
        )
    else:
        format_hint = (
            "Format jawaban WAJIB (ikuti persis):\n"
            "Jawaban: <jawaban satu paragraf ringkas namun informatif secara keseluruhan>\n"
            "Bukti:\n"
            "- <poin bukti 1> [CTX n]\n"
            "- <poin bukti 2> [CTX n]\n"
            "- <poin bukti 3> [CTX n]\n\n"
        )

    steps_rules = ""
    if q_intent == "TAHAP":
        steps_rules = (
            "\nKhusus TAHAPAN/LANGKAH:\n"
            "- Yang dimaksud tahapan adalah urutan langkah/fase (mis. planning, design, coding, testing, implementasi, pengujian, dsb) yang dinyatakan eksplisit di CTX.\n"
            "- JANGAN menganggap kelebihan/karakteristik metode (mis. 'waktu singkat', 'adaptasi cepat', 'incremental') sebagai tahapan.\n"
            "- Jika CTX hanya mengonfirmasi metode atau menjelaskan karakteristik umum tanpa daftar/urutan tahapan yang eksplisit,\n"
            "- jangan mengarang langkah. Tulis secara jujur bahwa daftar tahapan eksplisit tidak tampil pada potongan konteks yang tersedia.\n"
            "- Gunakan 'Tidak ditemukan pada dokumen.' hanya jika memang tidak ada dukungan konteks untuk metode tersebut.\n"
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
                "6) Jangan menyebut [CTX] yang tidak ada (hanya 1..N sesuai konteks).\n"
                "7) Jangan menuliskan ulang label 'Pertanyaan:' pada output. Output harus DIMULAI dari 'Jawaban:'.\n"
                f"8) Bagian Bukti maksimal {max_bullets} bullet.\n\n"
                "Pedoman kualitas:\n"
                "- Jika konteks berasal dari beberapa dokumen (ada penanda DOC/file/page), hindari frasa "
                "'penelitian ini' yang ambigu. Gunakan frasa seperti 'Dalam dokumen-dokumen yang relevan...'.\n"
                "- Jika pertanyaan meminta TAHAPAN/LANGKAH dan konteks memuat lebih dari satu metode, "
                "jawab secara TERPISAH per metode (contoh: 'RAD: ...; Prototyping: ...') dengan baris baru (line break).\n"
                "- Jika pertanyaan bertanya METODE, jangan mengubah daftar tahapan menjadi 'metode'.\n"
                f"{steps_rules}"
                "Khusus kasus (multi-metode):\n"
                f"- Target metode: {targets_str}\n"
                f"- Metode yang (di sisi retrieval) terindikasi tidak punya dukungan konteks: {missing_str}\n"
                "- Jika multi-metode aktif, kamu WAJIB menjawab TERPISAH per metode sesuai urutan target.\n"
                "- Jika suatu metode tidak didukung konteks, tulis: '<METODE>: Tidak ditemukan pada dokumen.' (tanpa mengarang).\n"
                "- Untuk bukti multi-metode, awali setiap bullet dengan '(METODE)' agar jelas.\n",
            ),
            (
                "human",
                "Jenis pertanyaan: {intent}\n\n"
                "Input user (jangan ditulis ulang sebagai 'Pertanyaan:' di output):\n{question}\n\n"
                "Konteks:\n{context}\n\n"
                "Tugas:\n"
                "- Jawab pertanyaan secara langsung.\n"
                "- Ambil bukti hanya dari konteks.\n\n"
                f"{format_hint}\n"
                f"Ketentuan Bukti:\n"
                f"- Maksimal {max_bullets} poin.\n"
                "- Tiap poin berisi fakta spesifik dan benar-benar didukung CTX yang sama (mis. nama metode/alat uji/tahapan).\n"
                "- Jika tidak ada dukungan konteks, tulis persis:\n"
                "Jawaban: Tidak ditemukan pada dokumen.\n"
                "Bukti: -\n",
            ),
        ]
    )

    if not out:
        chain = prompt | llm | StrOutputParser()
        out = chain.invoke(
            {
                "intent": q_intent,
                "question": (question or "").strip(),
                "context": ctx_block,
            }
        ).strip()

    # ======= Normalisasi awal =======
    # 0) bersihkan kemungkinan preamble "Pertanyaan:"
    out = _strip_question_preamble(out)

    # 1) Normalisasi sitasi + paksa format + paksa bukti jadi bullet (max. 3)
    out = _normalize_citations(out).strip()
    out = _enforce_qa_format(out)
    out = _ensure_bukti_bullets(out)
    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)

    # micro-fix: jika bullet [CTX] nyasar di Jawaban dan Bukti kosong, pindahkan ke Bukti
    out = _move_inline_cited_bullets_to_bukti(out)
    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)

    # ===== HARD NOT-FOUND (anti template evidence untuk query off-topic) =====
    # Jika pertanyaan tipe UMUM dan tidak ada sinyal dukungan yang spesifik di konteks,
    # jangan membiarkan model mengeluarkan template "Informasi relevan ditemukan..."
    if (
        used_ctx > 0
        and (q_intent == "UMUM")
        and (not (_looks_supported(question, contexts) or support_relaxed))
    ):
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # ===== Anti jawaban template untuk pertanyaan METODE =====
    # Jikalau model hanya menampilkan kutipan/dokumen ("Informasi relevan ditemukan..."),
    # paksa 1x agar MENYEBUT nama metode yang digunakan.
    if used_ctx > 0 and (q_intent == "METODE") and (not multi_method_mode):
        jaw = _extract_jawaban_text(out).lower()
        if "informasi relevan ditemukan" in jaw:
            methods_hint = (
                ", ".join([_nice_method(m) for m in (det_targets or [])]) if det_targets else "-"
            )
            method_focus_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Jawablah pertanyaan METODE dengan menyebutkan NAMA metode yang digunakan.\n"
                        "Dilarang menjawab dengan kalimat template seperti 'Informasi relevan ditemukan...'.\n"
                        "Aturan:\n"
                        "1) Output WAJIB mulai dari 'Jawaban:' (tanpa 'Pertanyaan:').\n"
                        "2) Jika konteks menyebut metode yang digunakan, sebutkan metodenya secara eksplisit.\n"
                        "3) Setiap klaim faktual WAJIB ada [CTX n].\n"
                        f"4) Bukti maksimal {max_bullets} bullet.\n"
                        f"5) CTX valid hanya 1..{used_ctx}.\n"
                        "6) Jangan menambah fakta baru.\n"
                        f"Hint target_methods (jika ada): {methods_hint}\n",
                    ),
                    (
                        "human",
                        "Pertanyaan:\n{question}\n\n" "Konteks:\n{context}\n\n" "Jawab sekarang:",
                    ),
                ]
            )
            forced_m = (method_focus_prompt | llm | StrOutputParser()).invoke(
                {"question": question.strip(), "context": ctx_block}
            )
            forced_m = _strip_question_preamble((forced_m or "").strip())
            forced_m = _normalize_citations(forced_m)
            forced_m = _enforce_qa_format(forced_m)
            forced_m = _ensure_bukti_bullets(forced_m)
            forced_m = _cap_and_dedupe_bukti(forced_m, max_points=max_bullets)

            forced_m = _move_inline_cited_bullets_to_bukti(forced_m)
            forced_m = _cap_and_dedupe_bukti(forced_m, max_points=max_bullets)
            if forced_m:
                out = forced_m

    # 2) normalisasi free-text tahapan agar tidak jadi kontradiksi saat deterministic enforce
    if multi_method_mode and targets:
        out = _normalize_multimethod_head(out, target_methods=targets)

    # Micro-fix: khusus TAHAP + multi-metode, bersihkan "bukti nyasar" di Jawaban
    if q_intent == "TAHAP":
        out = _clean_inline_method_bullets_in_jawaban(out)
        out = _cap_and_dedupe_bukti(out, max_points=max_bullets)

    # 3) cap per-method jika multi-mode
    if multi_method_mode and targets:
        out = _cap_bukti_per_method(
            out,
            target_methods=targets,
            per_method_bullets=per_method_bullets,
            max_total_bullets=max_bullets,
        )

    # ===== Multi-method enforcement repair (1x) =====
    # Tujuan:
    # 1) Pastikan semua metode target disebut pada bagian Jawaban sebagai header line "<METODE>:"
    # 2) Pastikan bullet Bukti diawali (METODE) agar bisa di-cap per metode dengan benar
    # 3) Jika ada missing_methods, WAJIB tampil "<METODE>: Tidak ditemukan pada dokumen."
    if used_ctx > 0 and multi_method_mode and targets:
        missing_clean = _prune_missing_methods_against_answer(
            out, targets, list(missing_clean or [])
        )
        missing = list(missing_clean)

        def _missing_line_ok(o: str, m: str) -> bool:
            lab = _nice_method(m)
            head = _jawaban_head_only(o)
            # harus ada line dan mengandung "Tidak ditemukan"
            mline = re.search(rf"(?im)^\s*{re.escape(lab)}\s*:\s*(.*)$", head)
            if not mline:
                return False
            return "tidak ditemukan pada dokumen" in (mline.group(1) or "").lower()

        need_multi_repair = False

        # jika model mengeluarkan jawaban template, paksa repair multi-metode
        if "informasi relevan ditemukan" in out.lower():
            need_multi_repair = True

        # strict: semua header harus ada (cek semua metode disebut di Jawaban)
        if not _all_method_headers_present(out, targets):
            need_multi_repair = True

        # strict: missing_methods harus benar-benar tertulis sebagai "Tidak ditemukan..."
        for mm in missing:
            if not _missing_line_ok(out, mm):
                need_multi_repair = True
                break

        # cek apakah ada bullet tanpa prefix (METODE) saat multi-method
        bukti_lines = _extract_bukti_lines(out)
        if bukti_lines:
            # minimal separuh bullet harus punya prefix metode agar jelas
            pref_ok = 0
            for b in bukti_lines:
                if re.match(r"^\(\s*[A-Za-z_ ]+\s*\)", b.strip()):
                    pref_ok += 1
            if pref_ok < max(1, len(bukti_lines) // 2):
                need_multi_repair = True

        if need_multi_repair:
            missing_str2 = ", ".join([_nice_method(x) for x in missing]) if missing else "-"

            multi_repair_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Perbaiki jawaban agar MULTI-METODE sesuai aturan.\n"
                        "Aturan WAJIB:\n"
                        "1) Output harus DIMULAI dari 'Jawaban:' (tanpa 'Pertanyaan:').\n"
                        "2) Pada bagian Jawaban, WAJIB berisi SEMUA metode target dan tulis TERPISAH per metode target (urut sesuai daftar target) sebagai baris:\n"
                        "   <METODE>: <isi>\n"
                        "3) Untuk metode yang ada di daftar MISSING, WAJIB tulis persis:\n"
                        "   <METODE>: Tidak ditemukan pada dokumen.\n"
                        f"4) CTX valid hanya 1..{used_ctx}.\n"
                        f"5) Bukti maksimal {max_bullets} bullet.\n"
                        "6) Setiap bullet Bukti HARUS diawali '(METODE)' dan mengandung [CTX n] agar jelas.\n"
                        "7) Jangan menambah fakta baru.\n",
                    ),
                    (
                        "human",
                        "Target metode: {targets}\n\n"
                        "Metode MISSING (wajib tulis Tidak ditemukan):\n{missing}\n\n"
                        "Pertanyaan:\n{question}\n\n"
                        "Konteks:\n{context}\n\n"
                        "Jawaban sebelumnya:\n{answer}\n\n"
                        "Perbaiki sekarang:",
                    ),
                ]
            )
            fixed_multi = (multi_repair_prompt | llm | StrOutputParser()).invoke(
                {
                    "targets": targets_str,
                    "missing": missing_str2,
                    "question": question.strip(),
                    "context": ctx_block,
                    "answer": out,
                }
            )
            fixed_multi = _strip_question_preamble((fixed_multi or "").strip())
            fixed_multi = _normalize_citations(fixed_multi)
            fixed_multi = _enforce_qa_format(fixed_multi)
            fixed_multi = _ensure_bukti_bullets(fixed_multi)
            fixed_multi = _cap_and_dedupe_bukti(fixed_multi, max_points=max_bullets)

            fixed_multi = _move_inline_cited_bullets_to_bukti(fixed_multi)
            fixed_multi = _cap_and_dedupe_bukti(fixed_multi, max_points=max_bullets)

            if fixed_multi:
                out = fixed_multi

            # setelah repair, pruning lagi agar missing_methods tidak meng-overwrite konten yang sudah ada
            missing_clean = _prune_missing_methods_against_answer(
                out, targets, list(missing_clean or [])
            )

            missing = list(missing_clean)   # sinkronkan ulang

        # FINAL deterministic enforce:
        # Kalau LLM masih saja keras kepala (mis. hanya isi RAD), saya paksa semua header harus ada,
        # dan missing_methods jadi "Tidak ditemukan..."
        # normalisasi dulu, baru enforce header + missing lines
        out = _normalize_multimethod_head(out, target_methods=targets)
        out = _ensure_method_headers_and_missing_lines(
            out, target_methods=targets, missing_methods=missing_clean
        )

        # ===== Anti-blank khusus TAHAP + multi-metode =====
        if multi_method_mode and targets and (q_intent == "TAHAP"):
            out = _ensure_multimethod_steps_output_not_blank(
                out,
                target_methods=targets,
                missing_methods=list(missing_clean or []),
            )

        # ===== buang section metode non-target (mis. "Scrum:") =====
        out = _prune_or_rename_non_target_method_sections(out, target_methods=targets)

        # ===== untuk TAHAP + multi-metode, selaraskan Jawaban dengan Bukti =====
        if q_intent == "TAHAP":
            out = _repair_steps_multimethod_head_using_bukti(
                out,
                target_methods=targets,
                max_items_per_method=3,
            )

        # rapikan lagi cap bukti per-metode
        out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
        out = _cap_bukti_per_method(
            out,
            target_methods=targets,
            per_method_bullets=per_method_bullets,
            max_total_bullets=max_bullets,
        )

    # Anti false-negative: kalau model bilang "Tidak ditemukan" padahal konteks terlihat relevan,
    # coba 1x "force answer" (tetap hanya dari konteks).
    if (
        used_ctx > 0
        and _is_global_not_found(out)
        and (_looks_supported(question, contexts) or support_relaxed)
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
                    f"2) Maksimal {max_bullets} bullet.\n"
                    f"3) CTX valid hanya 1..{used_ctx}.\n"
                    "4) Jangan menambah fakta baru.\n"
                    "5) Jangan menjawab 'Tidak ditemukan' jika ada bukti di konteks.\n"
                    "6) Jangan menulis ulang 'Pertanyaan:' pada output.\n",
                ),
                (
                    "human",
                    "Jenis pertanyaan: {intent}\n\n"
                    "Pertanyaan:\n{question}\n\n"
                    "Konteks:\n{context}\n\n"
                    "Jawab sekarang:",
                ),
            ]
        )
        forced = (force_prompt | llm | StrOutputParser()).invoke(
            {
                "intent": q_intent,
                "question": question.strip(),
                "context": ctx_block,
                "max_bullets": max_bullets,
            }
        )
        forced = _strip_question_preamble((forced or "").strip())
        forced = _normalize_citations(forced)
        forced = _enforce_qa_format(forced)
        forced = _ensure_bukti_bullets(forced)
        forced = _cap_and_dedupe_bukti(forced, max_points=max_bullets)

        forced = _move_inline_cited_bullets_to_bukti(forced)
        forced = _cap_and_dedupe_bukti(forced, max_points=max_bullets)
        if multi_method_mode and targets:
            forced = _cap_bukti_per_method(
                forced,
                target_methods=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )
        if forced:
            out = forced

    # ======= Repair format / citation-range =======
    # 2) Validasi format dasar
    need_repair_format = not _is_valid_answer_format(out) or (not _bukti_has_bullets_or_dash(out))

    # 3) Validasi range sitasi (kalau jawaban bukan "Tidak ditemukan")
    need_repair_range = False
    if used_ctx > 0 and (not _is_global_not_found(out)):
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
                    f"2) Maksimal {max_bullets} bullet di Bukti.\n"
                    f"3) CTX yang tersedia hanya 1 sampai {used_ctx}.\n"
                    "4) Jangan menambah fakta baru. Jangan mengubah makna.\n"
                    "5) Jangan menulis ulang 'Pertanyaan:' di output.\n"
                    "6) Jika tidak ada dukungan konteks, tulis persis:\n"
                    "   Jawaban: Tidak ditemukan pada dokumen.\n"
                    "   Bukti: -",
                ),
                (
                    "human",
                    "Jenis pertanyaan: {intent}\n\n"
                    "Pertanyaan:\n{question}\n\n"
                    "Konteks:\n{context}\n\n"
                    "Jawaban sebelumnya (perlu diperbaiki):\n{answer}\n\n"
                    "Perbaiki sekarang:",
                ),
            ]
        )
        fixed = (repair_prompt | llm | StrOutputParser()).invoke(
            {
                "intent": q_intent,
                "question": question.strip(),
                "context": ctx_block,
                "answer": out,
                "max_bullets": max_bullets,
            }
        )
        fixed = _strip_question_preamble((fixed or "").strip())
        fixed = _normalize_citations(fixed)
        fixed = _enforce_qa_format(fixed)
        fixed = _ensure_bukti_bullets(fixed)
        fixed = _cap_and_dedupe_bukti(fixed, max_points=max_bullets)

        fixed = _move_inline_cited_bullets_to_bukti(fixed)
        fixed = _cap_and_dedupe_bukti(fixed, max_points=max_bullets)
        if fixed:
            out = fixed

        # setelah repair format, pastikan struktur multi-metode tetap utuh
        if multi_method_mode and targets:
            out = _normalize_multimethod_head(out, target_methods=targets)
            out = _ensure_method_headers_and_missing_lines(
                out,
                target_methods=targets,
                missing_methods=missing_clean,
            )
            out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
            out = _cap_bukti_per_method(
                out,
                target_methods=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )

    # ======= Grounding check untuk anti klaim tidak presisi =======
    # Hanya berlaku jika bukan "Tidak ditemukan"
    if used_ctx > 0 and (not _is_global_not_found(out)):
        grounding_ok = _bukti_grounding_ok(out, contexts, used_ctx)
        answer_ok = _jawaban_grounding_ok(out, contexts)

        if (not grounding_ok) or (not answer_ok):
            # 1x grounding repair attempt: paksa bukti benar-benar "mengutip/meringkas langsung" CTX
            grounding_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Perbaiki jawaban agar SELURUH isi Jawaban dan setiap poin Bukti benar-benar DIDUKUNG oleh CTX yang dirujuk.\n"
                        "Aturan:\n"
                        "1) Output WAJIB mulai dari 'Jawaban:' (tanpa 'Pertanyaan:').\n"
                        f"2) Bukti maksimal {max_bullets} bullet.\n"
                        "3) Setiap bullet harus merangkum atau mengutip singkat isi CTX yang sama (bukan asumsi).\n"
                        "4) Jangan menyebut metode/hal lain jika tidak ada di CTX yang dirujuk.\n"
                        f"5) CTX valid hanya 1..{used_ctx}.\n",
                    ),
                    (
                        "human",
                        "Jenis pertanyaan: {intent}\n\n"
                        "Pertanyaan:\n{question}\n\n"
                        "Konteks:\n{context}\n\n"
                        "Jawaban sebelumnya (ada indikasi bukti kurang grounded):\n{answer}\n\n"
                        "Perbaiki sekarang:",
                    ),
                ]
            )
            grounded = (grounding_prompt | llm | StrOutputParser()).invoke(
                {
                    "intent": q_intent,
                    "question": question.strip(),
                    "context": ctx_block,
                    "answer": out,
                    "max_bullets": max_bullets,
                }
            )
            grounded = _strip_question_preamble((grounded or "").strip())
            grounded = _normalize_citations(grounded)
            grounded = _enforce_qa_format(grounded)
            grounded = _ensure_bukti_bullets(grounded)
            grounded = _cap_and_dedupe_bukti(grounded, max_points=max_bullets)

            grounded = _move_inline_cited_bullets_to_bukti(grounded)
            grounded = _cap_and_dedupe_bukti(grounded, max_points=max_bullets)

            if grounded and _bukti_grounding_ok(grounded, contexts, used_ctx):
                out = grounded

                # setelah grounding repair, pastikan struktur multi-metode tetap utuh
                if multi_method_mode and targets:
                    out = _normalize_multimethod_head(out, target_methods=targets)
                    out = _ensure_method_headers_and_missing_lines(
                        out,
                        target_methods=targets,
                        missing_methods=missing_clean,
                    )
                    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
                    out = _cap_bukti_per_method(
                        out,
                        target_methods=targets,
                        per_method_bullets=per_method_bullets,
                        max_total_bullets=max_bullets,
                    )
            else:
                # fallback aman:
                # - jika pertanyaan memang terlihat supported oleh konteks -> boleh tampilkan bukti ekstraktif
                # - jika tidak supported (off-topic) -> global not_found
                if _looks_supported(question, contexts) or support_relaxed:
                    return _extractive_fallback_from_contexts(
                        contexts,
                        used_ctx,
                        intent=q_intent,
                        targets=targets,
                        per_method_bullets=per_method_bullets,
                        max_total_bullets=max_bullets,
                    )
                return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # ======= Guardrails terakhir =======
    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
    if multi_method_mode and targets:
        out = _cap_bukti_per_method(
            out,
            target_methods=targets,
            per_method_bullets=per_method_bullets,
            max_total_bullets=max_bullets,
        )

    # 5) Guardrail: kalau jawab bukan "Tidak ditemukan" tapi tidak ada sitasi → fallback aman
    if used_ctx > 0 and out and (not _is_global_not_found(out)) and (not _has_citation(out)):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts) or support_relaxed:
            return _extractive_fallback_from_contexts(
                contexts,
                used_ctx,
                intent=q_intent,
                targets=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # 6) Guardrail: kalau sitasi keluar dari range → fallback aman
    if (
        used_ctx > 0
        and out
        and (not _is_global_not_found(out))
        and (not _citations_in_range(out, used_ctx))
    ):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts) or support_relaxed:
            return _extractive_fallback_from_contexts(
                contexts,
                used_ctx,
                intent=q_intent,
                targets=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # 7) Guardrail: format tidak valid → fallback aman
    if not _is_valid_answer_format(out) or (not _bukti_has_bullets_or_dash(out)):
        # kalau sebenarnya ada sinyal relevan di konteks, jangan kosongkan jawaban total
        if _looks_supported(question, contexts) or support_relaxed:
            return _extractive_fallback_from_contexts(
                contexts,
                used_ctx,
                intent=q_intent,
                targets=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    # memastikan struktur multi-metode tetap utuh sampai akhir
    if multi_method_mode and targets:
        out = _normalize_multimethod_head(out, target_methods=targets)
        out = _ensure_method_headers_and_missing_lines(
            out,
            target_methods=targets,
            missing_methods=missing_clean,
        )

        # Micro-fix final: bersihkan lagi bila TAHAP + multi-metode
        if q_intent == "TAHAP":
            out = _clean_inline_method_bullets_in_jawaban(out)
            out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
            out = _cap_bukti_per_method(
                out,
                target_methods=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )

    # ===== Humanize khusus TAHAP + multi-metode (hindari Jawaban==Bukti & daftar ';') =====
    if used_ctx > 0 and (q_intent == "TAHAP") and multi_method_mode and targets:
        humanize_on = _get_cfg(cfg, ["generation", "humanize_steps_answer"], True)
        if bool(humanize_on):
            # humanize hanya aktif jika output memang belum konsisten
            # (cegah pipeline berlebih pada output yang sudah rapi)
            sc = _output_sanity_check(out, target_methods=targets, used_ctx=used_ctx)
            needs_humanize = (not sc["steps_on_newline"]) or (not sc["no_duplicate_methods"])

            if needs_humanize:
                out2 = _humanize_multimethod_steps_answer(
                    out,
                    question=question,
                    target_methods=targets,
                    contexts=contexts,
                    used_ctx=used_ctx,
                    max_items_per_method=_as_int(
                        _get_cfg(cfg, ["generation", "steps_max_items_per_method"], 5), 5
                    ),
                )
                if out2 and (out2 != out):
                    out = out2
                    # re-apply guardrails ringan
                    out = _normalize_citations(out)
                    out = _enforce_qa_format(out)
                    out = _ensure_bukti_bullets(out)
                    out = _cap_and_dedupe_bukti(out, max_points=max_bullets)
                    out = _cap_bukti_per_method(
                        out,
                        target_methods=targets,
                        per_method_bullets=per_method_bullets,
                        max_total_bullets=max_bullets,
                    )

    # ===== jika Bukti kosong tapi Jawaban berisi [CTX], bangun Bukti dari Jawaban =====
    if used_ctx > 0 and (not _is_global_not_found(out)):
        out = _ensure_bukti_from_head_if_missing(
            out,
            target_methods=(targets if (multi_method_mode and targets) else []),
            max_points=max_bullets,
        )
        out = _ensure_bukti_bullets(out)
        out = _cap_and_dedupe_bukti(out, max_points=max_bullets)

        # kalau multi-metode, rapikan lagi cap per-metode (agar bucket tidak kacau)
        if multi_method_mode and targets:
            out = _cap_bukti_per_method(
                out,
                target_methods=targets,
                per_method_bullets=per_method_bullets,
                max_total_bullets=max_bullets,
            )

    # ===== FINAL CLEANUP =====
    # dipanggil SETELAH _ensure_bukti_from_head_if_missing agar
    # info [CTX] sudah berpindah ke Bukti sebelum dihapus dari Jawaban.
    if used_ctx > 0 and (not _is_global_not_found(out)):
        out = _fix_empty_numbered_lines(out)  # bersihkan nomor kosong
        out = _strip_ctx_from_jawaban(out)    # hapus [CTX] dari Jawaban

    return out


def build_generation_meta(
    *,
    question: str,
    contexts: List[str],
    answer_text: str,
    used_ctx: int,
    max_bullets: int = 3,
) -> Dict[str, Any]:
    """
    Metadata ringan untuk logging/audit.
    Tidak mengubah jawaban, hanya menganalisis:
    - format output (Jawaban/Bukti)
    - keberadaan sitasi dan range CTX
    - status akhir (ok / not_found / invalid_format / invalid_citation / error / empty)
    - sinyal relevansi (heuristik overlap keyword)

    Tambahan:
    - max_bullets: batas bullet saat turn ini
    - bukti_count: jumlah bullet Bukti yang terbaca
    - bukti_overflow: apakah bullet melewati max_bullets (harusnya false karena saya cap)
    - grounding_ok: semua bullet Bukti grounded ke CTX yang dirujuk
    - answer_grounding_ok: isi Jawaban grounded ke konteks gabungan (konservatif)
    - partial_not_found: ada sebagian metode yang not_found (untuk audit)
    """
    t = (answer_text or "").strip()
    supported = _looks_supported(question, contexts) if used_ctx > 0 else False

    bullets = _extract_bukti_lines(t)
    bukti_count = len(bullets) if bullets else 0
    max_bullets = int(max(1, min(max_bullets, 8)))

    # default meta
    meta: Dict[str, Any] = {
        "used_ctx": int(used_ctx),
        "max_bullets": int(max_bullets),
        "supported_heuristic": bool(supported),
        "format_ok": False,
        "bukti_ok": False,
        "bukti_count": int(bukti_count),
        "bukti_overflow": bool(bukti_count > max_bullets),
        "citations_present": False,
        "citations_in_range": False,
        "grounding_ok": True,  # default True jika tidak ada bukti / Bukti: -
        "answer_grounding_ok": True,
        "ctx_nums": [],
        "status": "empty" if not t else "ok",
    }

    if not t:
        return meta

    # =========================
    # Global not_found (ONLY)
    # =========================
    # Pakai helper global yang sama dengan generate_answer(),
    # agar status meta konsisten dengan guardrail generator.
    # dan mencegah false-trigger not_found saat hanya 1 metode yang "Tidak ditemukan".
    if _is_global_not_found(t):
        meta["status"] = "not_found"
        meta["format_ok"] = _is_valid_answer_format(t)
        meta["bukti_ok"] = _bukti_has_bullets_or_dash(t)
        meta["citations_present"] = _has_citation(t)
        meta["ctx_nums"] = _extract_ctx_nums(t)
        meta["citations_in_range"] = _citations_in_range(t, used_ctx)
        meta["grounding_ok"] = True
        meta["answer_grounding_ok"] = True
        meta["bukti_overflow"] = False
        return meta

    # =========================
    # Partial not_found (audit)
    # =========================
    head_only = _jawaban_head_only(t)
    head_body = re.sub(r"(?im)^\s*Jawaban\s*:\s*", "", head_only).strip()
    lines = [ln.strip() for ln in head_body.splitlines() if ln.strip()]

    methods_nf: List[str] = []
    for ln in lines:
        mline = re.match(r"(?im)^([A-Za-z_ ].+?)\s*:\s*(.*)$", ln)
        if mline and ("tidak ditemukan pada dokumen" in (mline.group(2) or "").lower()):
            methods_nf.append(mline.group(1).strip())

    if methods_nf:
        meta["partial_not_found"] = True
        meta["methods_not_found"] = methods_nf

    # =========================
    # Normal validation
    # =========================
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

    meta["answer_grounding_ok"] = _jawaban_grounding_ok(t, contexts) if used_ctx > 0 else True

    # =========================
    # Final status
    # turunkan status jika ada masalah
    # =========================
    if not meta["format_ok"] or not meta["bukti_ok"]:
        meta["status"] = "invalid_format"
    elif used_ctx > 0 and (not meta["citations_present"]):
        meta["status"] = "missing_citation"
    elif used_ctx > 0 and (not meta["citations_in_range"]):
        meta["status"] = "invalid_citation_range"
    elif used_ctx > 0 and (not meta["grounding_ok"]):
        meta["status"] = "weak_grounding_bukti"
    elif used_ctx > 0 and (not meta["answer_grounding_ok"]):
        meta["status"] = "weak_grounding_jawaban"
    elif meta["bukti_overflow"]:
        meta["status"] = "bukti_overflow"
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
