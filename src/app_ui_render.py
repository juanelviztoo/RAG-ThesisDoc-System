"""
app_ui_render.py
================
Render & display helpers yang diekstrak dari app_streamlit.py.
Berisi fungsi-fungsi yang bersifat pure Python / thin Streamlit render.

Diimpor oleh app_streamlit.py:
    from src.app_ui_render import (
        dist_to_sim, compute_retrieval_stats,
        parse_answer, is_global_not_found_answer, build_retrieval_only_answer,
        node_matches_filter, extract_terms_from_query, extract_terms_from_text,
        _prepare_jawaban_markdown, render_processing_box,
        build_contexts_with_meta,
    )

Tidak mengimpor dari app_streamlit.py (cegah circular import).
"""
from __future__ import annotations

import html
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import streamlit as st

# =========================
# Similarity helpers
# =========================

def dist_to_sim(dist: float) -> float:
    """Konversi distance ChromaDB → similarity (0..1). Lebih aman dari (1 - dist)."""
    return 1.0 / (1.0 + max(dist, 0.0))


def compute_retrieval_stats(nodes: List[Any]) -> Dict[str, Any]:
    """Hitung statistik ringkas dari hasil retrieval untuk keperluan logging."""
    if not nodes:
        return {
            "n_nodes": 0,
            "n_docs": 0,
            "doc_counts_top": {},
            "sim_top1": None,
            "sim_avg": None,
            "sim_gap12": None,
            "dist_top1": None,
        }

    sims = [dist_to_sim(float(n.score)) for n in nodes]
    sim_top1 = sims[0]
    dist_top1 = float(nodes[0].score)
    sim_avg = sum(sims) / max(len(sims), 1)
    sim_gap12 = (sims[0] - sims[1]) if len(sims) > 1 else None

    doc_ids = [
        str(
            getattr(n, "doc_id", "")
            or (n.metadata.get("doc_id") if getattr(n, "metadata", None) else "")
            or "unknown"
        )
        for n in nodes
    ]
    c = Counter(doc_ids)
    doc_counts_top = dict(c.most_common(5))

    return {
        "n_nodes": int(len(nodes)),
        "n_docs": int(len(c)),
        "doc_counts_top": doc_counts_top,
        "sim_top1": round(float(sim_top1), 6),
        "sim_avg": round(float(sim_avg), 6),
        "sim_gap12": round(float(sim_gap12), 6) if sim_gap12 is not None else None,
        "dist_top1": round(float(dist_top1), 6),
    }


# =========================
# Answer parsing & helpers
# =========================

def parse_answer(answer_text: str) -> Tuple[str, List[str]]:
    """
    Parse output generator dengan format:
        Jawaban: ...
        Bukti:
        - ... [CTX n]

    Robust terhadap preamble 'Pertanyaan:', dan variasi spasi/newline.
    - Jika ada blok "Pertanyaan:" sebelum "Jawaban:", buang semuanya sebelum "Jawaban:"
    - Ambil isi setelah "Jawaban:" sampai sebelum "Bukti:"
    - Bukti diambil dari bullet '-' setelah 'Bukti:

    Return: (jawaban_text, list_bukti)
    """
    text = (answer_text or "").strip()
    if not text:
        return "", []

    # Jika ada preamble "Pertanyaan:" sebelum "Jawaban:", buang semuanya sebelum "Jawaban:"
    m_jstart = re.search(r"(?im)^\s*Jawaban\s*:\s*", text)
    if m_jstart and m_jstart.start() > 0:
        text = text[m_jstart.start():].strip()

    # cari "Bukti:" (case-insensitive)
    m_bukti = re.search(r"\bBukti\s*:\s*", text, flags=re.IGNORECASE)
    if m_bukti:
        head = text[:m_bukti.start()].strip()
        tail = text[m_bukti.end():].strip()

        # ambil isi jawaban = setelah "Jawaban:"
        m_j = re.search(r"\bJawaban\s*:\s*", head, flags=re.IGNORECASE)
        if m_j:
            jawaban = head[m_j.end():].strip()
        else:
            # fallback: buang baris "Pertanyaan:" bila masih ada
            jawaban = re.sub(r"(?im)^\s*Pertanyaan\s*:\s*.*$", "", head).strip()

        # handle "Bukti: -" atau kosong
        if not tail or tail == "-":
            return jawaban, []

        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        bukti: List[str] = []
        for ln in lines:
            if ln == "-":
                continue
            if ln.startswith("-"):
                bukti.append(ln[1:].strip())
            else:
                bukti.append(ln)

        # kalau hasilnya tetap "-" aja
        if len(bukti) == 1 and bukti[0] == "-":
            bukti = []

        return jawaban, bukti

    # kalau tidak ada "Bukti:", coba ambil "Jawaban:" kalau ada
    m_j = re.search(r"^\s*Jawaban\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m_j:
        return m_j.group(1).strip(), []

    # fallback total: semua dianggap jawaban
    return text, []


_NOT_FOUND_GLOBAL_UI_RE = re.compile(
    r"(?is)^\s*Jawaban\s*:\s*Tidak\s+ditemukan\s+pada\s+dokumen\.\s*Bukti\s*:\s*-\s*$"
)


def is_global_not_found_answer(answer_text: str) -> bool:
    """True jika jawaban adalah global 'Tidak ditemukan pada dokumen.' (bukan partial)."""
    t = (answer_text or "").strip()
    if not t:
        return True
    # normalisasi ringan: pastikan "Bukti:\n-" dianggap sama dengan "Bukti: -"
    t = re.sub(r"(?im)^\s*Bukti\s*:\s*\n\s*-\s*$", "Bukti: -", t).strip()
    return bool(_NOT_FOUND_GLOBAL_UI_RE.match(t))


def build_retrieval_only_answer(nodes: List[Any], max_points: int = 3) -> str:
    """
    Jawaban extractive saat LLM dimatikan.
    Output mengikuti format generator agar UI tetap konsisten.
    """
    if not nodes:
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    bullets: List[str] = []
    for i, n in enumerate(nodes[:max_points], start=1):
        source_file = n.metadata.get("source_file", "")
        page = n.metadata.get("page", "?")

        snippet = (n.text or "").strip().replace("\n", " ")
        snippet = " ".join(snippet.split()) # rapikan whitespace
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")

        bullets.append(f"- ({source_file} p.{page}) {snippet} [CTX {i}]")

    return (
        "Jawaban: (Mode retrieval-only) Berikut bukti paling relevan dari dokumen untuk pertanyaan ini.\n"
        "Bukti:\n" + "\n".join(bullets)
    )


def node_matches_filter(n: Any, q: str) -> bool:
    """True jika node mengandung filter string q (dalam doc_id, source_file, atau text)."""
    if not q:
        return True
    ql = q.lower().strip()
    doc_id = (getattr(n, "doc_id", "") or "").lower()
    sf = (n.metadata.get("source_file", "") or "").lower()
    txt = (getattr(n, "text", "") or "").lower()
    return (ql in doc_id) or (ql in sf) or (ql in txt)


# =========================
# Highlight term extraction
# =========================

def extract_terms_from_query(q: str) -> List[str]:
    """Ambil keyword sederhana dari query untuk auto-highlight (default) di PDF viewer."""
    ql = (q or "").lower()
    tokens = re.findall(r"[a-z0-9]+", ql)

    stop = {
        "apa", "yang", "dan", "atau", "dengan", "dalam", "pada", "untuk",
        "dari", "ke", "ini", "itu", "adalah", "menurut", "jelaskan",
        "sebutkan", "minimal", "jika", "ada", "metode", "evaluasi",
        "sistem", "dokumen",
    }
    terms = [t for t in tokens if len(t) >= 4 and t not in stop]
    # ambil beberapa saja agar lebih ringan
    return terms[:5]


def extract_terms_from_text(text: str, max_terms: int = 6) -> List[str]:
    """Ambil keyword dari teks CTX untuk auto-highlight."""
    tokens = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    stop = {
        "yang", "dengan", "dalam", "pada", "untuk", "dari", "atau", "dan",
        "adalah", "ini", "itu", "sebagai", "dapat", "akan", "oleh",
        "menggunakan", "metode", "sistem", "penelitian", "hasil",
        "bab", "tabel", "gambar",
    }
    out, seen = [], set()
    for t in tokens:
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


# =========================
# Context builder
# =========================

def build_contexts_with_meta(nodes: List[Any]) -> List[str]:
    """
    Perkaya context dengan metadata dokumen agar LLM bisa eksplisit lintas dokumen.
    Format: DOC: <doc_id> | <source_file> p.<page>
    """
    out: List[str] = []
    for n in nodes:
        source_file = str((getattr(n, "metadata", {}) or {}).get("source_file", "") or "")
        page = (getattr(n, "metadata", {}) or {}).get("page", "?")
        try:
            page_str = str(int(page))
        except Exception:
            page_str = str(page) if page is not None else "?"

        doc_id = str(getattr(n, "doc_id", "") or "unknown")
        header = f"DOC: {doc_id} | {source_file} p.{page_str}".strip()
        body = str(getattr(n, "text", "") or "").strip()
        if header:
            out.append(f"{header}\n{body}".strip())
        else:
            out.append(body)
    return out


# =========================
# Jawaban Markdown formatter
# =========================

def _prepare_jawaban_markdown(jawaban: str) -> str:
    """
    Format teks jawaban multi-metode menjadi Markdown yang rapi di Streamlit.
    Strategi section-aware (bukan line-by-line):
    1. Split teks menjadi section per metode
    2. Tiap header metode di-bold: **RAD:**
    3. Tiap section diberi double newline
    4. Numbered list dipaksa mulai dari 1 dengan blank line sebelumnya
        dengan menyisipkan blank line di antara deskripsi dan list item pertama
    """
    t = (jawaban or "").strip()
    if not t:
        return t

    # --- Regex pattern untuk mengenali header metode ---
    _METHOD_NAMES = [
        "RAD", "Prototyping", "RUP",
        "Extreme Programming", "Waterfall", "Agile",
        "SDLC", "Scrum",
    ]
    _HDR_PAT = r"^(" + "|".join(re.escape(m) for m in _METHOD_NAMES) + r")\s*:\s*(.*)$"
    _HDR_RE = re.compile(_HDR_PAT, re.IGNORECASE)
    _NUM_RE = re.compile(r"^\s*(\d+)[\.\)]\s+(.+)$")

    # --- Langkah 1: Pastikan tiap header metode di baris sendiri ---
    for name in _METHOD_NAMES:
        escaped = re.escape(name)
        t = re.sub(
            rf"(?m)(?<=[^\n])(\s*)({escaped}\s*:)",
            r"\n\2",
            t,
            flags=re.IGNORECASE,
        )

    lines = t.splitlines()

    # --- Langkah 2: Parse menjadi section ---
    # Setiap section = {"header": str|None, "lines": [str]}
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"header": None, "rest": "", "lines": []}

    for ln in lines:
        stripped = ln.strip()
        m_hdr = _HDR_RE.match(stripped)
        if m_hdr:
            sections.append(current)
            label = m_hdr.group(1).strip()
            rest_txt = (m_hdr.group(2) or "").strip()
            current = {"header": label, "rest": rest_txt, "lines": []}
        else:
            current["lines"].append(ln)

    sections.append(current)

    # --- Langkah 3: Render tiap section ---
    output_parts: List[str] = []

    for sec in sections:
        hdr = sec.get("header")
        rest = sec.get("rest", "").strip()
        lines_sec = sec.get("lines", [])

        # Kumpulkan teks non-list dan list-item secara terpisah
        prose_lines: List[str] = []
        list_items: List[str] = []

        # Pre-split: pecah baris yang mengandung "N. item" inline di akhir prosa
        # Contoh: "...dari sistem. 1. Perancangan alur data"
        #      → ["...dari sistem.", "1. Perancangan alur data"]
        _INLINE_NUM_RE = re.compile(r"\s+(\d+[\.\)]\s+\S)")
        expanded_lines: List[str] = []
        for ln in lines_sec:
            stripped = ln.strip()
            if not stripped:
                continue
            # Cek apakah ada "N. teks" yang muncul di tengah/akhir baris (bukan di awal)
            if not _NUM_RE.match(stripped):
                m_inline = _INLINE_NUM_RE.search(stripped)
                if m_inline:
                    # Pisahkan: bagian sebelum "N." jadi prosa, sisanya menjadi baris sendiri
                    split_pos = m_inline.start()
                    before = stripped[:split_pos].strip()
                    after = stripped[split_pos:].strip()
                    if before:
                        expanded_lines.append(before)
                    # "after" bisa mengandung lebih dari satu item inline ("1. A 2. B")
                    # Pecah lebih lanjut
                    sub_parts = re.split(r"(?=\d+[\.\)]\s+\S)", after)
                    for part in sub_parts:
                        p = part.strip()
                        if p:
                            expanded_lines.append(p)
                else:
                    expanded_lines.append(stripped)
            else:
                expanded_lines.append(stripped)

        for ln in expanded_lines:
            if not ln.strip():
                continue
            m_num = _NUM_RE.match(ln.strip())
            if m_num:
                list_items.append(m_num.group(2).strip())
            else:
                prose_lines.append(ln.strip())

        # Bangun blok untuk section ini
        block_parts: List[str] = []

        # Header (bold jika ada)
        if hdr:
            block_parts.append(f"**{hdr}:**")

        # "rest" = teks setelah header di baris yang sama (mis. "Potongan konteks...")
        if rest:
            block_parts.append(rest)

        # Teks prosa lainnya
        if prose_lines:
            block_parts.append("\n".join(prose_lines))

        # Numbered list — selalu mulai dari 1, tiap item di baris baru
        # Blank line sebelum list (wajib agar Markdown render sebagai <ol>)
        if list_items:
            list_block = "\n".join(
                f"{i + 1}. {item}" for i, item in enumerate(list_items)
            )
            block_parts.append("")  # blank line sebelum list
            block_parts.append(list_block)

        if block_parts:
            output_parts.append("\n".join(block_parts))

    # --- Langkah 4: Gabung section dengan double newline ---
    return "\n\n".join(p for p in output_parts if p.strip())


# =========================
# Processing box renderer
# =========================

def render_processing_box(user_q: str, retrieval_q: str) -> None:
    """
    Render box ringkas yang menampilkan user query vs retrieval query (V1.5).
    Badge 'contextualized' / 'direct' menunjukkan apakah query direwrite.
    """
    user_q = (user_q or "").strip()
    retrieval_q = (retrieval_q or "").strip()

    if not user_q and not retrieval_q:
        return

    # escape agar aman jikalau ada karakter aneh
    user_q_esc = html.escape(user_q)
    retrieval_q_esc = html.escape(retrieval_q)

    # Badge kecil: apakah query direwrite?
    contextualized = bool(retrieval_q) and (retrieval_q != user_q)

    badge = (
        '<span style="background:#3b82f6; color:#fff; padding:2px 8px; '
        'border-radius:999px; font-size:12px; margin-left:8px;">contextualized</span>'
        if contextualized
        else '<span style="background:#16a34a; color:#fff; padding:2px 8px; '
        'border-radius:999px; font-size:12px; margin-left:8px;">direct</span>'
    )

    lines = []
    if user_q:
        lines.append(f"<div style='margin-top:6px;'><b>User Query:</b><br/>{user_q_esc}</div>")
    if retrieval_q and retrieval_q != user_q:
        lines.append(
            f"<div style='margin-top:10px;'><b>Retrieval Query (Contextualized):</b><br/>{retrieval_q_esc}</div>"
        )

    body = "\n".join(lines)

    st.markdown(
        f"""
<div style="
  background:#23262d;
  border:1px solid #3a3f47;
  padding:0.85rem 1rem;
  border-radius:0.75rem;
  margin:0.5rem 0 1rem 0;
">
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div style="font-weight:800;">Processing ⚙️ {badge}</div>
  </div>
  <div style="color:#e5e7eb; line-height:1.45;">
    {body}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )