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
        render_answer_card_header, render_bukti_section,
        render_ctx_mapping_header, render_sources_header,
        render_pdf_viewer_header, render_viewing_banner,
        sim_color_badge, render_ctx_mapping_table, render_source_score_pills,
    )

Tidak mengimpor dari app_streamlit.py (mencegah circular import).
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
    <div style="font-weight:800;">Processed ⚙️ {badge}</div>
  </div>
  <div style="color:#e5e7eb; line-height:1.45;">
    {body}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Styled section headers & answer cards
# =========================

def render_answer_card_header() -> None:
    """
    Render styled header bar untuk section Jawaban.
    Wajib dipanggil di dalam st.columns() bersama copy_to_clipboard_button()
    agar tombol copy tetap sejajar di kanan.
    """
    st.markdown(
        '<div style="'
        "border-left:4px solid #3b82f6;"
        "background:linear-gradient(90deg,#1e3a5f33,transparent);"
        "padding:0.45rem 0.85rem;"
        "border-radius:0 0.4rem 0.4rem 0;"
        "margin:0.5rem 0 0.15rem 0;"
        '">'
        '<span style="font-size:1.2rem;font-weight:800;color:#93c5fd;">'
        "💌 Jawaban"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_bukti_section(bukti: List[str]) -> None:
    """
    Render section Bukti lengkap: styled header + bullet items.
    Menggantikan st.subheader("Bukti") + for-loop bullets di app_streamlit.py.
    Konten bullet tetap memakai st.markdown() agar [CTX n] tampil benar (tidak di-escape).
    """
    # Styled header
    st.markdown(
        '<div style="'
        "border-left:4px solid #f59e0b;"
        "background:linear-gradient(90deg,#3f2a0033,transparent);"
        "padding:0.45rem 0.85rem;"
        "border-radius:0 0.4rem 0.4rem 0;"
        "margin:1.1rem 0 0.4rem 0;"
        '">'
        '<span style="font-size:1.1rem;font-weight:800;color:#fcd34d;">'
        "🗞️ Bukti"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not bukti:
        st.caption("— Tidak ada bukti sitasi.")
        return

    # Render semua bullet dalam 1 blok markdown (lebih efisien dari loop individual)
    # Tetap memakai st.markdown() agar [CTX n] tidak di-escape sebagai HTML
    with st.container(border=True):
        bullet_md = "\n".join(f"- {b}" for b in bukti)
        st.markdown(bullet_md)


def render_ctx_mapping_header() -> None:
    """
    Render styled header untuk section CTX Mapping.
    Menggantikan st.subheader("CTX Mapping") di app_streamlit.py.
    """
    st.markdown(
        '<div style="'
        "border-left:4px solid #8b5cf6;"
        "background:linear-gradient(90deg,#2e1a5533,transparent);"
        "padding:0.45rem 0.85rem;"
        "border-radius:0 0.4rem 0.4rem 0;"
        "margin:1.1rem 0 0.4rem 0;"
        '">'
        '<span style="font-size:1.1rem;font-weight:800;color:#c4b5fd;">'
        "🗂️ CTX Mapping"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sources_header() -> None:
    """
    Render styled header untuk section Sources (Top-k).
    Menggantikan st.subheader("Sources (Top-k)") di app_streamlit.py.
    """
    st.markdown(
        '<div style="'
        "border-left:4px solid #10b981;"
        "background:linear-gradient(90deg,#05302033,transparent);"
        "padding:0.45rem 0.85rem;"
        "border-radius:0 0.4rem 0.4rem 0;"
        "margin:1.1rem 0 0.4rem 0;"
        '">'
        '<span style="font-size:1.1rem;font-weight:800;color:#6ee7b7;">'
        "🌐 Sources (Top-k)"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_pdf_viewer_header(filename: str, page: int) -> None:
    """
    Render styled header untuk section PDF Viewer.
    Menggantikan st.subheader("PDF Viewer") di app_streamlit.py.
    """
    caption_text = f"Menampilkan <b>{filename} — halaman {page}</b>"
    st.markdown(
        '<div style="'
        "border-left:4px solid #ec4899;"
        "background:linear-gradient(90deg,#3f054433,transparent);"
        "padding:0.45rem 0.85rem;"
        "border-radius:0 0.4rem 0.4rem 0;"
        "margin:1.1rem 0 0.4rem 0;"
        '">'
        '<span style="font-size:1.1rem;font-weight:800;color:#f9a8d4;">'
        "📄 PDF Viewer"
        "</span>"
        f'<span style="font-size:0.8rem;color:#9ca3af;margin-left:0.75rem;">'
        f"{caption_text}"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# =========================
# CTX & Sources visual polish 
# =========================

# Threshold similarity (dipakai konsisten di semua fungsi)
_SIM_HIGH = 0.70    # >= HIGH  → hijau
_SIM_MED  = 0.55    # >= MED   → kuning / sedang
                    # <  MED   → merah  / rendah


def sim_color_badge(sim: float) -> str:
    """
    Return emoji badge berdasarkan nilai similarity.
    Dipakai di title Sources expander agar ada visual cue relevansi tanpa buka expander.

    Threshold:
        >= 0.70 → 🟢 (tinggi)
        >= 0.55 → 🟡 (sedang)
        <  0.55 → 🔴 (rendah)
    """
    if sim >= _SIM_HIGH:
        return "🟢"
    elif sim >= _SIM_MED:
        return "🟡"
    return "🔴"


def _sim_hex(sim: float) -> str:
    """Return hex warna untuk nilai sim (dipakai di HTML render internal)."""
    if sim >= _SIM_HIGH:
        return "#4ade80"   # green-400
    elif sim >= _SIM_MED:
        return "#facc15"   # yellow-400
    return "#f87171"       # red-400


def _sim_label(sim: float) -> str:
    """Return label teks untuk nilai sim."""
    if sim >= _SIM_HIGH:
        return "Tinggi"
    elif sim >= _SIM_MED:
        return "Sedang"
    return "Rendah"


def render_ctx_mapping_table(ctx_rows: List[Dict[str, Any]]) -> None:
    """
    Render CTX Mapping sebagai st.dataframe() native dengan kolom `sim` color-coded
    menggunakan pandas Styler.

    Kolom:  CTX | doc_id | page | source_file | sim ↑ | dist ↓ | chunk_id

    Threshold warna kolom `sim`:
        >= 0.70 → #4ade80  (hijau  / tinggi)
        >= 0.55 → #facc15  (kuning / sedang)
        <  0.55 → #f87171  (merah  / rendah)
    """
    if not ctx_rows:
        st.caption("— Tidak ada CTX untuk ditampilkan.")
        return

    try:
        import pandas as pd
    except ImportError:
        # Fallback aman: tampilkan sebagai plain dataframe tanpa styling
        st.dataframe(ctx_rows, use_container_width=True, hide_index=True)
        return

    df = pd.DataFrame(ctx_rows)

    # Pastikan urutan kolom konsisten
    col_order = ["CTX", "doc_id", "page", "source_file", "sim", "dist", "chunk_id"]
    df = df[[c for c in col_order if c in df.columns]]

    def _style_sim(val: float) -> str:
        """CSS inline untuk sel kolom sim."""
        if val >= _SIM_HIGH:
            return "color: #4ade80; font-weight: bold"
        elif val >= _SIM_MED:
            return "color: #facc15; font-weight: bold"
        return "color: #f87171; font-weight: bold"

    # Pylance-safe: try/except menghindari false-positive "None cannot be called"
    # type: ignore comments diperlukan karena Pylance salah infer tipe Scalar vs float
    try:
        styled = df.style.map(_style_sim, subset=["sim"])  # type: ignore[arg-type]
    except AttributeError:
        styled = df.style.applymap(_style_sim, subset=["sim"])  # type: ignore[attr-defined]

    st.dataframe(styled, width="stretch", hide_index=True)

    # Legend threshold (di bawah tabel, ukuran kecil muted)
    st.markdown(
        '<div style="display:flex;gap:1.1rem;margin-top:-0.8rem;margin-bottom:1.4rem;flex-wrap:wrap;'
        'font-size:0.75rem;color:#64748b;padding-left:0.1rem;">'
        "<span>🟢 sim &ge; 0.70 &nbsp;(Tinggi)</span>"
        "<span>🟡 sim &ge; 0.55 &nbsp;(Sedang)</span>"
        "<span>🔴 sim &lt; 0.55 &nbsp;(Rendah)</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_source_score_pills(sim: float, dist: float, ctx_label: str = "") -> None:
    """
    Render mini score pills (CTX label + sim + dist) di awal isi Sources expander.
    Memberikan visual feedback relevansi tanpa perlu membaca angka di title expander.

    Dipakai tepat sebelum st.caption("Preview: ...") di dalam st.expander() Sources.
    """
    sim_color = _sim_hex(sim)
    sim_lbl   = _sim_label(sim)

    # Pill CTX label
    ctx_pill = ""
    if ctx_label:
        ctx_escaped = html.escape(ctx_label)
        ctx_pill = (
            f'<span style="background:#1e3a5f;color:#93c5fd;'
            f'padding:2px 8px;border-radius:999px;font-size:0.75rem;'
            f'font-family:monospace;font-weight:600;">{ctx_escaped}</span>'
        )

    # Pill sim (warna sesuai threshold)
    sim_pill = (
        f'<span style="background:#0d1f17;border:1px solid {sim_color}55;'
        f'color:{sim_color};padding:2px 10px;border-radius:999px;'
        f'font-size:0.78rem;font-weight:700;font-family:monospace;">'
        f'sim&nbsp;{sim:.4f}'
        f'<span style="font-size:0.7rem;opacity:0.7;margin-left:4px;">({sim_lbl})</span>'
        f'</span>'
    )

    # Pill dist (abu-abu muted, tidak perlu threshold warna)
    dist_pill = (
        f'<span style="background:#1a1f2e;border:1px solid #33415555;'
        f'color:#64748b;padding:2px 10px;border-radius:999px;'
        f'font-size:0.78rem;font-family:monospace;">'
        f'dist&nbsp;{dist:.4f}'
        f'</span>'
    )

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.45rem;'
        f'margin:0.25rem 0 0.55rem 0;flex-wrap:wrap;">'
        f"{ctx_pill}{sim_pill}{dist_pill}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_viewing_banner(turn_id: int) -> None:
    """
    Render banner informatif saat user sedang melihat turn historis (bukan turn terbaru).
    Dipanggil di atas result area ketika viewing_turn_id != None.
    """
    st.markdown(
        f'<div style="'
        f"background:rgba(37,99,235,0.08);"
        f"border:1px solid #2563eb;"
        f"border-left:4px solid #3b82f6;"
        f"padding:0.55rem 1rem;"
        f"border-radius:0.6rem;"
        f"margin:0.4rem 0 0.8rem 0;"
        f"display:flex;align-items:center;gap:0.6rem;"
        f'">'
        f'<span style="font-size:1.05rem;">🔆</span>'
        f'<span style="color:#60a5fa;font-weight:700;font-size:0.95rem;">'
        f"Menampilkan Chat-Turn {turn_id}"
        f"</span>"
        f'<span style="color:#93c5fd;opacity:0.7;font-size:0.82rem;margin-left:0.25rem;">'
        f"— <i>Hasil di bawah merupakan historis chat, bukan turn terbaru.</i>"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )