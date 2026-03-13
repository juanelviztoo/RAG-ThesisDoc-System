from __future__ import annotations

import html as _html_mod
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.app_ui_render import (
    _prepare_jawaban_markdown,
    build_contexts_with_meta,
    build_retrieval_only_answer,
    compute_retrieval_stats,
    dist_to_sim,
    extract_terms_from_query,
    extract_terms_from_text,
    is_global_not_found_answer,
    node_matches_filter,
    parse_answer,
    render_answer_card_header,
    render_bukti_section,
    render_ctx_mapping_header,
    render_ctx_mapping_table,
    render_pdf_viewer_header,
    render_processing_box,
    render_source_score_pills,
    render_sources_header,
    render_viewing_banner,
    sim_color_badge,
)
from src.core.config import ConfigPaths, load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager
from src.core.ui_utils import (
    copy_to_clipboard_button,
    download_file_button,
    download_logs_zip_button,
    render_pdf_viewer,
    resolve_pdf_path,
    scroll_to_anchor,
)
from src.rag.generate import build_generation_meta, contextualize_question, generate_answer
from src.rag.metadata_router import maybe_route_metadata_query
from src.rag.method_detection import (
    METHOD_PATTERNS as _METHOD_PATTERNS,
)
from src.rag.method_detection import (
    detect_methods_in_contexts,
    detect_methods_in_text,
    has_steps_signal,
    is_anaphora_question,
    is_method_question,
    is_multi_doc_question,
    is_multi_target_question,
    is_steps_question,
    node_supports_method_for_coverage,
    pretty_method_name,
)
from src.rag.method_detection import (
    docid_hit_for_method as _docid_hit_for_method,
)
from src.rag.retrieve_dense import retrieve_dense

# =========================
# Doc-focus lock helpers
# =========================

def _node_doc_id(n: Any) -> str:
    return str(
        getattr(n, "doc_id", "")
        or (n.metadata.get("doc_id") if getattr(n, "metadata", None) else "")
        or "unknown"
    )


def _node_text(n: Any) -> str:
    return str(getattr(n, "text", "") or "")


def build_method_doc_map_from_nodes(nodes: List[Any], methods: List[str]) -> Dict[str, str]:
    """
    Map metode -> doc_id yang paling awal/kuat menyebut metode itu di nodes.
    Reuse _METHOD_PATTERNS yang sudah ada di file ini.
    """
    out: Dict[str, str] = {}
    if not nodes or not methods:
        return out

    for m in methods:
        pats = _METHOD_PATTERNS.get(m, [])
        if not pats:
            continue
        for n in nodes:
            txt = _node_text(n)
            if not txt:
                continue
            for p in pats:
                if re.search(p, txt, flags=re.IGNORECASE):
                    out[m] = _node_doc_id(n)
                    break
            if m in out:
                break
    return out


def apply_doc_focus_filter(
    nodes: List[Any],
    allowed_doc_ids: List[str],
    *,
    min_keep: int = 3,
    strict: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Filter nodes agar hanya berasal dari dokumen tertentu.
    - allowed_doc_ids boleh berisi doc_id ATAU source_file (mis. "...Prototyping.pdf")
    - strict=True: TIDAK fallback ke nodes awal walau hasil sedikit/kosong (hard lock)
    """
    meta: Dict[str, Any] = {
        "enabled": True,
        "strict": bool(strict),
        "doc_ids": [],
        "reason": None,
        "kept": 0,
        "total": len(nodes or []),
    }

    if not nodes:
        meta["reason"] = "no_nodes"
        return [], meta

    # sanitasi + dedupe
    allow_raw = [str(x).strip() for x in (allowed_doc_ids or []) if str(x).strip()]
    allow = list(dict.fromkeys(allow_raw))
    meta["doc_ids"] = allow

    if not allow:
        meta["reason"] = "empty_allowlist"
        return nodes, meta

    allow_l = [a.lower() for a in allow]

    kept: List[Any] = []
    for n in nodes:
        doc_id = str(getattr(n, "doc_id", "") or "").strip()
        doc_l = doc_id.lower()

        md = getattr(n, "metadata", {}) or {}
        sf = str(md.get("source_file", "") or "").strip()
        sf_l = sf.lower()

        hit = False
        for a in allow_l:
            if a.endswith(".pdf"):
                # match ke source_file
                if a == sf_l or a in sf_l:
                    hit = True
                    break
                # juga izinkan match ke doc_id (tanpa .pdf)
                a2 = a[:-4]
                if a2 and (a2 == doc_l or a2 in doc_l):
                    hit = True
                    break
            else:
                # match ke doc_id, dan juga ke source_file jikalau user menulis tanpa .pdf
                if a == doc_l or a in doc_l or a == sf_l or a in sf_l:
                    hit = True
                    break

        if hit:
            kept.append(n)

    meta["kept"] = len(kept)

    # HARD LOCK
    if strict:
        meta["reason"] = "strict_lock"
        return kept, meta

    # soft-lock: jikalau terlalu sedikit, fallback ke nodes awal (perilaku lama)
    if len(kept) < int(min_keep):
        meta["reason"] = "too_few_kept_fallback"
        return nodes, meta

    meta["reason"] = "filtered_ok"
    return kept, meta


# =========================
# Session / Run lifecycle
# =========================
def _new_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    """
    1 session Streamlit = 1 run_id (folder runs/<run_id>/).
    Jika config path berubah, buat run baru.
    _new_run() juga reset state penting agar UI & log tidak tercampur antar sesi.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])
    rm = RunManager(runs_dir=runs_dir, config=cfg).start()
    logger = setup_logger(rm.app_log_path)

    st.session_state["rm"] = rm
    st.session_state["logger"] = logger
    st.session_state["config_path"] = config_path
    st.session_state["turn_id"] = 0
    st.session_state["history"] = []  # list of dicts: {turn_id, query, answer}
    st.session_state["pdf_view"] = None  # {"path": str, "page": int}
    st.session_state["last_nodes"] = None
    st.session_state["last_answer_raw"] = None
    st.session_state["_scroll_answer"] = False
    st.session_state["last_applied_ui_cfg"] = None
    st.session_state["hl_terms"] = ""
    st.session_state["last_submitted_query"] = ""
    st.session_state["last_turn_id"] = 0
    st.session_state["last_user_query"] = ""
    st.session_state["last_retrieval_query"] = ""
    # doc-focus lock state (lintas turn)
    st.session_state["method_doc_map"] = {}  # method -> doc_id
    st.session_state["last_doc_focus_meta"] = None
    st.session_state["last_history_used"] = None
    st.session_state["last_contextualized"] = None
    st.session_state["turn_data"] = {}       # turn_id -> full render data per turn
    st.session_state["viewing_turn_id"] = None  # None = tampilkan turn terbaru
    return rm, logger


def get_or_create_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    # Jika belum ada run di session, buat
    if "rm" not in st.session_state or "logger" not in st.session_state:
        return _new_run(cfg, config_path)
    # Jika config path berubah, reset run baru
    if st.session_state.get("config_path") != config_path:
        return _new_run(cfg, config_path)
    return st.session_state["rm"], st.session_state["logger"]


def build_history_window_v2(
    history_pairs: List[Dict[str, Any]],
    *,
    use_memory: bool,
    mem_window: int,
    segment_on_topic_shift: bool,
    anaphora_skip_last_shift: bool,
    current_is_anaphora: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    History policy (robust):
    - Default: segment setelah topic_shift terakhir (include shift) agar topik baru tidak tercampur.
    - Kasus khusus (penting): jika query sekarang ANAFORA dan last history adalah TOPIC_SHIFT,
        maka memakai window SEBELUM shift (pre-shift), karena anafora biasanya merujuk topik sebelum shift.
    """
    meta: Dict[str, Any] = {
        "segment_on_topic_shift": bool(segment_on_topic_shift),
        "anaphora_skip_last_shift": bool(anaphora_skip_last_shift),
        "current_is_anaphora": bool(current_is_anaphora),
        "last_shift_turn_id": None,
        "used_pre_shift": False,
        "window_len": 0,
        "reason": None,
    }

    if not use_memory or mem_window <= 0 or not history_pairs:
        meta["reason"] = "no_memory_or_empty_history"
        return [], meta

    pairs = history_pairs

    # Cari last shift index
    last_shift_idx: Optional[int] = None
    for i in range(len(pairs) - 1, -1, -1):
        df = pairs[i].get("detect_flags", {}) or {}
        if bool(df.get("topic_shift")) or bool(pairs[i].get("topic_shift")):
            last_shift_idx = i
            meta["last_shift_turn_id"] = pairs[i].get("turn_id")
            break

    # Kasus khusus: anafora setelah shift -> pakai pre-shift
    if (
        current_is_anaphora
        and anaphora_skip_last_shift
        and last_shift_idx is not None
        and last_shift_idx == len(pairs) - 1  # shift adalah history paling akhir
    ):
        pre = pairs[:last_shift_idx]  # sebelum shift
        window = pre[-int(mem_window) :] if pre else []
        meta["used_pre_shift"] = True
        meta["reason"] = "anaphora_after_shift_use_pre_shift"
        meta["window_len"] = len(window)
        return window, meta

    # Default: segment setelah last shift (include shift)
    if segment_on_topic_shift and last_shift_idx is not None:
        pairs = pairs[last_shift_idx:]

    window = pairs[-int(mem_window) :]
    meta["reason"] = "default_segment"
    meta["window_len"] = len(window)
    return window, meta


def extract_doc_focus_from_query(q: str) -> List[str]:
    """
    Ambil sinyal doc-focus dari query user, misalnya:
    - ada string berakhiran .pdf
    - ada token doc_id seperti "ITERA_2023_SistemMonitoringTA_Prototyping"
    Return list allowlist (doc_id dan/atau source_file).
    """
    t = (q or "").strip()
    if not t:
        return []

    found: List[str] = []

    # 1) tangkap "....pdf" (baik pakai kutip atau tidak)
    for m in re.findall(r"([A-Za-z0-9_]+\.pdf)", t, flags=re.IGNORECASE):
        if m:
            found.append(m)
            # juga tambahkan doc_id tanpa .pdf
            if m.lower().endswith(".pdf"):
                found.append(m[:-4])

    # 2) tangkap doc_id style dataset: ITERA_2023_...
    for m in re.findall(r"\bITERA_\d{4}_[A-Za-z0-9_]+\b", t):
        if m:
            found.append(m)

    # dedupe + rapikan
    found = [x.strip() for x in found if x.strip()]
    return list(dict.fromkeys(found))


def build_steps_standalone_query(methods: List[str]) -> str:
    """
    Rewrite deterministic untuk pertanyaan tahapan yang anaphora:
    "tahapannya apa saja?" -> "Apa saja tahapan dari metode Prototyping?"
    """
    ms = [m for m in (methods or []) if m]
    if not ms:
        return ""
    labels = ", ".join(pretty_method_name(m) for m in ms)
    return f"Apa saja tahapan dari metode {labels}?"


def infer_target_methods(user_query: str, history_window: List[Dict[str, Any]]) -> List[str]:
    # 1) langsung dari query
    methods = detect_methods_in_text(user_query)
    if methods:
        return methods

    if not history_window:
        return []

    # 2) paling stabil: gunakan field "methods" yang disimpan di history (kalau ada)
    hist_methods: List[str] = []
    for h in history_window:
        ms = h.get("methods")
        if isinstance(ms, list):
            for m in ms:
                if isinstance(m, str) and m and (m not in hist_methods):
                    hist_methods.append(m)

    if hist_methods:
        return hist_methods

    # 3) fallback terakhir: scan query + answer_preview (cara lama)
    if (
        is_anaphora_question(user_query)
        or is_multi_target_question(user_query)
        or is_steps_question(user_query)
    ):
        hist_txt = " ".join(
            f"{h.get('query','')} {h.get('answer_preview','')}" for h in (history_window or [])
        )
        methods = detect_methods_in_text(hist_txt)
        if methods:
            return methods

    return []


def build_method_biased_query(base_query: str, method: str) -> str:
    q = (base_query or "").strip()
    return f"{q} metode {method}".strip()


def method_steps_query(base_query: str, method: str) -> str:
    """
    Query bias khusus untuk cari tahapan suatu metode.
    """
    q = (base_query or "").strip()
    m = (method or "").strip()
    # "tahapan" + metode lebih kuat daripada "... metode X"
    return f"tahapan {m}. {q}".strip()


def dedupe_nodes_by_chunk_id(nodes: List[Any]) -> List[Any]:
    best: Dict[str, Any] = {}
    for n in nodes:
        cid = getattr(n, "chunk_id", None) or (
            n.metadata.get("chunk_id") if getattr(n, "metadata", None) else None
        )
        if not cid:
            continue
        if cid not in best or float(n.score) < float(best[cid].score):
            best[cid] = n
    return sorted(best.values(), key=lambda x: float(x.score))


def select_nodes_with_coverage(
    nodes_sorted: List[Any],
    *,
    top_k: int,
    target_methods: List[str],
    max_per_doc: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    - Untuk setiap metode target, pilih 1 node paling relevan yang benar-benar mendukung metode tsb.
    - Prioritas: doc_id match (stabil) → strict text match → fallback (kecuali PROTOTYPING).
    """

    selected: List[Any] = []
    used = set()
    doc_count = Counter()

    coverage: Dict[str, Any] = {}

    def _pick_for_method(m: str) -> Optional[Any]:
        # Pass 1: doc_id hit + supports
        for n in nodes_sorted:
            if n.chunk_id in used:
                continue
            if _docid_hit_for_method(
                getattr(n, "doc_id", ""), m
            ) and node_supports_method_for_coverage(n, m):
                return n

        # Pass 2: supports (strict text / rule)
        for n in nodes_sorted:
            if n.chunk_id in used:
                continue
            if node_supports_method_for_coverage(n, m):
                return n

        # Pass 3: fallback ringan (untuk non-PROTOTYPING) pakai detect_methods_in_text biasa
        if m != "PROTOTYPING":
            for n in nodes_sorted:
                if n.chunk_id in used:
                    continue
                if m in set(detect_methods_in_text(getattr(n, "text", "") or "")):
                    return n

        return None

    # 1) minimal 1 node per metode
    for m in target_methods:
        pick = _pick_for_method(m)
        if pick:
            selected.append(pick)
            used.add(pick.chunk_id)
            doc_count[pick.doc_id] += 1
            coverage[m] = {
                "found": True,
                "chunk_id": pick.chunk_id,
                "doc_id": pick.doc_id,
                "docid_hit": _docid_hit_for_method(pick.doc_id, m),
            }
        else:
            coverage[m] = {"found": False, "chunk_id": None, "doc_id": None, "docid_hit": False}

    # 2) isi sisa slot dengan best-score, respect max_per_doc
    for n in nodes_sorted:
        if len(selected) >= int(top_k):
            break
        if n.chunk_id in used:
            continue
        if int(max_per_doc) > 0 and doc_count[n.doc_id] >= int(max_per_doc):
            continue
        selected.append(n)
        used.add(n.chunk_id)
        doc_count[n.doc_id] += 1

    meta = {"target_methods": target_methods, "coverage": coverage, "n_selected": len(selected)}
    return selected[: int(top_k)], meta


# =========================
# Topic Shift Detection
# =========================

_SHIFT_MARKERS = [
    "topik lain",
    "topik berbeda",
    "bahasan lain",
    "ganti topik",
    "sekarang bahas",
    "selanjutnya bahas",
    "ngomongin yang lain",
]

# stopwords ringan untuk deteksi overlap topik (bukan untuk retrieval)
_STOP_TOPIC = {
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
    "tersebut",
    "nya",
    "jelaskan",
    "sebutkan",
    "menurut",
    "dokumen",
    "skripsi",
    "sistem",
    "metode",
    "model",
    "bab",
    "tabel",
    "gambar",
}


def _content_terms(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {t for t in toks if t not in _STOP_TOPIC}


def detect_topic_shift(
    user_query: str,
    history_window: List[Dict[str, Any]],
    *,
    overlap_threshold: float = 0.15,
    min_terms: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Deteksi topic shift agar memory/contextualize tidak "memaksa nyambung" ke topik lama.
    Heuristik konservatif:
    - Jika ada marker eksplisit -> shift
    - Jika query anafora -> bukan shift
    - Jika overlap term penting query vs history rendah -> shift

    Return:
        (is_shift, meta_debug)
    """
    q = (user_query or "").strip()
    if not q:
        return False, {"reason": "empty_query"}
    if not history_window:
        return False, {"reason": "no_history"}

    t = q.lower()
    if any(m in t for m in _SHIFT_MARKERS):
        return True, {"reason": "explicit_marker", "marker_hit": True}

    # follow-up anafora: anggap masih topik sama
    if is_anaphora_question(q):
        return False, {"reason": "anaphora"}

    # gabungkan ringkas history (query + answer_preview)
    hist_txt = " ".join(f"{h.get('query','')} {h.get('answer_preview','')}" for h in history_window)

    q_terms = _content_terms(q)
    h_terms = _content_terms(hist_txt)

    if len(q_terms) < int(min_terms) or len(h_terms) < int(min_terms):
        return False, {
            "reason": "too_few_terms",
            "q_terms_n": len(q_terms),
            "h_terms_n": len(h_terms),
        }

    inter = q_terms & h_terms
    overlap_q = len(inter) / max(1, len(q_terms))  # fokus ke query
    is_shift = float(overlap_q) < float(overlap_threshold)

    return bool(is_shift), {
        "reason": "overlap_check",
        "overlap_threshold": float(overlap_threshold),
        "overlap_q": round(float(overlap_q), 6),
        "q_terms_n": int(len(q_terms)),
        "h_terms_n": int(len(h_terms)),
        "inter_n": int(len(inter)),
        "inter_terms_sample": sorted(list(inter))[:10],
    }


def decide_max_bullets(
    *,
    user_query: str,
    contexts: List[str],
    default_max: int = 3,
    expanded_max: int = 5,
) -> Tuple[int, Dict[str, Any]]:
    """
    Default 3, naik ke 5 hanya untuk kondisi tertentu.
    Return: (max_bullets, meta_debug)
    """
    q = (user_query or "").strip()
    methods = detect_methods_in_contexts(contexts)
    multi_methods = len(methods) >= 2

    multi_target = is_multi_target_question(q)
    anaphora = is_anaphora_question(q)

    # naik jika:
    # - pertanyaan multi-target
    # - atau pertanyaan anafora DAN konteks memuat >= 2 metode
    if multi_target or (anaphora and multi_methods):
        max_b = int(expanded_max)
    else:
        max_b = int(default_max)

    meta = {
        "multi_target": bool(multi_target),
        "anaphora": bool(anaphora),
        "methods_detected": methods,
        "multi_methods": bool(multi_methods),
        "max_bullets": int(max_b),
    }
    return max_b, meta


def main() -> None:
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", page_icon="🔍", layout="wide")
    st.title("🔍 RAG ThesisDoc System - V1.5")

    load_dotenv()  # baca .env (lokal)

    # Root project untuk resolve file
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> root
    # ambil data_raw dari config kalau ada, fallback "data_raw"
    # (akan set setelah cfg terbaca)
    # DATA_RAW_DIR = Path("data_raw")

    # ===== Sidebar: Controls =====
    st.sidebar.header("Controls & Information")

    # Config path
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/v1_5.yaml",
        help="Path YAML config yang dipakai untuk run ini (index, retrieval, llm, memory).",
    )
    cfg = load_config(config_path)
    data_raw_dir = Path(cfg.get("paths", {}).get("data_raw_dir", "data_raw"))

    # init/reuse run per session
    rm, logger = get_or_create_run(cfg, config_path)
    logger.info(f"Run active: {rm.run_id}")

    # Guard: pastikan state baru selalu ada (terutama saat code diupdate mid-session)
    if "turn_data" not in st.session_state:
        st.session_state["turn_data"] = {}
    if "viewing_turn_id" not in st.session_state:
        st.session_state["viewing_turn_id"] = None

    # info provider LLM
    provider = os.getenv("LLM_PROVIDER", "(not set)")
    groq_model = os.getenv("GROQ_MODEL", "(not set)")
    openai_model = os.getenv("OPENAI_MODEL", "(not set)")

    # Tombol new session (reset run_id + history)
    if st.sidebar.button("➕ New Session (Run Baru)"):
        rm, logger = _new_run(cfg, config_path)
        st.rerun()

    # =========================
    # Run Information
    # =========================
    st.sidebar.divider()
    st.sidebar.subheader("📰 Run Information")
    st.sidebar.write("Run ID:", rm.run_id)
    st.sidebar.write("Collection:", cfg["index"]["collection_name"])
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])

    st.sidebar.write("LLM Provider:", provider)
    if provider == "groq":
        st.sidebar.write("Model:", groq_model)
    elif provider == "openai":
        st.sidebar.write("Model:", openai_model)

    # --- Runtime status turn terakhir ---
    st.sidebar.divider()
    last_turn_status_slot = st.sidebar.empty()  # diisi setelah processing

    # ===================================================
    # Settings / Config
    # Override controls (UI-level) untuk eksperimen cepat
    # ===================================================
    st.sidebar.divider()
    st.sidebar.subheader("🛠️ Settings (Configuration)")
    # Toggle generator
    use_llm = st.sidebar.checkbox(
        "Use LLM (Generator)",
        value=True,
        help="Jika OFF, sistem hanya menampilkan jawaban retrieval-only (extractive) tanpa LLM.",
    )

    # Basic retrieval + prompt sizing
    top_k_ui = st.sidebar.slider(
        "top_k (override)",
        min_value=3,
        max_value=15,
        value=int(cfg["retrieval"]["top_k"]),
        help="Jumlah chunk (sumber) yang diambil dari vector DB untuk konteks jawaban.",
    )
    max_chars_ctx_ui = st.sidebar.slider(
        "max_chars_per_ctx (override)",
        min_value=600,
        max_value=3000,
        value=int(cfg.get("llm", {}).get("max_chars_per_ctx", 1800)),
        help="Batas karakter per-chunk konteks yang dikirim ke LLM (mencegah prompt terlalu panjang).",
    )

    # =========================
    # V1.5 Controls: Memory + Contextualization (Mode demo vs eval)
    # =========================
    # demo: memory/contextualization boleh ON untuk pengalaman chat lebih natural
    # eval: dipaksa OFF agar hasil retrieval stabil & mudah dibandingkan antar run
    mem_cfg = cfg.get("memory", {})
    mem_default = bool(mem_cfg.get("enabled", False))
    mem_window_default = int(mem_cfg.get("window", 4))

    with st.sidebar.expander("Memory & Context (V1.5)", expanded=True):
        run_mode = st.selectbox(
            "Mode",
            options=["demo", "eval"],
            help="eval = matikan memory/contextualization agar hasil konsisten untuk evaluasi.",
        )
        eval_mode = run_mode == "eval"
        # Toggle Use Memory + Window + Contextualize (untuk V1.5)
        use_memory = st.checkbox(
            "Use Memory (V1.5)",
            value=mem_default,
            help="Jika ON, pertanyaan lanjutan bisa memanfaatkan riwayat chat (untuk rewrite query).",
        )
        mem_window = st.slider(
            "Memory Window (Turns)",
            min_value=0,
            max_value=10,
            value=mem_window_default,
            help="Berapa turn terakhir yang dipakai untuk contextualization.",
        )
        contextualize_on = st.checkbox(
            "Contextualize Query (Rewrite Sebelum Retrieval)",
            value=True,
            disabled=(not use_memory),
            help="Jika ON, sistem rewrite pertanyaan lanjutan (mis. 'tahapannya?') menjadi standalone sebelum retrieval.",
        )

        # Eval mode: paksa OFF agar konsisten
        if eval_mode:
            use_memory = False
            contextualize_on = False
            mem_window = 0

    # =========================
    # Advanced Retrieval (Diversity)
    # =========================
    with st.sidebar.expander("Advanced Retrieval (Diversity)", expanded=False):
        # NOTE: eval_mode sudah terdefinisi dari expander Memory
        diverse_mode = st.selectbox(
            "Diversity Retrieval (Multi-Doc)",
            options=["Auto", "Off", "On"],
            index=0,
            disabled=eval_mode,
            help="Auto: Aktif jika pertanyaan mengandung 'beberapa dokumen/sumber'.",
        )
        max_per_doc = st.slider(
            "Max Chunks per-Dokumen",
            min_value=1,
            max_value=int(top_k_ui),
            value=min(2, int(top_k_ui)),
            disabled=eval_mode,
            help="Batasi jumlah chunk dari dokumen yang sama agar sources lebih beragam.",
        )
        pool_mult = st.slider(
            "Candidate Pool Multiplier",
            min_value=2,
            max_value=12,
            value=6,
            disabled=eval_mode,
            help="Jumlah kandidat awal = top_k * multiplier (lebih besar = lebih beragam tapi akan lebih lambat).",
        )

        if eval_mode:
            # paksa OFF saat eval agar konsisten
            diverse_mode = "Off"

    # indikator config berubah (render memakai placeholder dan update langsung setelah submit)
    current_ui_cfg = {
        "top_k": int(top_k_ui),
        "max_chars_per_ctx": int(max_chars_ctx_ui),
        "use_llm": bool(use_llm),
        "provider": provider,
        "run_mode": run_mode,
        "use_memory": bool(use_memory),
        "mem_window": int(mem_window),
        "contextualize_on": bool(contextualize_on),
        "diverse_mode": str(diverse_mode),
        "max_per_doc": int(max_per_doc),
        "pool_mult": int(pool_mult),
    }

    # snapshot terakhir yang BENAR-BENAR sudah dipakai untuk menghasilkan hasil terakhir
    if "last_applied_ui_cfg" not in st.session_state:
        st.session_state["last_applied_ui_cfg"] = None  # belum ada query sama sekali

    cfg_indicator_slot = st.sidebar.empty()  # <--- SLOT, akan diisi di bagian bawah setelah submit

    # Key status (tanpa menampilkan key)
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Diagnostics")
    st.sidebar.write("GROQ key set?:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("OPENAI key set?:", bool(os.getenv("OPENAI_API_KEY")))

    # apakah mau config snapshot per baris?
    include_cfg = st.sidebar.checkbox(
        "Log Config Snapshot per-Row",
        value=(os.getenv("LOG_INCLUDE_CONFIG_PER_ROW", "0") == "1"),
        help="Jika ON, setiap baris JSONL menyimpan snapshot config untuk audit (file log jadi lebih besar, namun audit lebih solid).",
    )

    # ===== Main Info =====
    st.success(
        "Now Running on **V1.5** - Dense Retrieval + Memory & Contextualization", icon="🚥"
    )

    _mem_icon = "🟢" if (use_memory and mem_window > 0) else "⚫"
    _ctx_icon = "🟢" if contextualize_on else "⚫"
    _llm_icon = "🟢" if use_llm else "⚫"
    _mode_badge = "🧪 eval" if eval_mode else "🎯 demo"
    _info_parts = [
        f"<b>Mode:</b> {_mode_badge}",
        f"{_llm_icon} Generator",
        f"{_mem_icon} Memory ({mem_window} turns)",
        f"{_ctx_icon} Contextualize",
    ]
    st.markdown(
        '<div style="background:#1e293b; border:1px solid #334155; padding:0.6rem 1rem; '
        'border-radius:0.6rem; margin-bottom:0.9rem; color:#94a3b8; font-size:0.875rem;">'
        + " &nbsp;|&nbsp; ".join(_info_parts)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ===== History slot (chat-like) =====
    if "history" not in st.session_state:
        st.session_state["history"] = []

    history_slot = st.empty()  # <--- SLOT (akan di-render ulang di bawah setelah submit)

    # ===== Input =====
    user_q = st.text_input("Halo! Ada yang Bisa Saya Bantu? Anda bisa Tanyakan di sini:")

    # Filter sources (UI)
    filter_q = st.text_input(
        "Filter Sources (doc_id / source_file / text) - BOLEH DIISI ATAU KOSONGKAN:", value=""
    )
    show_only_matches = st.checkbox("Show Only Matched Sources", value=True)

    # Tombol submit (pakai variabel agar logika show/hide benar di klik pertama)
    submitted = st.button("🪄 Kirim")

    if submitted and user_q.strip():
        query = user_q.strip()

        # reset viewing mode agar hasil terbaru tampil otomatis setelah submit
        st.session_state["viewing_turn_id"] = None

        # turn id per session
        st.session_state["turn_id"] += 1
        turn_id = int(st.session_state["turn_id"])

        # simpan query terakhir untuk auto-highlight
        st.session_state["last_submitted_query"] = query
        st.session_state["last_turn_id"] = turn_id

        with st.status("⏳ Processing Query... Please Wait a Moment :)", expanded=True) as _proc_status:

            st.write("🔧 Membangun konfigurasi runtime...")
            # Override cfg runtime (tanpa mengubah file yaml)
            cfg_runtime = dict(cfg)
            cfg_runtime["retrieval"] = dict(cfg.get("retrieval", {}))
            cfg_runtime["retrieval"]["top_k"] = int(top_k_ui)
            cfg_runtime["llm"] = dict(cfg.get("llm", {}))
            cfg_runtime["llm"]["max_chars_per_ctx"] = int(max_chars_ctx_ui)

            # ===== Memory window =====
            # ===== DETECT awal (berdasarkan query saja) =====
            detect_anaphora = is_anaphora_question(query)
            detect_multi_target = is_multi_target_question(query)

            # ===== History window (policy) =====
            # Ambil window percakapan terakhir (history) untuk membantu rewrite query.
            history_pairs = st.session_state.get("history", [])

            hist_cfg = cfg_runtime.get("history", {}) or {}
            segment_on_shift = bool(hist_cfg.get("segment_on_topic_shift", True))
            anaphora_skip_last_shift = bool(hist_cfg.get("anaphora_skip_last_shift", True))

            history_window, history_policy_meta = build_history_window_v2(
                history_pairs,
                use_memory=use_memory,
                mem_window=int(mem_window),
                segment_on_topic_shift=segment_on_shift,
                anaphora_skip_last_shift=anaphora_skip_last_shift,
                current_is_anaphora=detect_anaphora,
            )

            # Ringkasan memory yang dipakai untuk turn ini (untuk audit log)
            history_used = len(history_window) if (use_memory and mem_window > 0) else 0
            history_turn_ids = [h.get("turn_id") for h in history_window] if history_window else []

            st.write("💭 Mendeteksi konteks & riwayat percakapan...")
            # ===== DETECT (anaphora / multi-target / topic-shift) =====
            detect_anaphora = is_anaphora_question(query)
            detect_multi_target = is_multi_target_question(query)

            ts_cfg = cfg_runtime.get("topic_shift", {}) or {}
            ts_enabled = bool(ts_cfg.get("enabled", True))
            ts_overlap_threshold = float(ts_cfg.get("overlap_threshold", 0.15))
            ts_min_terms = int(ts_cfg.get("min_terms", 3))

            topic_shift = False
            topic_shift_meta: Dict[str, Any] = {"reason": "not_evaluated"}

            if not ts_enabled:
                topic_shift = False
                topic_shift_meta = {"reason": "disabled"}
            elif not (use_memory and history_window):
                topic_shift = False
                topic_shift_meta = {"reason": "no_memory_or_history"}
            else:
                topic_shift, topic_shift_meta = detect_topic_shift(
                    query,
                    history_window,
                    overlap_threshold=ts_overlap_threshold,
                    min_terms=ts_min_terms,
                )

            detect_flags = {
                "anaphora": bool(detect_anaphora),
                "multi_target": bool(detect_multi_target),
                "topic_shift": bool(topic_shift),
            }

            # ===== Contextualize / rewrite query =====
            # Contextualization akan menulis ulang pertanyaan user agar eksplisit (mis. coreference "itu", "tahapannya").
            # jika topic_shift True -> skip rewrite dan treat query sebagai fresh query
            retrieval_query = query
            contextualize_attempted = False
            contextualize_skipped = False
            contextualize_mode = "none"

            if use_memory and contextualize_on and history_window and (not topic_shift):
                # CASE 1: anaphora + steps -> deterministic rewrite (lebih stabil dari LLM)
                if is_steps_question(query) and is_anaphora_question(query):
                    tm = infer_target_methods(query, history_window)
                    rq2 = build_steps_standalone_query(tm)
                    if rq2:
                        retrieval_query = rq2
                        contextualize_attempted = True
                        contextualize_mode = "heuristic_steps"
                    else:
                        contextualize_attempted = True
                        contextualize_mode = "llm"
                        retrieval_query = contextualize_question(
                            cfg_runtime,
                            question=query,
                            history_pairs=history_window,
                            use_llm=use_llm,
                        )
                else:
                    # CASE 2: default -> LLM/heuristic bawaan contextualize_question()
                    contextualize_attempted = True
                    contextualize_mode = "llm"
                    retrieval_query = contextualize_question(
                        cfg_runtime,
                        question=query,
                        history_pairs=history_window,
                        use_llm=use_llm,
                    )

            elif use_memory and contextualize_on and history_window and topic_shift:
                contextualize_skipped = True
                contextualize_mode = "skipped_topic_shift"

            contextualize_decision = {
                "enabled": bool(contextualize_on),
                "attempted": bool(contextualize_attempted),
                "skipped": bool(contextualize_skipped),
                "mode": contextualize_mode,
                "skip_reason": (topic_shift_meta.get("reason") if contextualize_skipped else None),
                "topic_shift_enabled": bool(ts_enabled),
                "overlap_threshold": float(ts_overlap_threshold),
                "min_terms": int(ts_min_terms),
            }

            # ===== Metadata Routing (page_count) =====
            md_cfg = cfg_runtime.get("metadata_routing", {}) or {}
            md_enabled = bool(md_cfg.get("enabled", False))
            allow_in_eval = bool(md_cfg.get("allow_in_eval", False))
            if run_mode == "eval" and not allow_in_eval:
                md_enabled = False

            paths = ConfigPaths.from_cfg(cfg_runtime)
            persist_dir = paths.persist_dir

            metadata_routed = False
            metadata_meta: Dict[str, Any] = {"enabled": md_enabled, "handled": False}

            if md_enabled:
                md_res = maybe_route_metadata_query(query, persist_dir=persist_dir, cfg=cfg_runtime)
                if md_res and md_res.handled:
                    metadata_routed = True
                    metadata_meta = {
                        "enabled": True,
                        "handled": True,
                        "intent": md_res.intent,
                        "meta": md_res.meta,
                    }

                    st.write("✏️ Menjawab dari metadata catalog...")

                    # Untuk metadata routing: tidak ada nodes/context
                    nodes = []
                    contexts = []
                    retrieval_stats = compute_retrieval_stats(nodes)
                    strategy = "metadata_router"
                    ts = datetime.now().isoformat(timespec="seconds")

                    retrieval_record = {
                        "run_id": rm.run_id,
                        "turn_id": turn_id,
                        "timestamp": ts,
                        "retrieval_mode": "metadata_router",
                        "llm_provider": provider,
                        "use_llm": False,
                        "top_k": int(top_k_ui),
                        "user_query": query,
                        "retrieval_query": query,
                        "run_mode": run_mode,
                        "use_memory": use_memory,
                        "mem_window": int(mem_window),
                        "history_used": history_used,
                        "history_turn_ids": history_turn_ids,
                        "history_policy": history_policy_meta,
                        "contextualized": False,
                        "detect_flags": detect_flags,
                        "topic_shift_meta": topic_shift_meta,
                        "contextualize_decision": contextualize_decision,
                        "doc_focus": st.session_state.get("last_doc_focus_meta"),
                        "strategy": strategy,
                        "diversity": {
                            "mode": "Off",
                            "enabled": False,
                            "max_per_doc": 0,
                            "pool_mult": 0,
                            "candidate_k": None,
                        },
                        "retrieval_stats": retrieval_stats,
                        "retrieval_confidence": None,
                        "retrieved_nodes": [],
                        "metadata_route": metadata_meta,
                    }
                    rm.log_retrieval(retrieval_record, include_config_snapshot_per_row=include_cfg)

                    answer_raw = md_res.answer_text
                    generator_meta = {
                        "status": "metadata",
                        "format_ok": True,
                        "bukti_ok": True,
                        "citations_present": False,
                        "citations_in_range": True,
                        "grounding_ok": True,
                        "answer_grounding_ok": True,
                    }

                    answer_record = {
                        "run_id": rm.run_id,
                        "turn_id": turn_id,
                        "timestamp": ts,
                        "llm_provider": provider,
                        "use_llm": False,
                        "top_k": int(top_k_ui),
                        "user_query": query,
                        "retrieval_query": query,
                        "run_mode": run_mode,
                        "use_memory": use_memory,
                        "mem_window": int(mem_window),
                        "history_used": history_used,
                        "history_turn_ids": history_turn_ids,
                        "history_policy": history_policy_meta,
                        "contextualized": False,
                        "detect_flags": detect_flags,
                        "topic_shift_meta": topic_shift_meta,
                        "contextualize_decision": contextualize_decision,
                        "doc_focus": st.session_state.get("last_doc_focus_meta"),
                        "used_context_chunk_ids": [],
                        "answer": answer_raw,
                        "strategy": strategy,
                        "diversity": {
                            "mode": "Off",
                            "enabled": False,
                            "max_per_doc": 0,
                            "pool_mult": 0,
                            "candidate_k": None,
                        },
                        "retrieval_stats": retrieval_stats,
                        "retrieval_confidence": None,
                        "generator_strategy": "metadata_router",
                        "generator_status": "metadata",
                        "generator_meta": generator_meta,
                        "max_bullets": 1,
                        "bullet_policy": {},
                        "error": None,
                        "metadata_route": metadata_meta,
                    }
                    rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)

                    # append history (tambahkan detect_flags/topic_shift agar policy bisa bekerja)
                    jawaban, _ = parse_answer(answer_raw)
                    st.session_state["history"].append(
                        {
                            "turn_id": turn_id,
                            "query": query,
                            "answer_preview": (jawaban or answer_raw)[:220]
                            + ("..." if len(jawaban or answer_raw) > 220 else ""),
                            "detect_flags": detect_flags,
                            "topic_shift": bool(detect_flags.get("topic_shift")),
                            "metadata_intent": md_res.intent,
                            "methods": [],  # metadata route tidak relevan untuk chaining metode
                        }
                    )

                    # set last result untuk UI
                    st.session_state["last_answer_raw"] = answer_raw
                    st.session_state["last_turn_id"] = turn_id
                    st.session_state["last_nodes"] = []
                    st.session_state["last_strategy"] = strategy
                    st.session_state["_scroll_answer"] = True
                    st.session_state["last_user_query"] = query
                    st.session_state["last_retrieval_query"] = query
                    st.session_state["last_applied_ui_cfg"] = current_ui_cfg.copy()
                    st.session_state["last_history_used"] = history_used
                    st.session_state["last_contextualized"] = False

                    # simpan full render data untuk turn viewer
                    if "turn_data" not in st.session_state:
                        st.session_state["turn_data"] = {}
                    st.session_state["turn_data"][turn_id] = {
                        "nodes": [],
                        "answer_raw": answer_raw,
                        "user_query": query,
                        "retrieval_query": query,
                        "strategy": strategy,
                        "history_used": history_used,
                        "contextualized": False,
                    }

            if not metadata_routed:
                # =========================
                # Retrieval
                # =========================
                # Pakai retrieval_query (hasil rewrite) agar embedding search lebih tepat.
                # nodes -> dipakai untuk:
                # 1) konteks generator (contexts)
                # 2) CTX mapping + sources UI
                # 3) logging (retrieval.jsonl)
                # decide diversify
                if diverse_mode == "On":
                    diversify = True
                elif diverse_mode == "Off":
                    diversify = False
                else:
                    diversify = is_multi_doc_question(query)  # pakai user query (lebih natural)

                candidate_k = int(top_k_ui) * int(pool_mult) if diversify else None

                st.write("🔍 Retrieval dokumen dari vector store...")
                nodes = retrieve_dense(
                    cfg_runtime,
                    retrieval_query,
                    diversify=diversify,
                    max_per_doc=int(max_per_doc),
                    candidate_k=candidate_k,
                )

                # ===== Coverage per-metode (multi-query dense) =====
                cov_cfg = cfg_runtime.get("coverage", {}) or {}
                cov_enabled = bool(cov_cfg.get("enabled", True)) and (run_mode != "eval")
                per_method_k = int(cov_cfg.get("per_method_k", 3))
                min_methods = int(cov_cfg.get("min_methods", 2))

                # Jika topic_shift terdeteksi, JANGAN bawa metode dari history (hindari carry-over)
                history_for_methods = [] if topic_shift else history_window
                target_methods = infer_target_methods(query, history_for_methods)

                coverage_meta: Dict[str, Any] = {
                    "enabled": cov_enabled,
                    "performed": False,
                    "target_methods": target_methods,
                }

                if (
                    cov_enabled
                    and len(target_methods) >= min_methods
                    and (detect_anaphora or detect_multi_target)
                ):
                    # ===== steps-aware expansion =====
                    # Jika query adalah steps, cari tambahan PER METODE (bukan hanya yang missing),
                    # karena "metode terdeteksi" ≠ "tahapan tersedia".
                    steps_q = is_steps_question(query) or is_steps_question(retrieval_query)

                    methods_in_base = set(
                        detect_methods_in_text(" ".join([(n.text or "") for n in nodes]))
                    )
                    missing = [m for m in target_methods if m not in methods_in_base]

                    extra_nodes: List[Any] = []

                    # pilih metode yang akan di-query tambahan:
                    # - jika steps_q: semua target_methods (agar tiap metode punya peluang punya tahapan)
                    # - else: hanya missing (perilaku lama)
                    expand_methods = list(target_methods) if steps_q else list(missing)

                    for m in expand_methods:
                        cfg_tmp = dict(cfg_runtime)
                        cfg_tmp["retrieval"] = dict(cfg_runtime.get("retrieval", {}))
                        cfg_tmp["retrieval"]["top_k"] = int(per_method_k)

                        if steps_q:
                            q_m = method_steps_query(retrieval_query, m)
                        else:
                            q_m = build_method_biased_query(retrieval_query, m)

                        extra_nodes.extend(
                            retrieve_dense(
                                cfg_tmp,
                                q_m,
                                diversify=False,
                                max_per_doc=int(max_per_doc),
                                candidate_k=None,
                            )
                        )

                    merged = dedupe_nodes_by_chunk_id(nodes + extra_nodes)

                    # select nodes yang menjamin coverage per metode
                    nodes, cov_detail = select_nodes_with_coverage(
                        merged,
                        top_k=int(top_k_ui),
                        target_methods=target_methods,
                        max_per_doc=int(max_per_doc) if diversify else 0,
                    )

                    coverage_meta = {
                        "enabled": True,
                        "performed": True,
                        "target_methods": target_methods,
                        "missing_before": missing,
                        "detail": cov_detail,
                        "steps_query": bool(steps_q),
                        "expanded_methods": expand_methods,
                    }

                st.write("📊 Coverage & doc-focus analysis...")
                # =========================
                # Doc-focus lock (anaphora + steps)
                # =========================
                # 1) explicit doc-focus di query (mis. menyebut "...pdf" atau "ITERA_2023_..."):
                #    -> HARUS hard lock dan tidak boleh fallback ke dokumen lain
                # 2) jikalau user follow-up "tahapannya?" (anafóra), jangan sampai retrieval "loncat" ke dokumen lain.
                #    -> saya kunci nodes ke doc_id yang pernah diasosiasikan dengan metode (dari turn sebelumnya).
                # pastikan doc_focus_meta selalu ada (biar tidak error dan log konsisten)
                doc_focus_meta: Dict[str, Any] = {
                    "enabled": False,
                    "doc_ids": [],
                    "reason": None,
                    "kept": 0,
                    "total": len(nodes or []),
                    "strict": False,
                    "mode": None,
                    # tambahan agar generator-lock bisa dipakai
                    "locked": False,
                    "allowed_doc_ids": [],
                    "focus_methods": [],
                    "method_doc_map": {},
                    "method_bias_query": None,
                    "method_bias_added": 0,
                }

                steps_q_now = is_steps_question(query) or is_steps_question(retrieval_query)

                # --- (A) PRIORITAS 1: explicit doc-focus dari query user ---
                explicit_doc_focus = extract_doc_focus_from_query(query)  # list[str]
                if explicit_doc_focus:
                    # (1) HARD LOCK nodes ke dokumen yang disebut
                    nodes, dfm = apply_doc_focus_filter(
                        nodes,
                        list(explicit_doc_focus),
                        min_keep=0,
                        strict=True,  # HARD LOCK
                    )
                    doc_focus_meta.update(dfm)
                    doc_focus_meta["enabled"] = True
                    doc_focus_meta["mode"] = "explicit_query"

                    # tandai untuk dipakai generator lock
                    doc_focus_meta["locked"] = True
                    doc_focus_meta["allowed_doc_ids"] = list(explicit_doc_focus)

                    # (2) Jika pertanyaan METODE dan doc-focus explicit,
                    # lakukan 1 pass retrieval tambahan yang "method-biased",
                    # lalu hard-lock hasilnya ke dokumen yang sama, merge, dan ambil top_k terbaik.
                    if is_method_question(query):
                        focus_methods = detect_methods_in_text(query) or list(target_methods or [])
                        doc_focus_meta["focus_methods"] = list(focus_methods)

                        # query bias: memancing paragraf yang menyebut "metode yang digunakan"
                        method_hint = (
                            " ".join(pretty_method_name(m) for m in focus_methods)
                            if focus_methods
                            else ""
                        )
                        q_method_bias = (
                            f"metode pengembangan yang digunakan {method_hint} "
                            f"menggunakan metode {method_hint} "
                            f"metode yang digunakan"
                        ).strip()

                        cfg_m = dict(cfg_runtime)
                        cfg_m["retrieval"] = dict(cfg_runtime.get("retrieval", {}))

                        # ambil lebih banyak kandidat agar peluang kena paragraf "metode yang digunakan" lebih tinggi
                        cfg_m["retrieval"]["top_k"] = max(int(top_k_ui), 8)

                        extra_m = retrieve_dense(
                            cfg_m,
                            q_method_bias,
                            diversify=False,
                            max_per_doc=int(max_per_doc),
                            candidate_k=None,
                        )

                        # hard-lock extra ke doc yang sama
                        extra_m, _ = apply_doc_focus_filter(
                            extra_m,
                            list(explicit_doc_focus),
                            min_keep=0,
                            strict=True,
                        )

                        # merge + dedupe + ambil top_k terbaik (distance kecil = lebih relevan)
                        if extra_m:
                            nodes = dedupe_nodes_by_chunk_id(nodes + extra_m)
                            nodes = nodes[: int(top_k_ui)]

                        doc_focus_meta["method_bias_query"] = q_method_bias
                        doc_focus_meta["method_bias_added"] = int(len(extra_m or []))

                    else:
                        doc_focus_meta["focus_methods"] = []
                        doc_focus_meta["method_bias_query"] = None
                        doc_focus_meta["method_bias_added"] = 0

                # --- (B) PRIORITAS 2: anaphora+steps doc-focus dari method_doc_map (turn sebelumnya) ---
                elif detect_anaphora and steps_q_now and (not topic_shift):
                    if "method_doc_map" not in st.session_state:
                        st.session_state["method_doc_map"] = {}

                    md_map = st.session_state.get("method_doc_map") or {}

                    # metode target dari history_window (infer_target_methods sudah dipakai)
                    tm_focus = list(target_methods or [])
                    allowed_doc_ids: List[str] = []
                    for m in tm_focus:
                        d = md_map.get(m)
                        if isinstance(d, str) and d.strip():
                            allowed_doc_ids.append(d.strip())

                    # dedupe (biar bersih)
                    allowed_doc_ids = list(dict.fromkeys(allowed_doc_ids))

                    if allowed_doc_ids:
                        nodes, dfm = apply_doc_focus_filter(
                            nodes,
                            allowed_doc_ids,
                            min_keep=0,
                            strict=True,  # HARD LOCK juga (biar tidak bocor ke dokumen lain)
                        )
                        doc_focus_meta.update(dfm)
                        doc_focus_meta["enabled"] = True
                        doc_focus_meta["mode"] = "anaphora_steps"

                # simpan meta doc-focus untuk audit/debug UI
                st.session_state["last_doc_focus_meta"] = doc_focus_meta

                # contexts untuk LLM (kalau nanti key valid) dan perkaya dengan metadata dokumen
                contexts = build_contexts_with_meta(nodes)

                # =========================
                # Update method_doc_map (untuk doc-focus lock di turn berikutnya)
                # =========================
                if "method_doc_map" not in st.session_state:
                    st.session_state["method_doc_map"] = {}

                # update hanya jika ini BUKAN steps query (karena steps query rawan campur dokumen)
                if not (is_steps_question(query) or is_steps_question(retrieval_query)):
                    if target_methods:
                        md_new: Dict[str, str] = {}

                        # prioritas 1: dari coverage detail (paling stabil)
                        try:
                            if (
                                isinstance(coverage_meta, dict)
                                and coverage_meta.get("performed")
                                and isinstance(coverage_meta.get("detail"), dict)
                            ):
                                cov_map2 = coverage_meta["detail"].get("coverage") or {}
                                if isinstance(cov_map2, dict):
                                    for m, v in cov_map2.items():
                                        if isinstance(v, dict) and v.get("doc_id"):
                                            md_new[str(m)] = str(v["doc_id"])
                        except Exception:
                            pass

                        # fallback: scan nodes text
                        if not md_new:
                            md_new = build_method_doc_map_from_nodes(nodes, list(target_methods))

                        if md_new:
                            st.session_state["method_doc_map"].update(md_new)

                # Tentukan max bullet Bukti dinamis (3 default, bisa naik jadi 5)
                max_bullets, bullets_meta = decide_max_bullets(
                    user_query=query,  # pakai user query asli
                    contexts=contexts,
                    default_max=3,
                    expanded_max=5,
                )

                retrieval_stats = compute_retrieval_stats(nodes)

                strategy = cfg_runtime["retrieval"]["mode"]  # contoh: "dense"
                if retrieval_query != query:
                    strategy = f"{strategy}+rewrite"
                if diversify:
                    strategy = f"{strategy}+diverse"

                # timestamp per turn
                ts = datetime.now().isoformat(timespec="seconds")

                # log retrieval via RunManager
                retrieval_record = {
                    "run_id": rm.run_id,
                    "turn_id": turn_id,
                    "timestamp": ts,
                    "retrieval_mode": cfg_runtime["retrieval"]["mode"],
                    "llm_provider": provider,
                    "use_llm": use_llm,
                    "top_k": int(top_k_ui),
                    "user_query": query,
                    "retrieval_query": retrieval_query,
                    "run_mode": run_mode,
                    "use_memory": use_memory,
                    "mem_window": int(mem_window),
                    "history_used": history_used,
                    "history_turn_ids": history_turn_ids,
                    "contextualized": (retrieval_query != query),
                    "detect_flags": detect_flags,
                    "topic_shift_meta": topic_shift_meta,
                    "history_policy": history_policy_meta,
                    "coverage": coverage_meta,
                    "metadata_route": metadata_meta,
                    "contextualize_decision": contextualize_decision,
                    "strategy": strategy,
                    "diversity": {
                        "mode": diverse_mode,
                        "enabled": bool(diversify),
                        "max_per_doc": int(max_per_doc),
                        "pool_mult": int(pool_mult),
                        "candidate_k": int(candidate_k) if candidate_k is not None else None,
                    },
                    "retrieval_stats": retrieval_stats,
                    "retrieval_confidence": retrieval_stats.get("sim_top1"),
                    "retrieved_nodes": [
                        {
                            "doc_id": n.doc_id,
                            "chunk_id": n.chunk_id,
                            # NOTE: score dari Chroma umumnya adalah distance (lebih kecil = lebih relevan)
                            "distance": n.score,
                            "similarity": dist_to_sim(n.score),
                            "metadata": n.metadata,
                            "text": n.text,  # akan di-trim otomatis oleh RunManager.log_retrieval
                        }
                        for n in nodes
                    ],
                }
                rm.log_retrieval(retrieval_record, include_config_snapshot_per_row=include_cfg)

                # =========================
                # Generation
                # =========================
                # Kalau generator ON: jawab berdasarkan contexts top-k.
                # Kalau OFF: mode retrieval-only (extractive) agar UI tetap konsisten (Jawaban + Bukti).
                # Retrieval memakai retrieval_query (hasil rewrite) -> untuk mendapatkan konteks yang tepat.
                # Tetapi LLM harus menjawab pertanyaan USER (query asli) agar tidak mismatch.
                user_query = query  # selalu menjawab berdasarkan user_query
                used_ctx = min(len(contexts), min(int(top_k_ui), 8))  # generate.py memakai hard cap 8
                generator_error = None

                # ===== Generation policy (per-metode) =====
                gen_cfg = cfg_runtime.get("generation", {}) or {}
                per_method_bullets = int(gen_cfg.get("per_method_bullets", 2))
                max_total_bullets = int(gen_cfg.get("max_total_bullets", 6))
                force_method_list = bool(gen_cfg.get("force_per_method_for_method_list", True))

                # target_methods untuk generator:
                # - prioritas: dari coverage detail (paling reliable)
                # - fallback: target_methods dari infer_target_methods
                gen_targets: List[str] = []
                missing_methods: List[str] = []
                cov_map: Dict[str, Any] = {}  # supaya selalu terdefinisi

                if (
                    isinstance(coverage_meta, dict)
                    and coverage_meta.get("performed")
                    and isinstance(coverage_meta.get("detail"), dict)
                ):
                    gen_targets = list(coverage_meta["detail"].get("target_methods") or [])
                    cov_map = coverage_meta["detail"].get("coverage") or {}
                    if isinstance(cov_map, dict):
                        missing_methods = [
                            m
                            for m, v in cov_map.items()
                            if isinstance(v, dict) and (not v.get("found"))
                        ]
                else:
                    gen_targets = list(target_methods or [])

                # ===== Doc-focus lock: kunci target generator ke metode yang dilock =====
                dfm = st.session_state.get("last_doc_focus_meta") or {}
                if isinstance(dfm, dict) and dfm.get("locked"):
                    focus_methods = dfm.get("focus_methods") or []
                    method_doc_map = dfm.get("method_doc_map") or {}

                    if isinstance(focus_methods, list) and focus_methods:
                        # override gen_targets agar generator tidak melebar ke dokumen lain
                        gen_targets = [m for m in focus_methods if isinstance(m, str) and m]

                        # siapkan cov_map minimal untuk steps-aware missing (dipakai di bawah)
                        cov_map = {
                            m: {"found": True, "doc_id": str(method_doc_map.get(m, "") or "")}
                            for m in gen_targets
                        }

                        # reset missing_methods baseline; nanti dihitung lagi oleh blok steps-aware
                        missing_methods = []

                # kapan force per-metode?
                q_low = (query or "").lower()
                is_steps_q = bool(
                    re.search(r"\b(tahap|tahapan)\w*\b|\b(langkah|alur|prosedur)\w*\b", q_low)
                )
                is_method_list_q = bool(
                    re.search(r"\bmetode\b.*\bapa\s+saja\b|\bmetode\s+apa\s+saja\b", q_low)
                )

                # ===== fallback gen_targets untuk pertanyaan METODE (single-target) =====
                # kasus: "metode pengembangan apa yang digunakan?" -> detect_multi_target False -> infer_target_methods bisa kosong
                # supaya generator tidak jatuh ke jawaban template "Informasi relevan ditemukan..."
                if is_method_question(query) and (not gen_targets):
                    ctx_methods = detect_methods_in_contexts(contexts)
                    if ctx_methods:
                        # kalau bukan "metode apa saja" (list), ambil 1 metode dominan (top doc) agar tidak melebar
                        if (not is_method_list_q) and len(ctx_methods) > 1 and nodes:
                            top_doc = str(getattr(nodes[0], "doc_id", "") or "").lower()
                            picked = None
                            for m in ctx_methods:
                                if _docid_hit_for_method(top_doc, m):
                                    picked = m
                                    break
                            gen_targets = [picked or ctx_methods[0]]
                        else:
                            gen_targets = ctx_methods[:3]  # batasi agar tidak terlalu padat

                # ===== steps-aware missing_methods (lebih ketat & presisi) =====
                # Untuk query tahapan: sebuah metode dianggap "available" hanya jika konteks dari DOC metode tsb
                # memuat sinyal tahapan yang benar-benar eksplisit (has_steps_signal = True).
                if is_steps_q and gen_targets:
                    # doc_id per metode dari coverage detail → paling stabil
                    method_doc: Dict[str, str] = {}
                    if isinstance(cov_map, dict):
                        for m, v in cov_map.items():
                            if isinstance(v, dict) and v.get("doc_id"):
                                method_doc[m] = str(v["doc_id"])

                    # fallback tambahan: jikalau coverage tidak punya doc_id (atau coverage tidak performed),
                    # ambil dari method_doc_map lintas turn (doc-focus lock)
                    session_md = st.session_state.get("method_doc_map") or {}
                    for m in gen_targets:
                        if m not in method_doc and session_md.get(m):
                            method_doc[m] = str(session_md.get(m))

                    def _ctx_for_doc_id(doc_id: str) -> List[str]:
                        if not doc_id:
                            return []
                        # match tepat pada header pertama: "DOC: <doc_id> |"
                        pat = re.compile(rf"(?im)^DOC:\s*{re.escape(doc_id)}\s*\|")
                        return [c for c in (contexts or []) if pat.search(c or "")]

                    steps_missing: List[str] = []

                    for m in gen_targets:
                        doc_id = (method_doc.get(m, "") or "").strip()
                        ctx_for_m: List[str] = _ctx_for_doc_id(doc_id)

                        # fallback: kalau doc_id tidak ada (jarang), memakai deteksi metode dari teks konteks
                        if (not ctx_for_m) and (not doc_id):
                            for c in contexts or []:
                                try:
                                    if m in set(detect_methods_in_text(c or "")):
                                        ctx_for_m.append(c)
                                except Exception:
                                    pass

                        # dianggap missing jika:
                        # - tidak ada konteks yang jelas untuk metode tsb, ATAU
                        # - ada konteks tapi tidak ada sinyal tahapan eksplisit
                        if (not ctx_for_m) or (not any(has_steps_signal(x) for x in ctx_for_m)):
                            steps_missing.append(m)

                    # gabungkan: missing dari coverage + missing dari steps-signal (unique, urut stabil)
                    missing_methods = list(dict.fromkeys((missing_methods or []) + steps_missing))

                force_per_method = False
                if len(gen_targets) >= 2:
                    # steps query: selalu force (menghindari salah config/regex edge)
                    if is_steps_q:
                        force_per_method = True
                    # method-list query: pakai config flag (opsional)
                    elif is_method_list_q and force_method_list:
                        force_per_method = True
                    # multi-target eksplisit: force
                    elif detect_multi_target:
                        force_per_method = True

                # override max_bullets jika per-metode aktif
                if force_per_method and len(gen_targets) >= 2:
                    max_bullets = min(
                        int(max_total_bullets), int(per_method_bullets) * len(gen_targets)
                    )

                bullets_meta = dict(bullets_meta or {})
                bullets_meta["max_bullets"] = int(max_bullets)

                # enrich bullets_meta untuk logging
                bullets_meta.update(
                    {
                        "force_per_method": bool(force_per_method),
                        "target_methods_for_gen": gen_targets,
                        "per_method_bullets": int(per_method_bullets),
                        "max_total_bullets": int(max_total_bullets),
                        "missing_methods": missing_methods,
                        "is_steps_q": bool(is_steps_q),
                        "is_method_list_q": bool(is_method_list_q),
                    }
                )

                st.write("🎰 Generating jawaban dengan LLM..." if use_llm else "📄 Mode retrieval-only...")
                if use_llm:
                    try:
                        answer_raw = generate_answer(
                            cfg_runtime,
                            user_query,
                            contexts,
                            max_bullets=max_bullets,
                            target_methods=gen_targets,
                            missing_methods=missing_methods,
                            force_per_method=force_per_method,
                            per_method_bullets=per_method_bullets,
                            max_total_bullets=max_total_bullets,
                        )
                    except Exception as ex:
                        logger.exception("LLM call failed")
                        generator_error = {"type": type(ex).__name__, "message": str(ex)}
                        st.warning(f"LLM error: {type(ex).__name__}: {ex}")
                        answer_raw = "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"
                else:
                    answer_raw = build_retrieval_only_answer(nodes, max_points=3)

                # meta generator (dibuat setelah answer_raw final)
                generator_meta = build_generation_meta(
                    question=user_query,
                    contexts=contexts,
                    answer_text=answer_raw,
                    used_ctx=used_ctx,
                    max_bullets=max_bullets,
                )

                generator_status = generator_meta.get("status", "unknown")
                jawaban, bukti = parse_answer(answer_raw)

                # log answers via RunManager
                answer_record = {
                    "run_id": rm.run_id,
                    "turn_id": turn_id,
                    "timestamp": ts,
                    "llm_provider": provider,
                    "use_llm": use_llm,
                    "top_k": int(top_k_ui),
                    "user_query": query,
                    "retrieval_query": retrieval_query,
                    "run_mode": run_mode,
                    "use_memory": use_memory,
                    "mem_window": int(mem_window),
                    "history_used": history_used,
                    "history_turn_ids": history_turn_ids,
                    "contextualized": (retrieval_query != query),
                    "detect_flags": detect_flags,
                    "topic_shift_meta": topic_shift_meta,
                    "history_policy": history_policy_meta,
                    "coverage": coverage_meta,
                    "metadata_route": metadata_meta,
                    "contextualize_decision": contextualize_decision,
                    "used_context_chunk_ids": [n.chunk_id for n in nodes],
                    "answer": answer_raw,
                    "strategy": strategy,
                    "diversity": {
                        "mode": diverse_mode,
                        "enabled": bool(diversify),
                        "max_per_doc": int(max_per_doc),
                        "pool_mult": int(pool_mult),
                        "candidate_k": int(candidate_k) if candidate_k is not None else None,
                    },
                    "retrieval_stats": retrieval_stats,
                    "retrieval_confidence": retrieval_stats.get("sim_top1"),
                    "generator_strategy": "llm" if use_llm else "retrieval_only",
                    "generator_status": generator_status,
                    "generator_meta": generator_meta,
                    "max_bullets": int(max_bullets),
                    "bullet_policy": bullets_meta,  # berisi methods_detected, anaphora, multi_target, dll
                    "error": generator_error,
                }
                rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)

                # --- simpan metode ke history agar rewrite steps stabil ---
                turn_methods: List[str] = []
                try:
                    # 1) paling reliable: metode yang dipakai generator (gen_targets)
                    if isinstance(gen_targets, list) and gen_targets:
                        turn_methods = [m for m in gen_targets if isinstance(m, str) and m]

                    # 2) fallback: dari bullets_meta (kadang lebih kaya)
                    if not turn_methods and isinstance(bullets_meta, dict):
                        md = (
                            bullets_meta.get("target_methods_for_gen")
                            or bullets_meta.get("methods_detected")
                            or []
                        )
                        if isinstance(md, list):
                            turn_methods = [m for m in md if isinstance(m, str) and m]

                    # 3) fallback terakhir: scan jawaban raw
                    if not turn_methods:
                        turn_methods = detect_methods_in_text(answer_raw)

                except Exception:
                    turn_methods = []

                # dedupe + batasi
                turn_methods = list(dict.fromkeys(turn_methods))[:5]

                st.session_state["history"].append(
                    {
                        "turn_id": turn_id,
                        "query": query,
                        "answer_preview": (jawaban or answer_raw)[:220]
                        + ("..." if len(jawaban or answer_raw) > 220 else ""),
                        "detect_flags": detect_flags,
                        "topic_shift": bool(detect_flags.get("topic_shift")),
                        "metadata_intent": None,
                        "methods": turn_methods,
                    }
                )

                # persist last result (bersifat tetap hingga query baru dijalankan)
                st.session_state["last_answer_raw"] = answer_raw  # menyimpan last answer dari copy
                st.session_state["last_turn_id"] = turn_id
                st.session_state["last_nodes"] = nodes  # menyimpan last nodes untuk
                st.session_state["last_strategy"] = strategy
                st.session_state["_scroll_answer"] = True
                st.session_state["last_user_query"] = query
                st.session_state["last_retrieval_query"] = retrieval_query

                # tandai bahwa konfigurasi UI saat ini sudah dipakai untuk hasil terakhir
                st.session_state["last_applied_ui_cfg"] = current_ui_cfg.copy()
                st.session_state["last_history_used"] = history_used
                st.session_state["last_contextualized"] = bool(retrieval_query != query)

                # simpan full render data untuk turn viewer
                if "turn_data" not in st.session_state:
                    st.session_state["turn_data"] = {}
                st.session_state["turn_data"][turn_id] = {
                    "nodes": nodes,
                    "answer_raw": answer_raw,
                    "user_query": query,
                    "retrieval_query": retrieval_query,
                    "strategy": strategy,
                    "history_used": history_used,
                    "contextualized": bool(retrieval_query != query),
                }

            _proc_status.update(label="💡 Finished!", state="complete", expanded=False)

    # ===== Render Last Turn Status slot (setelah processing, nilainya sudah final) =====
    _turn_count  = int(st.session_state.get("turn_id", 0))
    _sid         = st.session_state.get("viewing_turn_id")
    _tds         = st.session_state.get("turn_data", {})
    _is_hist_sb  = (_sid is not None and _sid in _tds)

    if _is_hist_sb:
        # sidebar reflect turn yang sedang dilihat
        _sb_td       = _tds[_sid]
        _sb_turn_str = str(_sid)
        _sb_mem      = _sb_td.get("history_used")
        _sb_ctx      = _sb_td.get("contextualized")
        _sb_mem_str  = f"{_sb_mem} turns" if _sb_mem is not None else "—"
        _sb_ctx_str  = ("☑️" if _sb_ctx else "❌") if _sb_ctx is not None else "—"
    else:
        _sb_turn_str = str(_turn_count) if _turn_count > 0 else "—"
        _sb_mem      = st.session_state.get("last_history_used")
        _sb_ctx      = st.session_state.get("last_contextualized")
        _sb_mem_str  = f"{_sb_mem} turns" if _sb_mem is not None else "—"
        _sb_ctx_str  = ("☑️" if _sb_ctx else "❌") if _sb_ctx is not None else "—"

    with last_turn_status_slot.container():
        st.subheader("🚨 Last Turn Status")
        st.write("Turns (Session):", _sb_turn_str)
        st.write("Memory Used:", _sb_mem_str)
        st.write("Contextualized:", _sb_ctx_str)
        # Baris tambahan: hanya muncul saat user sedang melihat turn historis
        if _is_hist_sb:
            st.markdown(
                f'<div style="'
                f"background:rgba(37,99,235,0.08);"
                f"border-left:3px solid #3b82f6;"
                f"border-radius:0 0.3rem 0.3rem 0;"
                f"padding:0.3rem 0.6rem;"
                f"margin-top:0.35rem;"
                f'">'
                f'<span style="color:#93c5fd;font-size:0.8rem;">👁️ Viewing:</span>'
                f'<span style="color:#60a5fa;font-weight:700;font-size:0.8rem;'
                f'margin-left:0.4rem;">Chat-Turn {_sid}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ===== Downloads (Logs) - Extend Sidebar =====
    with st.sidebar.expander("Downloads (Run Logs)", expanded=False):
        manifest_path = rm.run_path / "manifest.json"
        download_file_button(
            "📥 Download manifest.json",
            manifest_path,
            "application/json",
            disabled_caption="Manifest selalu ada.",
        )
        download_file_button(
            "📥 Download retrieval.jsonl",
            rm.retrieval_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "📥 Download answers.jsonl",
            rm.answers_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "📥 Download app.log",
            rm.app_log_path,
            "text/plain",
            disabled_caption="Log aplikasi untuk debug.",
        )
        download_logs_zip_button(
            "📥 Download logs.zip (all)",
            files=[manifest_path, rm.retrieval_log_path, rm.answers_log_path, rm.app_log_path],
            zip_name=f"{rm.run_id}_logs.zip",
        )

    # ===== Compute has_result sekali =====
    has_result = (
        st.session_state.get("last_answer_raw") is not None
        and int(st.session_state.get("last_turn_id", 0)) > 0
    )

    # ===== Render indikator config (setelah submit) =====
    last_applied = st.session_state.get("last_applied_ui_cfg")
    config_dirty = (last_applied is not None) and (last_applied != current_ui_cfg)

    if not has_result:
        # belum ada query sama sekali → pesan netral (tidak membingungkan)
        cfg_indicator_slot.info("ℹ️ Belum ada Query. **Atur Konfigurasi** lalu Tekan **Kirim**.")
    else:
        if config_dirty:
            cfg_indicator_slot.warning(
                "⚠️ **Konfigurasi Diubah**! Tekan **Kirim** untuk Menerapkan Perubahan ke Hasil Berikutnya."
            )
        else:
            # hijau (success) supaya jelas “sudah diterapkan”
            cfg_indicator_slot.success("❇️ **Konfigurasi Aktif** sudah Diterapkan pada Hasil Terakhir.")

    # ===== Render history area (setelah submit) =====
    _viewing_id_now = st.session_state.get("viewing_turn_id")
    _last_turn_id_now = int(st.session_state.get("last_turn_id", 0))
    with history_slot.container():
        with st.expander("📚 Riwayat Tanya-Jawab (Session)", expanded=False):
            if not st.session_state["history"]:
                st.caption("Belum ada Pertanyaan pada Sesi ini!")
            else:
                # Sentinel: ID unik agar CSS :has() hanya menarget expander history ini,
                # tidak bocor ke expander lain (Sources, Downloads, dll.)
                st.markdown(
                    '<div id="hist-scope" style="display:none;"></div>',
                    unsafe_allow_html=True,
                )
                # CSS: style tombol history (card)
                st.markdown(
                    """
                    <style>
                    /* ── Scrollable area ── */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"] {
                        max-height: 360px !important;
                        overflow-y: auto !important;
                        overflow-x: hidden !important;
                        padding-right: 0.3rem !important;
                        /* Fade-out bawah: sinyal visual ada konten tersembunyi */
                        -webkit-mask-image: linear-gradient(
                            to bottom, black 0%, black 80%, transparent 100%
                        ) !important;
                        mask-image: linear-gradient(
                            to bottom, black 0%, black 80%, transparent 100%
                        ) !important;
                    }
                    /* Scrollbar */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar {
                        width: 5px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-track {
                        background: #0d1117 !important;
                        border-radius: 999px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb {
                        background: #1e3a52 !important;
                        border-radius: 999px !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb:hover {
                        background: #3b82f6 !important;
                    }
                    /* Firefox */
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div[data-testid="stExpanderDetails"] {
                        scrollbar-width: thin !important;
                        scrollbar-color: #1e3a52 #0d1117 !important;
                    }

                    /* ── Inactive turn: card button ── */
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button {
                        text-align: left !important;
                        white-space: pre-wrap !important;
                        height: auto !important;
                        min-height: 4.8rem !important;
                        padding: 0.75rem 1.05rem !important;
                        line-height: 1.6 !important;
                        background: #0d1117 !important;
                        border: 1px solid #1e2d40 !important;
                        border-radius: 0.55rem !important;
                        color: #cbd5e1 !important;
                        font-size: 0.875rem !important;
                        width: 100% !important;
                        margin-bottom: 0.65rem !important;
                        display: flex !important;
                        align-items: flex-start !important;
                        justify-content: flex-start !important;
                    }
                    /* Wildcard: semua child element (p, div, span) ikut rata kiri */
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button *,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button p,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button div,
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button span {
                        text-align: left !important;
                        width: 100% !important;
                        margin: 0 !important;
                        display: block !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button:hover {
                        background: #0f1e2e !important;
                        border-color: #3b82f6 !important;
                        color: #f1f5f9 !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope) div.stButton > button:hover * {
                        color: #f1f5f9 !important;
                    }
                    div[data-testid="stExpander"]:has(#hist-scope)
                        div.stButton > button:focus:not(:active) {
                        border-color: #60a5fa !important;
                        box-shadow: 0 0 0 2px #1d4ed833 !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                for item in st.session_state["history"][-10:]:
                    _tid       = item["turn_id"]
                    _is_active = (_viewing_id_now == _tid)
                    _q_raw     = item.get("query", "")
                    _a_raw     = item.get("answer_preview", "")
                    _is_latest = (_tid == _last_turn_id_now and _viewing_id_now is None)

                    if _is_active:
                        # ── Turn aktif: vibrant border, dark fill ──
                        _q_esc = _html_mod.escape(_q_raw)
                        _a_esc = _html_mod.escape(_a_raw)
                        _badge_label = "✦ Terbaru · Sedang Dilihat" if _is_latest else "👁️ Sedang Dilihat"
                        st.markdown(
                            f'<div style="'
                            f"border:1px solid #2563eb;"
                            f"border-left:4px solid #3b82f6;"
                            f"border-radius:0.55rem;"
                            f"background:rgba(37,99,235,0.07);"
                            f"padding:0.75rem 1.05rem;"
                            f"margin-bottom:0.65rem;"
                            f'">'
                            # ── Header row: Turn N + badge ──
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;margin-bottom:0.35rem;">'
                            f'<span style="color:#60a5fa;font-weight:700;font-size:0.8rem;'
                            f'letter-spacing:0.04em;text-transform:uppercase;">'
                            f'Chat-Turn {_tid}</span>'
                            f'<span style="background:rgba(59,130,246,0.15);color:#93c5fd;'
                            f'border:1px solid #1d4ed8;padding:1px 10px;border-radius:999px;'
                            f'font-size:0.73rem;font-weight:600;">'
                            f'{_badge_label}</span>'
                            f'</div>'
                            # ── Divider ──
                            f'<div style="border-top:1px solid #1e3a5f;'
                            f'margin-bottom:0.45rem;"></div>'
                            # ── Q row ──
                            f'<div style="color:#e2e8f0;font-size:0.875rem;'
                            f'line-height:1.6;margin-bottom:0.3rem;text-align:left;">'
                            f'<span style="color:#60a5fa;font-weight:700;">Q{_tid}:</span>'
                            f'&nbsp;{_q_esc}'
                            f'</div>'
                            # ── A row ──
                            f'<div style="color:#94a3b8;font-size:0.83rem;'
                            f'line-height:1.55;text-align:left;">'
                            f'<span style="color:#7dd3fc;font-weight:700;">A:</span>'
                            f'&nbsp;{_a_esc}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    else:
                        # ── Turn tidak aktif: button card left-aligned ──
                        _latest_marker = "  ✦" if _is_latest else ""
                        _q_disp = _q_raw[:100] + ("…" if len(_q_raw) > 100 else "")
                        _a_disp = _a_raw[:160] + ("…" if len(_a_raw) > 160 else "")
                        # Newline dipakai agar button Streamlit tampil multi-baris (pre-wrap)
                        _divider_line = "─" * 38
                        _btn_label = (
                            f"Chat-Turn {_tid}{_latest_marker}\n"
                            f"{_divider_line}\n"
                            f"Q{_tid}: {_q_disp}\n"
                            f"A:  {_a_disp}"
                        )
                        if st.button(
                            _btn_label,
                            key=f"view_turn_{_tid}",
                            use_container_width=True,
                        ):
                            st.session_state["viewing_turn_id"] = _tid
                            st.rerun()

    # ===== Render results (JIKA, sudah adanya query tersubmit DAN user belum/hendak mengirimkan query baru) =====
    if has_result:
        # --- Turn Viewer: tentukan sumber data yang akan di-render ---
        _viewing_id = st.session_state.get("viewing_turn_id")
        _turn_data_store = st.session_state.get("turn_data", {})
        _is_viewing_history = (
            _viewing_id is not None and _viewing_id in _turn_data_store
        )

        if _is_viewing_history:
            _td = _turn_data_store[_viewing_id]
            nodes: List[Any] = _td.get("nodes") or []
            answer_raw: str = _td.get("answer_raw", "")
            turn_id = int(_viewing_id or 0)
            uq = _td.get("user_query", "")
            rq = _td.get("retrieval_query", "")
            last_strategy = _td.get("strategy", "")
        else:
            nodes = st.session_state.get("last_nodes") or []
            answer_raw = st.session_state.get("last_answer_raw", "")
            turn_id = st.session_state.get("last_turn_id", 0)
            uq = st.session_state.get("last_user_query", "")
            rq = st.session_state.get("last_retrieval_query", "")
            last_strategy = st.session_state.get("last_strategy", "")

        jawaban, bukti = parse_answer(answer_raw)
        global_not_found = is_global_not_found_answer(answer_raw)

        # Anchor target
        st.markdown('<div id="answer-anchor"></div>', unsafe_allow_html=True)

        # Banner + tombol kembali jika sedang viewing historis
        if _is_viewing_history:
            render_viewing_banner(turn_id)
            if st.button("↩️ Kembali ke Hasil Terbaru", key="result_back_latest"):
                st.session_state["viewing_turn_id"] = None
                st.rerun()

        render_processing_box(uq, rq)

        # Answer section — styled card
        h1, h2 = st.columns([0.88, 0.12])
        with h1:
            render_answer_card_header()
        with h2:
            copy_to_clipboard_button(answer_raw, label="📑", key=f"copy_{rm.run_id}_{turn_id}")

        with st.container(border=True):
            if jawaban:
                # Format untuk render Markdown yang rapi (terutama multi-metode + numbered list)
                st.markdown(_prepare_jawaban_markdown(jawaban))
            else:
                st.write(answer_raw)

        # Evidence section — styled card
        render_bukti_section(bukti)

        # Auto-scroll after submit
        if st.session_state.get("_scroll_answer"):
            scroll_to_anchor("answer-anchor")
            st.session_state["_scroll_answer"] = False

        # CTX mapping + filter
        render_ctx_mapping_header()

        if last_strategy == "metadata_router":
            st.info(
                "Pertanyaan ini dijawab dari **doc_catalog (metadata routing)**, sehingga tidak ada CTX chunk yang ditampilkan."
            )
            highlight_terms = []  # tidak dipakai
        else:
            # ===== jika GLOBAL not_found, jangan tampilkan CTX mapping (default) =====
            if global_not_found:
                st.warning(
                    "⚠️ Jawaban berstatus **Tidak ditemukan pada dokumen**, sehingga **CTX Mapping tidak ditampilkan** "
                    "agar tidak misleading. Aktifkan mode debug jika ingin melihat sources yang sempat ter-retrieve."
                )
                show_debug = st.checkbox(
                    "Show CTX Mapping (Debug)",
                    value=False,
                    key=f"dbg_ctx_{rm.run_id}_{turn_id}",
                )
                if not show_debug:
                    highlight_terms = []
                else:
                    ctx_rows = []
                    for i, n in enumerate(nodes, start=1):
                        source_file = n.metadata.get("source_file", "")
                        page = n.metadata.get("page", "?")
                        ctx_rows.append(
                            {
                                "CTX": f"CTX {i}",
                                "doc_id": n.doc_id,
                                "page": page,
                                "source_file": source_file,
                                "sim": round(dist_to_sim(float(n.score)), 4),
                                "dist": round(float(n.score), 4),
                                "chunk_id": n.chunk_id,
                            }
                        )
                    render_ctx_mapping_table(ctx_rows)

                    # Highlight terms (manual override). Jika kosong -> auto dari query terakhir
                    hl_manual = st.text_input(
                        "Highlight Keyword di PDF (opsional, pisahkan dengan koma). Kosongkan = Auto dari Query.",
                        key="hl_terms",
                    )
                    manual_terms = [t.strip() for t in (hl_manual or "").split(",") if t.strip()]

                    last_query = st.session_state.get("last_submitted_query", "")
                    query_terms = extract_terms_from_query(last_query)

                    ctx1_text = nodes[0].text if nodes else ""
                    ctx_terms = extract_terms_from_text(ctx1_text, max_terms=6)

                    combined = []
                    seen = set()
                    for t in query_terms + ctx_terms:
                        tl = t.lower().strip()
                        if not tl or tl in seen:
                            continue
                        seen.add(tl)
                        combined.append(t)

                    highlight_terms = manual_terms if manual_terms else combined[:8]

            else:
                ctx_rows = []
                for i, n in enumerate(nodes, start=1):
                    source_file = n.metadata.get("source_file", "")
                    page = n.metadata.get("page", "?")
                    ctx_rows.append(
                        {
                            "CTX": f"CTX {i}",
                            "doc_id": n.doc_id,
                            "page": page,
                            "source_file": source_file,
                            "sim": round(dist_to_sim(float(n.score)), 4),
                            "dist": round(float(n.score), 4),
                            "chunk_id": n.chunk_id,
                        }
                    )
                render_ctx_mapping_table(ctx_rows)

                # Highlight terms (manual override). Jika kosong -> auto dari query terakhir
                hl_manual = st.text_input(
                    "Highlight Keyword di PDF (opsional, pisahkan dengan koma). Kosongkan = Auto dari Query.",
                    key="hl_terms",
                )
                manual_terms = [t.strip() for t in (hl_manual or "").split(",") if t.strip()]

                last_query = st.session_state.get("last_submitted_query", "")
                query_terms = extract_terms_from_query(last_query)

                ctx1_text = nodes[0].text if nodes else ""
                ctx_terms = extract_terms_from_text(ctx1_text, max_terms=6)

                combined = []
                seen = set()
                for t in query_terms + ctx_terms:
                    tl = t.lower().strip()
                    if not tl or tl in seen:
                        continue
                    seen.add(tl)
                    combined.append(t)

                highlight_terms = manual_terms if manual_terms else combined[:8]

        # Sources + PDF tools + filter (UI)
        render_sources_header()

        if last_strategy == "metadata_router":
            st.info(
                "Tidak ada Top-k Sources karena tidak dilakukan retrieval (jawaban diambil dari doc_catalog)."
            )
        else:
            # ===== jika GLOBAL not_found, jangan tampilkan Sources (default) =====
            if global_not_found:
                st.warning(
                    "⚠️ Jawaban berstatus **Tidak ditemukan pada dokumen**, sehingga **Sources (Top-k) tidak ditampilkan** "
                    "agar tidak misleading. Aktifkan mode debug jika ingin melihat sources yang sempat ter-retrieve."
                )
                show_debug_src = st.checkbox(
                    "Show Sources (Top-k) (Debug)",
                    value=False,
                    key=f"dbg_src_{rm.run_id}_{turn_id}",
                )
                if not show_debug_src:
                    indexed_nodes = []
                else:
                    indexed_nodes = list(enumerate(nodes, start=1))
            else:
                indexed_nodes = list(enumerate(nodes, start=1))

            if filter_q.strip():
                filtered = [(i, n) for i, n in indexed_nodes if node_matches_filter(n, filter_q)]
                if show_only_matches:
                    indexed_nodes = filtered
                st.caption(f"Filter match: {len(filtered)} dari {len(nodes)} sources.")

            for i, n in indexed_nodes:
                dist = float(n.score)
                sim = dist_to_sim(dist)
                source_file = n.metadata.get("source_file", "")
                page = int(n.metadata.get("page", 1) or 1)

                preview = (n.text or "").strip().replace("\n", " ")
                preview = preview[:260] + ("..." if len(preview) > 260 else "")

                _badge = sim_color_badge(sim)
                title = (
                    f"{_badge} {n.doc_id} | sim={sim:.4f} | dist={dist:.4f} | {source_file} p.{page}"
                )
                with st.expander(title):
                    render_source_score_pills(sim, dist, ctx_label=f"CTX {i}")
                    st.caption(f"Preview: {preview}")

                    # Resolve PDF path
                    pdf_path = resolve_pdf_path(
                        project_root=PROJECT_ROOT,
                        data_raw_dir=data_raw_dir,
                        metadata=n.metadata,
                    )

                    cols = st.columns([0.20, 0.20, 0.60], gap="small")
                    with cols[0]:
                        if pdf_path and st.button(
                            f"👀 View PDF (p.{page})", key=f"viewpdf_{rm.run_id}_{turn_id}_{i}"
                        ):
                            st.session_state["pdf_view"] = {
                                "path": str(pdf_path),
                                "page": int(page),
                                "terms": highlight_terms,
                            }
                            st.rerun()
                        if not pdf_path:
                            st.button("View PDF", disabled=True)

                    with cols[1]:
                        if pdf_path:
                            download_file_button(
                                "📥 Download PDF",
                                pdf_path,
                                "application/pdf",
                                file_name=pdf_path.name,
                                key=f"dlpdf_{rm.run_id}_{turn_id}_{i}",
                            )
                        else:
                            st.button("📥 Download PDF", disabled=True)

                    with cols[2]:
                        if pdf_path:
                            st.caption(f"PDF: {pdf_path.name} | page={page}")
                        else:
                            st.caption("PDF path tidak ditemukan dari metadata.")

                    st.write(n.text)

    # PDF Viewer section
    pdf_view = st.session_state.get("pdf_view")
    if pdf_view and isinstance(pdf_view, dict) and pdf_view.get("path"):
        pv_path = Path(str(pdf_view["path"]))
        pv_page = int(pdf_view.get("page", 1) or 1)
        render_pdf_viewer_header(pv_path.name, pv_page)

        if st.button("Close Viewer"):
            st.session_state["pdf_view"] = None
            st.rerun()

        if pv_path.exists():
            terms = pdf_view.get("terms") if isinstance(pdf_view, dict) else None
            render_pdf_viewer(pv_path, page=pv_page, height=720, highlight_terms=terms)
        else:
            st.warning("File PDF tidak ditemukan di path yang tersimpan.")

    if not has_result:
        # UI awal bersih
        st.caption(
            "Silahkan bertanya lalu klik **Kirim** untuk mendapatkan Jawaban, Bukti, CTX Mapping, dan Sources."
        )


if __name__ == "__main__":
    main()
