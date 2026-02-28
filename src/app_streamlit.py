from __future__ import annotations

import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st  # nanti dipasang di requirements
from dotenv import load_dotenv

from src.core.config import load_config
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
from src.rag.retrieve_dense import retrieve_dense


def dist_to_sim(dist: float) -> float:
    # distance -> similarity (0..1), lebih aman dari (1 - dist)
    return 1.0 / (1.0 + max(dist, 0.0))


def compute_retrieval_stats(nodes: List[Any]) -> Dict[str, Any]:
    if not nodes:
        return {
            "n_nodes": 0,
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

    # dibulatkan supaya log enak dibaca
    return {
        "n_nodes": int(len(nodes)),
        "sim_top1": round(float(sim_top1), 6),
        "sim_avg": round(float(sim_avg), 6),
        "sim_gap12": round(float(sim_gap12), 6) if sim_gap12 is not None else None,
        "dist_top1": round(float(dist_top1), 6),
    }


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
    return rm, logger


def get_or_create_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    # Jika belum ada run di session, buat
    if "rm" not in st.session_state or "logger" not in st.session_state:
        return _new_run(cfg, config_path)
    # Jika config path berubah, reset run baru
    if st.session_state.get("config_path") != config_path:
        return _new_run(cfg, config_path)
    return st.session_state["rm"], st.session_state["logger"]


def parse_answer(answer_text: str) -> Tuple[str, List[str]]:
    """
    Parse output generator yang idealnya:
      Jawaban: ...
      Bukti:
      - ... [CTX n]
    Tapi tetap toleran jika "Jawaban:" hilang, asalkan ada "Bukti:".
    """
    text = (answer_text or "").strip()
    if not text:
        return "", []

    # cari "Bukti:" (case-insensitive)
    m_bukti = re.search(r"\bBukti\s*:\s*", text, flags=re.IGNORECASE)
    if m_bukti:
        head = text[: m_bukti.start()].strip()
        tail = text[m_bukti.end() :].strip()

        # head bisa: "Jawaban: xxx" atau langsung "xxx"
        head2 = re.sub(r"^\s*Jawaban\s*:\s*", "", head, flags=re.IGNORECASE).strip()
        jawaban = head2

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

        # kalau hasilnya tetap "-" doang
        if len(bukti) == 1 and bukti[0] == "-":
            bukti = []

        return jawaban, bukti

    # kalau tidak ada "Bukti:", coba ambil "Jawaban:" kalau ada
    m_j = re.search(r"^\s*Jawaban\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m_j:
        jawaban = m_j.group(1).strip()
        return jawaban, []

    # fallback total: semua dianggap jawaban
    return text, []


def build_retrieval_only_answer(nodes: List[Any], max_points: int = 3) -> str:
    """
    Membuat jawaban "extractive" saat LLM dimatikan.
    Output mengikuti format generator agar UI tetap konsisten.
    """
    if not nodes:
        return "Jawaban: Tidak ditemukan pada dokumen.\nBukti: -"

    bullets: List[str] = []
    for i, n in enumerate(nodes[:max_points], start=1):
        source_file = n.metadata.get("source_file", "")
        page = n.metadata.get("page", "?")

        snippet = (n.text or "").strip().replace("\n", " ")
        snippet = " ".join(snippet.split())  # rapikan whitespace
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")

        bullets.append(f"- ({source_file} p.{page}) {snippet} [CTX {i}]")

    return (
        "Jawaban: (Mode retrieval-only) Berikut bukti paling relevan dari dokumen untuk pertanyaan ini.\n"
        "Bukti:\n" + "\n".join(bullets)
    )


def node_matches_filter(n: Any, q: str) -> bool:
    if not q:
        return True
    ql = q.lower().strip()
    doc_id = (getattr(n, "doc_id", "") or "").lower()
    sf = (n.metadata.get("source_file", "") or "").lower()
    txt = (getattr(n, "text", "") or "").lower()
    return (ql in doc_id) or (ql in sf) or (ql in txt)


def extract_terms_from_query(q: str) -> List[str]:
    """Ambil keyword sederhana dari query untuk highlight default."""
    ql = (q or "").lower()
    tokens = re.findall(r"[a-z0-9]+", ql)

    stop = {
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
        "evaluasi",
        "sistem",
        "dokumen",
    }
    terms = [t for t in tokens if len(t) >= 4 and t not in stop]
    # ambil beberapa saja biar ringan
    return terms[:5]


def extract_terms_from_text(text: str, max_terms: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    stop = {
        "yang",
        "dengan",
        "dalam",
        "pada",
        "untuk",
        "dari",
        "atau",
        "dan",
        "adalah",
        "ini",
        "itu",
        "sebagai",
        "dapat",
        "akan",
        "oleh",
        "menggunakan",
        "metode",
        "sistem",
        "penelitian",
        "hasil",
        "bab",
        "tabel",
        "gambar",
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


def render_processing_box(user_q: str, retrieval_q: str) -> None:
    """Box ringkas untuk menampilkan user query vs retrieval query (V1.5)."""
    user_q = (user_q or "").strip()
    retrieval_q = (retrieval_q or "").strip()

    if not user_q and not retrieval_q:
        return

    # escape biar aman kalau ada karakter aneh
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


def main() -> None:
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", layout="wide")
    st.title("RAG ThesisDoc System (V1: Dense Retrieval)")

    load_dotenv()  # baca .env (lokal)

    # Root project untuk resolve file
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> root
    # ambil data_raw dari config kalau ada, fallback "data_raw"
    # (akan set setelah cfg terbaca)
    # DATA_RAW_DIR = Path("data_raw")

    # ===== Sidebar: Controls =====
    st.sidebar.header("Controls")
    config_path = st.sidebar.text_input("Config path", value="configs/v1_5.yaml")
    cfg = load_config(config_path)
    data_raw_dir = Path(cfg.get("paths", {}).get("data_raw_dir", "data_raw"))

    # init/reuse run per session
    rm, logger = get_or_create_run(cfg, config_path)
    logger.info(f"Run active: {rm.run_id}")

    # info provider LLM
    provider = os.getenv("LLM_PROVIDER", "(not set)")
    groq_model = os.getenv("GROQ_MODEL", "(not set)")
    openai_model = os.getenv("OPENAI_MODEL", "(not set)")

    # Tombol new session (reset run_id + history)
    if st.sidebar.button("New Session (Run baru)"):
        rm, logger = _new_run(cfg, config_path)
        st.rerun()

    # info sidebar tambahan
    st.sidebar.divider()
    st.sidebar.write("Run ID:", rm.run_id)
    st.sidebar.write("Collection:", cfg["index"]["collection_name"])
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])

    st.sidebar.write("LLM Provider:", provider)
    if provider == "groq":
        st.sidebar.write("Model:", groq_model)
    elif provider == "openai":
        st.sidebar.write("Model:", openai_model)

    # Toggle generator
    use_llm = st.sidebar.checkbox("Use LLM (Generator)", value=True)

    # =========================
    # V1.5 Controls: memory + contextualization (Mode demo vs eval)
    # =========================
    # demo: memory/contextualization boleh ON untuk pengalaman chat lebih natural
    # eval: dipaksa OFF agar hasil retrieval stabil & mudah dibandingkan antar run
    run_mode = st.sidebar.selectbox(
        "Mode",
        options=["demo", "eval"],
        help="eval = matikan memory/contextualization agar hasil konsisten untuk evaluasi.",
    )

    mem_cfg = cfg.get("memory", {})
    mem_default = bool(mem_cfg.get("enabled", False))
    mem_window_default = int(mem_cfg.get("window", 4))

    use_memory = st.sidebar.checkbox("Use Memory (V1.5)", value=mem_default)
    mem_window = st.sidebar.slider("Memory Window (Turns)", 0, 10, value=mem_window_default)

    contextualize_on = st.sidebar.checkbox(
        "Contextualize Query (Rewrite Sebelum Retrieval)",
        value=True,
        disabled=(not use_memory),
    )

    # Eval mode: paksa off
    if run_mode == "eval":
        use_memory = False
        contextualize_on = False
        mem_window = 0

    # ===== Override controls (UI-level) untuk eksperimen cepat =====
    top_k_ui = st.sidebar.slider(
        "top_k (override)", min_value=3, max_value=15, value=int(cfg["retrieval"]["top_k"])
    )
    max_chars_ctx_ui = st.sidebar.slider(
        "max_chars_per_ctx (override)",
        min_value=600,
        max_value=3000,
        value=int(cfg.get("llm", {}).get("max_chars_per_ctx", 1800)),
    )
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
    }

    # snapshot terakhir yang BENAR-BENAR sudah dipakai untuk menghasilkan hasil terakhir
    if "last_applied_ui_cfg" not in st.session_state:
        st.session_state["last_applied_ui_cfg"] = None  # belum ada query sama sekali

    cfg_indicator_slot = st.sidebar.empty()  # <--- SLOT, akan diisi di bagian bawah setelah submit

    # Key status (tanpa menampilkan key)
    st.sidebar.divider()
    st.sidebar.write("GROQ key set?:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("OPENAI key set?:", bool(os.getenv("OPENAI_API_KEY")))

    st.sidebar.divider()
    # apakah mau config snapshot per baris?
    include_cfg = st.sidebar.checkbox(
        "Log config snapshot per row", value=(os.getenv("LOG_INCLUDE_CONFIG_PER_ROW", "0") == "1")
    )
    st.sidebar.caption("Jika ON, file log lebih besar tapi audit lebih kuat.")

    # ===== Main Info =====
    st.info(
        "V1 Dense Retrieval aktif. "
        "Generator opsional (Use LLM). "
        "Output selalu menampilkan Sources (Top-k) + log JSONL."
    )

    # ===== History slot (chat-like) =====
    if "history" not in st.session_state:
        st.session_state["history"] = []

    history_slot = st.empty()  # <--- SLOT (akan di-render ulang di bawah setelah submit)

    # ===== Input =====
    user_q = st.text_input("Ada yang bisa saya bantu? Anda bisa tanyakan di sini:")

    # Filter sources (UI)
    filter_q = st.text_input(
        "Filter sources (doc_id / source_file / text) - BOLEH DIISI ATAU KOSONGKAN:", value=""
    )
    show_only_matches = st.checkbox("Show only matched sources", value=True)

    # Tombol submit (pakai variabel agar logika show/hide benar di klik pertama)
    submitted = st.button("Kirim")

    if submitted and user_q.strip():
        query = user_q.strip()

        # turn id per session
        st.session_state["turn_id"] += 1
        turn_id = int(st.session_state["turn_id"])

        # simpan query terakhir untuk auto-highlight
        st.session_state["last_submitted_query"] = query
        st.session_state["last_turn_id"] = turn_id

        # Override cfg runtime (tanpa mengubah file yaml)
        cfg_runtime = dict(cfg)
        cfg_runtime["retrieval"] = dict(cfg.get("retrieval", {}))
        cfg_runtime["retrieval"]["top_k"] = int(top_k_ui)
        cfg_runtime["llm"] = dict(cfg.get("llm", {}))
        cfg_runtime["llm"]["max_chars_per_ctx"] = int(max_chars_ctx_ui)

        # ===== Memory window (ambil history terakhir) =====
        # Ambil window percakapan terakhir (history) untuk membantu rewrite query.
        history_pairs = st.session_state.get("history", [])
        history_window = []
        if use_memory and mem_window > 0 and history_pairs:
            history_window = history_pairs[-int(mem_window) :]
        # Ringkasan memory yang dipakai untuk turn ini (untuk audit log)
        history_used = len(history_window) if (use_memory and mem_window > 0) else 0
        history_turn_ids = [h.get("turn_id") for h in history_window] if history_window else []

        # ===== Contextualize / rewrite query =====
        # Contextualization akan menulis ulang pertanyaan user agar eksplisit (mis. coreference "itu", "tahapannya").
        retrieval_query = query
        if use_memory and contextualize_on and history_window:
            retrieval_query = contextualize_question(
                cfg_runtime,
                question=query,
                history_pairs=history_window,
                use_llm=use_llm,  # pakai LLM hanya jika generator ON (aman & hemat)
            )

        # =========================
        # Retrieval
        # =========================
        # Pakai retrieval_query (hasil rewrite) agar embedding search lebih tepat.
        # nodes -> dipakai untuk:
        # 1) konteks generator (contexts)
        # 2) CTX mapping + sources UI
        # 3) logging (retrieval.jsonl)
        nodes = retrieve_dense(cfg_runtime, retrieval_query)
        # contexts untuk LLM (kalau nanti key valid)
        contexts = [n.text for n in nodes]

        retrieval_stats = compute_retrieval_stats(nodes)

        strategy = cfg_runtime["retrieval"]["mode"]  # contoh: "dense"
        if retrieval_query != query:
            strategy = f"{strategy}+rewrite"

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
            "strategy": strategy,
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

        if use_llm:
            try:
                answer_raw = generate_answer(cfg_runtime, user_query, contexts)
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
            "used_context_chunk_ids": [n.chunk_id for n in nodes],
            "answer": answer_raw,
            "strategy": strategy,
            "retrieval_stats": retrieval_stats,
            "retrieval_confidence": retrieval_stats.get("sim_top1"),
            "generator_strategy": "llm" if use_llm else "retrieval_only",
            "generator_status": generator_status,
            "generator_meta": generator_meta,
            "error": generator_error,
        }
        rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)

        # Save to session history (preview singkat)
        st.session_state["history"].append(
            {
                "turn_id": turn_id,
                "query": query,
                "answer_preview": (jawaban or answer_raw)[:220]
                + ("..." if len(jawaban or answer_raw) > 220 else ""),
            }
        )

        # persist last result (bersifat tetap hingga query baru dijalankan)
        st.session_state["last_answer_raw"] = answer_raw  # menyimpan last answer dari copy
        st.session_state["last_turn_id"] = turn_id
        st.session_state["last_nodes"] = nodes  # menyimpan last nodes untuk rendering
        st.session_state["_scroll_answer"] = True
        st.session_state["last_user_query"] = query
        st.session_state["last_retrieval_query"] = retrieval_query

        # tandai bahwa konfigurasi UI saat ini sudah dipakai untuk hasil terakhir
        st.session_state["last_applied_ui_cfg"] = current_ui_cfg.copy()

    # ===== Downloads (Logs) - Extend Sidebar =====
    with st.sidebar.expander("Downloads (Run Logs)", expanded=False):
        manifest_path = rm.run_path / "manifest.json"
        download_file_button(
            "Download manifest.json",
            manifest_path,
            "application/json",
            disabled_caption="Manifest selalu ada.",
        )
        download_file_button(
            "Download retrieval.jsonl",
            rm.retrieval_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "Download answers.jsonl",
            rm.answers_log_path,
            "application/json",
            disabled_caption="Akan muncul setelah minimal 1 query.",
        )
        download_file_button(
            "Download app.log",
            rm.app_log_path,
            "text/plain",
            disabled_caption="Log aplikasi untuk debug.",
        )
        download_logs_zip_button(
            "Download logs.zip (all)",
            files=[manifest_path, rm.retrieval_log_path, rm.answers_log_path, rm.app_log_path],
            zip_name=f"{rm.run_id}_logs.zip",
        )

    # ===== Compute has_result sekali =====
    has_result = bool(st.session_state.get("last_answer_raw")) and bool(
        st.session_state.get("last_nodes")
    )

    # ===== Render indikator config (setelah submit) =====
    last_applied = st.session_state.get("last_applied_ui_cfg")
    config_dirty = (last_applied is not None) and (last_applied != current_ui_cfg)

    if not has_result:
        # belum ada query sama sekali → pesan netral (tidak membingungkan)
        cfg_indicator_slot.info("Belum ada query. Atur konfigurasi lalu klik **Kirim**.")
    else:
        if config_dirty:
            cfg_indicator_slot.warning(
                "Konfigurasi diubah. Klik **Kirim** untuk menerapkan perubahan ke hasil berikutnya."
            )
        else:
            # hijau (success) supaya jelas “sudah diterapkan”
            cfg_indicator_slot.success("Konfigurasi aktif sudah diterapkan pada hasil terakhir.")

    # ===== Render history area (setelah submit) =====
    with history_slot.container():
        with st.expander("Riwayat Tanya-Jawab (Session)", expanded=False):
            if not st.session_state["history"]:
                st.caption("Belum ada pertanyaan pada sesi ini.")
            else:
                for item in st.session_state["history"][-10:]:
                    st.markdown(f"**Q{item['turn_id']}:** {item['query']}")
                    st.markdown(f"**A:** {item['answer_preview']}")
                    st.divider()

    # ===== Render results (JIKA, sudah adanya query tersubmit DAN user belum/hendak mengirimkan query baru) =====
    if has_result:
        nodes: List[Any] = st.session_state["last_nodes"]
        answer_raw: str = st.session_state["last_answer_raw"]
        turn_id: int = st.session_state.get("last_turn_id", 0)
        jawaban, bukti = parse_answer(answer_raw)

        # Anchor target
        st.markdown('<div id="answer-anchor"></div>', unsafe_allow_html=True)

        uq = st.session_state.get("last_user_query", "")
        rq = st.session_state.get("last_retrieval_query", "")
        render_processing_box(uq, rq)

        # Answer section
        h1, h2 = st.columns([0.86, 0.14])
        with h1:
            st.subheader("Jawaban")
        with h2:
            copy_to_clipboard_button(answer_raw, label="📑", key=f"copy_{rm.run_id}_{turn_id}")

        st.write(jawaban if jawaban else answer_raw)

        # Evidence section
        st.subheader("Bukti")
        if bukti:
            for b in bukti:
                st.markdown(f"- {b}")
        else:
            st.write("-")

        # Auto-scroll after submit
        if st.session_state.get("_scroll_answer"):
            scroll_to_anchor("answer-anchor")
            st.session_state["_scroll_answer"] = False

        # CTX mapping + filter
        st.subheader("CTX Mapping")
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
        st.dataframe(ctx_rows, width="stretch", hide_index=True)
        # st.dataframe(ctx_rows, width="content", hide_index=True)

        # Highlight terms (manual override). Jika kosong -> auto dari query terakhir
        hl_manual = st.text_input(
            "Highlight keyword di PDF (opsional, pisahkan dengan koma). Kosongkan = auto dari query.",
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
        st.subheader("Sources (Top-k)")
        indexed_nodes: List[Tuple[int, Any]] = list(enumerate(nodes, start=1))

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

            title = f"{i}. {n.doc_id} | sim={sim:.4f} | dist={dist:.4f} | {source_file} p.{page}"
            with st.expander(title):
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
                        f"View PDF (p.{page})", key=f"viewpdf_{rm.run_id}_{turn_id}_{i}"
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
                            "Download PDF",
                            pdf_path,
                            "application/pdf",
                            file_name=pdf_path.name,
                            key=f"dlpdf_{rm.run_id}_{turn_id}_{i}",  # <-- unik per source
                        )
                    else:
                        st.button("Download PDF", disabled=True)

                with cols[2]:
                    if pdf_path:
                        st.caption(f"PDF: {pdf_path.name} | page={page}")
                    else:
                        st.caption("PDF path tidak ditemukan dari metadata.")

                st.write(n.text)

    # PDF Viewer section
    pdf_view = st.session_state.get("pdf_view")
    if pdf_view and isinstance(pdf_view, dict) and pdf_view.get("path"):
        st.subheader("PDF Viewer")
        pv_path = Path(str(pdf_view["path"]))
        pv_page = int(pdf_view.get("page", 1) or 1)

        cols = st.columns([0.2, 0.8])
        with cols[0]:
            if st.button("Close Viewer"):
                st.session_state["pdf_view"] = None
                st.rerun()
        with cols[1]:
            st.caption(f"Menampilkan: {pv_path.name} (page {pv_page})")

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
