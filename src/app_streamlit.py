from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st  # nanti dipasang di requirements
from dotenv import load_dotenv

from src.core.config import load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager
from src.rag.generate import generate_answer
from src.rag.retrieve_dense import retrieve_dense


def dist_to_sim(dist: float) -> float:
    # distance -> similarity (0..1), lebih aman dari (1 - dist)
    return 1.0 / (1.0 + max(dist, 0.0))


def get_or_create_run(cfg: Dict[str, Any], config_path: str) -> Tuple[RunManager, Any]:
    """
    1 session Streamlit = 1 run_id.
    Jika config path berubah, buat run baru.
    """
    runs_dir = Path(cfg["paths"]["runs_dir"])

    # Jika belum ada run di session, buat
    if "rm" not in st.session_state or "logger" not in st.session_state:
        rm = RunManager(runs_dir=runs_dir, config=cfg).start()
        logger = setup_logger(rm.app_log_path)

        st.session_state["rm"] = rm
        st.session_state["logger"] = logger
        st.session_state["config_path"] = config_path
        st.session_state["turn_id"] = 0
        return rm, logger

    # Jika config path berubah, reset run baru
    if st.session_state.get("config_path") != config_path:
        rm = RunManager(runs_dir=runs_dir, config=cfg).start()
        logger = setup_logger(rm.app_log_path)

        st.session_state["rm"] = rm
        st.session_state["logger"] = logger
        st.session_state["config_path"] = config_path
        st.session_state["turn_id"] = 0
        return rm, logger

    return st.session_state["rm"], st.session_state["logger"]


def main() -> None:
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", layout="wide")
    st.title("RAG ThesisDoc System (V1: Dense Retrieval)")

    load_dotenv()  # baca .env (lokal)

    config_path = st.sidebar.text_input("Config path", value="configs/v1.yaml")
    cfg = load_config(config_path)

    # init/reuse run per session
    rm, logger = get_or_create_run(cfg, config_path)
    logger.info(f"Run active: {rm.run_id}")

    # info sidebar
    st.sidebar.write("Run ID:", rm.run_id)
    st.sidebar.write("Collection:", cfg["index"]["collection_name"])
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])

    provider = os.getenv("LLM_PROVIDER", "(not set)")
    st.sidebar.write("LLM Provider (env):", provider)

    # Toggle generator
    use_llm = st.sidebar.checkbox("Use LLM (Generator)", value=True)

    # Key status (tanpa menampilkan key)
    st.sidebar.write("GROQ key set?:", bool(os.getenv("GROQ_API_KEY")))
    st.sidebar.write("OPENAI key set?:", bool(os.getenv("OPENAI_API_KEY")))

    st.info(
        "V1 Dense Retrieval aktif (testing). "
        "LLM opsional (jika API key valid). "
        "Output selalu menampilkan Sources (Top-k) + log JSONL."
    )

    user_q = st.text_input("Tanya sesuatu:")
    if st.button("Kirim") and user_q.strip():
        query = user_q.strip()

        # turn id per session
        st.session_state["turn_id"] += 1
        turn_id = int(st.session_state["turn_id"])

        # retrieval
        nodes = retrieve_dense(cfg, user_q.strip())
        # contexts untuk LLM (kalau nanti key valid)
        contexts = [n.text for n in nodes]

        # timestamp per turn
        ts = datetime.now().isoformat(timespec="seconds")

        # apakah mau config snapshot per baris?
        include_cfg = os.getenv("LOG_INCLUDE_CONFIG_PER_ROW", "0") == "1"

        # log retrieval via RunManager
        retrieval_record = {
            "run_id": rm.run_id,
            "turn_id": turn_id,
            "timestamp": ts,
            "query": query,
            "retrieval_mode": cfg["retrieval"]["mode"],
            "llm_provider": provider,
            "use_llm": use_llm,
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

        # Generate (controlled by toggle; kalau gagal, fallback placeholder)
        if use_llm:
            try:
                answer = generate_answer(cfg, query, contexts)
            except Exception as ex:
                logger.exception("LLM call failed")
                st.warning(f"LLM error: {type(ex).__name__}: {ex}")
                answer = "(LLM gagal / API key belum valid) Berikut adalah hasil retrieval top-k."
        else:
            answer = "(Mode retrieval-only) Berikut adalah hasil retrieval top-k."

        st.subheader("Jawaban")
        st.write(answer)

        # sources UI
        st.subheader("Sources (Top-k)")
        for i, n in enumerate(nodes, start=1):
            dist = float(n.score)
            sim = dist_to_sim(dist)
            source_file = n.metadata.get("source_file", "")
            page = n.metadata.get("page", "?")

            title = f"{i}. {n.doc_id} | sim={sim:.4f} | dist={dist:.4f} | {source_file} p{page}"
            with st.expander(title):
                st.write(n.text)

        # log answers via RunManager
        answer_record = {
            "run_id": rm.run_id,
            "turn_id": turn_id,
            "timestamp": ts,
            "query": query,
            "answer": answer,
            "llm_provider": provider,
            "use_llm": use_llm,
            "used_context_chunk_ids": [n.chunk_id for n in nodes],
        }
        rm.log_answer(answer_record, include_config_snapshot_per_row=include_cfg)


if __name__ == "__main__":
    main()
