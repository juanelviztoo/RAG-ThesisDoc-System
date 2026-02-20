from __future__ import annotations

import os
from pathlib import Path

import streamlit as st  # nanti dipasang di requirements
from dotenv import load_dotenv

from src.core.config import load_config
from src.core.logger import setup_logger
from src.core.run_manager import RunManager
from src.rag.generate import generate_answer


def main():
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", layout="wide")
    st.title("RAG ThesisDoc System (V1 Scaffold)")

    load_dotenv()  # baca .env (lokal)

    config_path = st.sidebar.text_input("Config path", value="configs/v1.yaml")
    cfg = load_config(config_path)

    # init run
    rm = RunManager(runs_dir=Path(cfg["paths"]["runs_dir"]), config=cfg).start()
    logger = setup_logger(rm.app_log_path)
    logger.info(f"Run started: {rm.run_id}")

    # info sidebar
    st.sidebar.write("Run ID:", rm.run_id)
    st.sidebar.write("Collection:", cfg["index"]["collection_name"])
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])
    st.sidebar.write("LLM Provider (env):", os.getenv("LLM_PROVIDER", "(not set)"))

    st.info(
        "UI scaffold siap. V1 Retrieval belum diimplementasikan, jadi saat ini konteks masih dummy."
    )

    user_q = st.text_input("Tanya sesuatu:")
    if st.button("Kirim") and user_q.strip():
        # dummy contexts sementara (nanti diganti retrieval hasil Chroma)
        dummy_contexts = [
            "RAG adalah Retrieval-Augmented Generation: menggabungkan retrieval dokumen + generasi LLM."
        ]

        try:
            answer = generate_answer(cfg, user_q.strip(), dummy_contexts)
            st.subheader("Jawaban")
            st.write(answer)

            st.subheader("Sources (dummy)")
            st.write("- [CTX 1] Dummy context")

        except Exception as e:
            st.error(
                "Gagal memanggil LLM. Ini normal jika API key masih placeholder.\n\n"
                f"Detail: {type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()
