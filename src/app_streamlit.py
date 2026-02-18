from __future__ import annotations

import streamlit as st  # nanti dipasang di requirements
from src.core.config import load_config
from src.core.run_manager import RunManager
from src.core.logger import setup_logger


def main():
    st.set_page_config(page_title="Sistem RAG Dokumen Skripsi", layout="wide")
    st.title("RAG Skripsi System (Scaffold)")

    config_path = st.sidebar.text_input("Config path", value="configs/v1.yaml")
    cfg = load_config(config_path)

    # init run
    rm = RunManager(runs_dir=cfg["paths"]["runs_dir"], config=cfg).start()
    logger = setup_logger(rm.app_log_path)
    logger.info(f"Run started: {rm.run_id}")

    st.sidebar.write("Run ID:", rm.run_id)
    st.sidebar.write("Collection:", cfg["index"]["collection_name"])
    st.sidebar.write("Mode:", cfg["retrieval"]["mode"])

    st.info("UI scaffold siap. Implementasi retrieval/generation akan diisi di tahap coding V1.0.")

    user_q = st.text_input("Tanya sesuatu:")
    if st.button("Kirim") and user_q.strip():
        st.write("Jawaban (placeholder):")
        st.write("Belum diimplementasikan.")
        st.write("Sources (placeholder):")
        st.write("- chunk_id: ...")


if __name__ == "__main__":
    main()