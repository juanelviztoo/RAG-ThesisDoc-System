from __future__ import annotations

import argparse
from typing import Any, Dict

from src.core.config import ConfigPaths, load_config


def ingest(cfg: Dict[str, Any]) -> None:
    """
    Offline ingestion:
    - baca PDF dari data_raw/
    - extract text
    - chunk
    - embed
    - simpan ke Chroma persist_dir + collection_name
    """
    paths = ConfigPaths.from_cfg(cfg)
    collection_name = cfg["index"]["collection_name"]
    print(
        f"[INGEST] data_raw_dir={paths.data_raw_dir} persist_dir={paths.persist_dir} collection={collection_name}"
    )

    # TODO (V1): implement PDF loader + naive chunking + embedding + chroma upsert
    # Nanti diisikan saat mulai coding versi 1.0
    raise NotImplementedError("Ingest belum diimplementasikan (scaffold).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path config yaml (e.g., configs/v1.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ingest(cfg)


if __name__ == "__main__":
    main()
