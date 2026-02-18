from __future__ import annotations

import argparse
from pathlib import Path

# TODO: nanti import ragas setelah requirements di-setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    run_path = Path(args.runs_dir) / args.run_id
    dataset_path = run_path / "eval_dataset.json"

    # TODO: implement RAGAS metrics read from dataset_path
    raise NotImplementedError("RAGAS evaluation belum diimplementasikan (scaffold).")


if __name__ == "__main__":
    main()