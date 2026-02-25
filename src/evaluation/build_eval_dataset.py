from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    run_path = Path(args.runs_dir) / args.run_id
    answers_path = run_path / "answers.jsonl"

    out = []
    for row in read_jsonl(answers_path):
        # format minimal yang biasanya dibutuhkan evaluator (nanti disesuaikan dgn RAGAS)
        out.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row.get("contexts", []),
            }
        )

    (run_path / "eval_dataset.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {run_path / 'eval_dataset.json'}")


if __name__ == "__main__":
    main()
