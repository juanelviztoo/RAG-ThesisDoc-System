from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def now_iso_jakarta() -> str:
    # Timezone user Asia/Jakarta; sederhana (tanpa pytz) -> offset manual
    # Nanti bisa diganti pakai zoneinfo saat requirements.
    return datetime.now().astimezone().isoformat()


def sha256_json(data: Dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass
class RunManager:
    runs_dir: Path
    config: Dict[str, Any]

    run_id: str = ""
    run_path: Path = Path(".")
    config_hash: str = ""

    retrieval_log_path: Path = Path(".")
    answers_log_path: Path = Path(".")
    app_log_path: Path = Path(".")

    def start(self) -> "RunManager":
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_path = Path(self.runs_dir) / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)

        self.config_hash = sha256_json(self.config)

        # files
        self.retrieval_log_path = self.run_path / "retrieval.jsonl"
        self.answers_log_path = self.run_path / "answers.jsonl"
        self.app_log_path = self.run_path / "app.log"

        # manifest: snapshot config + git hash
        manifest = {
            "run_id": self.run_id,
            "timestamp": now_iso_jakarta(),
            "git_commit": git_commit_hash(),
            "config_hash": self.config_hash,
            "config": self.config,  # snapshot sekali per-run
        }
        (self.run_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self

    def log_retrieval(
        self,
        record: Dict[str, Any],
        include_config_snapshot_per_row: bool = False,
        text_preview_len: int = 500,
    ) -> None:
        # Potong text panjang biar log tidak bengkak
        nodes = record.get("retrieved_nodes", [])
        for n in nodes:
            if "text" in n and isinstance(n["text"], str) and len(n["text"]) > text_preview_len:
                n["text"] = n["text"][:text_preview_len] + " ...<trimmed>"

        record["config_hash"] = self.config_hash
        if include_config_snapshot_per_row:
            record["config_snapshot"] = self.config

        write_jsonl(self.retrieval_log_path, record)

    def log_answer(
        self,
        record: Dict[str, Any],
        include_config_snapshot_per_row: bool = False,
    ) -> None:
        record["config_hash"] = self.config_hash
        if include_config_snapshot_per_row:
            record["config_snapshot"] = self.config

        write_jsonl(self.answers_log_path, record)
