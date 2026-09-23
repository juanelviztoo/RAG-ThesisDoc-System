"""Selective, provenance-safe OCR preprocessing for Corpus v1.

Safety contract
---------------
- NEVER writes to ``data_raw``.
- Does not import ``src.*`` and never runs RAG ingestion/indexing.
- OCR is driven only by an explicit JSON plan.
- Non-OCR parts are copied byte-for-byte into a complete derived thesis view.
- OCR outputs are staged first, validated, then published to ``data_derived``.
- Raw SHA-256 hashes are rechecked after processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF belum tersedia. Install requirements-ocr.txt terlebih dahulu."
    ) from exc


OCR_PIPELINE_VERSION = "0.1.1"

CANONICAL_PARTS: Tuple[Tuple[str, str], ...] = (
    ("BAB_1", "bab_1.pdf"),
    ("BAB_2", "bab_2.pdf"),
    ("BAB_3", "bab_3.pdf"),
    ("BAB_4", "bab_4.pdf"),
    ("BAB_5", "bab_5.pdf"),
    ("DAFTAR_PUSTAKA", "daftar_pustaka.pdf"),
)
PART_TO_FILENAME = dict(CANONICAL_PARTS)
DOC_ID_RE = re.compile(r"^UNS_INF_[A-Z0-9]+$", re.I)

DEFAULT_LANGUAGES = "ind+eng"
DEFAULT_PSM = 3
DEFAULT_OEM = 1
DEFAULT_JOBS = 1
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_FORCE_OVERSAMPLE_DPI = 300


@dataclass(frozen=True)
class PlanPart:
    part_type: str
    pages: str
    mode: str
    reason: str
    oversample: Optional[int] = None


@dataclass(frozen=True)
class OcrPlan:
    plan_version: str
    doc_id: str
    decision: str
    rationale: str
    copy_unlisted_parts: bool
    parts: Tuple[PlanPart, ...]


@dataclass
class FileRecord:
    pipeline_version: str
    plan_version: str
    doc_id: str
    part_type: str
    canonical_filename: str
    action: str
    pages_spec: str
    mode: str
    reason: str
    source_path: str
    derived_path: str
    sidecar_path: str
    source_sha256: str
    derived_sha256: str
    source_size_bytes: int
    derived_size_bytes: int
    source_page_count: int
    derived_page_count: int
    source_total_chars: int
    derived_total_chars: int
    text_gain_chars: int
    languages: str
    tesseract_psm: int
    tesseract_oem: int
    source_unchanged: str
    command: str
    status: str
    notes: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_version(distribution: str) -> str:
    """Return installed distribution version for provenance."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def git_dirty(project_root: Path) -> bool:
    """Return True when tracked/untracked non-ignored changes exist."""
    try:
        output = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(project_root),
                    "status",
                    "--porcelain",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return bool(output)
    except Exception:
        # Unknown Git state is treated conservatively.
        return True


def tessdata_model_manifest(
    tessdata_dir: Optional[Path],
    languages: str,
) -> Dict[str, Dict[str, Any]]:
    """Record exact OCR language-model files used by Tesseract."""
    if tessdata_dir is None:
        return {}

    models: Dict[str, Dict[str, Any]] = {}

    for language in [item.strip() for item in languages.split("+") if item.strip()]:
        model_path = tessdata_dir / f"{language}.traineddata"

        if not model_path.is_file():
            raise FileNotFoundError(f"Tessdata model tidak ditemukan: {model_path}")

        models[language] = {
            "filename": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }

    return models


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rel(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def pdf_stats(path: Path) -> Tuple[int, int]:
    """Return ``(page_count, total_extracted_chars)``."""
    doc = fitz.open(str(path))
    try:
        if bool(doc.needs_pass):
            raise RuntimeError(f"PDF membutuhkan password: {path}")
        total_chars = 0
        for page_index in range(int(doc.page_count)):
            page = doc.load_page(page_index)
            raw = page.get_text("text")
            if raw is None:
                text = ""
            elif isinstance(raw, str):
                text = " ".join(raw.split())
            else:
                raise TypeError(
                    "PyMuPDF get_text('text') menghasilkan tipe tak terduga: "
                    f"{type(raw).__name__}"
                )
            total_chars += len(text)
        return int(doc.page_count), total_chars
    finally:
        doc.close()


def run_capture(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=check,
    )


def first_output_line(command: Sequence[str], env: Dict[str, str]) -> str:
    result = run_capture(command, env=env)
    return next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        "",
    )


def list_tesseract_languages(env: Dict[str, str]) -> List[str]:
    result = run_capture(["tesseract", "--list-langs"], env=env)
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    ]


def normalize_pages(value: Any) -> str:
    if isinstance(value, str):
        pages = value.strip()
    elif isinstance(value, list) and all(isinstance(item, int) for item in value):
        pages = ",".join(str(item) for item in value)
    else:
        raise ValueError("Field pages harus 'all', string seperti '1-3,5', atau list integer.")

    if pages.lower() == "all":
        return "all"
    pages = re.sub(r"\s+", "", pages)
    if not pages or not re.fullmatch(r"[0-9,\-]+", pages):
        raise ValueError(f"Format pages tidak valid: {pages!r}")
    return pages


def parse_plan(path: Path) -> OcrPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("OCR plan harus berupa JSON object.")

    plan_version = str(raw.get("plan_version", "")).strip()
    doc_id = str(raw.get("doc_id", "")).strip().upper()
    decision = str(raw.get("decision", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    copy_unlisted = bool(raw.get("copy_unlisted_parts", True))
    raw_parts = raw.get("parts")

    if not plan_version:
        raise ValueError("plan_version wajib.")
    if not DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f"doc_id tidak valid: {doc_id}")
    if not isinstance(raw_parts, dict) or not raw_parts:
        raise ValueError("parts wajib berupa object yang tidak kosong.")

    parts: List[PlanPart] = []
    for key, value in raw_parts.items():
        part_type = str(key).strip().upper()
        if part_type not in PART_TO_FILENAME:
            raise ValueError(f"Part type tidak dikenal: {part_type}")
        if not isinstance(value, dict):
            raise ValueError(f"Plan {part_type} harus berupa object.")

        pages = normalize_pages(value.get("pages", "all"))
        mode = str(value.get("mode", "redo")).strip().lower()
        if mode not in {"redo", "force"}:
            raise ValueError(f"Mode {part_type} harus redo/force, bukan {mode!r}.")

        reason = str(value.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"reason wajib untuk {part_type}.")

        oversample_raw = value.get("oversample")
        oversample = int(oversample_raw) if oversample_raw is not None else None
        if oversample is not None and oversample <= 0:
            raise ValueError("oversample harus > 0.")
        if mode == "redo" and oversample is not None:
            raise ValueError(
                f"{part_type}: mode redo tidak memakai oversample. "
                "Gunakan force hanya sebagai fallback terkontrol."
            )
        if mode == "force" and oversample is None:
            oversample = DEFAULT_FORCE_OVERSAMPLE_DPI

        parts.append(
            PlanPart(
                part_type=part_type,
                pages=pages,
                mode=mode,
                reason=reason,
                oversample=oversample,
            )
        )

    return OcrPlan(
        plan_version=plan_version,
        doc_id=doc_id,
        decision=decision,
        rationale=rationale,
        copy_unlisted_parts=copy_unlisted,
        parts=tuple(parts),
    )


def validate_pages(pages: str, page_count: int) -> None:
    if pages == "all":
        return

    selected: set[int] = set()
    for token in pages.split(","):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Page range terbalik: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = sorted(p for p in selected if p < 1 or p > page_count)
    if invalid:
        raise ValueError(f"Page selection di luar rentang 1-{page_count}: {invalid}")


def build_ocr_command(
    source: Path,
    output: Path,
    sidecar: Path,
    plan_part: PlanPart,
    *,
    languages: str,
    jobs: int,
    timeout_seconds: float,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "ocrmypdf",
        "--mode",
        plan_part.mode,
        "--language",
        languages,
        "--tesseract-pagesegmode",
        str(DEFAULT_PSM),
        "--tesseract-oem",
        str(DEFAULT_OEM),
        "--output-type",
        "pdf",
        "--optimize",
        "0",
        "--rasterizer",
        "pypdfium",
        "--pdf-renderer",
        "fpdf2",
        "--jobs",
        str(jobs),
        "--tesseract-timeout",
        str(timeout_seconds),
        "--sidecar",
        str(sidecar),
    ]
    if plan_part.pages != "all":
        command.extend(["--pages", plan_part.pages])
    if plan_part.mode == "force" and plan_part.oversample is not None:
        command.extend(["--oversample", str(plan_part.oversample)])
    command.extend([str(source), str(output)])
    return command


def shell_join(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_csv(path: Path, records: List[FileRecord]) -> None:
    rows = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_commit(project_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def publish_staging(
    staging_doc: Path,
    derived_doc: Path,
    run_dir: Path,
    *,
    overwrite: bool,
) -> None:
    derived_doc.parent.mkdir(parents=True, exist_ok=True)
    backup: Optional[Path] = None

    if derived_doc.exists():
        if not overwrite:
            raise FileExistsError(
                f"Derived thesis sudah ada: {derived_doc}. " "Refuse overwrite secara default."
            )
        backup = run_dir / "_previous_derived" / derived_doc.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(derived_doc), str(backup))

    try:
        shutil.move(str(staging_doc), str(derived_doc))
    except Exception:
        if backup is not None and backup.exists() and not derived_doc.exists():
            shutil.move(str(backup), str(derived_doc))
        raise


def process(args: argparse.Namespace) -> Tuple[List[FileRecord], Path]:
    project_root = Path(args.project_root).resolve()
    source_root = (project_root / args.source_root).resolve()
    derived_root = (project_root / args.derived_root).resolve()
    runs_root = (project_root / args.runs_root).resolve()
    plan_path = (project_root / args.plan).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root tidak ditemukan: {source_root}")
    if source_root == derived_root:
        raise RuntimeError("source_root dan derived_root tidak boleh sama.")
    if is_within(derived_root, source_root) or is_within(source_root, derived_root):
        raise RuntimeError("source_root dan derived_root tidak boleh saling nested.")
    if not plan_path.is_file():
        raise FileNotFoundError(f"OCR plan tidak ditemukan: {plan_path}")

    plan_text = plan_path.read_text(encoding="utf-8")
    plan = parse_plan(plan_path)
    source_doc = source_root / plan.doc_id
    derived_doc = derived_root / plan.doc_id

    if not source_doc.is_dir():
        raise FileNotFoundError(f"Folder thesis tidak ditemukan: {source_doc}")

    source_paths = {part_type: source_doc / filename for part_type, filename in CANONICAL_PARTS}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Core PDF tidak ditemukan: {path}")

    timestamp = datetime.now().astimezone()
    run_name = args.run_name or (f"{plan.doc_id.lower()}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir = runs_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory sudah berisi data: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    staging_doc = run_dir / "_staging" / plan.doc_id
    staging_doc.mkdir(parents=True, exist_ok=False)
    sidecar_dir = run_dir / "sidecar" / plan.doc_id
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    tessdata_dir: Optional[Path] = None

    if args.tessdata_dir:
        resolved_tessdata_dir = (project_root / args.tessdata_dir).resolve()

        if not resolved_tessdata_dir.is_dir():
            raise FileNotFoundError(f"tessdata tidak ditemukan: {resolved_tessdata_dir}")

        tessdata_dir = resolved_tessdata_dir
        env["TESSDATA_PREFIX"] = str(resolved_tessdata_dir)

    ocrmypdf_version = first_output_line([sys.executable, "-m", "ocrmypdf", "--version"], env)
    tesseract_version = first_output_line(["tesseract", "--version"], env)
    available_languages = list_tesseract_languages(env)
    required_languages = args.languages.split("+")
    missing_languages = sorted(set(required_languages) - set(available_languages))
    if missing_languages:
        raise RuntimeError(
            "Tesseract language model tidak tersedia: " + ", ".join(missing_languages)
        )

    tessdata_models = tessdata_model_manifest(
        tessdata_dir,
        args.languages,
    )

    source_hash_before = {part_type: sha256_file(path) for part_type, path in source_paths.items()}
    source_stats = {part_type: pdf_stats(path) for part_type, path in source_paths.items()}

    plan_by_part = {item.part_type: item for item in plan.parts}
    records: List[FileRecord] = []
    execution_log: List[str] = []

    for part_type, filename in CANONICAL_PARTS:
        source = source_paths[part_type]
        staged = staging_doc / filename
        final_derived = derived_doc / filename
        source_pages, source_chars = source_stats[part_type]
        plan_part = plan_by_part.get(part_type)

        if plan_part is None:
            if not plan.copy_unlisted_parts:
                raise ValueError(f"{part_type} tidak ada di plan dan copy_unlisted_parts=false.")
            shutil.copy2(source, staged)
            derived_pages, derived_chars = pdf_stats(staged)
            derived_hash = sha256_file(staged)

            if derived_hash != source_hash_before[part_type] or derived_pages != source_pages:
                raise RuntimeError(f"Byte-for-byte copy validation gagal: {filename}")

            records.append(
                FileRecord(
                    OCR_PIPELINE_VERSION,
                    plan.plan_version,
                    plan.doc_id,
                    part_type,
                    filename,
                    "COPY_ORIGINAL",
                    "",
                    "",
                    "Part bukan OCR candidate.",
                    rel(source, project_root),
                    rel(final_derived, project_root),
                    "",
                    source_hash_before[part_type],
                    derived_hash,
                    source.stat().st_size,
                    staged.stat().st_size,
                    source_pages,
                    derived_pages,
                    source_chars,
                    derived_chars,
                    0,
                    args.languages,
                    DEFAULT_PSM,
                    DEFAULT_OEM,
                    "PENDING",
                    "",
                    "PASS",
                    "Byte-for-byte copy dari authoritative raw source.",
                )
            )
            continue

        validate_pages(plan_part.pages, source_pages)
        sidecar = sidecar_dir / f"{Path(filename).stem}.txt"
        command = build_ocr_command(
            source,
            staged,
            sidecar,
            plan_part,
            languages=args.languages,
            jobs=args.jobs,
            timeout_seconds=args.tesseract_timeout,
        )
        command_text = shell_join(command)
        execution_log.append(f"$ {command_text}")

        if args.dry_run:
            records.append(
                FileRecord(
                    OCR_PIPELINE_VERSION,
                    plan.plan_version,
                    plan.doc_id,
                    part_type,
                    filename,
                    "OCR_DRY_RUN",
                    plan_part.pages,
                    plan_part.mode,
                    plan_part.reason,
                    rel(source, project_root),
                    rel(final_derived, project_root),
                    rel(sidecar, project_root),
                    source_hash_before[part_type],
                    "",
                    source.stat().st_size,
                    0,
                    source_pages,
                    0,
                    source_chars,
                    0,
                    0,
                    args.languages,
                    DEFAULT_PSM,
                    DEFAULT_OEM,
                    "PENDING",
                    command_text,
                    "DRY_RUN",
                    "",
                )
            )
            continue

        result = run_capture(command, env=env, check=False)
        execution_log.append(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"OCRmyPDF gagal untuk {filename} (exit {result.returncode}). "
                "Lihat execution.log."
            )
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError(f"OCR output kosong/tidak ada: {staged}")

        derived_pages, derived_chars = pdf_stats(staged)
        if derived_pages != source_pages:
            raise RuntimeError(
                f"Page count berubah {filename}: " f"{source_pages} -> {derived_pages}"
            )

        text_gain = derived_chars - source_chars
        status = "PASS" if text_gain > 0 else "REVIEW"
        notes = (
            "Selective OCR selesai. Inclusion final tetap ditentukan oleh "
            "corpus_qc.py pada complete derived thesis view."
        )
        if text_gain <= 0:
            notes += " Tidak ada positive extracted-text gain."

        records.append(
            FileRecord(
                OCR_PIPELINE_VERSION,
                plan.plan_version,
                plan.doc_id,
                part_type,
                filename,
                "OCR",
                plan_part.pages,
                plan_part.mode,
                plan_part.reason,
                rel(source, project_root),
                rel(final_derived, project_root),
                rel(sidecar, project_root),
                source_hash_before[part_type],
                sha256_file(staged),
                source.stat().st_size,
                staged.stat().st_size,
                source_pages,
                derived_pages,
                source_chars,
                derived_chars,
                text_gain,
                args.languages,
                DEFAULT_PSM,
                DEFAULT_OEM,
                "PENDING",
                command_text,
                status,
                notes,
            )
        )

    source_hash_after = {part_type: sha256_file(path) for part_type, path in source_paths.items()}
    mutated = sorted(
        part_type
        for part_type in source_hash_before
        if source_hash_before[part_type] != source_hash_after[part_type]
    )
    if mutated:
        raise RuntimeError("CRITICAL: raw source berubah setelah OCR: " + ", ".join(mutated))

    for record in records:
        record.source_unchanged = "YES"

    write_csv(run_dir / "ocr_manifest.csv", records)
    (run_dir / "execution.log").write_text("\n\n".join(execution_log), encoding="utf-8")

    summary = {
        "ocr_pipeline_version": OCR_PIPELINE_VERSION,
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "script_path": rel(Path(__file__).resolve(), project_root),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "plan_path": rel(plan_path, project_root),
        "plan_sha256": sha256_text(plan_text),
        "plan_version": plan.plan_version,
        "doc_id": plan.doc_id,
        "decision": plan.decision,
        "rationale": plan.rationale,
        "toolchain": {
            "python": sys.version.split()[0],
            "ocrmypdf": ocrmypdf_version,
            "pymupdf": package_version("PyMuPDF"),
            "pypdfium2": package_version("pypdfium2"),
            "fpdf2": package_version("fpdf2"),
            "tesseract": tesseract_version,
            "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
            "tessdata_models": tessdata_models,
            "languages": args.languages,
            "psm": DEFAULT_PSM,
            "oem": DEFAULT_OEM,
            "rasterizer": "pypdfium",
            "pdf_renderer": "fpdf2",
            "output_type": "pdf",
            "optimize": 0,
            "jobs": args.jobs,
            "tesseract_timeout_seconds": args.tesseract_timeout,
        },
        "counts": {
            "parts_total": len(records),
            "parts_copy_original": sum(r.action == "COPY_ORIGINAL" for r in records),
            "parts_ocr": sum(r.action == "OCR" for r in records),
            "parts_dry_run": sum(r.action == "OCR_DRY_RUN" for r in records),
            "pass": sum(r.status == "PASS" for r in records),
            "review": sum(r.status == "REVIEW" for r in records),
        },
        "raw_source_immutable": not mutated,
        "next_gate": (
            "Run corpus_qc.py against the complete derived thesis view. "
            "OCR utility PASS bukan corpus inclusion PASS."
        ),
    }
    (run_dir / "ocr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print("\nSelective OCR Dry Run\n---------------------")
        print(f"Doc ID            : {plan.doc_id}")
        print(f"OCR candidates    : {summary['counts']['parts_dry_run']}")
        print(f"Raw immutable     : {summary['raw_source_immutable']}")
        print(f"Run output        : {run_dir}")
        return records, run_dir

    publish_staging(
        staging_doc,
        derived_doc,
        run_dir,
        overwrite=args.overwrite_derived,
    )

    print("\nSelective OCR Summary\n---------------------")
    print(f"Pipeline version  : {OCR_PIPELINE_VERSION}")
    print(f"Doc ID            : {plan.doc_id}")
    print(f"OCRmyPDF          : {ocrmypdf_version}")
    print(f"Tesseract         : {tesseract_version}")
    print(
        "Parts copy/OCR   : "
        f"{summary['counts']['parts_copy_original']}/"
        f"{summary['counts']['parts_ocr']}"
    )
    print("PASS/REVIEW      : " f"{summary['counts']['pass']}/" f"{summary['counts']['review']}")
    print(f"Raw immutable     : {summary['raw_source_immutable']}")
    print(f"Derived thesis    : {derived_doc}")
    print(f"Run output        : {run_dir}")
    return records, run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Selective OCR preprocessing for Corpus v1. "
            "Read-only terhadap data_raw; output hanya ke data_derived."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-root", default="data_raw/corpus_v1")
    parser.add_argument("--derived-root", default="data_derived/corpus_v1_ocr")
    parser.add_argument("--runs-root", default="runs/corpus_ocr")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument("--tessdata-dir", default=None)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument(
        "--tesseract-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--overwrite-derived", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        print("[FATAL] --jobs harus >= 1", file=sys.stderr)
        return 1
    if args.tesseract_timeout <= 0:
        print("[FATAL] --tesseract-timeout harus > 0", file=sys.stderr)
        return 1

    try:
        records, _ = process(args)
    except KeyboardInterrupt:
        print("[INTERRUPTED] OCR dihentikan pengguna.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[FATAL] Selective OCR gagal: {exc}", file=sys.stderr)
        return 1

    if any(record.status == "REVIEW" for record in records):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
