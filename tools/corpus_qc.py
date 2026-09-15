"""Read-only Automated PDF-QC for Corpus v1.

Utility ini sengaja dipisahkan dari runtime RAG: tidak mengimpor ``src.*``,
tidak menyentuh Chroma/BM25/embedding/LLM, tidak menjalankan OCR, dan tidak
memodifikasi PDF sumber maupun Google Sheet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF belum tersedia. Install requirements.txt terlebih dahulu.") from exc

SPEC_VERSION = "1.0.0"
MIN_FILE_BYTES = 1024
MIN_TOTAL_CHARS = 500
SUBSTANTIVE_CHARS_PER_PAGE = 100
TEXT_COVERAGE_YES = 0.60
TEXT_COVERAGE_PARTIAL = 0.25
MIN_MEDIAN_CHARS_YES = 100
STRUCTURE_SCAN_PAGES = 3
DOC_ID_RE = re.compile(r"^UNS_INF_[A-Z0-9]+$", re.I)


@dataclass(frozen=True)
class PartSpec:
    part_type: str
    filename: str
    stream: str
    bab_label: str
    patterns: Tuple[str, ...]


PARTS: Tuple[PartSpec, ...] = (
    PartSpec("BAB_1", "bab_1.pdf", "narasi", "BAB_I", (r"\bBAB\s*(?:I|1)\b",)),
    PartSpec("BAB_2", "bab_2.pdf", "narasi", "BAB_II", (r"\bBAB\s*(?:II|2)\b",)),
    PartSpec("BAB_3", "bab_3.pdf", "narasi", "BAB_III", (r"\bBAB\s*(?:III|3)\b",)),
    PartSpec("BAB_4", "bab_4.pdf", "narasi", "BAB_IV", (r"\bBAB\s*(?:IV|4)\b",)),
    PartSpec("BAB_5", "bab_5.pdf", "narasi", "BAB_V", (r"\bBAB\s*(?:V|5)\b",)),
    PartSpec(
        "DAFTAR_PUSTAKA",
        "daftar_pustaka.pdf",
        "sitasi",
        "DAFTAR_PUSTAKA",
        (r"\bDAFTAR\s+(?:PUSTAKA|REFERENSI)\b", r"\bREFERENCES?\b", r"\bBIBLIOGRAPHY\b"),
    ),
)
PART_BY_TYPE = {x.part_type: x for x in PARTS}
ANY_MARKERS = (
    ("BAB_I", r"\bBAB\s*(?:I|1)\b"),
    ("BAB_II", r"\bBAB\s*(?:II|2)\b"),
    ("BAB_III", r"\bBAB\s*(?:III|3)\b"),
    ("BAB_IV", r"\bBAB\s*(?:IV|4)\b"),
    ("BAB_V", r"\bBAB\s*(?:V|5)\b"),
    ("DAFTAR_PUSTAKA", r"\bDAFTAR\s+(?:PUSTAKA|REFERENSI)\b"),
    ("REFERENCES", r"\bREFERENCES?\b"),
    ("BIBLIOGRAPHY", r"\bBIBLIOGRAPHY\b"),
)


@dataclass
class ExpectedFile:
    file_id: str
    doc_id: str
    nim: str
    part_type: str
    filename: str
    relpath: str
    path: Path


@dataclass
class QcResult:
    qc_spec_version: str
    file_id: str
    doc_id: str
    nim: str
    part_type: str
    canonical_filename: str
    planned_relative_path: str
    file_exists: str
    file_size_bytes: int
    page_count: int
    openable: str
    encrypted: str
    total_chars: int
    total_words: int
    median_chars_per_page: float
    mean_chars_per_page: float
    min_chars_per_page: int
    max_chars_per_page: int
    substantive_text_pages: int
    substantive_text_coverage: float
    low_text_pages: int
    image_bearing_pages: int
    image_bearing_ratio: float
    images_total: int
    text_extractable: str
    ocr_dependency_flag: str
    expected_marker_found_first_pages: str
    expected_marker_found_anywhere: str
    first_section_marker_detected: str
    structure_ok: str
    sha256: str
    duplicate_of: str
    file_qc_status: str
    qc_inclusion_recommendation: str
    qc_flags: str
    notes: str


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def row_get(row: Dict[str, str], name: str) -> str:
    target = norm_key(name)
    for key, value in row.items():
        if norm_key(key) == target:
            return (value or "").strip()
    return ""


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def expected_contract(
    corpus_root: Path, project_root: Path, selected: set[str]
) -> Tuple[List[ExpectedFile], List[str]]:
    dirs = [
        path
        for path in sorted(corpus_root.iterdir())
        if path.is_dir()
        and DOC_ID_RE.match(path.name)
        and (not selected or path.name in selected)
    ]
    found = {path.name for path in dirs}
    missing_dirs = sorted(selected - found)
    if missing_dirs:
        raise RuntimeError(f"Doc ID folder tidak ditemukan: {', '.join(missing_dirs)}")
    if not dirs:
        raise RuntimeError(f"Tidak ada folder UNS_INF_* pada {corpus_root}")

    output: List[ExpectedFile] = []
    unexpected: List[str] = []
    expected_names = {part.filename for part in PARTS}
    for directory in dirs:
        nim = directory.name.removeprefix("UNS_INF_")
        for spec in PARTS:
            path = directory / spec.filename
            output.append(
                ExpectedFile(
                    f"{directory.name}__{spec.part_type}",
                    directory.name,
                    nim,
                    spec.part_type,
                    spec.filename,
                    rel(path, project_root),
                    path,
                )
            )
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() == ".pdf" and path.name not in expected_names:
                unexpected.append(rel(path, project_root))

    # Penting: file PDF lama yang berada langsung di data_raw/ tidak dipindai.
    # Hanya corpus_root yang menjadi scope QC.
    for path in sorted(corpus_root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdf":
            unexpected.append(rel(path, project_root))
    return output, unexpected


def expected_manifest(
    csv_path: Path, project_root: Path, corpus_root: Path, selected: set[str]
) -> Tuple[List[ExpectedFile], List[str]]:
    output: List[ExpectedFile] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    matched_docs: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("Manifest CSV tidak memiliki header")
        for row_number, row in enumerate(reader, start=2):
            doc_id = row_get(row, "Doc ID")
            if not doc_id or (selected and doc_id not in selected):
                continue
            if row_get(row, "Requirement Role").upper() not in ("", "CORE_INDEXED"):
                continue
            if row_get(row, "Indexing Policy").upper() not in ("", "INDEXED"):
                continue

            part_type = row_get(row, "Part Type").upper()
            if part_type not in PART_BY_TYPE:
                raise RuntimeError(f"Row {row_number}: Part Type tidak valid: {part_type}")
            spec = PART_BY_TYPE[part_type]
            filename = row_get(row, "Canonical Filename") or spec.filename
            if filename != spec.filename:
                raise RuntimeError(f"Row {row_number}: filename {filename} != {spec.filename}")

            planned = row_get(row, "Planned Relative Path")
            path = (
                (project_root / planned).resolve()
                if planned
                else (corpus_root / doc_id / filename).resolve()
            )
            if not inside(path, corpus_root):
                raise RuntimeError(f"Row {row_number}: path keluar corpus_root: {path}")

            file_id = row_get(row, "File ID") or f"{doc_id}__{part_type}"
            if file_id in seen_ids or path in seen_paths:
                raise RuntimeError(f"Row {row_number}: duplicate File ID/path")
            seen_ids.add(file_id)
            seen_paths.add(path)
            matched_docs.add(doc_id)
            nim = row_get(row, "NIM") or doc_id.removeprefix("UNS_INF_")
            output.append(
                ExpectedFile(
                    file_id,
                    doc_id,
                    nim,
                    part_type,
                    filename,
                    planned or rel(path, project_root),
                    path,
                )
            )

    missing_docs = sorted(selected - matched_docs)
    if missing_docs:
        raise RuntimeError(f"Doc ID tidak ditemukan pada manifest: {', '.join(missing_docs)}")
    if not output:
        raise RuntimeError("Tidak ada row CORE_INDEXED/INDEXED yang terpilih")

    expected_paths = {item.path.resolve() for item in output}
    unexpected: List[str] = []
    for doc_id in sorted({item.doc_id for item in output}):
        directory = corpus_root / doc_id
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() == ".pdf" and path.resolve() not in expected_paths:
                unexpected.append(rel(path, project_root))
    return output, unexpected


def marker_found(text: str, spec: PartSpec) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in spec.patterns)


def first_marker(text: str) -> str:
    hits: List[Tuple[int, str]] = []
    for label, pattern in ANY_MARKERS:
        match = re.search(pattern, text, re.I)
        if match:
            hits.append((match.start(), label))
    return min(hits)[1] if hits else ""


def text_class(
    total_chars: int, page_count: int, substantive_pages: int, median_chars: float
) -> Tuple[str, str, List[str]]:
    coverage = substantive_pages / page_count if page_count else 0.0
    flags: List[str] = []
    if total_chars < MIN_TOTAL_CHARS or coverage < TEXT_COVERAGE_PARTIAL:
        return "NO", "REQUIRED", ["CORE_TEXT_NOT_SUFFICIENTLY_EXTRACTABLE"]
    if coverage < TEXT_COVERAGE_YES or median_chars < MIN_MEDIAN_CHARS_YES:
        if coverage < TEXT_COVERAGE_YES:
            flags.append("PARTIAL_TEXT_COVERAGE")
        if median_chars < MIN_MEDIAN_CHARS_YES:
            flags.append("LOW_MEDIAN_TEXT_PER_PAGE")
        return "PARTIAL", "POSSIBLE", flags
    return "YES", "NO", flags


def structure_class(
    spec: PartSpec, first_text: str, full_text: str
) -> Tuple[str, str, str, str, List[str]]:
    first_ok = marker_found(first_text, spec)
    anywhere_ok = first_ok or marker_found(full_text, spec)
    detected = first_marker(first_text)
    if first_ok:
        return "YES", "YES", "YES", detected, []
    if anywhere_ok:
        return "REVIEW", "NO", "YES", detected, ["EXPECTED_SECTION_MARKER_FOUND_LATE"]
    if detected and detected != spec.bab_label:
        reference_equivalent = (
            spec.bab_label == "DAFTAR_PUSTAKA" and detected in {"REFERENCES", "BIBLIOGRAPHY"}
        )
        if not reference_equivalent:
            return "NO", "NO", "NO", detected, [f"WRONG_SECTION_MARKER:{detected}"]
    return "REVIEW", "NO", "NO", detected, ["EXPECTED_SECTION_MARKER_NOT_FOUND"]


def blank_result(expected: ExpectedFile, flag: str, status: str = "FAIL") -> QcResult:
    return QcResult(
        SPEC_VERSION,
        expected.file_id,
        expected.doc_id,
        expected.nim,
        expected.part_type,
        expected.filename,
        expected.relpath,
        "NO",
        0,
        0,
        "NO",
        "UNKNOWN",
        0,
        0,
        0.0,
        0.0,
        0,
        0,
        0,
        0.0,
        0,
        0,
        0.0,
        0,
        "NO",
        "REQUIRED",
        "NO",
        "NO",
        "",
        "PENDING",
        "",
        "",
        status,
        "BLOCKED",
        flag,
        "",
    )


def audit_file(expected: ExpectedFile) -> QcResult:
    path = expected.path
    if not path.is_file():
        return blank_result(expected, "MISSING_FILE")

    size = int(path.stat().st_size)
    try:
        digest = sha256_file(path)
    except Exception as exc:
        result = blank_result(expected, "HASH_READ_ERROR")
        result.file_exists = "YES"
        result.file_size_bytes = size
        result.notes = str(exc)
        return result

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        result = blank_result(expected, "PDF_OPEN_ERROR")
        result.file_exists = "YES"
        result.file_size_bytes = size
        result.sha256 = digest
        result.notes = str(exc)
        return result

    try:
        if bool(getattr(document, "needs_pass", False)):
            result = blank_result(expected, "PDF_PASSWORD_REQUIRED")
            result.file_exists = "YES"
            result.file_size_bytes = size
            result.sha256 = digest
            result.encrypted = "YES"
            return result

        page_count = int(document.page_count)
        if page_count <= 0:
            result = blank_result(expected, "ZERO_PAGE_PDF")
            result.file_exists = "YES"
            result.file_size_bytes = size
            result.sha256 = digest
            result.encrypted = "NO"
            return result

        texts: List[str] = []
        chars_per_page: List[int] = []
        words_per_page: List[int] = []
        images_per_page: List[int] = []
        for page in document:
            text = norm_text(page.get_text("text") or "")
            texts.append(text)
            chars_per_page.append(len(text))
            words_per_page.append(len(text.split()))
            try:
                images_per_page.append(len(page.get_images(full=True)))
            except Exception:
                images_per_page.append(0)
    finally:
        document.close()

    total_chars = sum(chars_per_page)
    total_words = sum(words_per_page)
    median_chars = float(statistics.median(chars_per_page)) if chars_per_page else 0.0
    mean_chars = float(statistics.mean(chars_per_page)) if chars_per_page else 0.0
    substantive_pages = sum(value >= SUBSTANTIVE_CHARS_PER_PAGE for value in chars_per_page)
    coverage = substantive_pages / page_count

    text_extractable, ocr_flag, flags = text_class(
        total_chars, page_count, substantive_pages, median_chars
    )
    spec = PART_BY_TYPE[expected.part_type]
    first_text = "\n".join(texts[:STRUCTURE_SCAN_PAGES])
    full_text = "\n".join(texts)
    structure_ok, found_first, found_anywhere, detected_marker, structure_flags = structure_class(
        spec, first_text, full_text
    )
    flags += structure_flags
    if size < MIN_FILE_BYTES:
        flags.append("FILE_TOO_SMALL")

    status = (
        "PASS"
        if not flags and text_extractable == "YES" and structure_ok == "YES"
        else "REVIEW"
    )
    recommendation = "ELIGIBLE" if status == "PASS" else "MANUAL_REVIEW"
    image_pages = sum(value > 0 for value in images_per_page)
    note = ""
    if image_pages:
        note = (
            f"Komponen visual terdeteksi pada {image_pages}/{page_count} halaman; "
            "tidak otomatis memengaruhi PASS selama teks inti tetap extractable."
        )

    return QcResult(
        SPEC_VERSION,
        expected.file_id,
        expected.doc_id,
        expected.nim,
        expected.part_type,
        expected.filename,
        expected.relpath,
        "YES",
        size,
        page_count,
        "YES",
        "NO",
        total_chars,
        total_words,
        round(median_chars, 2),
        round(mean_chars, 2),
        min(chars_per_page),
        max(chars_per_page),
        substantive_pages,
        round(coverage, 4),
        page_count - substantive_pages,
        image_pages,
        round(image_pages / page_count, 4),
        sum(images_per_page),
        text_extractable,
        ocr_flag,
        found_first,
        found_anywhere,
        detected_marker,
        structure_ok,
        digest,
        "",
        status,
        recommendation,
        ";".join(dict.fromkeys(flags)),
        note,
    )


def apply_duplicates(results: List[QcResult]) -> Dict[str, List[str]]:
    groups: Dict[str, List[QcResult]] = {}
    for result in results:
        if result.sha256:
            groups.setdefault(result.sha256, []).append(result)

    duplicates: Dict[str, List[str]] = {}
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        ids = [result.file_id for result in group]
        duplicates[digest] = ids
        anchor = ids[0]
        for result in group[1:]:
            result.duplicate_of = anchor
            result.qc_flags = ";".join(
                value for value in [result.qc_flags, "EXACT_DUPLICATE_SHA256"] if value
            )
            if result.file_qc_status == "PASS":
                result.file_qc_status = "REVIEW"
                result.qc_inclusion_recommendation = "MANUAL_REVIEW"
    return duplicates


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(
    results: List[QcResult],
    expected: List[ExpectedFile],
    unexpected: List[str],
    duplicates: Dict[str, List[str]],
    mode: str,
    project_root: Path,
    corpus_root: Path,
    manifest: Optional[Path],
) -> Dict[str, Any]:
    status = {
        value: sum(result.file_qc_status == value for result in results)
        for value in ("PASS", "REVIEW", "FAIL")
    }
    text_status = {
        value: sum(result.text_extractable == value for result in results)
        for value in ("YES", "PARTIAL", "NO")
    }
    ocr_status = {
        value: sum(result.ocr_dependency_flag == value for result in results)
        for value in ("NO", "POSSIBLE", "REQUIRED")
    }
    structure_status = {
        value: sum(result.structure_ok == value for result in results)
        for value in ("YES", "REVIEW", "NO", "PENDING")
    }

    per_doc: Dict[str, Dict[str, Any]] = {}
    for result in results:
        doc = per_doc.setdefault(
            result.doc_id, {"expected": 0, "pass": 0, "review": 0, "fail": 0}
        )
        doc["expected"] += 1
        doc[result.file_qc_status.lower()] += 1
    for doc in per_doc.values():
        doc["all_core_pass"] = doc["expected"] == len(PARTS) and doc["pass"] == len(PARTS)

    return {
        "qc_spec_version": SPEC_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "git_commit": git_hash(project_root),
        "project_root": str(project_root),
        "corpus_root": str(corpus_root),
        "manifest": str(manifest) if manifest else None,
        "thresholds": {
            "min_file_bytes": MIN_FILE_BYTES,
            "min_total_chars": MIN_TOTAL_CHARS,
            "substantive_chars_per_page": SUBSTANTIVE_CHARS_PER_PAGE,
            "text_coverage_yes": TEXT_COVERAGE_YES,
            "text_coverage_partial": TEXT_COVERAGE_PARTIAL,
            "min_median_chars_yes": MIN_MEDIAN_CHARS_YES,
            "structure_scan_pages": STRUCTURE_SCAN_PAGES,
        },
        "counts": {
            "expected_files": len(expected),
            "found_files": sum(result.file_exists == "YES" for result in results),
            "missing_files": sum(result.file_exists == "NO" for result in results),
            "total_pages": sum(result.page_count for result in results),
            "status": status,
            "text_extractable": text_status,
            "ocr_dependency_flag": ocr_status,
            "structure_ok": structure_status,
            "unexpected_pdf_count": len(unexpected),
            "duplicate_hash_groups": len(duplicates),
            "docs_all_core_pass": sum(bool(doc["all_core_pass"]) for doc in per_doc.values()),
            "docs_total": len(per_doc),
        },
        "unexpected_pdfs": unexpected,
        "duplicate_groups": duplicates,
        "per_doc": per_doc,
        "policy": [
            "Visual elements do not auto-fail a PDF when substantive text is extractable.",
            "OCR is diagnostic only and is never executed by this utility.",
            "Content/structure anomalies route to REVIEW; hard integrity blockers route to FAIL.",
            "PDFs outside corpus_root are ignored by design.",
        ],
    }


def run(args: argparse.Namespace) -> Tuple[List[QcResult], Dict[str, Any], Path]:
    project_root = Path(args.project_root).resolve()
    corpus_root = Path(args.corpus_root)
    corpus_root = (
        (project_root / corpus_root).resolve()
        if not corpus_root.is_absolute()
        else corpus_root.resolve()
    )
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root tidak ditemukan: {corpus_root}")
    if not inside(corpus_root, project_root) and not args.allow_external_corpus_root:
        raise RuntimeError(
            "corpus_root di luar project_root; gunakan --allow-external-corpus-root jika disengaja"
        )

    selected = {value.strip() for value in (args.doc_id or []) if value.strip()}
    manifest: Optional[Path] = None
    if args.manifest:
        manifest = Path(args.manifest)
        manifest = (
            (project_root / manifest).resolve()
            if not manifest.is_absolute()
            else manifest.resolve()
        )
        expected, unexpected = expected_manifest(manifest, project_root, corpus_root, selected)
        mode = "manifest"
    else:
        expected, unexpected = expected_contract(corpus_root, project_root, selected)
        mode = "contract"

    results = [audit_file(item) for item in expected]
    duplicates = apply_duplicates(results)
    summary = make_summary(
        results, expected, unexpected, duplicates, mode, project_root, corpus_root, manifest
    )

    output_root = Path(args.output_dir)
    output_root = (
        (project_root / output_root).resolve()
        if not output_root.is_absolute()
        else output_root.resolve()
    )
    run_dir = output_root / (args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(result) for result in results]
    fields = list(rows[0].keys())
    write_csv(run_dir / "qc_report.csv", rows, fields)
    write_csv(
        run_dir / "qc_review_queue.csv",
        [row for row in rows if row["file_qc_status"] != "PASS"],
        fields,
    )
    (run_dir / "qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = summary["counts"]
    status = counts["status"]
    print("\nCorpus PDF-QC Summary\n---------------------")
    print(f"QC spec           : {SPEC_VERSION}")
    print(f"Mode              : {mode}")
    print(f"Expected files    : {counts['expected_files']}")
    print(f"Found / Missing   : {counts['found_files']} / {counts['missing_files']}")
    print(f"Total pages       : {counts['total_pages']}")
    print(f"PASS/REVIEW/FAIL  : {status['PASS']}/{status['REVIEW']}/{status['FAIL']}")
    print(f"Unexpected PDFs   : {counts['unexpected_pdf_count']}")
    print(f"Duplicate groups  : {counts['duplicate_hash_groups']}")
    print(f"Docs all-core PASS: {counts['docs_all_core_pass']}/{counts['docs_total']}")
    print(f"Output            : {run_dir}")
    return results, summary, run_dir


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Automated PDF-QC for Corpus v1 (no OCR, no RAG ingest)."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--corpus-root", default="data_raw/corpus_v1")
    parser.add_argument(
        "--manifest", default=None, help="Optional CSV export dari Corpus v1 - File Manifest"
    )
    parser.add_argument(
        "--doc-id", action="append", default=None, help="Batasi ke Doc ID tertentu; dapat diulang"
    )
    parser.add_argument("--output-dir", default="runs/corpus_qc")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--allow-external-corpus-root", action="store_true")
    parser.add_argument(
        "--fail-on-qc", action="store_true", help="Exit 2 jika ada REVIEW/FAIL/anomali"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        _, summary, _ = run(args)
    except Exception as exc:
        print(f"[FATAL] Corpus QC gagal: {exc}", file=sys.stderr)
        return 1

    if args.fail_on_qc:
        counts = summary["counts"]
        status = counts["status"]
        if (
            counts["missing_files"]
            or status["REVIEW"]
            or status["FAIL"]
            or counts["unexpected_pdf_count"]
            or counts["duplicate_hash_groups"]
        ):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
