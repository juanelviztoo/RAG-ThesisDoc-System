"""Read-only Automated PDF-QC for Corpus v1.

This utility is intentionally isolated from the RAG runtime: it does not import
``src.*`` modules, does not touch Chroma/BM25/embeddings/LLM, does not run OCR,
and never modifies the source PDFs or Google Sheet.
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

SPEC_VERSION = "1.1.0"
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
        (
            r"\bDAFTAR\s+(?:PUSTAKA|REFERENSI)\b",
            r"\bREFERENSI\b",
            r"\bREFERENCES?\b",
            r"\bBIBLIOGRAPHY\b",
        ),
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
    ("REFERENSI", r"\bREFERENSI\b"),
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
    for k, v in row.items():
        if norm_key(k) == target:
            return (v or "").strip()
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
        p for p in sorted(corpus_root.iterdir())
        if p.is_dir() and DOC_ID_RE.match(p.name) and (not selected or p.name in selected)
    ]
    found = {p.name for p in dirs}
    missing_dirs = sorted(selected - found)
    if missing_dirs:
        raise RuntimeError(f"Doc ID folder tidak ditemukan: {', '.join(missing_dirs)}")
    if not dirs:
        raise RuntimeError(f"Tidak ada folder UNS_INF_* pada {corpus_root}")

    out: List[ExpectedFile] = []
    unexpected: List[str] = []
    expected_names = {x.filename for x in PARTS}
    for d in dirs:
        nim = d.name.removeprefix("UNS_INF_")
        for spec in PARTS:
            p = d / spec.filename
            out.append(ExpectedFile(
                f"{d.name}__{spec.part_type}", d.name, nim, spec.part_type,
                spec.filename, rel(p, project_root), p,
            ))
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() == ".pdf" and p.name not in expected_names:
                unexpected.append(rel(p, project_root))
    for p in sorted(corpus_root.iterdir()):
        if p.is_file() and p.suffix.lower() == ".pdf":
            unexpected.append(rel(p, project_root))
    return out, unexpected


def expected_manifest(
    csv_path: Path, project_root: Path, corpus_root: Path, selected: set[str]
) -> Tuple[List[ExpectedFile], List[str]]:
    out: List[ExpectedFile] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    matched_docs: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Manifest CSV tidak memiliki header")
        for n, row in enumerate(reader, start=2):
            doc_id = row_get(row, "Doc ID")
            if not doc_id or (selected and doc_id not in selected):
                continue
            if row_get(row, "Requirement Role").upper() not in ("", "CORE_INDEXED"):
                continue
            if row_get(row, "Indexing Policy").upper() not in ("", "INDEXED"):
                continue
            part = row_get(row, "Part Type").upper()
            if part not in PART_BY_TYPE:
                raise RuntimeError(f"Row {n}: Part Type tidak valid: {part}")
            spec = PART_BY_TYPE[part]
            filename = row_get(row, "Canonical Filename") or spec.filename
            if filename != spec.filename:
                raise RuntimeError(f"Row {n}: filename {filename} != {spec.filename}")
            planned = row_get(row, "Planned Relative Path")
            path = (project_root / planned).resolve() if planned else (corpus_root / doc_id / filename).resolve()
            if not inside(path, corpus_root):
                raise RuntimeError(f"Row {n}: path keluar corpus_root: {path}")
            file_id = row_get(row, "File ID") or f"{doc_id}__{part}"
            if file_id in seen_ids or path in seen_paths:
                raise RuntimeError(f"Row {n}: duplicate File ID/path")
            seen_ids.add(file_id)
            seen_paths.add(path)
            matched_docs.add(doc_id)
            nim = row_get(row, "NIM") or doc_id.removeprefix("UNS_INF_")
            out.append(ExpectedFile(file_id, doc_id, nim, part, filename, planned or rel(path, project_root), path))
    missing_docs = sorted(selected - matched_docs)
    if missing_docs:
        raise RuntimeError(f"Doc ID tidak ditemukan pada manifest: {', '.join(missing_docs)}")
    if not out:
        raise RuntimeError("Tidak ada row CORE_INDEXED/INDEXED yang terpilih")

    expected_paths = {x.path.resolve() for x in out}
    unexpected: List[str] = []
    for doc_id in sorted({x.doc_id for x in out}):
        d = corpus_root / doc_id
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() == ".pdf" and p.resolve() not in expected_paths:
                unexpected.append(rel(p, project_root))
    return out, unexpected


def marker_found(text: str, spec: PartSpec) -> bool:
    return any(re.search(p, text, re.I) for p in spec.patterns)


def first_marker(text: str) -> str:
    hits: List[Tuple[int, str]] = []
    for label, pattern in ANY_MARKERS:
        m = re.search(pattern, text, re.I)
        if m:
            hits.append((m.start(), label))
    return min(hits)[1] if hits else ""


def text_class(total: int, pages: int, substantive: int, median_chars: float) -> Tuple[str, str, List[str]]:
    coverage = substantive / pages if pages else 0.0
    flags: List[str] = []
    if total < MIN_TOTAL_CHARS or coverage < TEXT_COVERAGE_PARTIAL:
        return "NO", "REQUIRED", ["CORE_TEXT_NOT_SUFFICIENTLY_EXTRACTABLE"]
    if coverage < TEXT_COVERAGE_YES or median_chars < MIN_MEDIAN_CHARS_YES:
        if coverage < TEXT_COVERAGE_YES:
            flags.append("PARTIAL_TEXT_COVERAGE")
        if median_chars < MIN_MEDIAN_CHARS_YES:
            flags.append("LOW_MEDIAN_TEXT_PER_PAGE")
        return "PARTIAL", "POSSIBLE", flags
    return "YES", "NO", flags


def structure_class(spec: PartSpec, first_text: str, full_text: str) -> Tuple[str, str, str, str, List[str]]:
    first_ok = marker_found(first_text, spec)
    any_ok = first_ok or marker_found(full_text, spec)
    detected = first_marker(first_text)
    if first_ok:
        return "YES", "YES", "YES", detected, []
    if any_ok:
        return "REVIEW", "NO", "YES", detected, ["EXPECTED_SECTION_MARKER_FOUND_LATE"]
    if detected and detected != spec.bab_label:
        ref_eq = spec.bab_label == "DAFTAR_PUSTAKA" and detected in {"REFERENSI", "REFERENCES", "BIBLIOGRAPHY"}
        if not ref_eq:
            return "NO", "NO", "NO", detected, [f"WRONG_SECTION_MARKER:{detected}"]
    return "REVIEW", "NO", "NO", detected, ["EXPECTED_SECTION_MARKER_NOT_FOUND"]


def blank_result(e: ExpectedFile, flag: str, status: str = "FAIL") -> QcResult:
    return QcResult(
        SPEC_VERSION, e.file_id, e.doc_id, e.nim, e.part_type, e.filename, e.relpath,
        "NO", 0, 0, "NO", "UNKNOWN", 0, 0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0,
        "NO", "REQUIRED", "NO", "NO", "", "PENDING", "", "", status, "BLOCKED", flag, "",
    )


def extract_page_text(page: Any) -> str:
    """Extract plain text from one PyMuPDF page with runtime type narrowing.

    PyMuPDF's type stubs expose ``Page.get_text`` as a union because other
    extraction modes can return lists or dictionaries. In ``"text"`` mode the
    runtime contract we require is ``str``. Any unexpected type is treated as
    a page-level extraction anomaly instead of being silently coerced.
    """
    raw_text = page.get_text("text")
    if raw_text is None:
        return ""
    if not isinstance(raw_text, str):
        raise TypeError(
            "PyMuPDF get_text('text') menghasilkan tipe tak terduga: "
            f"{type(raw_text).__name__}"
        )
    return norm_text(raw_text)


def audit_file(e: ExpectedFile) -> QcResult:
    p = e.path
    if not p.is_file():
        return blank_result(e, "MISSING_FILE")

    size = int(p.stat().st_size)
    try:
        digest = sha256_file(p)
    except Exception as exc:
        r = blank_result(e, "HASH_READ_ERROR")
        r.file_exists = "YES"
        r.file_size_bytes = size
        r.notes = str(exc)
        return r

    try:
        doc = fitz.open(str(p))
    except Exception as exc:
        r = blank_result(e, "PDF_OPEN_ERROR")
        r.file_exists = "YES"
        r.file_size_bytes = size
        r.sha256 = digest
        r.notes = str(exc)
        return r

    texts: List[str] = []
    chars: List[int] = []
    words: List[int] = []
    imgs: List[int] = []
    extraction_error_pages: List[int] = []
    extraction_error_notes: List[str] = []

    try:
        if bool(doc.needs_pass):
            r = blank_result(e, "PDF_PASSWORD_REQUIRED")
            r.file_exists = "YES"
            r.file_size_bytes = size
            r.sha256 = digest
            r.encrypted = "YES"
            return r

        page_count = int(doc.page_count)
        if page_count <= 0:
            r = blank_result(e, "ZERO_PAGE_PDF")
            r.file_exists = "YES"
            r.file_size_bytes = size
            r.sha256 = digest
            r.encrypted = "NO"
            return r

        for page_index in range(page_count):
            try:
                page = doc.load_page(page_index)
            except Exception as exc:
                # Jika satu halaman bahkan gagal dimuat, catat sebagai anomaly
                # tetapi jangan hentikan pemeriksaan seluruh PDF/corpus.
                texts.append("")
                chars.append(0)
                words.append(0)
                imgs.append(0)

                extraction_error_pages.append(page_index + 1)
                extraction_error_notes.append(
                    f"p{page_index + 1}:"
                    f"PAGE_LOAD_ERROR:"
                    f"{type(exc).__name__}:{exc}"
                )
                continue

            try:
                text = extract_page_text(page)
            except Exception as exc:
                # Isolasi kegagalan ekstraksi teks per halaman:
                # satu halaman anomali tidak boleh menghentikan seluruh batch.
                text = ""
                extraction_error_pages.append(page_index + 1)
                extraction_error_notes.append(
                    f"p{page_index + 1}:"
                    f"TEXT_EXTRACTION_ERROR:"
                    f"{type(exc).__name__}:{exc}"
                )

            texts.append(text)
            chars.append(len(text))
            words.append(len(text.split()))

            try:
                imgs.append(len(page.get_images(full=True)))
            except Exception:
                imgs.append(0)
    finally:
        doc.close()

    total = sum(chars)
    total_words = sum(words)
    median_chars = float(statistics.median(chars)) if chars else 0.0
    mean_chars = float(statistics.mean(chars)) if chars else 0.0
    substantive = sum(x >= SUBSTANTIVE_CHARS_PER_PAGE for x in chars)
    coverage = substantive / page_count
    text_extractable, ocr_flag, flags = text_class(
        total, page_count, substantive, median_chars
    )

    if extraction_error_pages:
        flags.append(f"PAGE_TEXT_EXTRACTION_ERROR:{len(extraction_error_pages)}")
        # Sebagian halaman gagal diekstrak => jangan otomatis FAIL selama
        # core text lainnya masih usable. Flag ke REVIEW untuk inspeksi manusia.
        if len(extraction_error_pages) < page_count and ocr_flag == "NO":
            ocr_flag = "POSSIBLE"
        # Semua halaman gagal => core text secara praktis tidak tersedia.
        if len(extraction_error_pages) == page_count:
            text_extractable = "NO"
            ocr_flag = "REQUIRED"

    spec = PART_BY_TYPE[e.part_type]
    first_text = "\n".join(texts[:STRUCTURE_SCAN_PAGES])
    full_text = "\n".join(texts)
    structure, exp_first, exp_any, detected, sflags = structure_class(
        spec, first_text, full_text
    )
    flags += sflags

    if size < MIN_FILE_BYTES:
        flags.append("FILE_TOO_SMALL")

    if len(extraction_error_pages) == page_count:
        status = "FAIL"
        recommendation = "BLOCKED"
    else:
        status = (
            "PASS"
            if not flags and text_extractable == "YES" and structure == "YES"
            else "REVIEW"
        )
        recommendation = "ELIGIBLE" if status == "PASS" else "MANUAL_REVIEW"

    image_pages = sum(x > 0 for x in imgs)
    note_parts: List[str] = []
    if image_pages:
        note_parts.append(
            f"Komponen visual terdeteksi pada {image_pages}/{page_count} halaman; "
            "tidak otomatis memengaruhi PASS selama teks inti tetap extractable."
        )
    if extraction_error_pages:
        shown_pages = ",".join(str(x) for x in extraction_error_pages[:10])
        suffix = "..." if len(extraction_error_pages) > 10 else ""
        note_parts.append(
            "Text extraction gagal pada halaman "
            f"{shown_pages}{suffix}; batch QC tetap dilanjutkan. "
            f"Detail: {' | '.join(extraction_error_notes[:3])}"
        )

    return QcResult(
        SPEC_VERSION,
        e.file_id,
        e.doc_id,
        e.nim,
        e.part_type,
        e.filename,
        e.relpath,
        "YES",
        size,
        page_count,
        "YES",
        "NO",
        total,
        total_words,
        round(median_chars, 2),
        round(mean_chars, 2),
        min(chars),
        max(chars),
        substantive,
        round(coverage, 4),
        page_count - substantive,
        image_pages,
        round(image_pages / page_count, 4),
        sum(imgs),
        text_extractable,
        ocr_flag,
        exp_first,
        exp_any,
        detected,
        structure,
        digest,
        "",
        status,
        recommendation,
        ";".join(dict.fromkeys(flags)),
        " ".join(note_parts),
    )


def apply_duplicates(results: List[QcResult]) -> Dict[str, List[str]]:
    groups: Dict[str, List[QcResult]] = {}
    for r in results:
        if r.sha256:
            groups.setdefault(r.sha256, []).append(r)
    duplicates: Dict[str, List[str]] = {}
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        ids = [r.file_id for r in group]
        duplicates[digest] = ids
        anchor = ids[0]
        for r in group[1:]:
            r.duplicate_of = anchor
            r.qc_flags = ";".join(x for x in [r.qc_flags, "EXACT_DUPLICATE_SHA256"] if x)
            if r.file_qc_status == "PASS":
                r.file_qc_status = "REVIEW"
                r.qc_inclusion_recommendation = "MANUAL_REVIEW"
    return duplicates


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def make_summary(
    results: List[QcResult], expected: List[ExpectedFile], unexpected: List[str],
    duplicates: Dict[str, List[str]], mode: str, project_root: Path, corpus_root: Path,
    manifest: Optional[Path],
) -> Dict[str, Any]:
    status = {x: sum(r.file_qc_status == x for r in results) for x in ("PASS", "REVIEW", "FAIL")}
    text = {x: sum(r.text_extractable == x for r in results) for x in ("YES", "PARTIAL", "NO")}
    ocr = {x: sum(r.ocr_dependency_flag == x for r in results) for x in ("NO", "POSSIBLE", "REQUIRED")}
    structure = {x: sum(r.structure_ok == x for r in results) for x in ("YES", "REVIEW", "NO", "PENDING")}
    per_doc: Dict[str, Dict[str, Any]] = {}
    for r in results:
        d = per_doc.setdefault(r.doc_id, {"expected": 0, "pass": 0, "review": 0, "fail": 0})
        d["expected"] += 1
        d[r.file_qc_status.lower()] += 1
    for d in per_doc.values():
        d["all_core_pass"] = d["expected"] == len(PARTS) and d["pass"] == len(PARTS)
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
            "found_files": sum(r.file_exists == "YES" for r in results),
            "missing_files": sum(r.file_exists == "NO" for r in results),
            "total_pages": sum(r.page_count for r in results),
            "status": status,
            "text_extractable": text,
            "ocr_dependency_flag": ocr,
            "structure_ok": structure,
            "unexpected_pdf_count": len(unexpected),
            "duplicate_hash_groups": len(duplicates),
            "docs_all_core_pass": sum(bool(d["all_core_pass"]) for d in per_doc.values()),
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
    corpus_root = (project_root / corpus_root).resolve() if not corpus_root.is_absolute() else corpus_root.resolve()
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root tidak ditemukan: {corpus_root}")
    if not inside(corpus_root, project_root) and not args.allow_external_corpus_root:
        raise RuntimeError("corpus_root di luar project_root; gunakan --allow-external-corpus-root jika disengaja")
    selected = {x.strip() for x in (args.doc_id or []) if x.strip()}
    manifest: Optional[Path] = None
    if args.manifest:
        manifest = Path(args.manifest)
        manifest = (project_root / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
        expected, unexpected = expected_manifest(manifest, project_root, corpus_root, selected)
        mode = "manifest"
    else:
        expected, unexpected = expected_contract(corpus_root, project_root, selected)
        mode = "contract"
    results = [audit_file(x) for x in expected]
    duplicates = apply_duplicates(results)
    summary = make_summary(results, expected, unexpected, duplicates, mode, project_root, corpus_root, manifest)

    output_root = Path(args.output_dir)
    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    run_dir = output_root / (args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(x) for x in results]
    fields = list(rows[0].keys())
    write_csv(run_dir / "qc_report.csv", rows, fields)
    write_csv(run_dir / "qc_review_queue.csv", [r for r in rows if r["file_qc_status"] != "PASS"], fields)
    (run_dir / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    c = summary["counts"]
    s = c["status"]
    print("\nCorpus PDF-QC Summary\n---------------------")
    print(f"QC spec           : {SPEC_VERSION}\nMode              : {mode}\nExpected files    : {c['expected_files']}")
    print(f"Found / Missing   : {c['found_files']} / {c['missing_files']}\nTotal pages       : {c['total_pages']}")
    print(f"PASS/REVIEW/FAIL  : {s['PASS']}/{s['REVIEW']}/{s['FAIL']}")
    print(f"Unexpected PDFs   : {c['unexpected_pdf_count']}\nDuplicate groups  : {c['duplicate_hash_groups']}")
    print(f"Docs all-core PASS: {c['docs_all_core_pass']}/{c['docs_total']}\nOutput            : {run_dir}")
    return results, summary, run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only Automated PDF-QC for Corpus v1 (no OCR, no RAG ingest).")
    p.add_argument("--project-root", default=".")
    p.add_argument("--corpus-root", default="data_raw/corpus_v1")
    p.add_argument("--manifest", default=None, help="Optional CSV export dari Corpus v1 - File Manifest")
    p.add_argument("--doc-id", action="append", default=None, help="Batasi ke Doc ID tertentu; dapat diulang")
    p.add_argument("--output-dir", default="runs/corpus_qc")
    p.add_argument("--run-name", default=None)
    p.add_argument("--allow-external-corpus-root", action="store_true")
    p.add_argument("--fail-on-qc", action="store_true", help="Exit 2 jika ada REVIEW/FAIL/anomali")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        _, summary, _ = run(args)
    except Exception as exc:
        print(f"[FATAL] Corpus QC gagal: {exc}", file=sys.stderr)
        return 1
    if args.fail_on_qc:
        c = summary["counts"]
        s = c["status"]
        if c["missing_files"] or s["REVIEW"] or s["FAIL"] or c["unexpected_pdf_count"] or c["duplicate_hash_groups"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
