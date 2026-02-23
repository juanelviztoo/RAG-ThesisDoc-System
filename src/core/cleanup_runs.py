from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    path: Path
    timestamp: datetime
    size_bytes: int
    keep: bool  # protected (.keep / KEEP)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_dt(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # anggap local -> ubah ke UTC
            return dt.astimezone(timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _run_timestamp(run_dir: Path) -> datetime:
    """
    Prefer timestamp dari manifest.json; fallback ke nama folder; terakhir fallback ke mtime.
    """
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            ts = obj.get("timestamp")
            if isinstance(ts, str):
                dt = _parse_iso_dt(ts)
                if dt:
                    return dt
        except Exception:
            pass

    # parse dari run_id (YYYY-MM-DD_HH-MM-SS)
    m = RUN_ID_RE.match(run_dir.name)
    if m:
        try:
            dt_naive = datetime.strptime(run_dir.name, "%Y-%m-%d_%H-%M-%S")
            return dt_naive.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # fallback ke modified time
    try:
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return _now_utc()


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _is_protected(run_dir: Path) -> bool:
    # kalau ada marker, jangan dihapus
    return (
        (run_dir / ".keep").exists()
        or (run_dir / "KEEP").exists()
        or (run_dir / "keep.txt").exists()
    )


def _list_runs(runs_dir: Path) -> List[RunInfo]:
    if not runs_dir.exists():
        return []

    out: List[RunInfo] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if not RUN_ID_RE.match(d.name):
            continue
        ts = _run_timestamp(d)
        size_b = _dir_size_bytes(d)
        keep = _is_protected(d)
        out.append(RunInfo(run_id=d.name, path=d, timestamp=ts, size_bytes=size_b, keep=keep))

    # newest first
    out.sort(key=lambda x: x.timestamp, reverse=True)
    return out


def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def _zip_run(run: RunInfo, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archive_dir / f"{run.run_id}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run.path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(run.path)
                zf.write(p, arcname=str(Path(run.run_id) / rel))

    return zip_path


def plan_cleanup(
    runs: List[RunInfo],
    *,
    keep_last_n: int,
    max_age_days: Optional[int],
    max_total_mb: Optional[int],
) -> Tuple[List[RunInfo], str]:
    """
    Policy:
    - Selalu keep N run terbaru
    - Jangan pernah hapus run yang protected (.keep/KEEP/keep.txt)
    - Hapus kandidat jika:
      a) lebih tua dari max_age_days (jika diset), atau
      b) total size melebihi max_total_mb (jika diset) -> delete oldest sampai <= limit
    """
    if keep_last_n < 0:
        keep_last_n = 0

    # split: protected + unprotected
    protected = [r for r in runs if r.keep]
    normal = [r for r in runs if not r.keep]

    # keep N terbaru dari normal
    keep_set = set([r.run_id for r in normal[:keep_last_n]]) | set([r.run_id for r in protected])

    # candidates = normal yg tidak termasuk keep_set
    candidates = [r for r in normal if r.run_id not in keep_set]

    # mark by age
    to_remove_ids = set()
    now = _now_utc()
    if max_age_days is not None:
        cutoff = now - timedelta(days=int(max_age_days))
        for r in candidates:
            if r.timestamp < cutoff:
                to_remove_ids.add(r.run_id)

    # size enforcement (delete oldest first among candidates)
    if max_total_mb is not None:
        limit_b = int(max_total_mb) * 1024 * 1024
        total_b = sum(r.size_bytes for r in runs)
        if total_b > limit_b:
            # prefer remove those already marked; then oldest remaining candidates
            candidates_oldest = sorted(candidates, key=lambda x: x.timestamp)  # oldest first
            # build removal order: first age-marked old ones in oldest order, then rest oldest
            age_marked = [r for r in candidates_oldest if r.run_id in to_remove_ids]
            rest = [r for r in candidates_oldest if r.run_id not in to_remove_ids]
            removal_order = age_marked + rest

            cur = total_b
            for r in removal_order:
                if cur <= limit_b:
                    break
                to_remove_ids.add(r.run_id)
                cur -= r.size_bytes

    to_remove = [r for r in runs if r.run_id in to_remove_ids and not r.keep]
    to_remove.sort(key=lambda x: x.timestamp)  # oldest first

    summary = []
    summary.append(f"Total runs: {len(runs)}")
    summary.append(f"Protected: {len(protected)}")
    summary.append(f"Keep last N: {keep_last_n}")
    summary.append(f"Planned remove: {len(to_remove)}")
    summary.append(f"Total size: {_bytes_to_mb(sum(r.size_bytes for r in runs)):.1f} MB")
    if max_total_mb is not None:
        summary.append(f"Max total: {max_total_mb} MB")
    if max_age_days is not None:
        summary.append(f"Max age: {max_age_days} days")

    return to_remove, " | ".join(summary)


def apply_cleanup(
    to_remove: List[RunInfo],
    *,
    archive_dir: Optional[Path],
    delete_only: bool,
) -> None:
    for r in to_remove:
        if archive_dir is not None and not delete_only:
            zip_path = _zip_run(r, archive_dir=archive_dir)
            print(f"[ARCHIVED] {r.run_id} -> {zip_path}")
        # delete folder
        shutil.rmtree(r.path, ignore_errors=True)
        print(f"[DELETED] {r.run_id}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cleanup runs/ directory (safe default: dry-run).")
    p.add_argument("--runs-dir", type=str, default="runs", help="Path to runs directory")
    p.add_argument("--keep-last", type=int, default=30, help="Always keep last N most recent runs")
    p.add_argument("--max-age-days", type=int, default=None, help="Remove runs older than N days")
    p.add_argument("--max-total-mb", type=int, default=None, help="Keep total runs size under N MB")
    p.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="If set, archive removed runs into zip files here before deleting",
    )
    p.add_argument(
        "--delete-only",
        action="store_true",
        help="If set, do not archive even if --archive-dir is provided (delete only)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform deletion/archiving. Without this: dry-run.",
    )
    p.add_argument("--list", action="store_true", help="List runs and exit (no plan).")
    args = p.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs = _list_runs(runs_dir)

    if args.list:
        for r in runs:
            flag = "KEEP" if r.keep else "    "
            print(
                f"{flag} {r.run_id} | {r.timestamp.isoformat()} | {_bytes_to_mb(r.size_bytes):.1f} MB | {r.path}"
            )
        return 0

    to_remove, summary = plan_cleanup(
        runs,
        keep_last_n=args.keep_last,
        max_age_days=args.max_age_days,
        max_total_mb=args.max_total_mb,
    )

    print(summary)
    if not to_remove:
        print("Nothing to remove.")
        return 0

    print("\nPlanned removals (oldest -> newest):")
    for r in to_remove:
        age_days = int((_now_utc() - r.timestamp).total_seconds() // 86400)
        print(f"- {r.run_id} | age={age_days}d | {_bytes_to_mb(r.size_bytes):.1f} MB")

    if not args.apply:
        print("\nDry-run only. Add --apply to perform.")
        return 0

    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    apply_cleanup(to_remove, archive_dir=archive_dir, delete_only=bool(args.delete_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
