#!/usr/bin/env python3
"""Run the Community Notes scoring algorithm using code and data from a specific date.

This script:
1. Checks out the most recent CN repo commit on main before the target date
2. Dynamically extracts expected column schemas from that version's constants.py
3. Filters raw data to the target date
4. Strips columns to match the checked-out version's expectations
5. Merges ratings files and runs the scoring algorithm
6. Restores the CN repo to its original commit

Usage:
    python run_at_date.py --date 2024-03-01
    python run_at_date.py --date 2024-03-01 --skip-filter
    python run_at_date.py --date 2024-03-01 --skip-scoring
"""

import argparse
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# --- Path constants ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # students/gaal/
REPO_ROOT = PROJECT_ROOT.parent.parent  # 494-user-trajectories/
LOCAL_DATA = REPO_ROOT / "local-data"
CN_DIR = PROJECT_ROOT / "communitynotes"
# Python from the CN repo's venv (has numpy, pandas, torch, etc.)
CN_PYTHON = str(CN_DIR / ".venv" / "bin" / "python")

# --- Timestamp column mapping for filtering ---
TIMESTAMP_COL = {
    "notes": "createdAtMillis",
    "notes-status-history": "createdAtMillis",
    "ratings": "createdAtMillis",
    "notes-request-data": "noteRequestFeedEligibleAtMillis",
}
COPY_WITHOUT_FILTER = {"user-enrollment"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CN scoring algorithm for a specific date."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD format (e.g. 2024-03-01)",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filtering (use existing filtered data)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring (only filter and strip columns)",
    )
    return parser.parse_args()


def compute_cutoff_ms(date_str: str) -> int:
    """Compute end-of-day epoch milliseconds for the given date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def date_label(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYY-MM for directory naming."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m")


# --- Git operations ---


def get_current_commit(cn_dir: Path) -> str:
    """Get the current HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cn_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_commit_for_date(cn_dir: Path, date_str: str) -> str:
    """Find the most recent commit on origin/main on or before the given date."""
    result = subprocess.run(
        ["git", "log", "origin/main", f"--before={date_str}T23:59:59",
         "--format=%H", "-1"],
        cwd=cn_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"No commit found on origin/main before {date_str}")
    return commit


def checkout_commit(cn_dir: Path, commit: str):
    """Checkout a specific commit in the CN repo."""
    subprocess.run(
        ["git", "checkout", commit],
        cwd=cn_dir,
        check=True,
        capture_output=True,
    )


# --- Directory layout detection ---


def find_scoring_paths(cn_dir: Path) -> tuple[Path, Path]:
    """Return (main_py_path, scoring_package_parent) for the checked-out version.

    The CN repo changed layout from sourcecode/ to scoring/src/ on June 30, 2025.
    """
    if (cn_dir / "scoring" / "src" / "main.py").exists():
        return cn_dir / "scoring" / "src" / "main.py", cn_dir / "scoring" / "src"
    elif (cn_dir / "sourcecode" / "main.py").exists():
        return cn_dir / "sourcecode" / "main.py", cn_dir / "sourcecode"
    else:
        raise RuntimeError("Cannot find main.py in checked-out CN code")


# --- Dynamic column extraction ---


def extract_columns(scoring_parent: Path) -> dict[str, list[str]]:
    """Extract expected column lists from the checked-out constants.py.

    Runs in a subprocess to avoid module cache contamination between dates.
    """
    script = textwrap.dedent(f"""\
        import sys, json
        sys.path.insert(0, {str(scoring_parent)!r})
        from scoring.constants import (
            noteTSVColumns,
            ratingTSVColumns,
            noteStatusHistoryTSVColumns,
        )
        try:
            from scoring.constants import userEnrollmentExpandedTSVColumns
            enrollment_cols = list(userEnrollmentExpandedTSVColumns)
        except ImportError:
            from scoring.constants import userEnrollmentTSVColumns
            enrollment_cols = list(userEnrollmentTSVColumns)
        print(json.dumps({{
            "notes": list(noteTSVColumns),
            "ratings": list(ratingTSVColumns),
            "notes-status-history": list(noteStatusHistoryTSVColumns),
            "user-enrollment": enrollment_cols,
        }}))
    """)
    result = subprocess.run(
        [CN_PYTHON, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to extract columns from constants.py")
    return json.loads(result.stdout)


# --- Data filtering ---


def filter_data(local_data: Path, out_root: Path, cutoff_ms: int):
    """Filter all data directories to the cutoff timestamp."""
    for subdir in sorted(local_data.iterdir()):
        if not subdir.is_dir():
            continue

        out_dir = out_root / subdir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy without filtering for user-enrollment
        if subdir.name in COPY_WITHOUT_FILTER:
            print(f"\n  Copying {subdir.name}/ without filtering...")
            for tsv in sorted(subdir.glob("*.tsv")):
                out_path = out_dir / tsv.name
                lf = pl.scan_csv(tsv, separator="\t", infer_schema_length=0)
                lf.sink_csv(out_path, separator="\t")
                rows = lf.select(pl.len()).collect()[0, 0]
                print(f"    Copied {out_path.name} ({rows:,} rows)")
            continue

        ts_col = TIMESTAMP_COL.get(subdir.name)
        if ts_col is None:
            print(f"\n  Skipping {subdir.name}/ (no timestamp column configured)")
            continue

        print(f"\n  Filtering {subdir.name}/ on {ts_col} <= {cutoff_ms} ...")

        for tsv in sorted(subdir.glob("*.tsv")):
            out_path = out_dir / tsv.name

            lf = pl.scan_csv(tsv, separator="\t", infer_schema_length=0)
            schema = lf.collect_schema().names()

            if ts_col not in schema:
                print(f"    Skipping {tsv.name} (no {ts_col} column)")
                continue

            filtered = (
                lf.with_columns(
                    pl.col(ts_col).cast(pl.Int64, strict=False).alias(ts_col)
                )
                .filter(pl.col(ts_col).is_not_null())
                .filter(pl.col(ts_col) <= cutoff_ms)
            )

            filtered.sink_csv(out_path, separator="\t")
            rows = filtered.select(pl.len()).collect()[0, 0]
            print(f"    Wrote {out_path.name} ({rows:,} rows)")


# --- Column stripping ---


def strip_columns(filtered_dir: Path, expected_columns: dict[str, list[str]]):
    """Strip extra columns from filtered data to match the scoring schema."""
    for data_type, expected_cols in expected_columns.items():
        input_dir = filtered_dir / data_type
        if not input_dir.exists():
            print(f"  {data_type}/: directory not found, skipping")
            continue

        for tsv in sorted(input_dir.glob("*.tsv")):
            df = pl.read_csv(tsv, separator="\t", infer_schema_length=0)
            current_cols = df.columns

            # Keep columns that exist in both raw data and expected schema
            cols_to_keep = [c for c in expected_cols if c in current_cols]
            extra_cols = [c for c in current_cols if c not in expected_cols]

            if extra_cols:
                print(f"  {data_type}/{tsv.name}: removing {len(extra_cols)} extra columns: {extra_cols}")
                df = df.select(cols_to_keep)
                df.write_csv(tsv, separator="\t")
            else:
                print(f"  {data_type}/{tsv.name}: no extra columns")


# --- Ratings merging ---


def merge_ratings(filtered_dir: Path, output_dir: Path) -> Path:
    """Merge multiple rating TSV files into a single file."""
    ratings_dir = filtered_dir / "ratings"
    merged_path = output_dir / "merged_ratings.tsv"
    files = sorted(ratings_dir.glob("*.tsv"))

    if not files:
        raise RuntimeError(f"No rating files found in {ratings_dir}")

    print(f"  Merging {len(files)} ratings files...")
    dfs = [pl.read_csv(f, separator="\t", infer_schema_length=0) for f in files]
    merged = pl.concat(dfs)
    merged.write_csv(merged_path, separator="\t")
    print(f"  Merged ratings: {len(merged):,} rows -> {merged_path.name}")
    return merged_path


# --- Scoring ---


def run_scoring(
    main_py: Path,
    scoring_parent: Path,
    filtered_dir: Path,
    output_dir: Path,
    cutoff_ms: int,
    ratings_path: Path,
):
    """Invoke the CN scoring algorithm."""
    cmd = [
        CN_PYTHON,
        str(main_py),
        "--notes",
        str(filtered_dir / "notes" / "notes-00000.tsv"),
        "--ratings",
        str(ratings_path),
        "--status",
        str(filtered_dir / "notes-status-history" / "noteStatusHistory-00000.tsv"),
        "--enrollment",
        str(filtered_dir / "user-enrollment" / "userEnrollment-00000.tsv"),
        "--epoch-millis",
        str(cutoff_ms),
        "--nostrict-columns",
        "--nopseudoraters",
        "--outdir",
        str(output_dir),
    ]
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(scoring_parent), check=True)


# --- Main ---


def main():
    args = parse_args()
    target_date = args.date
    label = date_label(target_date)
    cutoff_ms = compute_cutoff_ms(target_date)

    filtered_dir = REPO_ROOT / "data" / "filtered" / label
    output_dir = PROJECT_ROOT / "output" / label

    print(f"=== CN Scoring for {target_date} ===")
    print(f"  Date label:    {label}")
    print(f"  Cutoff millis: {cutoff_ms}")
    print(f"  Filtered data: {filtered_dir}")
    print(f"  Output dir:    {output_dir}")
    print()

    # Step 1: Find and checkout the right CN commit
    print("Step 1: Checking out CN repo at target date...")
    original_commit = get_current_commit(CN_DIR)
    target_commit = get_commit_for_date(CN_DIR, target_date)
    print(f"  Original commit: {original_commit[:12]}")
    print(f"  Target commit:   {target_commit[:12]}")
    checkout_commit(CN_DIR, target_commit)
    print(f"  Checked out {target_commit[:12]}")
    print()

    try:
        # Step 2: Detect directory layout
        print("Step 2: Detecting directory layout...")
        main_py, scoring_parent = find_scoring_paths(CN_DIR)
        print(f"  main.py:        {main_py.relative_to(CN_DIR)}")
        print(f"  scoring parent: {scoring_parent.relative_to(CN_DIR)}")
        print()

        # Step 3: Extract expected columns
        print("Step 3: Extracting expected columns from constants.py...")
        expected_columns = extract_columns(scoring_parent)
        for dtype, cols in expected_columns.items():
            print(f"  {dtype}: {len(cols)} columns")
        print()

        # Step 4: Filter data
        if not args.skip_filter:
            print("Step 4: Filtering raw data...")
            filter_data(LOCAL_DATA, filtered_dir, cutoff_ms)
            print()

            # Step 5: Strip extra columns
            print("Step 5: Stripping extra columns...")
            strip_columns(filtered_dir, expected_columns)
            print()
        else:
            print("Step 4-5: Skipping filter and strip (--skip-filter)")
            print()

        # Step 6-7: Merge ratings and run scoring
        if not args.skip_scoring:
            print("Step 6: Preparing output...")
            output_dir.mkdir(parents=True, exist_ok=True)
            ratings_path = merge_ratings(filtered_dir, output_dir)
            print()

            print("Step 7: Running scoring algorithm...")
            run_scoring(main_py, scoring_parent, filtered_dir, output_dir, cutoff_ms, ratings_path)
            print()
            print(f"=== Scoring complete for {target_date}! ===")
        else:
            print("Step 6-7: Skipping scoring (--skip-scoring)")
            print()
            print(f"=== Filter/strip complete for {target_date}! ===")

    finally:
        # Always restore original commit
        print(f"\nRestoring CN repo to {original_commit[:12]}...")
        checkout_commit(CN_DIR, original_commit)
        print("  Restored.")


if __name__ == "__main__":
    main()
