# main.py
import sys
import polars as pl
import pandas as pd
import subprocess
from datetime import datetime
import os
import gc
from commits import get_commit
from src.filter import filter_by_date
from src.prepare import prepare_notes, prepare_ratings, prepare_status, prepare_enrollment
from src.load_schema import load_scorer_schema

print(f"Running Community Notes Data Preprocessing for {sys.argv[1]}")

# 1. Get the right commit YYYY-MM-DD
print(sys.argv)
date_str = sys.argv[1]
date_obj = datetime.strptime(date_str, "%Y-%m-%d")
print(date_obj)
year = date_obj.year
month = date_obj.month
commit = get_commit("communitynotes", date_str)
print(f"Finding commit for date: {commit}")

# 2. Checkout that commit in submodule
subprocess.run(["git", "-C", "communitynotes", "checkout", commit])


def find_scorer_entrypoint(repo_path):
    """
    Detects the scorer entrypoint for the checked-out commit.
    Returns the path to the Python file to run.
    """

    candidates = [
        "scoring/src/main.py",
        "sourcecode/main.py",
        "main.py",
        "src/main.py",
        "run.py",
        "score.py",
    ]

    for rel in candidates:
        full = os.path.join(repo_path, rel)
        if os.path.exists(full):
            return full

    raise FileNotFoundError(
        f"No scorer entrypoint found in {repo_path}. "
        "Commit may be too old or repo structure changed."
    )


# 3. Load data
print("Loading notes data...")
notes = pl.read_csv("org-data/notes-00000.tsv", separator='\t')
print("Loading status data...")
status = pl.read_csv("org-data/noteStatusHistory-00000.tsv", separator='\t')
print("Loading enrollment data...")
enrollment = pl.read_csv("org-data/userEnrollment-00000.tsv", separator='\t')

# 2. Load schema from that commit 
constants = load_scorer_schema("communitynotes")

# 4. Filter by date
print("Filtering data...")
notes = filter_by_date(notes, year, month)
status = filter_by_date(status, year, month)

# 5. Prepare data (drop columns)
print("Preparing data...")
notes = prepare_notes(notes, constants)
status = prepare_status(status, constants)
enrollment = prepare_enrollment(enrollment, constants)

# region rating preparation - STREAMING VERSION
print("Loading ratings data with Polars (streaming)...")

os.makedirs("data", exist_ok=True)
first_file = True
total_ratings = 0

for i in range(20):
    print(f"  Processing ratings-{i:05d}.tsv...")

    # Read, filter, prepare one file at a time
    rdf = pl.read_csv(f"org-data/ratings/ratings-{i:05d}.tsv", separator='\t')
    rdf = filter_by_date(rdf, year, month)
    rdf = prepare_ratings(rdf, constants)

    total_ratings += len(rdf)

    # Write immediately and free memory
    if first_file:
        rdf.write_csv("data/ratings-combined.tsv", separator='\t')
        first_file = False
    else:
        # Append to existing file
        with open("data/ratings-combined.tsv", "ab") as f:
            rdf.write_csv(f, separator='\t', include_header=False)

    del rdf
    gc.collect()

print(f"Total ratings: {total_ratings}")
# endregion

# 6. Save other data
print("Saving filtered data...")
notes.write_csv("data/notes-00000.tsv", separator='\t')
status.write_csv("data/noteStatusHistory-00000.tsv", separator='\t')
enrollment.write_csv("data/userEnrollment-00000.tsv", separator='\t')

# Free memory before running CN
del notes, status, enrollment
gc.collect()


# 7. Run CN algorithm
print("Running Community Notes algorithm...")
entry = find_scorer_entrypoint("communitynotes")
print(f"Detected scorer entrypoint: {entry}")

print(f'python {entry}')

# result = subprocess.run([
#     "python", entry,
#     "--enrollment", "data/userEnrollment-00000.tsv",
#     "--notes", "data/notes-00000.tsv",
#     "--ratings", "data/ratings-combined.tsv",
#     "--status", "data/noteStatusHistory-00000.tsv",
#     "--outdir", "data"
# ], capture_output=True, text=True)


result = subprocess.run(
    [
        "python", os.path.relpath(entry, "communitynotes"),
        "--enrollment", "../data/userEnrollment-00000.tsv",
        "--notes", "../data/notes-00000.tsv",
        "--ratings", "../data/ratings-combined.tsv",
        "--status", "../data/noteStatusHistory-00000.tsv",
        "--outdir", "../data",
    ],
    cwd="communitynotes",
    capture_output=True,
    text=True,
)

print("Done!")

# Print stdout and stderr
if result.stdout:
    print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Check return code
if result.returncode != 0:
    print(f"ERROR: Process exited with code {result.returncode}")
else:
    print("Process completed successfully")
    # Check if output files were created
    output_files = os.listdir("data")
    print(f"Output files created: {output_files}")
