# %% [markdown]
"""Notebook 01 -- Cohort Filtering & Distribution Analysis

Demonstrates loading the Speall MRI manifest, applying per-sequence and
per-grade filters, and plotting distribution charts.  Runs as a plain
Python script or can be converted to a Jupyter notebook with jupytext.

    jupytext --to notebook notebooks/01_cohort_filtering.py

Prerequisites
-------------
    pip install pyarrow pandas matplotlib

If you downloaded the dataset from HuggingFace you can also drive this
notebook from the streaming interface:
    pip install datasets
"""

# %% Setup
# pip install pyarrow pandas matplotlib

from pathlib import Path
from typing import Any

# %%
# Adjust DATASET_ROOT to wherever you unpacked the dataset download.
DATASET_ROOT = Path("/data/speall-mri")  # <-- change this
MANIFEST_PATH = DATASET_ROOT / "manifest.parquet"


# %% Helper -- load manifest without polars/torch dependency
def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load manifest.parquet into a list of dicts."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        cols = table.to_pydict()
        n = len(table)
        return [{col: cols[col][i] for col in cols} for i in range(n)]
    except Exception:
        pass
    import pandas as pd

    return pd.read_parquet(str(path)).to_dict(orient="records")


# %% Load
if not MANIFEST_PATH.exists():
    print(f"[DEMO] manifest not found at {MANIFEST_PATH}. Using synthetic data.")
    # Synthetic fallback for demo purposes
    import random

    random.seed(42)
    SEQ_TYPES = ["DWI", "FLAIR", "T1", "T2", "SWAN", "TOF"]
    GRADES = ["A", "B", "C", "D", "F"]
    rows = []
    for i in range(500):
        rows.append(
            {
                "study_id": f"study_{i // 5:04d}",
                "series_uid": f"uid_{i:06d}",
                "sequence_type": random.choice(SEQ_TYPES),
                "quality_grade": random.choice(GRADES),
                "ml_score": random.uniform(20, 95),
                "field_strength_T": random.choice([1.5, 3.0]),
                "modality": "MR",
                "b_value": 1000.0 if i % 5 == 0 else None,
            }
        )
    all_rows = rows
else:
    all_rows = load_manifest(MANIFEST_PATH)

print(f"Total series: {len(all_rows)}")


# %% Filter by sequence type
def filter_by_sequence(rows: list[dict], seq: str) -> list[dict]:
    """Return rows matching a sequence type (case-insensitive)."""
    seq_upper = seq.upper()
    return [r for r in rows if (r.get("sequence_type") or "").upper() == seq_upper]


dwi_rows = filter_by_sequence(all_rows, "DWI")
flair_rows = filter_by_sequence(all_rows, "FLAIR")
t1_rows = filter_by_sequence(all_rows, "T1")
print(f"DWI: {len(dwi_rows)}  FLAIR: {len(flair_rows)}  T1: {len(t1_rows)}")


# %% Filter by quality grade
def filter_by_grade(rows: list[dict], grades: list[str]) -> list[dict]:
    """Return rows where quality_grade is in the given list."""
    upper_grades = {g.upper() for g in grades}
    return [r for r in rows if (r.get("quality_grade") or "").upper() in upper_grades]


premium = filter_by_grade(all_rows, ["A", "B"])
print(f"Grade A/B (premium): {len(premium)} series")

# %% Plot sequence type distribution
try:
    from collections import Counter

    import matplotlib.pyplot as plt

    seq_counts = Counter((r.get("sequence_type") or "Unknown") for r in all_rows)
    labels, values = zip(*sorted(seq_counts.items(), key=lambda x: -x[1]), strict=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: sequence type
    axes[0].bar(labels, values, color="steelblue")
    axes[0].set_title("Series count by sequence type")
    axes[0].set_xlabel("Sequence type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Bar chart: quality grade
    grade_counts = Counter((r.get("quality_grade") or "?") for r in all_rows)
    g_labels = ["A", "B", "C", "D", "F", "?"]
    g_values = [grade_counts.get(g, 0) for g in g_labels]
    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#c0392b", "#95a5a6"]
    axes[1].bar(g_labels, g_values, color=colors)
    axes[1].set_title("Series count by quality grade")
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("cohort_distribution.png", dpi=120)
    print("Saved cohort_distribution.png")
    plt.show()

except ImportError:
    print("matplotlib not installed; skipping plots.")


# %% Filter example: DWI grade A/B for 3T scanners
def filter_cohort(
    rows: list[dict],
    sequence_type: str | None = None,
    grades: list[str] | None = None,
    field_strength: float | None = None,
) -> list[dict]:
    """Composable filter returning a cohort of series rows."""
    result = rows
    if sequence_type:
        result = filter_by_sequence(result, sequence_type)
    if grades:
        result = filter_by_grade(result, grades)
    if field_strength is not None:
        result = [r for r in result if r.get("field_strength_T") == field_strength]
    return result


training_cohort = filter_cohort(
    all_rows, sequence_type="DWI", grades=["A", "B"], field_strength=3.0
)
print(f"DWI 3T grade A/B cohort: {len(training_cohort)} series")

# %% Summary statistics
ml_scores = [r.get("ml_score") for r in training_cohort if r.get("ml_score") is not None]
if ml_scores:
    print(
        f"ML score -- mean: {sum(ml_scores) / len(ml_scores):.1f}, min: {min(ml_scores):.1f}, max: {max(ml_scores):.1f}"
    )

# %% Next steps
# -----------------------------------------------------------------------------
# NEXT STEPS:
#   1. Replace DATASET_ROOT with your actual download path.
#   2. Save `training_cohort` to a filtered parquet for reproducibility:
#          import pandas as pd
#          pd.DataFrame(training_cohort).to_parquet("dwi_3T_AB_cohort.parquet")
#   3. Pass the filtered list to SpeallMRIDataset via a custom manifest.
#   4. See notebooks/02_pytorch_training_loop.py for the training skeleton.
# -----------------------------------------------------------------------------
