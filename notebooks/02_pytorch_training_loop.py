# %% [markdown]
"""Notebook 02 -- PyTorch Training Loop Skeleton

End-to-end demonstration: load the Speall MRI dataset via SpeallMRIDataset,
define a small CNN, and run one training epoch.  Synthetic quality-grade
labels are used so the demo runs without a GPU.  Swap in your own model and
real data paths before production training.

    jupytext --to notebook notebooks/02_pytorch_training_loop.py

Prerequisites
-------------
    pip install "micom[torch]" pyarrow
    # or individually:
    pip install torch pyarrow
"""

# %% Setup
# pip install torch pyarrow

import sys
from pathlib import Path

# Add repo root so `src.loaders.pytorch` is importable when running as script
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# %% Check torch availability
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    _TORCH = True
    print(f"torch {torch.__version__} available -- device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
except ImportError:
    _TORCH = False
    print("torch not installed. Install with: pip install torch")
    print("Running in no-op demonstration mode.")

# %% Import the Speall loader
from src.loaders.pytorch import SpeallMRIDataset, make_dataloader  # noqa: E402

# %% Configuration
DATASET_ROOT = Path("/data/speall-mri")       # <-- change this
MANIFEST_PATH = DATASET_ROOT / "manifest.parquet"
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

DEVICE = "mps" if (_TORCH and torch.backends.mps.is_available()) else "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
print(f"Using device: {DEVICE}")

# %% Build a synthetic dataset when real data is absent (demo mode)
_DEMO_MODE = not MANIFEST_PATH.exists()

if _DEMO_MODE:
    print("[DEMO] manifest.parquet not found -- building in-memory synthetic dataset.")
    import random
    import tempfile
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    random.seed(0)
    _SEQ = ["DWI", "FLAIR", "T1", "T2"]
    _GRADES = ["A", "B", "C", "D"]
    _tmp = Path(tempfile.mkdtemp())
    _rows = {
        "study_id": [f"study_{i//4:04d}" for i in range(80)],
        "series_uid": [f"uid_{i:06d}" for i in range(80)],
        "series_number": list(range(80)),
        "series_description": [_SEQ[i % 4] for i in range(80)],
        "sequence_type": [_SEQ[i % 4] for i in range(80)],
        "sequence_confidence": ["high"] * 80,
        "modality": ["MR"] * 80,
        "file_count": [50] * 80,
        "tr_ms": [6000.0] * 80,
        "te_ms": [75.0] * 80,
        "ti_ms": [None] * 80,
        "fa_deg": [90.0] * 80,
        "b_value": [1000.0 if i % 4 == 0 else None for i in range(80)],
        "field_strength_T": [3.0] * 80,
        "plane": ["axial"] * 80,
        "volume_shape": [[50, 256, 256]] * 80,
        "spacing_mm": [[1.09, 1.09, 2.94]] * 80,
        "fov_mm": [[147.0, 280.0, 280.0]] * 80,
        "volume_snr": [random.uniform(0.3, 0.9) for _ in range(80)],
        "volume_cnr": [random.uniform(500, 3000) for _ in range(80)],
        "volume_entropy": [random.uniform(1.0, 3.0) for _ in range(80)],
        "quality_grade": [_GRADES[i % 4] for i in range(80)],
        "quality_score": [random.uniform(40, 90) for _ in range(80)],
        "ml_score": [random.uniform(30, 95) for _ in range(80)],
        "ml_grade": [_GRADES[i % 4] for i in range(80)],
        "commercial_tier": ["standard"] * 80,
        "detail_path": [""] * 80,
        "montage_path": [None] * 80,
        "has_tar_shard": [False] * 80,
    }
    schema = pa.schema([
        ("study_id", pa.string()),
        ("series_uid", pa.string()),
        ("series_number", pa.int64()),
        ("series_description", pa.string()),
        ("sequence_type", pa.string()),
        ("sequence_confidence", pa.string()),
        ("modality", pa.string()),
        ("file_count", pa.int64()),
        ("tr_ms", pa.float64()),
        ("te_ms", pa.float64()),
        ("ti_ms", pa.float64()),
        ("fa_deg", pa.float64()),
        ("b_value", pa.float64()),
        ("field_strength_T", pa.float64()),
        ("plane", pa.string()),
        ("volume_shape", pa.list_(pa.int64())),
        ("spacing_mm", pa.list_(pa.float64())),
        ("fov_mm", pa.list_(pa.float64())),
        ("volume_snr", pa.float64()),
        ("volume_cnr", pa.float64()),
        ("volume_entropy", pa.float64()),
        ("quality_grade", pa.string()),
        ("quality_score", pa.float64()),
        ("ml_score", pa.float64()),
        ("ml_grade", pa.string()),
        ("commercial_tier", pa.string()),
        ("detail_path", pa.string()),
        ("montage_path", pa.string()),
        ("has_tar_shard", pa.bool_()),
    ])
    table = pa.table(_rows, schema=schema)
    MANIFEST_PATH = _tmp / "manifest.parquet"
    DATASET_ROOT = _tmp
    pq.write_table(table, MANIFEST_PATH)
    print(f"[DEMO] Wrote synthetic manifest to {MANIFEST_PATH}")


# %% Create dataset and dataloader
dataset = SpeallMRIDataset(
    manifest_path=MANIFEST_PATH,
    root=DATASET_ROOT,
    split="train",
    sequence_type="DWI",  # train on DWI only
)
print(f"Training series: {len(dataset)}")

if _TORCH:
    from src.loaders.pytorch import make_dataloader
    loader = make_dataloader(
        manifest=MANIFEST_PATH,
        root=DATASET_ROOT,
        split="train",
        batch_size=BATCH_SIZE,
        num_workers=0,        # set >0 for production
        sequence_type="DWI",
    )
    print(f"DataLoader batches: {len(loader)}")


# %% Define a minimal CNN for quality-grade classification
# Accepts (B, 1, D, H, W) volumes or (B, 1, H, W) 2D slices.

if _TORCH:
    class SmallMRIClassifier(nn.Module):
        """Tiny 3D CNN: grades MRI volumes into A/B/C/D/F (5 classes)."""

        def __init__(self, num_classes: int = 5) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((4, 4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return self.head(self.encoder(x))

    model = SmallMRIClassifier(num_classes=5).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Grade string -> class index mapping
    _GRADE_IDX = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}


# %% Training loop (1 epoch, synthetic volume tensors)
if _TORCH:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        grades = [_GRADE_IDX.get((g or "F").upper(), 0) for g in batch.get("quality_grade", [])]
        labels = torch.tensor(grades, dtype=torch.long).to(DEVICE)

        # Synthetic volume: replace with batch["volume"] when load_volume=True
        B = labels.shape[0]
        volume = torch.randn(B, 1, 16, 64, 64, device=DEVICE)

        optimizer.zero_grad()
        logits = model(volume)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx:3d}/{len(loader)} -- loss: {loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    print(f"Epoch 1 complete. Average loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    torch.save(checkpoint, "speall_mri_checkpoint.pt")
    print("Checkpoint saved: speall_mri_checkpoint.pt")

else:
    print("[no-op] torch not installed -- skipping training.")


# %% Next steps
# -----------------------------------------------------------------------------
# NEXT STEPS:
#   1. Set DATASET_ROOT to your real data path and set load_volume=True
#      in the SpeallMRIDataset constructor to load actual DICOM volumes.
#   2. Replace SmallMRIClassifier with a pretrained backbone (e.g. Med3D,
#      or torchvision 2D + slice-aggregation for lighter compute).
#   3. Add a validation loop; see notebooks/01_cohort_filtering.py for
#      how to filter a grade-A/B validation cohort.
#   4. For segmentation, see notebooks/03_monai_segmentation.py.
# -----------------------------------------------------------------------------
