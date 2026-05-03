# %% [markdown]
"""Notebook 03 -- MONAI Brain Segmentation Skeleton

Demonstrates using ``to_monai_dict_dataset`` to bridge Speall MRI manifest
data into MONAI's standard dict-dataset pipeline, then training a MONAI UNet
for brain segmentation.  Synthetic labels are used so the skeleton runs
without manual annotations -- swap in your own segmentation masks.

    jupytext --to notebook notebooks/03_monai_segmentation.py

Prerequisites
-------------
    pip install "micom[monai]" pyarrow
    # or individually:
    pip install monai torch nibabel pyarrow
"""

# %% Setup
# pip install "micom[monai]" pyarrow

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# %% Check framework availability
_MONAI_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
    print(f"torch {torch.__version__}")
except ImportError:
    print("torch not installed -- running no-op demo. pip install torch")

try:
    import monai
    from monai.data import Dataset, DataLoader
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.losses import DiceLoss
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Spacingd,
        ScaleIntensityRanged,
        RandFlipd,
        RandRotate90d,
        ToTensord,
        EnsureTyped,
    )
    _MONAI_AVAILABLE = True
    print(f"monai {monai.__version__}")
except ImportError:
    print("monai not installed -- running no-op demo. pip install monai")

# %% Import helpers
from src.loaders.monai import to_monai_dict_dataset, recommend_transforms  # noqa: E402

# %% Configuration
DATASET_ROOT = Path("/data/speall-mri")   # <-- change this
MANIFEST_PATH = DATASET_ROOT / "manifest.parquet"
BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
SEQUENCE = "FLAIR"     # FLAIR is a natural choice for brain segmentation

DEVICE = "mps" if (_TORCH_AVAILABLE and torch.backends.mps.is_available()) else "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
print(f"Device: {DEVICE}")

# %% Show recommended transforms for this sequence
transforms_code = recommend_transforms(SEQUENCE)
print(f"\nRecommended transforms for {SEQUENCE}:")
for line in transforms_code:
    print(f"  {line}")

# %% Build data dicts (or use synthetic data for demo)
_DEMO_MODE = not MANIFEST_PATH.exists()

if _DEMO_MODE:
    print("\n[DEMO] manifest.parquet not found -- using synthetic data_dicts.")
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    import random

    random.seed(1)
    _tmp = Path(tempfile.mkdtemp())
    n = 40
    _rows = {
        "study_id": [f"study_{i//4:04d}" for i in range(n)],
        "series_uid": [f"uid_{i:06d}" for i in range(n)],
        "series_number": list(range(n)),
        "series_description": ["Ax T2 FLAIR"] * n,
        "sequence_type": ["FLAIR"] * n,
        "sequence_confidence": ["high"] * n,
        "modality": ["MR"] * n,
        "file_count": [40] * n,
        "tr_ms": [9000.0] * n,
        "te_ms": [90.0] * n,
        "ti_ms": [2500.0] * n,
        "fa_deg": [90.0] * n,
        "b_value": [None] * n,
        "field_strength_T": [3.0] * n,
        "plane": ["axial"] * n,
        "volume_shape": [[40, 256, 256]] * n,
        "spacing_mm": [[1.0, 1.0, 3.0]] * n,
        "fov_mm": [[256.0, 256.0, 120.0]] * n,
        "volume_snr": [random.uniform(0.5, 0.9) for _ in range(n)],
        "volume_cnr": [random.uniform(800, 2500) for _ in range(n)],
        "volume_entropy": [random.uniform(1.5, 2.5) for _ in range(n)],
        "quality_grade": [["A", "A", "B", "B"][i % 4] for i in range(n)],
        "quality_score": [random.uniform(65, 92) for _ in range(n)],
        "ml_score": [random.uniform(70, 95) for _ in range(n)],
        "ml_grade": [["A", "A", "B", "B"][i % 4] for i in range(n)],
        "commercial_tier": ["premium"] * n,
        "detail_path": [""] * n,
        "montage_path": [None] * n,
        "has_tar_shard": [False] * n,
    }
    schema = pa.schema([
        ("study_id", pa.string()), ("series_uid", pa.string()),
        ("series_number", pa.int64()), ("series_description", pa.string()),
        ("sequence_type", pa.string()), ("sequence_confidence", pa.string()),
        ("modality", pa.string()), ("file_count", pa.int64()),
        ("tr_ms", pa.float64()), ("te_ms", pa.float64()),
        ("ti_ms", pa.float64()), ("fa_deg", pa.float64()),
        ("b_value", pa.float64()), ("field_strength_T", pa.float64()),
        ("plane", pa.string()), ("volume_shape", pa.list_(pa.int64())),
        ("spacing_mm", pa.list_(pa.float64())), ("fov_mm", pa.list_(pa.float64())),
        ("volume_snr", pa.float64()), ("volume_cnr", pa.float64()),
        ("volume_entropy", pa.float64()), ("quality_grade", pa.string()),
        ("quality_score", pa.float64()), ("ml_score", pa.float64()),
        ("ml_grade", pa.string()), ("commercial_tier", pa.string()),
        ("detail_path", pa.string()), ("montage_path", pa.string()),
        ("has_tar_shard", pa.bool_()),
    ])
    pq.write_table(pa.table(_rows, schema=schema), _tmp / "manifest.parquet")
    MANIFEST_PATH = _tmp / "manifest.parquet"
    DATASET_ROOT = _tmp

data_dicts = to_monai_dict_dataset(
    manifest_path=MANIFEST_PATH,
    root=DATASET_ROOT,
    split="train",
    sequence_type=SEQUENCE,
    quality_grades=["A", "B"],
)
print(f"\nCohort: {len(data_dicts)} {SEQUENCE} series (grade A/B)")

if data_dicts:
    print("Example record keys:", list(data_dicts[0].keys()))
    print("Example image path:", data_dicts[0]["image"])
    print("Example label:", data_dicts[0]["label"])

# %% Define MONAI transforms
if _MONAI_AVAILABLE:
    # For demo: skip LoadImaged since paths are synthetic.
    # In production, set keys=["image", "label"] and ensure label is a NIfTI mask.
    train_transforms = Compose([
        EnsureTyped(keys=["label"]),
    ])

    # Full production transform pipeline (requires real NIfTI files):
    _production_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandRotate90d(keys=["image"], prob=0.3, max_k=3),
        ToTensord(keys=["image"]),
    ])
    print("\nFull production transforms (requires real NIfTI):")
    for t in transforms_code:
        print(f"  {t}")


# %% Build MONAI Dataset and DataLoader
if _MONAI_AVAILABLE and data_dicts:
    monai_dataset = Dataset(data=data_dicts, transform=train_transforms)
    monai_loader = DataLoader(monai_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"\nMONAI DataLoader: {len(monai_loader)} batches")


# %% Define MONAI UNet
if _MONAI_AVAILABLE and _TORCH_AVAILABLE:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,           # background + brain
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(DEVICE)
    print(f"UNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %% Training loop skeleton (synthetic volumes -- swap for real data)
if _MONAI_AVAILABLE and _TORCH_AVAILABLE and data_dicts:
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_i, batch in enumerate(monai_loader):
            B = len(batch["label"])
            # Synthetic inputs: (B, 1, D, H, W) -- replace with batch["image"] for real data
            volume = torch.randn(B, 1, 32, 64, 64, device=DEVICE)
            # Synthetic binary mask (0/1)
            mask = torch.randint(0, 2, (B, 1, 32, 64, 64), dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            pred = model(volume)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_i % 5 == 0:
                print(f"  Epoch {epoch+1} batch {batch_i:3d}/{len(monai_loader)} -- Dice loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -- avg loss: {epoch_loss / max(len(monai_loader), 1):.4f}")

    torch.save(model.state_dict(), "speall_unet_weights.pt")
    print("Saved: speall_unet_weights.pt")

elif not _MONAI_AVAILABLE or not _TORCH_AVAILABLE:
    print("[no-op] monai or torch not installed -- skipping training loop.")


# %% Next steps
# -----------------------------------------------------------------------------
# NEXT STEPS:
#   1. Provide real NIfTI segmentation masks and add "label" paths to each
#      data_dict: data_dict["label"] = str(path_to_mask.nii.gz)
#   2. Add the segmentation mask path to LoadImaged: keys=["image", "label"]
#   3. Use monai.metrics.DiceMetric for per-class validation metrics.
#   4. For multi-class segmentation (tumour, white matter, etc.), set
#      out_channels to the number of classes and update DiceLoss accordingly.
#   5. Export the trained model to ONNX for clinical deployment:
#          torch.onnx.export(model, sample_input, "speall_unet.onnx")
# -----------------------------------------------------------------------------
