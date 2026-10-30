


import torch
import numpy as np
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet


# ============================================================
# Configuration
# ============================================================

BASE_DIR = "data/raw/sen1floods11"

CSV_FILE = (
    "data/raw/sen1floods11/"
    "metadata/valid_data.csv"
)

MODEL_PATH = "models/geopulse_unet_best.pth"

THRESHOLD = 0.85

DEVICE = torch.device("cpu")


# ============================================================
# Load complete validation dataset
# ============================================================

print("Loading complete validation dataset...")

dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=BASE_DIR,
    num_samples=1000,
    indices=None
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples loaded: {len(dataset)}")


# ============================================================
# Load trained model
# ============================================================

print("\nLoading best model...")

model = UNet().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")
print(f"Decision threshold: {THRESHOLD}")


# ============================================================
# Evaluate all validation samples
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0
total_valid_pixels = 0

print("\nEvaluating validation dataset...")

with torch.no_grad():

    for images, masks, valid_masks in loader:

        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).squeeze(1).cpu().numpy()[0].astype(bool)

        actual = masks.numpy()[0].astype(bool)
        valid = valid_masks.numpy()[0].astype(bool)

        # Exclude invalid pixels
        predictions = predictions[valid]
        actual = actual[valid]

        tp = int(np.sum(predictions & actual))
        fp = int(np.sum(predictions & ~actual))
        fn = int(np.sum(~predictions & actual))
        tn = int(np.sum(~predictions & ~actual))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        total_valid_pixels += len(actual)


# ============================================================
# Calculate dataset-wide metrics
# ============================================================

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0 else 0.0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0 else 0.0
)

dice = (
    2 * total_tp / (2 * total_tp + total_fp + total_fn)
    if 2 * total_tp + total_fp + total_fn > 0 else 0.0
)

iou = (
    total_tp / (total_tp + total_fp + total_fn)
    if total_tp + total_fp + total_fn > 0 else 0.0
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("GEOPULSE: COMPLETE VALIDATION EVALUATION")
print("=" * 60)

print(f"Samples evaluated : {len(dataset)}")
print(f"Threshold         : {THRESHOLD:.2f}")
print(f"Valid pixels      : {total_valid_pixels:,}")

print("-" * 60)

print(f"True Positives    : {total_tp:,}")
print(f"False Positives   : {total_fp:,}")
print(f"False Negatives   : {total_fn:,}")
print(f"True Negatives    : {total_tn:,}")

print("-" * 60)

print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"Dice              : {dice:.4f}")
print(f"IoU               : {iou:.4f}")

print("=" * 60)
print("Validation evaluation complete!")
