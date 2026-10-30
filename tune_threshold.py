

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from flood_dataset import FloodDataset
from unet import UNet


# ============================================================
# Configuration
# ============================================================

TOTAL_SAMPLES = 89

# Evaluate every validation sample (indices are zero-based).
VAL_INDICES = list(range(TOTAL_SAMPLES))

BASE_DIR = "data/raw/sen1floods11"

CSV_FILE = (
    "data/raw/sen1floods11/"
    "metadata/valid_data.csv"
)

MODEL_PATH = "models/geopulse_unet_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.825,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95
]

# ============================================================
# Check files
# ============================================================

if not Path(CSV_FILE).is_file():
    raise FileNotFoundError(f"Validation CSV not found: {CSV_FILE}")

if not Path(MODEL_PATH).is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")


# ============================================================
# Load validation dataset
# ============================================================

print("Loading validation dataset...")

dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=BASE_DIR,
    num_samples=TOTAL_SAMPLES,
    indices=VAL_INDICES
)

if len(dataset) != TOTAL_SAMPLES:
    raise ValueError(
        f"Expected {TOTAL_SAMPLES} samples, "
        f"but loaded {len(dataset)}."
    )

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"Validation images loaded: {len(dataset)}")
print(f"Device: {DEVICE}")


# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

model = UNet().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Generate predictions once
# ============================================================

all_probabilities = []
all_targets = []
all_valid_masks = []

print("\nGenerating predictions for all validation images...")

with torch.no_grad():

    for i, (images, masks, valid_masks) in enumerate(loader):

        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.squeeze(1).cpu().numpy()[0]
        )

        all_targets.append(
            masks.cpu().numpy()[0]
        )

        all_valid_masks.append(
            (valid_masks.cpu().numpy()[0] == 1)
        )

        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            print(f"Processed {i + 1}/{len(dataset)} images")

print("All predictions generated.")


# ============================================================
# Test thresholds using aggregate pixel counts
# ============================================================

print("\nAggregate threshold comparison")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Dice':<12}"
    f"{'IoU':<12}"
    f"{'Flood %':<12}"
)

best_threshold = None
best_dice = -1.0
results = []

for threshold in THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_valid_pixels = 0
    total_predicted_flood = 0
    total_actual_flood = 0

    for probabilities, targets, valid in zip(
        all_probabilities,
        all_targets,
        all_valid_masks
    ):

        predictions = probabilities >= threshold
        actual = targets == 1

        # Exclude invalid / unlabeled pixels.
        predictions_valid = predictions & valid
        actual_valid = actual & valid

        total_tp += int(
            np.sum(predictions_valid & actual_valid)
        )

        total_fp += int(
            np.sum(predictions_valid & ~actual_valid & valid)
        )

        total_fn += int(
            np.sum(~predictions_valid & actual_valid)
        )

        total_valid_pixels += int(np.sum(valid))
        total_predicted_flood += int(np.sum(predictions_valid))
        total_actual_flood += int(np.sum(actual_valid))

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

    predicted_flood_percentage = (
        100 * total_predicted_flood / total_valid_pixels
        if total_valid_pixels > 0 else 0.0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "flood_percentage": predicted_flood_percentage
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{dice:<12.4f}"
        f"{iou:<12.4f}"
        f"{predicted_flood_percentage:<12.2f}"
    )

    if dice > best_dice:
        best_dice = dice
        best_threshold = threshold


# ============================================================
# Report best threshold
# ============================================================

best_result = next(
    result for result in results
    if result["threshold"] == best_threshold
)

print("\n" + "=" * 90)
print("BEST VALIDATION THRESHOLD")
print("=" * 90)

print(f"Best threshold          : {best_threshold:.2f}")
print(f"Aggregate validation Dice: {best_result['dice']:.4f}")
print(f"Precision               : {best_result['precision']:.4f}")
print(f"Recall                  : {best_result['recall']:.4f}")
print(f"IoU                     : {best_result['iou']:.4f}")
print(
    f"Predicted flood area    : "
    f"{best_result['flood_percentage']:.2f}%"
)

print("\nThreshold tuning complete.")
print("The test dataset has not been used.")
