import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

OUTPUT_DIR = Path("outputs/validation_predictions")

THRESHOLD = 0.85
TOTAL_SAMPLES = 89
NUM_VISUALIZATIONS = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Check paths
# ============================================================

if not Path(CSV_FILE).is_file():
    raise FileNotFoundError(
        f"Validation CSV not found: {CSV_FILE}"
    )

if not Path(MODEL_PATH).is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load validation dataset
# ============================================================

dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=BASE_DIR,
    num_samples=TOTAL_SAMPLES,
    indices=list(range(TOTAL_SAMPLES))
)

if len(dataset) != TOTAL_SAMPLES:
    raise ValueError(
        f"Expected {TOTAL_SAMPLES} validation samples, "
        f"but loaded {len(dataset)}."
    )

print(f"Validation samples loaded: {len(dataset)}")
print(f"Device: {DEVICE}")
print(f"Decision threshold: {THRESHOLD:.2f}")


# ============================================================
# Load best model
# ============================================================

model = UNet()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Best model loaded successfully.")


# ============================================================
# Aggregate confusion-matrix counts
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

total_valid_pixels = 0
total_actual_flood_pixels = 0
total_predicted_flood_pixels = 0


# ============================================================
# Evaluate validation images
# ============================================================

# Keep this counter OUTSIDE the loop so it accumulates across samples.
saved_visualizations = 0

with torch.no_grad():

    for i in range(len(dataset)):

        image, actual_mask_tensor, valid_mask_tensor = dataset[i]

        input_image = image.unsqueeze(0).to(DEVICE)

        logits = model(input_image)
        probabilities = torch.sigmoid(logits)

        predicted_mask = (
            (probabilities >= THRESHOLD)
            .squeeze()
            .cpu()
            .numpy()
            .astype(bool)
        )

        actual_mask = (
            actual_mask_tensor.cpu().numpy() == 1
        )

        valid_mask = (
            valid_mask_tensor.cpu().numpy() == 1
        )

        # ----------------------------------------------------
        # Validate array shapes
        # ----------------------------------------------------

        if not (
            predicted_mask.shape
            == actual_mask.shape
            == valid_mask.shape
        ):
            raise ValueError(
                f"Shape mismatch at sample {i + 1}: "
                f"predicted={predicted_mask.shape}, "
                f"actual={actual_mask.shape}, "
                f"valid={valid_mask.shape}"
            )

        # ----------------------------------------------------
        # Confusion matrix: count only valid pixels
        # ----------------------------------------------------

        actual_flood = actual_mask & valid_mask
        predicted_flood = predicted_mask & valid_mask
        
        if i == 2:
            print("\n--- SAMPLE 3 LABEL DIAGNOSTICS ---")
            print("Image/label:", dataset.data.iloc[i].to_dict())
            print("Actual mask shape:", actual_mask.shape)
            print("Actual mask values and counts:",
                  np.unique(actual_mask, return_counts=True))
            print("Valid mask values and counts:",
                  np.unique(valid_mask, return_counts=True))
            print("Actual flood pixels:", int(actual_mask.sum()))
            print("Valid pixels:", int(valid_mask.sum()))
            print("Flood pixels among valid labels:",
                  int((actual_mask & valid_mask).sum()))


        tp = int(
            (actual_flood & predicted_flood).sum()
        )

        fp = int(
            (~actual_mask & predicted_mask & valid_mask).sum()
        )

        fn = int(
            (actual_mask & ~predicted_mask & valid_mask).sum()
        )

        tn = int(
            (~actual_mask & ~predicted_mask & valid_mask).sum()
        )

        valid_count = int(valid_mask.sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        total_valid_pixels += valid_count

        total_actual_flood_pixels += int(
            actual_flood.sum()
        )

        total_predicted_flood_pixels += int(
            predicted_flood.sum()
        )

        # ----------------------------------------------------
        # Diagnostic output and visualizations
        # ----------------------------------------------------
        # Save only the first NUM_VISUALIZATIONS samples with valid labels.
        if valid_count > 0 and saved_visualizations < NUM_VISUALIZATIONS:

            # Display the first SAR band (VV).
            vv = image[0].cpu().numpy().astype(float)

            print(f"\nSample {i + 1} diagnostics:")
            print("VV shape:", vv.shape)
            print(
                "VV finite pixels:",
                int(np.isfinite(vv).sum())
            )
            print(
                "Valid pixels:",
                int(valid_mask.sum())
            )
            print(
                "Actual mask unique values:",
                np.unique(actual_mask)
            )
            print(
                "Predicted mask unique values:",
                np.unique(predicted_mask)
            )
            print(
                "VV min/max:",
                np.nanmin(vv),
                np.nanmax(vv)
            )

            # Use NaN for invalid pixels.
            vv_display = vv.copy()
            vv_display[
                (~valid_mask) | (~np.isfinite(vv_display))
            ] = np.nan

            actual_display = actual_mask.astype(float)
            actual_display[~valid_mask] = np.nan

            predicted_display = predicted_mask.astype(float)
            predicted_display[~valid_mask] = np.nan

            # ------------------------------------------------
            # Plot three panels
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(15, 5)
            )

            # SAR VV: robust contrast avoids a washed-out image.
            finite_values = vv_display[
                np.isfinite(vv_display)
            ]

            if finite_values.size > 0:
                vmin, vmax = np.percentile(
                    finite_values,
                    [2, 98]
                )

                if vmin >= vmax:
                    vmin = float(finite_values.min())
                    vmax = float(finite_values.max())

                if vmin < vmax:
                    axes[0].imshow(
                        vv_display,
                        cmap="gray",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest"
                    )
                else:
                    axes[0].imshow(
                        vv_display,
                        cmap="gray",
                        interpolation="nearest"
                    )
            else:
                print(
                    "WARNING: No finite VV pixels "
                    f"found in sample {i + 1}."
                )

            axes[0].set_title("SAR VV")

            # Actual flood mask: white = flood, black = non-flood.
            axes[1].imshow(
                np.ma.masked_invalid(actual_display),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            axes[1].set_title("Actual Flood")

            # Predicted flood mask: white = predicted flood.
            axes[2].imshow(
                np.ma.masked_invalid(predicted_display),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            axes[2].set_title(
                f"Predicted Flood (threshold={THRESHOLD:.2f})"
            )

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()

            # Use sequential filenames: validation_sample_1.png to _6.png.
            output_path = (
                OUTPUT_DIR / f"validation_sample_{saved_visualizations + 1}.png"
            )

            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white"
            )

            plt.close(fig)

            print(f"Visualization saved: {output_path}")
            saved_visualizations += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            print(
                f"Evaluated {i + 1}/{len(dataset)} images"
            )


# ============================================================
# Calculate aggregate metrics
# ============================================================

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0.0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0.0
)

dice = (
    2 * total_tp
    / (2 * total_tp + total_fp + total_fn)
    if 2 * total_tp + total_fp + total_fn > 0
    else 0.0
)

iou = (
    total_tp / (total_tp + total_fp + total_fn)
    if total_tp + total_fp + total_fn > 0
    else 0.0
)

actual_flood_percentage = (
    100 * total_actual_flood_pixels / total_valid_pixels
    if total_valid_pixels > 0
    else 0.0
)

predicted_flood_percentage = (
    100 * total_predicted_flood_pixels / total_valid_pixels
    if total_valid_pixels > 0
    else 0.0
)


# ============================================================
# Print final validation report
# ============================================================

print("\n")
print("=" * 65)
print("GEOPULSE - AGGREGATE VALIDATION RESULTS")
print("=" * 65)

print(f"Validation images       : {len(dataset)}")
print(f"Model checkpoint        : {MODEL_PATH}")
print(f"Decision threshold      : {THRESHOLD:.2f}")
print(f"Total valid pixels      : {total_valid_pixels}")

print("\nConfusion Matrix Counts")
print("-" * 65)
print(f"True Positives          : {total_tp}")
print(f"False Positives         : {total_fp}")
print(f"False Negatives         : {total_fn}")
print(f"True Negatives          : {total_tn}")

print("\nAggregate Metrics")
print("-" * 65)
print(f"Precision               : {precision:.4f}")
print(f"Recall                  : {recall:.4f}")
print(f"Dice / F1               : {dice:.4f}")
print(f"IoU                     : {iou:.4f}")

print("\nFlood Area")
print("-" * 65)
print(f"Actual flood pixels     : {total_actual_flood_pixels}")
print(f"Predicted flood pixels  : {total_predicted_flood_pixels}")
print(f"Actual flood percentage : {actual_flood_percentage:.2f}%")
print(f"Predicted flood percent : {predicted_flood_percentage:.2f}%")

print("\nVisualizations saved to:")
print(OUTPUT_DIR.resolve())

print("\nValidation evaluation complete!")
