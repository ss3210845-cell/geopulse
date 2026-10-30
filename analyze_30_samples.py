import pandas as pd
import rasterio
import numpy as np
from pathlib import Path


BASE_DIR = Path("data/raw/sen1floods11")

CSV_FILE = (
    BASE_DIR / "metadata" / "train_data.csv"
)

IMAGE_DIR = BASE_DIR / "S1Hand"
LABEL_DIR = BASE_DIR / "LabelHand"


# --------------------------------------------------
# Load first 30 training samples
# --------------------------------------------------

df = pd.read_csv(
    CSV_FILE,
    header=None,
    names=["image", "label"]
).head(30)


total_flood = 0
total_valid = 0
total_ignore = 0
flood_images = 0


print()
print("GeoPulse — 30 Sample Flood Analysis")
print("=" * 70)

print(
    f"{'Sample':<8}"
    f"{'Flood':>12}"
    f"{'Valid':>12}"
    f"{'Ignore':>12}"
    f"{'Flood %':>12}"
)

print("-" * 70)


for index, row in df.iterrows():

    label_path = LABEL_DIR / row["label"]

    with rasterio.open(label_path) as src:
        label = src.read(1)

    flood_pixels = np.sum(label == 1)
    valid_pixels = np.sum(label != -1)
    ignore_pixels = np.sum(label == -1)

    if flood_pixels > 0:
        flood_images += 1

    total_flood += flood_pixels
    total_valid += valid_pixels
    total_ignore += ignore_pixels

    if valid_pixels > 0:
        flood_percentage = (
            flood_pixels / valid_pixels
        ) * 100
    else:
        flood_percentage = 0.0

    print(
        f"{index + 1:<8}"
        f"{flood_pixels:>12}"
        f"{valid_pixels:>12}"
        f"{ignore_pixels:>12}"
        f"{flood_percentage:>11.2f}%"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 70)

print("SUMMARY")
print("=" * 70)

print(
    "Total samples:",
    len(df)
)

print(
    "Images containing flood:",
    flood_images
)

print(
    "Images without flood:",
    len(df) - flood_images
)

print(
    "Total flood pixels:",
    total_flood
)

print(
    "Total valid pixels:",
    total_valid
)

print(
    "Total ignored pixels:",
    total_ignore
)

if total_valid > 0:

    overall_flood_percentage = (
        total_flood / total_valid
    ) * 100

    print(
        "Overall flood percentage:",
        f"{overall_flood_percentage:.2f}%"
    )

print()
print("Analysis complete!")