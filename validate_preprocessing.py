import pandas as pd
import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path("data/raw/sen1floods11")

df = pd.read_csv(
    BASE_DIR / "metadata" / "train_data.csv",
    header=None,
    names=["image", "label"]
).head(20)

print("GeoPulse Preprocessing Validation")
print("=" * 70)

total_nan = 0
total_ignore = 0
total_valid = 0
total_flood = 0

successful = 0

for i, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]
    label_path = BASE_DIR / "LabelHand" / row["label"]

    try:

        # Load image
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # Load label
        with rasterio.open(label_path) as src:
            label = src.read(1)

        vv = image[0]
        vh = image[1]

        # Check shape
        if image.shape != (2, 512, 512):
            print(f"Sample {i+1}: WARNING - unexpected image shape")
            continue

        if label.shape != (512, 512):
            print(f"Sample {i+1}: WARNING - unexpected label shape")
            continue

        # NaN mask
        nan_mask = np.isnan(vv) | np.isnan(vh)

        # Valid label pixels
        valid_mask = label != -1

        # Final training mask
        training_mask = valid_mask & ~nan_mask

        # Statistics
        nan_count = np.sum(nan_mask)
        ignore_count = np.sum(label == -1)
        valid_count = np.sum(training_mask)
        flood_count = np.sum(label[training_mask] == 1)

        total_nan += nan_count
        total_ignore += ignore_count
        total_valid += valid_count
        total_flood += flood_count

        successful += 1

        print(
            f"Sample {i+1:2d} | "
            f"NaN: {nan_count:6d} | "
            f"Ignore: {ignore_count:6d} | "
            f"Usable: {valid_count:6d} | "
            f"Flood: {flood_count:6d}"
        )

    except Exception as e:
        print(f"Sample {i+1}: ERROR - {e}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Successfully processed:", successful, "/", len(df))
print("Total NaN pixels:", total_nan)
print("Total ignored pixels:", total_ignore)
print("Total usable pixels:", total_valid)
print("Total flood pixels:", total_flood)

if total_valid > 0:
    print(
        "Flood percentage among usable pixels:",
        total_flood / total_valid * 100,
        "%"
    )

print("\nValidation complete!")