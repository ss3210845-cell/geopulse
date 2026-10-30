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

vv_sum = 0.0
vv_sum_sq = 0.0
vv_count = 0

vh_sum = 0.0
vh_sum_sq = 0.0
vh_count = 0

print("Calculating Global SAR Statistics")
print("=" * 60)

for i, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]
    label_path = BASE_DIR / "LabelHand" / row["label"]

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    vv = image[0]
    vh = image[1]

    # Only use labeled pixels
    valid_mask = label != -1

    # Also exclude NaN pixels
    vv_mask = valid_mask & ~np.isnan(vv)
    vh_mask = valid_mask & ~np.isnan(vh)

    vv_valid = vv[vv_mask]
    vh_valid = vh[vh_mask]

    vv_sum += np.sum(vv_valid)
    vv_sum_sq += np.sum(vv_valid ** 2)
    vv_count += len(vv_valid)

    vh_sum += np.sum(vh_valid)
    vh_sum_sq += np.sum(vh_valid ** 2)
    vh_count += len(vh_valid)

    print(f"Processed sample {i + 1:2d}/20")

# Calculate mean
vv_mean = vv_sum / vv_count
vh_mean = vh_sum / vh_count

# Calculate standard deviation
vv_std = np.sqrt(
    (vv_sum_sq / vv_count) - (vv_mean ** 2)
)

vh_std = np.sqrt(
    (vh_sum_sq / vh_count) - (vh_mean ** 2)
)

print("\n" + "=" * 60)
print("GLOBAL TRAINING STATISTICS")
print("=" * 60)

print(f"VV pixels used : {vv_count}")
print(f"VV mean        : {vv_mean:.6f}")
print(f"VV std         : {vv_std:.6f}")

print()

print(f"VH pixels used : {vh_count}")
print(f"VH mean        : {vh_mean:.6f}")
print(f"VH std         : {vh_std:.6f}")

print("\nStatistics calculation complete!")