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

total_no_data = 0
total_non_flood = 0
total_flood = 0

for _, row in df.iterrows():

    label_path = BASE_DIR / "LabelHand" / row["label"]

    with rasterio.open(label_path) as src:
        label = src.read(1)

    total_no_data += np.sum(label == -1)
    total_non_flood += np.sum(label == 0)
    total_flood += np.sum(label == 1)

total_pixels = total_no_data + total_non_flood + total_flood

print("Label Distribution - 20 Samples")
print("-" * 40)

print("No-data (-1):", total_no_data)
print("Non-flood (0):", total_non_flood)
print("Flood (1):   ", total_flood)
print("Total pixels:", total_pixels)

print("\nPercentages")
print("-" * 40)

print("No-data:   ", total_no_data / total_pixels * 100, "%")
print("Non-flood: ", total_non_flood / total_pixels * 100, "%")
print("Flood:     ", total_flood / total_pixels * 100, "%")