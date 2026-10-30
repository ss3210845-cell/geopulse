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

nan_label_0 = 0
nan_label_1 = 0
nan_label_minus1 = 0

for _, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]
    label_path = BASE_DIR / "LabelHand" / row["label"]

    with rasterio.open(image_path) as src:
        image = src.read()

    with rasterio.open(label_path) as src:
        label = src.read(1)

    nan_mask = np.isnan(image[0])

    nan_label_minus1 += np.sum(nan_mask & (label == -1))
    nan_label_0 += np.sum(nan_mask & (label == 0))
    nan_label_1 += np.sum(nan_mask & (label == 1))

print("NaN Pixel Label Distribution")
print("-" * 45)

total = nan_label_minus1 + nan_label_0 + nan_label_1

print("NaN + Label -1:", nan_label_minus1)
print("NaN + Label  0:", nan_label_0)
print("NaN + Label  1:", nan_label_1)

print("\nTotal NaN pixels:", total)

print("\nPercentages:")
print("Label -1:", nan_label_minus1 / total * 100, "%")
print("Label  0:", nan_label_0 / total * 100, "%")
print("Label  1:", nan_label_1 / total * 100, "%")