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

nan_and_ignore = 0
nan_and_valid = 0

for _, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]
    label_path = BASE_DIR / "LabelHand" / row["label"]

    with rasterio.open(image_path) as src:
        image = src.read()

    with rasterio.open(label_path) as src:
        label = src.read(1)

    # NaN pixels in the VV channel
    nan_mask = np.isnan(image[0])

    # Label -1 pixels
    ignore_mask = label == -1

    nan_and_ignore += np.sum(nan_mask & ignore_mask)
    nan_and_valid += np.sum(nan_mask & ~ignore_mask)

print("NaN vs Label Analysis")
print("-" * 45)

print("NaN pixels with label -1:", nan_and_ignore)
print("NaN pixels with valid label:", nan_and_valid)

print("\nTotal NaN pixels:", nan_and_ignore + nan_and_valid)

print(
    "\nPercentage of NaN pixels with label -1:",
    nan_and_ignore / (nan_and_ignore + nan_and_valid) * 100,
    "%"
)