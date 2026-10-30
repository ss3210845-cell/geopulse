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

total_vv_nan = 0
total_vh_nan = 0
total_pixels = 0

for _, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]

    with rasterio.open(image_path) as src:
        image = src.read()

    vv = image[0]
    vh = image[1]

    total_vv_nan += np.isnan(vv).sum()
    total_vh_nan += np.isnan(vh).sum()
    total_pixels += vv.size

print("Sentinel-1 NaN Analysis")
print("-" * 45)

print("Total pixels per channel:", total_pixels)

print("\nVV:")
print("NaN pixels:", total_vv_nan)
print("NaN percentage:", total_vv_nan / total_pixels * 100, "%")

print("\nVH:")
print("NaN pixels:", total_vh_nan)
print("NaN percentage:", total_vh_nan / total_pixels * 100, "%")