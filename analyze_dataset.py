import pandas as pd
import rasterio
from pathlib import Path
import numpy as np

BASE_DIR = Path("data/raw/sen1floods11")

df = pd.read_csv(
    BASE_DIR / "metadata" / "train_data.csv",
    header=None,
    names=["image", "label"]
).head(20)

print("Sample | Image Shape | Flood Pixels | Valid Pixels | Flood %")
print("-" * 70)

for i, row in df.iterrows():

    image_path = BASE_DIR / "S1Hand" / row["image"]
    label_path = BASE_DIR / "LabelHand" / row["label"]

    with rasterio.open(image_path) as src:
        image_shape = src.shape
        bands = src.count

    with rasterio.open(label_path) as src:
        label = src.read(1)

    flood_pixels = np.sum(label == 1)
    valid_pixels = np.sum(label != -1)

    flood_percentage = (flood_pixels / valid_pixels) * 100

    print(
        f"{i+1:2} | "
        f"({bands}, {image_shape[0]}, {image_shape[1]}) | "
        f"{flood_pixels:12} | "
        f"{valid_pixels:12} | "
        f"{flood_percentage:6.2f}%"
    )