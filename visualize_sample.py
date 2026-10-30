import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

BASE_DIR = Path("data/raw/sen1floods11")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(
    BASE_DIR / "metadata" / "train_data.csv",
    header=None,
    names=["image", "label"]
).head(20)

# Sample 13
row = df.iloc[12]

image_path = BASE_DIR / "S1Hand" / row["image"]
label_path = BASE_DIR / "LabelHand" / row["label"]

with rasterio.open(image_path) as src:
    image = src.read()

with rasterio.open(label_path) as src:
    label = src.read(1)

vv = image[0]

# Normalize only for visualization
vv_norm = (vv - vv.min()) / (vv.max() - vv.min())

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(vv_norm, cmap="gray")
plt.title("Sentinel-1 VV")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label, cmap="gray")
plt.title("Flood Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(vv_norm, cmap="gray")
plt.imshow(
    np.ma.masked_where(label != 1, label),
    cmap="Reds",
    alpha=0.5
)
plt.title("VV + Flood Overlay")
plt.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "sample_13_flood_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Visualization saved to: {output_path}")