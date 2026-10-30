import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("data/raw/sen1floods11")

image_path = BASE_DIR / "S1Hand" / "Ghana_103272_S1Hand.tif"

with rasterio.open(image_path) as src:
    image = src.read()

vv = image[0]

nan_mask = np.isnan(vv)

# Replace NaN only for visualization
vv_display = np.nan_to_num(vv, nan=0)

plt.figure(figsize=(8, 8))

plt.imshow(vv_display, cmap="gray")

# Highlight NaN regions
plt.imshow(
    np.ma.masked_where(~nan_mask, nan_mask),
    cmap="Reds",
    alpha=0.7
)

plt.title("VV Image + NaN Regions")
plt.axis("off")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "nan_overlay.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Visualization saved to: {output_path}")