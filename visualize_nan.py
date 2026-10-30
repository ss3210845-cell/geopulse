import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("data/raw/sen1floods11")

image_path = BASE_DIR / "S1Hand" / "Ghana_103272_S1Hand.tif"
label_path = BASE_DIR / "LabelHand" / "Ghana_103272_LabelHand.tif"

# Read satellite image
with rasterio.open(image_path) as src:
    image = src.read()

# Read flood label
with rasterio.open(label_path) as src:
    label = src.read(1)

vv = image[0]

# Find NaN pixels
nan_mask = np.isnan(vv)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. VV image
axes[0].imshow(vv, cmap="gray")
axes[0].set_title("Sentinel-1 VV")
axes[0].axis("off")

# 2. NaN mask
axes[1].imshow(nan_mask, cmap="gray")
axes[1].set_title("NaN Pixels")
axes[1].axis("off")

# 3. Flood label
axes[2].imshow(label, cmap="gray")
axes[2].set_title("Flood Label")
axes[2].axis("off")

plt.tight_layout()

# Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "nan_visualization.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Visualization saved to: {output_path}")