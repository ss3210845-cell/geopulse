import rasterio
import numpy as np
from pathlib import Path

BASE_DIR = Path("data/raw/sen1floods11")

image_path = BASE_DIR / "S1Hand" / "Ghana_103272_S1Hand.tif"
label_path = BASE_DIR / "LabelHand" / "Ghana_103272_LabelHand.tif"

# -----------------------------
# 1. Load image
# -----------------------------

with rasterio.open(image_path) as src:
    image = src.read().astype(np.float32)

# Image shape = (2, 512, 512)
vv = image[0]
vh = image[1]


# -----------------------------
# 2. Load label
# -----------------------------

with rasterio.open(label_path) as src:
    label = src.read(1)


# -----------------------------
# 3. Create valid mask
# -----------------------------

# -1 means ignore
valid_mask = label != -1


# -----------------------------
# 4. Handle NaN values
# -----------------------------

# Find pixels where either VV or VH is NaN
nan_mask = np.isnan(vv) | np.isnan(vh)

# Pixels used for training must:
# 1. Have a valid label
# 2. Have valid satellite measurements

training_mask = valid_mask & ~nan_mask


# -----------------------------
# 5. Normalize VV and VH
# -----------------------------

vv_valid = vv[training_mask]
vh_valid = vh[training_mask]

vv_mean = np.mean(vv_valid)
vv_std = np.std(vv_valid)

vh_mean = np.mean(vh_valid)
vh_std = np.std(vh_valid)

vv_normalized = (vv - vv_mean) / vv_std
vh_normalized = (vh - vh_mean) / vh_std


# -----------------------------
# 6. Replace remaining NaNs
# -----------------------------

# NaNs are not training pixels, so their
# value will not affect training.
vv_normalized = np.nan_to_num(vv_normalized, nan=0.0)
vh_normalized = np.nan_to_num(vh_normalized, nan=0.0)


# -----------------------------
# 7. Print results
# -----------------------------

print("GeoPulse Preprocessing Test")
print("=" * 50)

print("\nOriginal image:")
print("Shape:", image.shape)

print("\nOriginal label:")
print("Shape:", label.shape)

print("\nLabel pixels:")
print("Ignore (-1):", np.sum(label == -1))
print("Non-flood (0):", np.sum(label == 0))
print("Flood (1):", np.sum(label == 1))

print("\nImage NaNs:")
print("NaN pixels:", np.sum(nan_mask))

print("\nTraining pixels:")
print("Valid pixels:", np.sum(training_mask))

print("\nNormalization statistics:")
print("VV mean:", vv_mean)
print("VV std:", vv_std)
print("VH mean:", vh_mean)
print("VH std:", vh_std)

print("\nAfter preprocessing:")
print("VV NaNs:", np.isnan(vv_normalized).sum())
print("VH NaNs:", np.isnan(vh_normalized).sum())

print("\nPreprocessing test complete!")