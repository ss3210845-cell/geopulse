import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from flood_dataset import FloodDataset
from unet import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------
THRESHOLD = 0.85
DATA_DIR = "data/raw/sen1floods11"

CSV_FILE = (
    "data/raw/sen1floods11/metadata/valid_data.csv"
)
MODEL_PATH = (
    "models/geopulse_unet_best.pth"
)
device = torch.device("cpu")


# --------------------------------------------------
# Load validation sample
# --------------------------------------------------

dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=DATA_DIR,
    num_samples=89,
    indices=[12, 19, 22, 24]
)
# Select first validation image
image, flood_mask, valid_mask = dataset[0]

print("Validation sample loaded")

print("Image shape:", image.shape)
print("Flood mask shape:", flood_mask.shape)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully")


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

input_image = image.unsqueeze(0)

input_image = input_image.to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    output = model(input_image)

    probability = torch.sigmoid(output)

    prediction = (
    probability >= THRESHOLD
   ).float()


# Remove batch/channel dimensions

probability = probability.squeeze().numpy()

prediction = prediction.squeeze().numpy()

actual_mask = flood_mask.numpy()

valid = valid_mask.numpy()


# --------------------------------------------------
# Calculate flood pixels
# --------------------------------------------------

actual_flood = (
    (actual_mask == 1)
    & (valid == 1)
).sum()

predicted_flood = (
    (prediction == 1)
    & (valid == 1)
).sum()


print()
print("=" * 60)

print("Prediction Results")

print("=" * 60)

print(
    "Actual flood pixels:",
    actual_flood
)

print(
    "Predicted flood pixels:",
    predicted_flood
)

# --------------------------------------------------
# Pixel-level evaluation metrics
# --------------------------------------------------

# Exclude invalid pixels using the valid mask
valid_pixels = valid == 1

actual_binary = actual_mask == 1
predicted_binary = prediction == 1

tp = (predicted_binary & actual_binary & valid_pixels).sum()
fp = (predicted_binary & ~actual_binary & valid_pixels).sum()
fn = (~predicted_binary & actual_binary & valid_pixels).sum()
tn = (~predicted_binary & ~actual_binary & valid_pixels).sum()

precision = tp / (tp + fp) if tp + fp > 0 else 0.0
recall = tp / (tp + fn) if tp + fn > 0 else 0.0
dice = (
    2 * tp / (2 * tp + fp + fn)
    if 2 * tp + fp + fn > 0 else 0.0
)
iou = (
    tp / (tp + fp + fn)
    if tp + fp + fn > 0 else 0.0
)

print("\nPixel-level evaluation")
print("-" * 40)
print(f"True Positives  : {tp}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Negatives  : {tn}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"Dice            : {dice:.4f}")
print(f"IoU             : {iou:.4f}")



# --------------------------------------------------
# Visualization
# --------------------------------------------------

with rasterio.open(
    DATA_DIR + "/S1Hand/" +
    dataset.data.iloc[0]["image"]
) as src:

    sar = src.read()

vv = sar[0]


# Normalize VV for visualization

vv_min = np.nanpercentile(vv, 2)

vv_max = np.nanpercentile(vv, 98)

vv_display = np.clip(
    (vv - vv_min) /
    (vv_max - vv_min),
    0,
    1
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    vv_display,
    cmap="gray"
)

plt.title("SAR VV")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    actual_mask,
    cmap="gray"
)

plt.title("Actual Flood Mask")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    prediction,
    cmap="gray"
)

plt.title("Predicted Flood Mask")

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/prediction_epoch1.png",
    dpi=150
)

plt.show()


print()
print("Prediction visualization saved to:")

print(
    "outputs/prediction_best_threshold_030.png"
)