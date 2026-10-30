import pandas as pd
import requests
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"

BASE_DIR = Path("data/raw/sen1floods11")
IMAGE_DIR = BASE_DIR / "S1Hand"
LABEL_DIR = BASE_DIR / "LabelHand"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    BASE_DIR / "metadata" / "train_data.csv",
    header=None,
    names=["image", "label"]
)

# First 20 training samples
# First 30 training samples
df = df.head(30)
for _, row in df.iterrows():

    image_name = row["image"]
    label_name = row["label"]

    image_path = IMAGE_DIR / image_name
    label_path = LABEL_DIR / label_name

    image_url = f"{BASE_URL}/S1Hand/{image_name}"
    label_url = f"{BASE_URL}/LabelHand/{label_name}"

    if not image_path.exists():
        print(f"Downloading image: {image_name}")
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        image_path.write_bytes(response.content)
    else:
        print(f"Image already exists: {image_name}")

    if not label_path.exists():
        print(f"Downloading label: {label_name}")
        response = requests.get(label_url, timeout=120)
        response.raise_for_status()
        label_path.write_bytes(response.content)
    else:
        print(f"Label already exists: {label_name}")

print("\nDownload complete!")