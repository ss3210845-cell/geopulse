
import pandas as pd
import rasterio
import numpy as np
from pathlib import Path

base_dir = Path("data/raw/sen1floods11")
csv_path = base_dir / "metadata" / "valid_data.csv"

df = pd.read_csv(
    csv_path,
    header=None,
    names=["image", "label"]
)

no_valid = []
no_flood = []
no_nonflood = []
valid_counts = []

for i, row in df.iterrows():

    label_path = base_dir / "LabelHand" / row["label"]

    with rasterio.open(label_path) as src:
        label = src.read(1)

    valid = label != -1
    valid_count = int(valid.sum())

    flood_count = int((label == 1).sum())
    nonflood_count = int((label == 0).sum())

    valid_counts.append(valid_count)

    if valid_count == 0:
        no_valid.append(row["image"])

    if valid_count > 0 and flood_count == 0:
        no_flood.append(row["image"])

    if valid_count > 0 and nonflood_count == 0:
        no_nonflood.append(row["image"])

print("\n" + "=" * 60)
print("GEOPULSE VALIDATION LABEL AUDIT")
print("=" * 60)

print(f"Total validation samples: {len(df)}")
print(f"Samples with zero valid pixels: {len(no_valid)}")
print(f"Samples with no flood pixels: {len(no_flood)}")
print(f"Samples with no non-flood pixels: {len(no_nonflood)}")

print(
    f"Minimum valid pixels: {min(valid_counts):,}"
)
print(
    f"Maximum valid pixels: {max(valid_counts):,}"
)

if no_valid:
    print("\nSamples with zero valid pixels:")
    for name in no_valid:
        print(" ", name)

print("\nLabel audit complete.")
