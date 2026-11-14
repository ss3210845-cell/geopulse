import json
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [("2026-08-28", "outputs/patna_before_mask.tif"),
        ("2026-09-09", "outputs/patna_flood_mask.tif"),
        ("2026-09-21", "outputs/p85_sep21_mask.tif")]

masks = []
for d, p in runs:
    with rasterio.open(p) as s:
        masks.append(s.read(1))
        px = abs(s.transform.a * s.transform.e) / 1e6
valid = np.all([m != 255 for m in masks], axis=0)

rows = []
base = (masks[0] == 1) & valid
for (d, _), m in zip(runs, masks):
    w = (m == 1) & valid
    rows.append({"date": d, "orbit": 85, "water_km2": round(float(w.sum() * px), 1),
                 "new_vs_baseline_km2": round(float((w & ~base).sum() * px), 1)})
    print(rows[-1])
json.dump({"orbit": 85, "valid_km2": round(float(valid.sum() * px), 1), "dates": rows},
          open("outputs/timeline_orbit85.json", "w"), indent=2)

plt.figure(figsize=(7, 4))
plt.bar([r["date"] for r in rows], [r["water_km2"] for r in rows], color="#2b6cb0")
plt.ylabel("Water found by model (km2)")
plt.title("Patna area, Sentinel-1 orbit 85, Aug-Sep 2026")
plt.tight_layout()
plt.savefig("outputs/timeline_orbit85.png", dpi=130)
print("Saved outputs/timeline_orbit85.json and .png")
