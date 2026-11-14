"""Remove tiny isolated patches from the change layer."""
import argparse
import numpy as np
import rasterio
from scipy import ndimage

p = argparse.ArgumentParser()
p.add_argument("--inp", default="outputs/patna_change.tif")
p.add_argument("--out", default="outputs/patna_change_clean.tif")
p.add_argument("--min_px", type=int, default=100, help="100 px = 1 ha at 10 m")
a = p.parse_args()

with rasterio.open(a.inp) as s:
    c = s.read(1)
    prof = s.profile
    px_km2 = abs(s.transform.a * s.transform.e) / 1e6

out = c.copy()
names = {1: "Water before", 2: "NEW flood", 3: "Receded"}
for code in (1, 2, 3):
    lab, n = ndimage.label(c == code, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones(lab.shape, np.uint8), lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= a.min_px])
    out[(c == code) & ~keep[lab]] = 0
    print(f"{names[code]:<14}: {(c == code).sum() * px_km2:8.2f} -> {(out == code).sum() * px_km2:8.2f} km2"
          f"  ({n - int(keep.sum() - 1)} small patches removed)")

with rasterio.open(a.out, "w", **prof) as dst:
    dst.write(out, 1)
print("Saved", a.out)
