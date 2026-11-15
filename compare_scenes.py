"""Compare before/flood masks: area of new flood, permanent water, receded water, plus a map."""
import argparse
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument("--before_mask", default="outputs/patna_before_mask.tif")
p.add_argument("--flood_mask", default="outputs/patna_flood_mask.tif")
p.add_argument("--before_sar", default="data/scenes/patna_20260828_o85.tif")
p.add_argument("--flood_sar", default="data/scenes/patna_20260909_o85.tif")
p.add_argument("--out", default="outputs/patna_change")
p.add_argument("--decimate", type=int, default=6)
p.add_argument("--before_label", default="28 Aug 2026")
p.add_argument("--flood_label", default="9 Sep 2026")
a = p.parse_args()

with rasterio.open(a.before_mask) as s:
    b = s.read(1)
    prof, tf, crs = s.profile, s.transform, s.crs
with rasterio.open(a.flood_mask) as s:
    f = s.read(1)

valid = (b != 255) & (f != 255)
bw, fw = (b == 1) & valid, (f == 1) & valid

# 0 = no water, 1 = permanent water, 2 = NEW flood, 3 = receded
change = np.zeros(b.shape, np.uint8)
change[bw & fw] = 1
change[~bw & fw] = 2
change[bw & ~fw] = 3
change[~valid] = 255

px_km2 = abs(tf.a * tf.e) / 1e6
for name, code in [("Permanent water", 1), ("NEW flood", 2), ("Receded water", 3)]:
    print(f"{name:<16}: {(change == code).sum() * px_km2:8.2f} km2")
print(f"{'Water, before':<16}: {bw.sum() * px_km2:8.2f} km2")
print(f"{'Water, flood':<16}: {fw.sum() * px_km2:8.2f} km2")
print(f"{'Net change':<16}: {(fw.sum() - bw.sum()) * px_km2:8.2f} km2")

prof.update(dtype="uint8", nodata=255, count=1, compress="lzw")
with rasterio.open(a.out + ".tif", "w", **prof) as dst:
    dst.write(change, 1)
print("Saved", a.out + ".tif")

d = a.decimate
def vv(path):
    with rasterio.open(path) as s:
        arr = s.read(1, out_shape=(s.height // d, s.width // d))
    arr = np.where(arr == -9999.0, np.nan, arr)
    lo, hi = np.nanpercentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo), 0, 1)

cols = ["#f2f2f2", "#2b6cb0", "#e53e3e", "#38a169"]
cm = ListedColormap(cols)
ch_small = change[::d, ::d].astype(float)
ch_small[ch_small == 255] = np.nan

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(vv(a.before_sar), cmap="gray"); ax[0].set_title(f"Sentinel-1 VV  {a.before_label} (before)")
ax[1].imshow(vv(a.flood_sar), cmap="gray");  ax[1].set_title(f"Sentinel-1 VV  {a.flood_label} (flood)")
ax[2].imshow(ch_small, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
ax[2].set_title("Model: water change")
ax[2].legend(handles=[Patch(color=cols[1], label=f"Water on {a.before_label} (before)"),
                      Patch(color=cols[2], label="New flood"),
                      Patch(color=cols[3], label="Receded")],
             loc="lower left", fontsize=9)
for x in ax: x.axis("off")
plt.tight_layout()
plt.savefig(a.out + ".png", dpi=130)
print("Saved", a.out + ".png")
