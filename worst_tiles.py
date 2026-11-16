"""Plot the worst test tiles (and a few from one region): radar, label, prediction, errors."""
import argparse
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/aug/best.pth")
p.add_argument("--base_dir", default="data/raw/sen1floods11")
p.add_argument("--csv", default="data/raw/sen1floods11/metadata/test_data.csv")
p.add_argument("--threshold", type=float, default=0.40)
p.add_argument("--worst", type=int, default=5)
p.add_argument("--region", default="Pakistan")
p.add_argument("--region_n", type=int, default=3)
p.add_argument("--out", default="docs/worst_tiles.png")
a = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = sum(1 for line in open(a.csv) if line.strip())
ds = FloodDataset(a.csv, a.base_dir, num_samples=n, indices=list(range(n)))
loader = DataLoader(ds, batch_size=1, shuffle=False)
model = UNet(in_channels=2).to(device)
model.load_state_dict(torch.load(a.weights, map_location=device))
model.eval()

tiles = []
with torch.no_grad():
    for i, (img, flood, valid) in enumerate(loader):
        prob = torch.sigmoid(model(img.to(device))).squeeze().cpu().numpy()
        v = valid.squeeze().numpy() == 1
        t = flood.squeeze().numpy() == 1
        if v.sum() == 0 or (t & v).sum() == 0:
            continue                       # only tiles that have water in the label
        pr = prob >= a.threshold
        tp, fp, fn = (pr & t & v).sum(), (pr & ~t & v).sum(), (~pr & t & v).sum()
        tiles.append(dict(name=ds.data.iloc[i]["image"], t=t, v=v, pr=pr,
                          dice=2 * tp / (2 * tp + fp + fn)))

tiles.sort(key=lambda r: r["dice"])
sel = tiles[:a.worst]
extra = [r for r in tiles if r["name"].startswith(a.region + "_") and r not in sel][:a.region_n]
sel += extra


def vv_img(name):
    with rasterio.open(f"{a.base_dir}/S1Hand/{name}") as s:
        vv = s.read(1).astype(np.float32)
    lo, hi = np.nanpercentile(vv, [2, 98])
    return np.clip((np.nan_to_num(vv, nan=lo) - lo) / (hi - lo + 1e-6), 0, 1)


def rgb(mask, v, color):
    out = np.ones(mask.shape + (3,), np.float32)
    out[mask & v] = color
    out[~v] = (0.7, 0.7, 0.7)
    return out


def err_img(t, pr, v):
    out = np.ones(t.shape + (3,), np.float32)
    out[t & pr & v] = (0.22, 0.63, 0.34)      # hit (green)
    out[~t & pr & v] = (0.90, 0.24, 0.24)     # false alarm (red)
    out[t & ~pr & v] = (0.95, 0.60, 0.10)     # missed water (orange)
    out[~v] = (0.7, 0.7, 0.7)
    return out


print(f"{'tile':<34}{'dice':>6}{'valid%':>8}{'label water%':>14}{'predicted%':>12}")
fig, ax = plt.subplots(len(sel), 4, figsize=(14, 3.3 * len(sel)))
ax = np.atleast_2d(ax)
for k, r in enumerate(sel):
    v, t, pr = r["v"], r["t"], r["pr"]
    lw, pw = 100 * (t & v).sum() / v.sum(), 100 * (pr & v).sum() / v.sum()
    print(f"{r['name']:<34}{r['dice']:>6.3f}{100 * v.mean():>8.1f}{lw:>14.1f}{pw:>12.1f}")
    panels = [(vv_img(r["name"]), "gray"), (rgb(t, v, (0.17, 0.42, 0.69)), None),
              (rgb(pr, v, (0.17, 0.42, 0.69)), None), (err_img(t, pr, v), None)]
    titles = ["Radar VV", "Label (blue = water, grey = no data)", "Prediction",
              "Green hit, red false alarm, orange missed"]
    for j, ((im, cm), ttl) in enumerate(zip(panels, titles)):
        ax[k, j].imshow(im[::2, ::2], cmap=cm)
        ax[k, j].axis("off")
        if k == 0:
            ax[k, j].set_title(ttl, fontsize=9)
    ax[k, 0].text(4, 20, f"{r['name'].replace('_S1Hand.tif', '')}  Dice {r['dice']:.2f}",
                  color="yellow", fontsize=9, backgroundcolor="black")
plt.tight_layout()
plt.savefig(a.out, dpi=70)
print("\nSaved", a.out)
