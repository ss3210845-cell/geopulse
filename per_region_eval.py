"""Per-region and per-tile evaluation on a split, at a fixed threshold."""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/aug/best.pth")
p.add_argument("--base_dir", default="data/raw/sen1floods11")
p.add_argument("--csv", default="data/raw/sen1floods11/metadata/test_data.csv")
p.add_argument("--threshold", type=float, default=0.40)
p.add_argument("--out", default="docs/per_region_test.json")
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
        if v.sum() == 0:
            continue                      # no valid labels: cannot be scored
        pr = prob >= a.threshold
        tiles.append(dict(
            name=ds.data.iloc[i]["image"],
            region=ds.data.iloc[i]["image"].split("_")[0],
            tp=int((pr & t & v).sum()), fp=int((pr & ~t & v).sum()),
            fn=int((~pr & t & v).sum()), valid=int(v.sum())))

def metrics(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return dict(tiles=len(rows),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                dice=2 * tp / max(2 * tp + fp + fn, 1), iou=tp / max(tp + fp + fn, 1))

def tile_dice(r):
    d = 2 * r["tp"] + r["fp"] + r["fn"]
    return None if d == 0 else 2 * r["tp"] / d

by_region = defaultdict(list)
for r in tiles:
    by_region[r["region"]].append(r)

print(f"Threshold {a.threshold} | {len(tiles)} scored tiles | weights {a.weights}\n")
print(f"{'region':<12}{'tiles':>6}{'prec':>8}{'recall':>8}{'dice':>8}{'iou':>8}")
out = {"threshold": a.threshold, "regions": {}}
for reg, rows in sorted(by_region.items(), key=lambda kv: -len(kv[1])):
    m = metrics(rows)
    out["regions"][reg] = m
    print(f"{reg:<12}{m['tiles']:>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['dice']:>8.3f}{m['iou']:>8.3f}")
m = metrics(tiles)
out["pooled"] = m
print(f"{'ALL (pooled)':<12}{m['tiles']:>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['dice']:>8.3f}{m['iou']:>8.3f}")

has_water = [r for r in tiles if r["tp"] + r["fn"] > 0]
no_water = [r for r in tiles if r["tp"] + r["fn"] == 0]
dices = [tile_dice(r) for r in has_water]
print(f"\nTiles with water in the label: {len(has_water)} | per-tile mean Dice {np.mean(dices):.3f}, "
      f"median {np.median(dices):.3f}, worst {min(dices):.3f}")
fp_tiles = [r for r in no_water if r["fp"] > 0]
print(f"Tiles with NO water in the label: {len(no_water)} | of these, {len(fp_tiles)} have false positives "
      f"({sum(r['fp'] for r in fp_tiles):,} px)")
worst = sorted(has_water, key=tile_dice)[:5]
print("\nFive worst tiles (by Dice):")
for r in worst:
    print(f"  {r['name']:<34} dice {tile_dice(r):.3f}  prec {r['tp']/max(r['tp']+r['fp'],1):.2f}  "
          f"recall {r['tp']/max(r['tp']+r['fn'],1):.2f}")
out["per_tile"] = dict(tiles_with_water=len(has_water), mean_dice=float(np.mean(dices)),
                       median_dice=float(np.median(dices)), tiles_without_water=len(no_water),
                       false_positive_tiles=len(fp_tiles))
json.dump(out, open(a.out, "w"), indent=2)
print("\nSaved", a.out)
