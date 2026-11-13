"""Sweep decision thresholds over a split and report pooled metrics."""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/aug/best.pth")
    p.add_argument("--base_dir", default="data/raw/sen1floods11")
    p.add_argument("--csv", default="data/raw/sen1floods11/metadata/valid_data.csv")
    p.add_argument("--out", default="outputs/threshold_sweep.json")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = sum(1 for line in open(a.csv) if line.strip())
    ds = FloodDataset(a.csv, a.base_dir, num_samples=n, indices=list(range(n)))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet(in_channels=2).to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()

    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tp = np.zeros(len(ths)); fp = np.zeros(len(ths)); fn = np.zeros(len(ths))

    with torch.no_grad():
        for img, flood, valid in loader:
            prob = torch.sigmoid(model(img.to(device))).squeeze().cpu().numpy()
            v = valid.squeeze().numpy() == 1
            t = flood.squeeze().numpy() == 1
            if v.sum() == 0:
                continue
            for i, th in enumerate(ths):
                pr = prob >= th
                tp[i] += (pr & t & v).sum()
                fp[i] += (pr & ~t & v).sum()
                fn[i] += (~pr & t & v).sum()

    rows = []
    print(f"{'thr':>5} {'prec':>7} {'recall':>7} {'dice':>7} {'iou':>7}")
    for i, th in enumerate(ths):
        prec = tp[i] / max(tp[i] + fp[i], 1)
        rec = tp[i] / max(tp[i] + fn[i], 1)
        dice = 2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1)
        iou = tp[i] / max(tp[i] + fp[i] + fn[i], 1)
        rows.append(dict(threshold=float(th), precision=prec, recall=rec, dice=dice, iou=iou))
        print(f"{th:5.2f} {prec:7.4f} {rec:7.4f} {dice:7.4f} {iou:7.4f}")

    best = max(rows, key=lambda r: r["dice"])
    print(f"\nBest pooled Dice: {best['dice']:.4f} at threshold {best['threshold']}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
