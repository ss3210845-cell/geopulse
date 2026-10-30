import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet
from loss_functions import CombinedLoss
from metrics import calculate_dice, calculate_iou


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_rows(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="data/raw/sen1floods11")
    p.add_argument("--train_csv", default="data/raw/sen1floods11/metadata/train_data.csv")
    p.add_argument("--val_csv", default="data/raw/sen1floods11/metadata/valid_data.csv")
    p.add_argument("--max_train", type=int, default=None, help="limit train tiles (for quick tests)")
    p.add_argument("--max_val", type=int, default=None, help="limit val tiles (for quick tests)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--flood_weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--augment", action="store_true", help="random flips/rotations on train tiles")
    p.add_argument("--add_diff", action="store_true", help="add VV-VH difference as 3rd channel")
    p.add_argument("--out_dir", default="runs/baseline")
    return p.parse_args()


def make_dataset(csv_file, base_dir, limit, augment=False, add_diff=False):
    n = count_rows(csv_file)
    if limit is not None:
        n = min(n, limit)
    return FloodDataset(
        csv_file=csv_file,
        base_dir=base_dir,
        num_samples=n,
        indices=list(range(n)),
        augment=augment,
        add_diff=add_diff,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = make_dataset(args.train_csv, args.base_dir, args.max_train,
                            augment=args.augment, add_diff=args.add_diff)
    val_ds = make_dataset(args.val_csv, args.base_dir, args.max_val,
                          augment=False, add_diff=args.add_diff)
    print(f"Device: {device} | train tiles: {len(train_ds)} | val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    model = UNet(in_channels=3 if args.add_diff else 2).to(device)
    criterion = CombinedLoss(dice_weight=args.dice_weight, flood_weight=args.flood_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "seconds"])

    best_dice, best_epoch = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for images, masks, valid_masks in train_loader:
            images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
            if valid_masks.sum() == 0:
                continue
            loss = criterion(model(images), masks, valid_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = dice_sum = iou_sum = 0.0
        counted = 0
        with torch.no_grad():
            for images, masks, valid_masks in val_loader:
                images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
                if valid_masks.sum() == 0:
                    continue  # tiles with no valid labels cannot be scored
                outputs = model(images)
                val_loss += criterion(outputs, masks, valid_masks).item()
                dice_sum += float(calculate_dice(outputs, masks, valid_masks))
                iou_sum += float(calculate_iou(outputs, masks, valid_masks))
                counted += 1
        counted = max(counted, 1)
        val_loss, val_dice, val_iou = val_loss / counted, dice_sum / counted, iou_sum / counted

        secs = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| Dice {val_dice:.4f} | IoU {val_iou:.4f} | {secs:.0f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_dice, val_iou, round(secs, 1)])

        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pth"))
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"  new best (Dice {best_dice:.4f})")

    print(f"Done. Best epoch {best_epoch}, Dice {best_dice:.4f}, saved to {args.out_dir}/best.pth")


if __name__ == "__main__":
    main()
