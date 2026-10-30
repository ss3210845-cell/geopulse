import pandas as pd
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FloodDataset(Dataset):

    def __init__(self, csv_file, base_dir, num_samples=20, indices=None,
                 augment=False, add_diff=False):

        self.base_dir = Path(base_dir)
        self.augment = augment
        self.add_diff = add_diff

        self.data = pd.read_csv(
            csv_file, header=None, names=["image", "label"]
        ).head(num_samples)

        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        # Global statistics (from the first 20 training samples)
        self.vv_mean = -9.564593
        self.vv_std = 2.717001
        self.vh_mean = -15.899832
        self.vh_std = 3.138511

        # Rough stats for the VV-VH difference in dB (to be recomputed later)
        self.diff_mean = 6.33
        self.diff_std = 2.0

    def __len__(self):
        return len(self.data)

    def _augment(self, image, flood_mask, valid_mask):
        # image: (C, H, W); masks: (H, W). Same transform for all three.
        if torch.rand(1).item() < 0.5:
            image, flood_mask, valid_mask = (
                image[:, :, ::-1], flood_mask[:, ::-1], valid_mask[:, ::-1])
        if torch.rand(1).item() < 0.5:
            image, flood_mask, valid_mask = (
                image[:, ::-1, :], flood_mask[::-1, :], valid_mask[::-1, :])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            flood_mask = np.rot90(flood_mask, k, axes=(0, 1))
            valid_mask = np.rot90(valid_mask, k, axes=(0, 1))
        return (np.ascontiguousarray(image),
                np.ascontiguousarray(flood_mask),
                np.ascontiguousarray(valid_mask))

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.base_dir / "S1Hand" / row["image"]
        label_path = self.base_dir / "LabelHand" / row["label"]

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(label_path) as src:
            label = src.read(1)

        vv_raw = image[0]
        vh_raw = image[1]

        valid_mask = (label != -1) & ~np.isnan(vv_raw) & ~np.isnan(vh_raw)

        vv = np.nan_to_num((vv_raw - self.vv_mean) / self.vv_std, nan=0.0)
        vh = np.nan_to_num((vh_raw - self.vh_mean) / self.vh_std, nan=0.0)

        channels = [vv, vh]
        if self.add_diff:
            diff = (vv_raw - vh_raw - self.diff_mean) / self.diff_std
            channels.append(np.nan_to_num(diff, nan=0.0))
        image = np.stack(channels, axis=0).astype(np.float32)

        flood_mask = (label == 1).astype(np.float32)
        valid_mask = valid_mask.astype(np.float32)

        if self.augment:
            image, flood_mask, valid_mask = self._augment(
                image, flood_mask, valid_mask)

        return (torch.tensor(image, dtype=torch.float32),
                torch.tensor(flood_mask, dtype=torch.float32),
                torch.tensor(valid_mask, dtype=torch.float32))
