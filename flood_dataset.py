import pandas as pd
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FloodDataset(Dataset):

    def __init__(
        self,
        csv_file,
        base_dir,
        num_samples=20,
        indices=None
    ):

        self.base_dir = Path(base_dir)

        self.data = pd.read_csv(
            csv_file,
            header=None,
            names=["image", "label"]
        ).head(num_samples)

        # Select specific samples if indices are provided
        if indices is not None:
            self.data = self.data.iloc[
                indices
            ].reset_index(drop=True)

        # Global statistics calculated from
        # the first 20 training samples
        self.vv_mean = -9.564593
        self.vv_std = 2.717001

        self.vh_mean = -15.899832
        self.vh_std = 3.138511

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.base_dir
            / "S1Hand"
            / row["image"]
        )

        label_path = (
            self.base_dir
            / "LabelHand"
            / row["label"]
        )

        # ------------------------------------------
        # Read SAR image
        # ------------------------------------------

        with rasterio.open(image_path) as src:

            image = src.read().astype(
                np.float32
            )

        # ------------------------------------------
        # Read flood label
        # ------------------------------------------

        with rasterio.open(label_path) as src:

            label = src.read(1)

        vv = image[0]
        vh = image[1]

        # ------------------------------------------
        # Valid pixel mask
        # ------------------------------------------

        valid_mask = (
            (label != -1)
            & ~np.isnan(vv)
            & ~np.isnan(vh)
        )

        # ------------------------------------------
        # Normalize SAR channels
        # ------------------------------------------

        vv = (
            vv - self.vv_mean
        ) / self.vv_std

        vh = (
            vh - self.vh_mean
        ) / self.vh_std

        # ------------------------------------------
        # Replace remaining NaN values
        # ------------------------------------------

        vv = np.nan_to_num(
            vv,
            nan=0.0
        )

        vh = np.nan_to_num(
            vh,
            nan=0.0
        )

        # ------------------------------------------
        # Stack VV + VH
        # ------------------------------------------

        image = np.stack(
            [vv, vh],
            axis=0
        )

        # ------------------------------------------
        # Create flood mask
        # ------------------------------------------

        flood_mask = (
            label == 1
        ).astype(np.float32)

        valid_mask = (
            valid_mask.astype(np.float32)
        )

        # ------------------------------------------
        # Convert to PyTorch tensors
        # ------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        flood_mask = torch.tensor(
            flood_mask,
            dtype=torch.float32
        )

        valid_mask = torch.tensor(
            valid_mask,
            dtype=torch.float32
        )

        return (
            image,
            flood_mask,
            valid_mask
        )