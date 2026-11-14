"""Reusable scene inference (used by the API). Same logic as predict_scene.py."""
import numpy as np
import rasterio
import torch

from unet import UNet
from predict_scene import (starts, blend_window, row_area_km2,
                           VV_MEAN, VV_STD, VH_MEAN, VH_STD)

_MODELS = {}


def load_model(weights, device):
    key = (weights, str(device))
    if key not in _MODELS:
        m = UNet(in_channels=2).to(device)
        m.load_state_dict(torch.load(weights, map_location=device))
        m.eval()
        _MODELS[key] = m
    return _MODELS[key]


def predict_scene(path, weights="runs/aug/best.pth", threshold=0.4, tile=512,
                  overlap=64, batch_size=4, vv_band=1, vh_band=2, linear=False,
                  progress=None):
    """Returns dict: mask (uint8: 1 flood, 0 none, 255 nodata), prob, crs, transform,
    valid_km2, flood_km2, flood_pixels, valid_pixels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with rasterio.open(path) as src:
        vv = src.read(vv_band).astype(np.float32)
        vh = src.read(vh_band).astype(np.float32)
        crs, tf, nodata = src.crs, src.transform, src.nodata
        areas = row_area_km2(src, src.height)
    H, W = vv.shape

    if linear:
        bad = (vv <= 0) | (vh <= 0)
        vv = 10 * np.log10(np.where(vv > 0, vv, 1.0))
        vh = 10 * np.log10(np.where(vh > 0, vh, 1.0))
        vv[bad] = np.nan
        vh[bad] = np.nan
    if nodata is not None:
        vv[vv == nodata] = np.nan
        vh[vh == nodata] = np.nan

    valid = ~np.isnan(vv) & ~np.isnan(vh)
    img = np.stack([np.nan_to_num((vv - VV_MEAN) / VV_STD, nan=0.0),
                    np.nan_to_num((vh - VH_MEAN) / VH_STD, nan=0.0)]).astype(np.float32)

    T = tile
    ph, pw = max(T - H, 0), max(T - W, 0)
    if ph or pw:
        img = np.pad(img, ((0, 0), (0, ph), (0, pw)))
    PH, PW = img.shape[1:]

    model = load_model(weights, device)
    stride = T - overlap
    coords = [(y, x) for y in starts(PH, T, stride) for x in starts(PW, T, stride)]
    win = blend_window(T, overlap)
    acc = np.zeros((PH, PW), np.float32)
    wsum = np.zeros((PH, PW), np.float32)

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([img[:, y:y + T, x:x + T] for y, x in chunk])
            p = torch.sigmoid(model(torch.from_numpy(batch).to(device))).squeeze(1).cpu().numpy()
            for (y, x), pp in zip(chunk, p):
                acc[y:y + T, x:x + T] += pp * win
                wsum[y:y + T, x:x + T] += win
            if progress:
                progress(min(i + batch_size, len(coords)), len(coords))

    prob = (acc / np.maximum(wsum, 1e-6))[:H, :W]
    mask = (prob >= threshold).astype(np.uint8)
    mask[~valid] = 255
    flood = mask == 1

    out = dict(mask=mask, prob=prob, crs=crs, transform=tf,
               valid_pixels=int(valid.sum()), flood_pixels=int(flood.sum()),
               valid_km2=None, flood_km2=None)
    if areas is not None:
        out["flood_km2"] = float((flood * areas[:, None]).sum())
        out["valid_km2"] = float((valid * areas[:, None]).sum())
    return out
