"""Run the flood model on a Sentinel-1 GeoTIFF of any size.

Input : GeoTIFF with VV and VH bands (in dB, like Sen1Floods11).
Output: <out>_mask.tif (1 = flood, 0 = no flood, 255 = no data)
        <out>_prob.tif (flood probability), both georeferenced like the input.
"""
import argparse
import numpy as np
import rasterio
import torch

from unet import UNet

# Same constants as flood_dataset.py (the model was trained with these)
VV_MEAN, VV_STD = -9.564593, 2.717001
VH_MEAN, VH_STD = -15.899832, 3.138511


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--weights", default="runs/aug/best.pth")
    p.add_argument("--out", default="outputs/scene")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--vv_band", type=int, default=1)
    p.add_argument("--vh_band", type=int, default=2)
    p.add_argument("--linear", action="store_true",
                   help="input is linear power, not dB: convert with 10*log10")
    return p.parse_args()


def starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def blend_window(tile, overlap):
    r = np.minimum(np.arange(tile), np.arange(tile)[::-1]) + 1.0
    w1 = np.clip(r / max(overlap, 1), 1e-3, 1.0)
    return np.outer(w1, w1).astype(np.float32)


def row_area_km2(src, height):
    """Area of one pixel (km^2) for each row of the scene."""
    t = src.transform
    if src.crs is None:
        print("WARNING: no CRS in file, cannot compute area.")
        return None
    if src.crs.is_geographic:
        lat = t.f + (np.arange(height) + 0.5) * t.e
        dx = abs(t.a) * 111.320 * np.cos(np.radians(lat))
        dy = abs(t.e) * 110.574
        return dx * dy
    if src.crs.linear_units.lower() not in ("metre", "meter"):
        print("WARNING: CRS units are not metres, area may be wrong.")
    return np.full(height, abs(t.a * t.e) / 1e6)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with rasterio.open(a.input) as src:
        vv = src.read(a.vv_band).astype(np.float32)
        vh = src.read(a.vh_band).astype(np.float32)
        profile_crs, profile_tf = src.crs, src.transform
        areas = row_area_km2(src, src.height)
        nodata = src.nodata
    H, W = vv.shape
    print(f"Scene: {W} x {H} px | device: {device}")

    if a.linear:
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

    # pad small scenes up to one tile
    T = a.tile
    ph, pw = max(T - H, 0), max(T - W, 0)
    if ph or pw:
        img = np.pad(img, ((0, 0), (0, ph), (0, pw)))
    PH, PW = img.shape[1:]

    model = UNet(in_channels=2).to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()

    stride = T - a.overlap
    coords = [(y, x) for y in starts(PH, T, stride) for x in starts(PW, T, stride)]
    print(f"Tiles: {len(coords)}")
    win = blend_window(T, a.overlap)
    acc = np.zeros((PH, PW), np.float32)
    wsum = np.zeros((PH, PW), np.float32)

    with torch.no_grad():
        for i in range(0, len(coords), a.batch_size):
            chunk = coords[i:i + a.batch_size]
            batch = np.stack([img[:, y:y + T, x:x + T] for y, x in chunk])
            p = torch.sigmoid(model(torch.from_numpy(batch).to(device)))
            p = p.squeeze(1).cpu().numpy()
            for (y, x), pp in zip(chunk, p):
                acc[y:y + T, x:x + T] += pp * win
                wsum[y:y + T, x:x + T] += win
            print(f"  {min(i + a.batch_size, len(coords))}/{len(coords)} tiles", end="\r")
    print()

    prob = (acc / np.maximum(wsum, 1e-6))[:H, :W]
    mask = (prob >= a.threshold).astype(np.uint8)
    mask[~valid] = 255
    prob_out = prob.copy()
    prob_out[~valid] = np.nan

    base = dict(driver="GTiff", height=H, width=W, count=1,
                crs=profile_crs, transform=profile_tf, compress="lzw")
    with rasterio.open(a.out + "_mask.tif", "w", dtype="uint8", nodata=255, **base) as dst:
        dst.write(mask, 1)
    with rasterio.open(a.out + "_prob.tif", "w", dtype="float32", nodata=np.nan, **base) as dst:
        dst.write(prob_out.astype(np.float32), 1)

    flood = (mask == 1)
    n_valid, n_flood = int(valid.sum()), int(flood.sum())
    print("-" * 50)
    print(f"Valid pixels  : {n_valid:,}")
    print(f"Flood pixels  : {n_flood:,} ({100 * n_flood / max(n_valid, 1):.2f}% of valid)")
    if areas is not None:
        flood_km2 = float((flood * areas[:, None]).sum())
        total_km2 = float((valid * areas[:, None]).sum())
        print(f"Flooded area  : {flood_km2:.2f} km2 of {total_km2:.2f} km2 valid")
    print(f"Saved: {a.out}_mask.tif and {a.out}_prob.tif")


if __name__ == "__main__":
    main()
