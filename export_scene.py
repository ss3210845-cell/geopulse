"""Download a Sentinel-1 scene (VV, VH in dB) from Earth Engine as one GeoTIFF."""
import argparse, io, os, zipfile
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from rasterio.warp import transform_bounds

p = argparse.ArgumentParser()
p.add_argument("--aoi", nargs=4, type=float, default=[84.9, 25.45, 85.5, 25.85])
p.add_argument("--orbit", type=int, required=True)
p.add_argument("--date", required=True, help="YYYY-MM-DD")
p.add_argument("--out", required=True)
p.add_argument("--crs", default="EPSG:32645")   # UTM zone 45N (Patna)
p.add_argument("--tile_px", type=int, default=1200)
a = p.parse_args()

ee.Initialize(project="geopulse-514608")
NODATA = -9999.0

# AOI in UTM metres, snapped to the 10 m grid
w, s, e, n = transform_bounds("EPSG:4326", a.crs, *a.aoi)
w, s = np.floor(w / 10) * 10, np.floor(s / 10) * 10
e, n = np.ceil(e / 10) * 10, np.ceil(n / 10) * 10
W, H = int((e - w) / 10), int((n - s) / 10)
print(f"Scene size: {W} x {H} px (about {W * H * 8 / 1e6:.0f} MB)")

d = ee.Date(a.date)
aoi = ee.Geometry.Rectangle(a.aoi)
col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
       .filterDate(d, d.advance(1, "day"))
       .filter(ee.Filter.eq("instrumentMode", "IW"))
       .filter(ee.Filter.eq("relativeOrbitNumber_start", a.orbit))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
print("Source frames:", col.size().getInfo())
img = col.select(["VV", "VH"]).mosaic().unmask(NODATA).toFloat()

step = a.tile_px * 10
tiles = []
x = w
while x < e:
    y = s
    while y < n:
        tiles.append((x, y, min(x + step, e), min(y + step, n)))
        y += step
    x += step
print("Tiles to download:", len(tiles))

tmp = a.out + "_tiles"
os.makedirs(tmp, exist_ok=True)


def get(i):
    path = f"{tmp}/t{i:03d}.tif"
    if os.path.exists(path):
        return path
    b = tiles[i]
    region = ee.Geometry.Rectangle(list(b), a.crs, False)
    url = img.getDownloadURL({"region": region, "scale": 10, "crs": a.crs,
                              "format": "GEO_TIFF"})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"PK":   # zipped per-band files: stack them
        z = zipfile.ZipFile(io.BytesIO(data))
        names = sorted(z.namelist())
        vv = [m for m in names if ".VV." in m][0]
        vh = [m for m in names if ".VH." in m][0]
        with rasterio.open(io.BytesIO(z.read(vv))) as sv, rasterio.open(io.BytesIO(z.read(vh))) as sh:
            prof = sv.profile
            prof.update(count=2)
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(sv.read(1), 1)
                dst.write(sh.read(1), 2)
    else:
        open(path, "wb").write(data)
    print(f"  tile {i + 1}/{len(tiles)}")
    return path


with ThreadPoolExecutor(max_workers=4) as ex:
    paths = list(ex.map(get, range(len(tiles))))

srcs = [rasterio.open(p_) for p_ in paths]
mosaic, transform = merge(srcs, nodata=NODATA)
prof = srcs[0].profile
prof.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
            count=2, dtype="float32", nodata=NODATA, compress="lzw")
for s_ in srcs:
    s_.close()
with rasterio.open(a.out, "w", **prof) as dst:
    dst.write(mosaic.astype("float32"))
    dst.set_band_description(1, "VV")
    dst.set_band_description(2, "VH")
print("Saved", a.out, "| size", mosaic.shape[2], "x", mosaic.shape[1])
