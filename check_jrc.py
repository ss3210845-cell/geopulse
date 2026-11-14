"""Compare model water masks with JRC Global Surface Water occurrence (1984-2021)."""
import io, zipfile
import ee, numpy as np, rasterio, requests
from rasterio.warp import reproject, Resampling

ee.Initialize(project="geopulse-514608")
with rasterio.open("outputs/patna_before_mask.tif") as s:
    before, tf, crs, bounds = s.read(1), s.transform, s.crs, s.bounds
with rasterio.open("outputs/patna_change.tif") as s:
    change = s.read(1)

region = ee.Geometry.Rectangle([bounds.left, bounds.bottom, bounds.right, bounds.top],
                               str(crs), False)
jrc = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toByte())
url = jrc.getDownloadURL({"region": region, "scale": 30, "crs": str(crs), "format": "GEO_TIFF"})
data = requests.get(url, timeout=300).content
if data[:2] == b"PK":
    z = zipfile.ZipFile(io.BytesIO(data))
    data = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
with rasterio.open(io.BytesIO(data)) as s:
    occ, otf, ocrs = s.read(1), s.transform, s.crs

def to_jrc_grid(mask_bool):
    out = np.zeros(occ.shape, np.float32)
    reproject(mask_bool.astype(np.float32), out, src_transform=tf, src_crs=crs,
              dst_transform=otf, dst_crs=ocrs, resampling=Resampling.average)
    return out > 0.5

px = abs(otf.a * otf.e) / 1e6
perm, new = to_jrc_grid(change == 1), to_jrc_grid(change == 2)
usual = occ >= 50          # water at least half of the time, 1984-2021
ever = occ > 0             # water at least once

def line(name, m):
    a = m.sum() * px
    print(f"{name:<28}{a:8.1f} km2 | on JRC usual water: {100*(m & usual).sum()/max(m.sum(),1):5.1f}% "
          f"| on JRC ever-water: {100*(m & ever).sum()/max(m.sum(),1):5.1f}%")

print(f"JRC usual water (>=50% of time) in this area: {usual.sum()*px:.1f} km2")
line("Model permanent water", perm)
line("Model NEW flood", new)
print(f"Share of JRC usual water the model marks as permanent water: "
      f"{100*(perm & usual).sum()/max(usual.sum(),1):.1f}%")
