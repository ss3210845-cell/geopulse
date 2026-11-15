import argparse
import ee

p = argparse.ArgumentParser()
p.add_argument("--aoi", nargs=4, type=float, default=[84.9, 25.45, 85.5, 25.85],
               metavar=("W", "S", "E", "N"))
p.add_argument("--orbit", type=int, default=85)
p.add_argument("--dates", nargs="+", default=["2026-08-28", "2026-09-09"])
a = p.parse_args()

ee.Initialize(project="geopulse-514608")
aoi = ee.Geometry.Rectangle(a.aoi)

print(f"AOI W,S,E,N = {a.aoi} | orbit {a.orbit}")
print(f"{'date':<12}{'frames':<8}{'coverage':<10}{'mean VV':<10}{'mean VH'}")
for day in a.dates:
    d = ee.Date(day)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi)
           .filterDate(d, d.advance(1, "day"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", a.orbit))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    n = col.size().getInfo()
    if n == 0:
        print(f"{day:<12}{'none found'}")
        continue
    img = col.select(["VV", "VH"]).mosaic()
    cover = img.select("VV").mask().reduceRegion(ee.Reducer.mean(), aoi, 100).get("VV").getInfo()
    means = img.reduceRegion(ee.Reducer.mean(), aoi, 100).getInfo()
    print(f"{day:<12}{n:<8}{cover * 100:>6.1f}%   {means['VV']:<10.2f}{means['VH']:.2f}")
