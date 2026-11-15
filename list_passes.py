import argparse
import ee

p = argparse.ArgumentParser()
p.add_argument("--lon", type=float, default=85.14)
p.add_argument("--lat", type=float, default=25.61)
p.add_argument("--start", default="2026-08-15")
p.add_argument("--end", default="2026-09-25")
a = p.parse_args()

ee.Initialize(project="geopulse-514608")
pt = ee.Geometry.Point([a.lon, a.lat])

col = (ee.ImageCollection("COPERNICUS/S1_GRD")
       .filterBounds(pt)
       .filterDate(a.start, a.end)
       .filter(ee.Filter.eq("instrumentMode", "IW"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
       .sort("system:time_start"))

rows = col.map(lambda im: ee.Feature(None, {
    "date": im.date().format("YYYY-MM-dd HH:mm"),
    "pass": im.get("orbitProperties_pass"),
    "orbit": im.get("relativeOrbitNumber_start"),
    "sat": im.get("platform_number"),
})).getInfo()["features"]

print(f"{'date (UTC)':<18}{'pass':<12}{'rel.orbit':<10}{'sat'}")
for f in rows:
    q = f["properties"]
    print(f"{q['date']:<18}{str(q['pass']):<12}{str(q['orbit']):<10}{q['sat']}")
print(f"\n{len(rows)} images")
