"""GeoPulse flood API: upload a Sentinel-1 GeoTIFF (VV, VH in dB), get a flood mask and area."""
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from inference import predict_scene

log = logging.getLogger("geopulse")
WEIGHTS = os.environ.get("GEOPULSE_WEIGHTS", "runs/aug/best.pth")
MAX_UPLOAD_MB = 600
WORK = Path(os.environ.get("GEOPULSE_WORK", "jobs"))
WORK.mkdir(exist_ok=True)

app = FastAPI(title="GeoPulse flood API")
pool = ThreadPoolExecutor(max_workers=1)   # one scene at a time
JOBS = {}


def make_overlay(mask, crs, tf, out_png):
    """Flood pixels as a red transparent PNG in lat/lon, plus its bounds for a web map."""
    h, w = mask.shape
    flood = (mask == 1).astype(np.uint8)
    dst_tf, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, *array_bounds(h, w, tf))
    dst = np.zeros((dh, dw), np.uint8)
    reproject(flood, dst, src_transform=tf, src_crs=crs, dst_transform=dst_tf,
              dst_crs="EPSG:4326", resampling=Resampling.nearest)
    f = max(1, -(-max(dh, dw) // 2000))     # keep the PNG at most ~2000 px wide
    if f > 1:
        hh, ww = dh // f * f, dw // f * f
        dst = dst[:hh, :ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3))
    rgba = np.zeros((*dst.shape, 4), np.uint8)
    rgba[dst == 1] = (229, 62, 62, 170)
    Image.fromarray(rgba, "RGBA").save(out_png)
    west, south, east, north = array_bounds(dh, dw, dst_tf)
    return [[south, west], [north, east]]    # Leaflet order: [[south, west], [north, east]]


def run_job(job_id, threshold):
    job, d = JOBS[job_id], WORK / job_id
    try:
        job["status"] = "running"

        def progress(i, n):
            job["progress"] = round(i / n, 3)

        r = predict_scene(str(d / "input.tif"), weights=WEIGHTS,
                          threshold=threshold, progress=progress)
        mask = r["mask"]
        with rasterio.open(d / "mask.tif", "w", driver="GTiff", height=mask.shape[0],
                           width=mask.shape[1], count=1, dtype="uint8", crs=r["crs"],
                           transform=r["transform"], nodata=255, compress="lzw") as dst:
            dst.write(mask, 1)
        bounds = make_overlay(mask, r["crs"], r["transform"], d / "overlay.png")
        valid = r["valid_pixels"]
        job["result"] = {
            "flood_km2": None if r["flood_km2"] is None else round(r["flood_km2"], 2),
            "valid_km2": None if r["valid_km2"] is None else round(r["valid_km2"], 2),
            "flood_percent_of_valid": round(100 * r["flood_pixels"] / max(valid, 1), 2),
            "threshold": threshold,
            "bounds": bounds,
            "note": "Model estimate. Includes all open water in the scene, "
                    "including rivers and lakes that are normally there.",
        }
        job["status"] = "done"
    except Exception:
        log.exception("job %s failed", job_id)
        job["status"] = "error"
        job["error"] = "Processing failed. Check that the file is a Sentinel-1 GeoTIFF."
    finally:
        (d / "input.tif").unlink(missing_ok=True)


FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS}


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), threshold: float = Form(0.4)):
    if not 0.05 <= threshold <= 0.95:
        raise HTTPException(422, "threshold must be between 0.05 and 0.95")
    job_id = uuid.uuid4().hex[:12]
    d = WORK / job_id
    d.mkdir()
    src, size, too_big = d / "input.tif", 0, False
    with open(src, "wb") as f:                      # our own filename, never the user's
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        with rasterio.open(src) as s:
            ok = s.count >= 2 and s.crs is not None
    except Exception:
        ok = False
    if not ok:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Upload a GeoTIFF with VV (band 1) and VH (band 2) in dB "
                                 "and a coordinate system.")
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}
    pool.submit(run_job, job_id, threshold)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


def get_job(job_id):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id)


@app.get("/jobs/{job_id}/mask")
def job_mask(job_id: str):
    if get_job(job_id)["status"] != "done":
        raise HTTPException(409, "Job not finished")
    return FileResponse(WORK / job_id / "mask.tif", media_type="image/tiff",
                        filename=f"flood_mask_{job_id}.tif")


@app.get("/jobs/{job_id}/overlay.png")
def job_overlay(job_id: str):
    if get_job(job_id)["status"] != "done":
        raise HTTPException(409, "Job not finished")
    return FileResponse(WORK / job_id / "overlay.png", media_type="image/png")


# ---------------------------------------------------------------- change detection
from scipy import ndimage


def _overlay_classes(cls, crs, tf, out_png, palette):
    """Class raster (0 none, 1/2 classes, 255 nodata) as a transparent lat/lon PNG."""
    h, w = cls.shape
    cls = np.where(cls == 255, 0, cls).astype(np.uint8)
    dst_tf, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, *array_bounds(h, w, tf))
    dst = np.zeros((dh, dw), np.uint8)
    reproject(cls, dst, src_transform=tf, src_crs=crs, dst_transform=dst_tf,
              dst_crs="EPSG:4326", resampling=Resampling.nearest)
    f = max(1, -(-max(dh, dw) // 2000))
    if f > 1:
        hh, ww = dh // f * f, dw // f * f
        dst = dst[:hh, :ww].reshape(hh // f, f, ww // f, f).max(axis=(1, 3))
    rgba = np.zeros((*dst.shape, 4), np.uint8)
    for code, color in palette.items():
        rgba[dst == code] = color
    Image.fromarray(rgba, "RGBA").save(out_png)
    west, south, east, north = array_bounds(dh, dw, dst_tf)
    return [[south, west], [north, east]]


def run_change_job(job_id, threshold, min_px=100):
    job, d = JOBS[job_id], WORK / job_id
    try:
        job["status"] = "running"
        res = {}
        for k, name in enumerate(["before", "flood"]):
            def progress(i, n, k=k):
                job["progress"] = round((k + i / n) / 2, 3)
            res[name] = predict_scene(str(d / f"{name}.tif"), weights=WEIGHTS,
                                      threshold=threshold, progress=progress)
            res[name].pop("prob")            # free memory
        mb, mf = res["before"]["mask"], res["flood"]["mask"]
        tf, crs = res["flood"]["transform"], res["flood"]["crs"]
        valid = (mb != 255) & (mf != 255)
        bw, fw = (mb == 1) & valid, (mf == 1) & valid
        new = fw & ~bw
        lab, n = ndimage.label(new, structure=np.ones((3, 3)))
        sizes = ndimage.sum(np.ones(lab.shape, np.uint8), lab, index=np.arange(1, n + 1))
        keep = np.concatenate([[False], sizes >= min_px])
        new = keep[lab]
        cls = np.zeros(mb.shape, np.uint8)
        cls[bw & fw] = 1
        cls[new] = 2
        cls[~valid] = 255
        px = abs(tf.a * tf.e) / 1e6
        with rasterio.open(d / "mask.tif", "w", driver="GTiff", height=cls.shape[0],
                           width=cls.shape[1], count=1, dtype="uint8", crs=crs,
                           transform=tf, nodata=255, compress="lzw") as dst:
            dst.write(cls, 1)
        bounds = _overlay_classes(cls, crs, tf, d / "overlay.png",
                                  {1: (43, 108, 176, 150), 2: (229, 62, 62, 210)})
        job["result"] = {
            "kind": "change",
            "new_flood_km2": round(float(new.sum() * px), 2),
            "water_before_km2": round(float(bw.sum() * px), 2),
            "water_flood_km2": round(float(fw.sum() * px), 2),
            "valid_km2": round(float(valid.sum() * px), 2),
            "threshold": threshold,
            "bounds": bounds,
            "note": "Blue: water on both dates. Red: new water on the later date "
                    "(patches under 1 ha removed). Model estimate, not validated "
                    "against an independent flood map. Mask download: 0 none, "
                    "1 water on both dates, 2 new flood, 255 no data.",
        }
        job["status"] = "done"
    except Exception:
        log.exception("change job %s failed", job_id)
        job["status"] = "error"
        job["error"] = "Processing failed. Check that both files are Sentinel-1 GeoTIFFs."
    finally:
        (d / "before.tif").unlink(missing_ok=True)
        (d / "flood.tif").unlink(missing_ok=True)


async def _save_upload(upload, path):
    size = 0
    with open(path, "wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                return False
            f.write(chunk)
    return True


@app.post("/changes", status_code=202)
async def create_change_job(before: UploadFile = File(...), flood: UploadFile = File(...),
                            threshold: float = Form(0.4)):
    if not 0.05 <= threshold <= 0.95:
        raise HTTPException(422, "threshold must be between 0.05 and 0.95")
    job_id = uuid.uuid4().hex[:12]
    d = WORK / job_id
    d.mkdir()
    for upload, name in ((before, "before"), (flood, "flood")):
        if not await _save_upload(upload, d / f"{name}.tif"):
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    problem = None
    try:
        with rasterio.open(d / "before.tif") as a, rasterio.open(d / "flood.tif") as b:
            if a.count < 2 or b.count < 2 or a.crs is None or b.crs is None:
                problem = "Each file needs VV (band 1), VH (band 2) and a coordinate system."
            elif not a.crs.is_projected:
                problem = "Scenes must use a projected (metre) coordinate system such as UTM."
            elif not (a.crs == b.crs and a.transform == b.transform
                      and a.width == b.width and a.height == b.height):
                problem = ("The two scenes must cover exactly the same grid "
                           "(same area, size and projection), from the same orbit.")
    except Exception:
        problem = "Could not read one of the files as a GeoTIFF."
    if problem:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, problem)
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}
    pool.submit(run_change_job, job_id, threshold)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}
