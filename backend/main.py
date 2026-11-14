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
