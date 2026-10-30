
import pandas as pd
import urllib.request
from pathlib import Path
from urllib.error import URLError, HTTPError

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path("data/raw/sen1floods11")
CSV_FILE = BASE_DIR / "metadata" / "valid_data.csv"

URL_BASE = (
    "https://storage.googleapis.com/sen1floods11/"
    "v1.1/data/flood_events/HandLabeled"
)

IMAGE_DIR = BASE_DIR / "S1Hand"
LABEL_DIR = BASE_DIR / "LabelHand"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Read all validation entries
df = pd.read_csv(
    CSV_FILE,
    header=None,
    names=["image", "label"]
)

total_downloaded = 0
total_skipped = 0
total_failed = 0


def download_file(filename, folder, category):
    """Download a missing TIFF without overwriting existing files."""
    global total_downloaded, total_skipped, total_failed

    destination = folder / filename

    if destination.is_file() and destination.stat().st_size > 0:
        total_skipped += 1
        return True

    url = f"{URL_BASE}/{category}/{filename}"
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoPulse-Dataset-Downloader"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            with open(temporary, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if not temporary.exists() or temporary.stat().st_size == 0:
            raise OSError("Downloaded file is empty")

        temporary.replace(destination)
        total_downloaded += 1
        return True

    except (HTTPError, URLError, OSError, TimeoutError) as error:
        print(f"\nFAILED: {filename} — {error}")
        total_failed += 1

        if temporary.exists():
            temporary.unlink()

        return False


# --------------------------------------------------
# Download missing validation images and labels
# --------------------------------------------------

print(f"Validation entries: {len(df)}")
print("Downloading missing validation TIFF files...")
print("Existing non-empty files will be preserved.\n")

for index, row in df.iterrows():
    image_name = str(row["image"]).strip()
    label_name = str(row["label"]).strip()

    image_ok = download_file(
        image_name, IMAGE_DIR, "S1Hand"
    )

    label_ok = download_file(
        label_name, LABEL_DIR, "LabelHand"
    )

    if (index + 1) % 10 == 0 or index + 1 == len(df):
        print(
            f"Processed {index + 1}/{len(df)} entries | "
            f"Downloaded: {total_downloaded} | "
            f"Skipped: {total_skipped} | "
            f"Failed: {total_failed}"
        )

print("\nDownload summary")
print("=" * 45)
print(f"New files downloaded : {total_downloaded}")
print(f"Existing files kept  : {total_skipped}")
print(f"Failed downloads     : {total_failed}")

if total_failed:
    print("\nSome downloads failed. Review the errors above.")
    print("Rerun this script after resolving the errors.")
else:
    print("\nAll requested validation files are available.")
