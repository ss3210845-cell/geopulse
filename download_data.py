import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
SPLIT_URL = "https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled"

# local csv name -> name on the server
SPLITS = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def fetch(url, dest):
    """Download url to dest unless it already exists. Returns True if downloaded."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    tmp.write_bytes(r.content)
    tmp.rename(dest)  # only appears under the final name when complete
    return True


def get_split_csv(split, meta_dir):
    local = meta_dir / f"{split}_data.csv"
    if not local.exists():
        print(f"Fetching split list: {split}")
        fetch(f"{SPLIT_URL}/{SPLITS[split]}", local)
    return pd.read_csv(local, header=None, names=["image", "label"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="data/raw/sen1floods11")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   choices=list(SPLITS))
    p.add_argument("--limit", type=int, default=None, help="only first N tiles per split")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    base = Path(args.base_dir)
    img_dir, lab_dir, meta_dir = base / "S1Hand", base / "LabelHand", base / "metadata"
    for d in (img_dir, lab_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split in args.splits:
        df = get_split_csv(split, meta_dir)
        if args.limit:
            df = df.head(args.limit)
        print(f"{split}: {len(df)} tiles")
        for _, row in df.iterrows():
            jobs.append((f"{DATA_URL}/S1Hand/{row['image']}", img_dir / row["image"]))
            jobs.append((f"{DATA_URL}/LabelHand/{row['label']}", lab_dir / row["label"]))

    jobs = list(dict.fromkeys(jobs))  # drop duplicates
    done = new = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for got in ex.map(lambda j: fetch(*j), jobs):
            done += 1
            new += got
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files checked, {new} newly downloaded")
    print("Download complete!")


if __name__ == "__main__":
    main()
