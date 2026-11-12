#!/usr/bin/env python
"""Download and extract the public UVP6Net dataset from SEANOE.

UVP6Net: plankton images captured with the UVP6.
Picheral et al. (2024), SEANOE. https://doi.org/10.17882/101948
Licence: CC BY-NC 4.0 (non-commercial use, attribution required).

Uses only the Python standard library, so it can be run before installing
anything else. Downloads resume if interrupted, and are checked against the
size reported by the server before extraction.

The dataset comes in three archives:

    data   labels, train/valid/test split and 62 handcrafted features per object
           (always downloaded: every approach needs the labels and split)
    imgs   the raw images, one folder per class (needed for CNNs / deep learning)
    uvpec  the same objects on a black background, coarser 20-class scheme

Pick what fits your approach:

    # Feature-based models only (e.g. random forest, XGBoost): ~425 MB
    python scripts/download_uvp6net.py --parts data

    # Images for deep learning (default): ~1 GB
    python scripts/download_uvp6net.py

    # Everything, into a custom location (e.g. a scratch disk on a cluster)
    python scripts/download_uvp6net.py --parts all --data-dir /scratch/$USER/uvp6net
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://www.seanoe.org/data/00908/101948/data"

# name -> (SEANOE file id, archive filename, description)
PARTS = {
    "data": ("119451", "UVP6Net_data.tar", "metadata, labels, splits and handcrafted features (~425 MB)"),
    "imgs": ("119452", "UVP6Net_imgs.tar", "images, 110 classes (~560 MB)"),
    "uvpec": ("119453", "UVPEC_imgs.tar", "images on black background, 20 classes (~590 MB)"),
}
DEFAULT_PARTS = ["data", "imgs"]
CHUNK = 1 << 20  # 1 MiB


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` file if present."""
    total = remote_size(url)
    if dest.exists() and (total is None or dest.stat().st_size == total):
        print(f"  already downloaded: {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored the Range header
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    done, t0 = have, time.time()
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        _progress(done, total, t0, have)
            print()
            break
        except (OSError, urllib.error.URLError) as err:
            print(f"\n  attempt {attempt}/{retries} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    if total is not None and tmp.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: expected {total} bytes, got {tmp.stat().st_size}")
    tmp.replace(dest)


def _progress(done: int, total: int | None, t0: float, start: int) -> None:
    speed = (done - start) / max(time.time() - t0, 1e-6) / 1e6
    if total:
        pct = 100 * done / total
        bar = "#" * int(pct // 4)
        msg = f"\r  [{bar:<25}] {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB  {speed:5.1f} MB/s"
    else:
        msg = f"\r  {done / 1e6:7.1f} MB  {speed:5.1f} MB/s"
    sys.stdout.write(msg)
    sys.stdout.flush()


_UNSAFE = str.maketrans({c: "_" for c in '<>:"|?*'})


def safe_path(path: str) -> str:
    """Make an archive path valid on every OS: class folders like ``other<living``
    become ``other_living``. Applied identically on Linux, macOS and Windows so
    paths in ``labels.csv`` are the same everywhere."""
    return path.translate(_UNSAFE)


def extract(archive: Path, out_dir: Path) -> None:
    print(f"  extracting {archive.name} -> {out_dir}")
    root = out_dir.resolve()
    with tarfile.open(archive) as tar:
        for member in tar:
            member.name = safe_path(member.name)
            if not (root / member.name).resolve().is_relative_to(root):  # refuse paths escaping out_dir
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if member.isfile() or member.isdir():  # skip links and device files
                tar.extract(member, out_dir)


def write_labels(data_dir: Path) -> None:
    """Write ``labels.csv``: one row per object with labels, split and image path."""
    src = data_dir / "UVP6Net" / "taxa.csv.gz"
    dst = data_dir / "labels.csv"
    cols = ["objid", "taxon_level1", "taxon_level2", "plankton", "set", "img_path"]
    with gzip.open(src, "rt", newline="", encoding="utf-8") as fin:  # noqa: SIM117 - keep 3.9 syntax
        with open(dst, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in csv.DictReader(fin):
                row["img_path"] = safe_path(row["img_path"])
                writer.writerow(row)
    print(f"  wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/uvp6net"),
                        help="where to put the dataset (default: data/uvp6net)")
    parser.add_argument("--parts", nargs="+", default=DEFAULT_PARTS, choices=[*PARTS, "all"],
                        help="which archives to fetch (default: data imgs)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .tar files after extraction")
    args = parser.parse_args()

    parts = list(PARTS) if "all" in args.parts else ["data", *(p for p in args.parts if p != "data")]
    archive_dir = args.data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("UVP6Net (Picheral et al. 2024, https://doi.org/10.17882/101948), licence CC BY-NC 4.0\n")
    for name in parts:
        file_id, filename, desc = PARTS[name]
        marker = args.data_dir / f".{name}.extracted"
        print(f"[{name}] {filename}: {desc}")
        if marker.exists():
            print("  already extracted, skipping")
            continue
        archive = archive_dir / filename
        download(f"{BASE_URL}/{file_id}.tar", archive)
        extract(archive, args.data_dir)
        if name == "data":
            write_labels(args.data_dir)
        marker.touch()
        if not args.keep_archives:
            archive.unlink()

    if not args.keep_archives and archive_dir.exists() and not any(archive_dir.iterdir()):
        shutil.rmtree(archive_dir)
    print(f"\nDone. Dataset is in {args.data_dir.resolve()}")


if __name__ == "__main__":
    if sys.version_info < (3, 9):  # noqa: UP036 - runs before the package (3.10+) is installed
        sys.exit("Python 3.9 or newer is required.")
    main()
