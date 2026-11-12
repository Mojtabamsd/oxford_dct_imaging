#!/usr/bin/env python
"""Download the NOC UVP6 test images (released in the final week of the challenge).

The images come from a UVP6 deployment by the National Oceanography Centre and
were labelled by an expert. Labels are NOT included: you predict them, and your
predictions are scored against the hidden expert labels.

Example
-------
    python scripts/download_noc_uvp6.py
    python scripts/download_noc_uvp6.py --data-dir /scratch/$USER/noc_uvp6
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

from download_uvp6net import download  # same folder

# Zenodo record: https://zenodo.org/records/24160827
URL = "https://zenodo.org/records/24160827/files/noc_uvp6_test.zip?download=1"
SHA256 = "e32b4641018f4d3b76a31b8abb8c87bd553392cc313a9e30cf66c1381324cb72"  # from build_noc_release.py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="parent folder; images go to <data-dir>/noc_uvp6_test (default: data)")
    parser.add_argument("--url", default=URL, help="override the download URL")
    args = parser.parse_args()

    if not args.url:
        sys.exit("The NOC test set has not been released yet. It will be released in week 4.")

    out = args.data_dir / "noc_uvp6_test"
    if (out / "metadata.csv").exists():
        print(f"Already downloaded: {out.resolve()}")
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "noc_uvp6_test.zip"
    print("Downloading NOC UVP6 test images (~34 MB)")
    download(args.url, archive)

    if args.url == URL and sha256(archive) != SHA256:
        archive.unlink()
        sys.exit("Checksum mismatch: the download is corrupt. Please run the script again.")

    with zipfile.ZipFile(archive) as z:
        z.extractall(args.data_dir)
    archive.unlink()
    n = sum(1 for _ in (out / "images").glob("*.png"))
    print(f"Done: {n} images in {out.resolve()}")


if __name__ == "__main__":
    main()
