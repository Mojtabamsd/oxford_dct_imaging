#!/usr/bin/env python
"""Build the NOC UVP6 test-set release (maintainer only, not for students).

Produces two things from the raw NOC export:

1. ``noc_uvp6_test.zip``: the package students download. It holds the images
   and a ``metadata.csv``: sampling context (sample, date/time, position,
   depth) and the 62 UVPapp features, named as in UVP6Net. It holds NO
   labels, model outputs, sampling weights or annotator details.
2. ``answer_key.csv``: object_id, expert label, UVP6Net level-2 label and
   sampling weight. Keep this private, and never commit it or share it.

Example
-------
    python scripts/maintainer/build_noc_release.py \
        --images data/noc_uvp6/images \
        --labels data/noc_uvp6/validation_merged.csv \
        --out-dir release/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Expert label -> UVP6Net level-2 class. Kept with the private data, outside git: it would
# reveal which classes are in the hidden test set, and which one is the novel 'unknown' class.
DEFAULT_MAPPING = REPO / "data" / "noc_uvp6" / "label_mapping.csv"

# Columns copied into the public metadata.csv as  public name -> column in the NOC export.
# Everything else is deliberately left out: expert/model labels, scores, annotation
# details (incl. annotator name/email), sampling weights (they leak label
# information), empty columns, placeholder IDs and duplicated columns.
CONTEXT_COLUMNS = {
    "object_id": "object_id",
    "sample_id": "sample_id",
    "date": "object_date",
    "time": "object_time",
    "lat": "object_lat",
    "lon": "object_lon",
    "depth_m": "object_depth_min",  # depth_min == depth_max for every object
    "frame_x": "object_vig_left_position",  # position of the crop in the full UVP frame
    "frame_y": "object_vig_top_position",
}
# The 62 UVPapp features, named exactly as in UVP6Net's features_native.csv.gz
NATIVE_FEATURES = [
    "area", "mean", "stddev", "mode", "min", "max", "perim.", "width", "height", "major", "minor",
    "angle", "circ.", "feret", "intden", "median", "skew", "kurt", "%area", "area_exc", "fractal",
    "skelarea", "slope", "histcum1", "histcum2", "histcum3", "nb1", "nb2", "nb3", "symetrieh",
    "symetriev", "symetriehc", "symetrievc", "convperim", "convarea", "fcons", "thickr", "elongation",
    "range", "meanpos", "cv", "sr", "perimareaexc", "feretareaexc", "perimferet", "perimmajor",
    "circex", "kurt_mean", "skew_mean", "convperim_perim", "convarea_area", "symetrieh_area",
    "symetriev_area", "nb1_area", "nb2_area", "nb3_area", "nb1_range", "nb2_range", "nb3_range",
    "median_mean", "median_mean_range", "skeleton_area",
]
PUBLIC_COLUMNS = ["object_id", "image", *list(CONTEXT_COLUMNS)[1:], *NATIVE_FEATURES]


def load_mapping(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row["expert_label"]: row["uvp6net_level2"] for row in csv.DictReader(f)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="folder of NOC .png images")
    parser.add_argument("--labels", type=Path, required=True, help="validation_merged.csv export")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                        help="expert label -> UVP6Net level 2 CSV (default: data/noc_uvp6/label_mapping.csv)")
    parser.add_argument("--out-dir", type=Path, default=Path("release"))
    parser.add_argument("--key-only", action="store_true",
                        help="only rewrite answer_key.csv (e.g. after editing the label mapping); "
                             "leaves the published zip, and its checksum, untouched")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unmapped = sorted({r["expert_label_raw"] for r in rows} - set(mapping))
    if unmapped:
        raise SystemExit(f"Labels missing from {args.mapping.name}: {unmapped}")

    public, key = [], []
    for r in sorted(rows, key=lambda r: r["object_id"]):
        image = f"{r['object_id']}.png"
        if not (args.images / image).is_file():
            raise SystemExit(f"Missing image: {image}")
        row = {name: r[col] for name, col in CONTEXT_COLUMNS.items()}
        row["image"] = f"images/{image}"
        row.update({feat: r[f"object_{feat}"] for feat in NATIVE_FEATURES})
        public.append(row)
        key.append({"object_id": r["object_id"], "expert_label": r["expert_label_raw"],
                    "label": mapping[r["expert_label_raw"]], "weight": r["population_weight"]})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    key_path = args.out_dir / "answer_key.csv"
    with open(key_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    print(f"PRIVATE key     : {key_path}  <- do not commit or share")
    if args.key_only:
        return

    zip_path = args.out_dir / "noc_uvp6_test.zip"
    meta_path = args.out_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC_COLUMNS)
        w.writeheader()
        w.writerows(public)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        z.write(meta_path, "noc_uvp6_test/metadata.csv")
        for row in public:
            z.write(args.images / Path(row["image"]).name, f"noc_uvp6_test/{row['image']}")
    meta_path.unlink()

    print(f"{len(public)} images")
    print(f"student package : {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  sha256        : {sha256(zip_path)}  <- paste into scripts/download_noc_uvp6.py")


if __name__ == "__main__":
    main()
