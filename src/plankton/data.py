"""Load the UVP6Net training data and the NOC test data as pandas tables.

Every table has one row per object (image) and an ``object_id`` column, so
predictions can be written straight to the CSV format the evaluation script and
the leaderboard expect (see :func:`write_predictions`).

Data locations default to ``data/uvp6net`` and ``data/noc_uvp6_test`` in the
repository. Set the ``PLANKTON_DATA_DIR`` environment variable to use another
parent folder (e.g. a scratch disk on a cluster).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLASSES_FILE = REPO / "configs" / "uvp6net_classes.csv"

# The 62 handcrafted UVPapp features, available for both datasets.
FEATURES = [
    "area", "mean", "stddev", "mode", "min", "max", "perim.", "width", "height", "major", "minor",
    "angle", "circ.", "feret", "intden", "median", "skew", "kurt", "%area", "area_exc", "fractal",
    "skelarea", "slope", "histcum1", "histcum2", "histcum3", "nb1", "nb2", "nb3", "symetrieh",
    "symetriev", "symetriehc", "symetrievc", "convperim", "convarea", "fcons", "thickr", "elongation",
    "range", "meanpos", "cv", "sr", "perimareaexc", "feretareaexc", "perimferet", "perimmajor",
    "circex", "kurt_mean", "skew_mean", "convperim_perim", "convarea_area", "symetrieh_area",
    "symetriev_area", "nb1_area", "nb2_area", "nb3_area", "nb1_range", "nb2_range", "nb3_range",
    "median_mean", "median_mean_range", "skeleton_area",
]

UNKNOWN = "unknown"  # label for objects of a class absent from the training data


def data_dir() -> Path:
    return Path(os.environ.get("PLANKTON_DATA_DIR", REPO / "data"))


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


# ---- classes -----------------------------------------------------------------------

def load_classes() -> pd.DataFrame:
    """The UVP6Net class list: level1 (54 classes), level2 (32), plankton flag, split counts."""
    return pd.read_csv(CLASSES_FILE)


def class_names(level: int = 1) -> list[str]:
    """Sorted class names at taxonomic level 1 (54 classes) or 2 (32 classes)."""
    return sorted(load_classes()[f"level{level}"].unique(), key=str.lower)


def level1_to_level2() -> dict[str, str]:
    c = load_classes()
    return dict(zip(c["level1"], c["level2"], strict=True))


# ---- UVP6Net (training data) --------------------------------------------------------

def load_uvp6net(split: str | None = None, root: Path | None = None) -> pd.DataFrame:
    """UVP6Net labels, one row per image.

    Columns: object_id, level1, level2, plankton, split, path (absolute image path).

    Args:
        split: "train", "valid" or "test" to select one split; None for all.
        root: dataset folder (default: <data dir>/uvp6net).
    """
    root = Path(root) if root else data_dir() / "uvp6net"
    labels = _require(root / "labels.csv", "Run: python scripts/download_uvp6net.py")
    df = pd.read_csv(labels, dtype={"objid": str})
    df = df.rename(columns={"objid": "object_id", "taxon_level1": "level1",
                            "taxon_level2": "level2", "set": "split"})
    df["path"] = [str(root / p) for p in df.pop("img_path")]
    if split is not None:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"split must be 'train', 'valid' or 'test', not {split!r}")
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def load_uvp6net_features(split: str | None = None, root: Path | None = None) -> pd.DataFrame:
    """:func:`load_uvp6net` joined with the 62 handcrafted features (no images needed)."""
    root = Path(root) if root else data_dir() / "uvp6net"
    feats = pd.read_csv(_require(root / "UVP6Net" / "features_native.csv.gz",
                                 "Run: python scripts/download_uvp6net.py --parts data"),
                        dtype={"objid": str})
    feats = feats.rename(columns={"objid": "object_id"})
    return load_uvp6net(split, root).merge(feats, on="object_id", how="left")


# ---- NOC (test data) ----------------------------------------------------------------

def load_noc(root: Path | None = None) -> pd.DataFrame:
    """NOC UVP6 test set, one row per image. No labels: that is what you predict.

    Columns: object_id, sample_id, date, time, lat, lon, depth_m, frame_x, frame_y,
    the 62 handcrafted features, and path (absolute image path).
    """
    root = Path(root) if root else data_dir() / "noc_uvp6_test"
    meta = _require(root / "metadata.csv", "Run: python scripts/download_noc_uvp6.py")
    df = pd.read_csv(meta, dtype={"object_id": str, "sample_id": str})
    df["path"] = [str(root / p) for p in df.pop("image")]
    return df


# ---- predictions ---------------------------------------------------------------------

def write_predictions(object_ids, labels, path: str | Path) -> Path:
    """Write a predictions CSV (object_id,label) for scripts/evaluate.py or the leaderboard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"object_id": list(object_ids), "label": list(labels)})
    if out["object_id"].duplicated().any():
        raise ValueError("object_ids must be unique")
    out.to_csv(path, index=False)
    return path
