#!/usr/bin/env python
"""Score a predictions file against ground-truth labels.

Predictions format
------------------
A CSV with one row per image and two columns:

    object_id,label
    294999840,detritus
    297999905,Copepoda
    ...

* ``object_id``: the UVP6Net ``objid`` or the NOC ``object_id``.
* ``label``: a UVP6Net class name, level 2 (32 classes) or level 1 (54 classes),
  or ``unknown`` for an object you think belongs to none of the training classes.
  Level-1 names are mapped to level 2 automatically when scoring at level 2.
  See configs/uvp6net_classes.csv for the full list.

Examples
--------
    # Your model on the UVP6Net test split (labels are public)
    python scripts/evaluate.py predictions.csv --truth uvp6net-test

    # Same, at the finer level 1, saving per-class results and confusion matrix
    python scripts/evaluate.py predictions.csv --truth uvp6net-test --level 1 --out-dir results/

    # NOC test set (organisers only: needs the private answer key)
    python scripts/evaluate.py predictions.csv --truth answer_key.csv

Metrics
-------
Accuracy is reported but is misleading here: predicting "detritus" for
everything already scores ~80% on UVP6Net. The headline metrics are:

* macro F1: F1 averaged over classes, each class counting equally
* balanced accuracy: recall averaged over classes

Both are averaged over the classes present in the ground truth.

Open-set ("unknown") scoring: the NOC test set contains one class that is absent
from UVP6Net. Its images have the true label ``unknown``, which counts as one
extra class in the macro metrics. When the truth contains ``unknown`` the report
also gives:

* closed-set metrics: computed only on images of known classes
* unknown recall: share of novel images you labelled ``unknown``
* false-unknown rate: share of known-class images you wrongly labelled ``unknown``

If the truth file has a ``weight`` column (the NOC answer key does), weighted
metrics are also reported. The NOC images were not sampled at random: rare
predicted classes were over-sampled. The weights undo that, so weighted metrics
estimate performance on the full, unsampled deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[1]
CLASSES = REPO / "configs" / "uvp6net_classes.csv"
UNKNOWN = "unknown"
MISSING = "<missing>"


class SubmissionError(ValueError):
    """A predictions file that cannot be scored; the message tells the user why."""


# ---- loading --------------------------------------------------------------------

def load_classes(path: Path = CLASSES) -> pd.DataFrame:
    """UVP6Net class list: columns level1, level2, plankton, n_train, n_valid, n_test."""
    return pd.read_csv(path)


def load_truth(truth: str, level: int, data_dir: Path) -> pd.DataFrame:
    """Return a frame with columns object_id, label and optionally weight.

    ``truth`` is ``uvp6net-test`` / ``uvp6net-valid`` or a path to a CSV with
    columns object_id, label and optionally weight.
    """
    if truth.startswith("uvp6net-"):
        split = truth.split("-", 1)[1]
        labels = data_dir / "labels.csv"
        if not labels.exists():
            raise SubmissionError(f"{labels} not found. Run scripts/download_uvp6net.py first.")
        df = pd.read_csv(labels, usecols=["objid", f"taxon_level{level}", "set"], dtype={"objid": str})
        df = df[df["set"] == split]
        if df.empty:
            raise SubmissionError(f"No images in split '{split}'. Use uvp6net-test or uvp6net-valid.")
        return df.rename(columns={"objid": "object_id", f"taxon_level{level}": "label"})[["object_id", "label"]]

    if level != 2:
        raise SubmissionError("The NOC answer key is labelled at level 2 only; use --level 2.")
    df = pd.read_csv(truth, dtype={"object_id": str})
    return df[[c for c in ("object_id", "label", "weight") if c in df.columns]]


def read_predictions(source) -> pd.DataFrame:
    """Read a predictions CSV from a path or file-like object."""
    try:
        pred = pd.read_csv(source, dtype=str)
    except Exception as err:  # noqa: BLE001 - any parse failure is the user's file
        raise SubmissionError(f"Could not read the predictions file as CSV: {err}") from err
    pred.columns = [c.strip() for c in pred.columns]
    missing_cols = {"object_id", "label"} - set(pred.columns)
    if missing_cols:
        raise SubmissionError(
            f"Predictions file must have columns 'object_id' and 'label' (missing: {sorted(missing_cols)}).")
    pred = pred[["object_id", "label"]].copy()
    pred["object_id"] = pred["object_id"].str.strip()
    pred["label"] = pred["label"].fillna("").str.strip()
    return pred


def clean_predictions(pred: pd.DataFrame, level: int, classes: pd.DataFrame) -> pd.DataFrame:
    """Check class names and duplicates; map level-1 names to level 2 if scoring at level 2."""
    dupes = pred["object_id"][pred["object_id"].duplicated()]
    if len(dupes):
        raise SubmissionError(f"{len(dupes)} duplicated object_id(s) in predictions, e.g. {dupes.iloc[0]}.")

    pred = pred.copy()
    level1, level2 = set(classes["level1"]), set(classes["level2"])
    if level == 2:
        to_level2 = dict(zip(classes["level1"], classes["level2"], strict=True))
        pred["label"] = [lab if lab in level2 else to_level2.get(lab, lab) for lab in pred["label"]]
        valid = level2 | {UNKNOWN}
    else:
        valid = level1 | {UNKNOWN}

    bad = sorted(set(pred["label"]) - valid)
    if bad:
        raise SubmissionError(f"Unknown level-{level} class name(s) in predictions: {bad[:10]}. "
                              "See configs/uvp6net_classes.csv for valid names.")
    return pred


# ---- scoring --------------------------------------------------------------------

def score(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], weight: np.ndarray | None) -> dict:
    """Accuracy, balanced accuracy, macro F1 and a per-class table over ``classes``."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, sample_weight=weight, zero_division=0)
    w = np.ones(len(y_true)) if weight is None else weight
    return {
        "accuracy": float(np.sum(w * (y_true == y_pred)) / np.sum(w)),
        "balanced_accuracy": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=classes),
    }


def open_set_scores(y_true: np.ndarray, y_pred: np.ndarray, present: list[str], weight: np.ndarray | None) -> dict:
    """Closed-set metrics on known classes, plus how well ``unknown`` was detected."""
    w = np.ones(len(y_true)) if weight is None else weight
    known, flagged = y_true != UNKNOWN, y_pred == UNKNOWN
    closed = score(y_true[known], y_pred[known], [c for c in present if c != UNKNOWN], w[known])
    return {
        "closed_macro_f1": closed["macro_f1"],
        "closed_balanced_accuracy": closed["balanced_accuracy"],
        "unknown_recall": float(np.sum(w * (flagged & ~known)) / np.sum(w * ~known)),
        "false_unknown_rate": float(np.sum(w * (flagged & known)) / np.sum(w * known)),
    }


def evaluate(truth: pd.DataFrame, pred: pd.DataFrame, level: int, classes: pd.DataFrame,
             allow_missing: bool = False) -> dict:
    """Score cleaned predictions against the truth.

    Returns a dict with ``metrics`` ({"unweighted": {...}, "weighted": {...}}, the
    latter only if the truth has weights), a per-class ``table``, a ``confusion``
    matrix and image counts.
    """
    extra = set(pred["object_id"]) - set(truth["object_id"])
    if extra:
        raise SubmissionError(f"{len(extra)} predicted object_id(s) are not in the test set, "
                              f"e.g. {next(iter(extra))}. Are you predicting on the right images?")
    df = truth.merge(pred, on="object_id", how="left", suffixes=("_true", "_pred"))
    n_missing = int(df["label_pred"].isna().sum())
    if n_missing and not allow_missing:
        raise SubmissionError(f"{n_missing} of {len(df)} images have no prediction.")
    df["label_pred"] = df["label_pred"].fillna(MISSING)

    y_true, y_pred = df["label_true"].to_numpy(), df["label_pred"].to_numpy()
    order = [*dict.fromkeys(classes[f"level{level}"]), UNKNOWN]
    present = [c for c in order if c in set(y_true)]
    weightings = {"unweighted": None}
    if "weight" in df.columns:
        weightings["weighted"] = df["weight"].to_numpy(dtype=float)

    metrics, table = {}, pd.DataFrame({"support": pd.Series(y_true).value_counts().reindex(present).astype(int)})
    for kind, w in weightings.items():
        res = score(y_true, y_pred, present, w)
        table = table.join(res.pop("per_class").add_suffix("" if kind == "unweighted" else "_w"))
        if UNKNOWN in present:
            res.update(open_set_scores(y_true, y_pred, present, w))
        metrics[kind] = res

    cm_labels = present + sorted(set(y_pred) - set(present), key=lambda c: (c == MISSING, c.lower()))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=cm_labels), index=cm_labels, columns=cm_labels)

    return {
        "level": level,
        "n_images": len(df),
        "n_missing": n_missing,
        "n_classes": len(present),
        "n_unknown": int((y_true == UNKNOWN).sum()),
        "metrics": metrics,
        "table": table.rename_axis("class"),
        "confusion": cm.loc[present].rename_axis("true \\ predicted"),
    }


# ---- command line ---------------------------------------------------------------

def print_report(result: dict, truth_name: str) -> None:
    n_missing = result["n_missing"]
    print(f"\nScored {result['n_images']} images at level {result['level']} against '{truth_name}' "
          f"({result['n_classes']} classes present{f', {n_missing} missing predictions' if n_missing else ''})\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(result["table"].to_string())
    print()
    for kind, m in result["metrics"].items():
        print(f"{kind:>10}:  macro F1 {m['macro_f1']:.4f}   balanced acc {m['balanced_accuracy']:.4f}   "
              f"accuracy {m['accuracy']:.4f}")
    if result["n_unknown"]:
        print(f"\nOpen set ({result['n_unknown']} images of a class absent from training):")
        for kind, m in result["metrics"].items():
            print(f"{kind:>10}:  closed-set macro F1 {m['closed_macro_f1']:.4f}   "
                  f"unknown recall {m['unknown_recall']:.4f}   false-unknown rate {m['false_unknown_rate']:.4f}")


def save_report(result: dict, truth_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"truth": truth_name, **{k: result[k] for k in ("level", "n_images", "n_missing", "n_classes")},
               **result["metrics"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    result["table"].to_csv(out_dir / "per_class.csv", float_format="%.4f")
    result["confusion"].to_csv(out_dir / "confusion_matrix.csv")
    print(f"\nSaved results to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="CSV with columns object_id,label")
    parser.add_argument("--truth", required=True,
                        help="'uvp6net-test', 'uvp6net-valid', or a CSV with columns object_id,label[,weight]")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="taxonomic level to score at (default: 2, the 32-class level)")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "uvp6net",
                        help="UVP6Net folder made by download_uvp6net.py (default: data/uvp6net)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="score images without a prediction as wrong instead of stopping")
    parser.add_argument("--out-dir", type=Path, help="also save summary.json, per_class.csv, confusion_matrix.csv")
    args = parser.parse_args()

    try:
        classes = load_classes()
        truth = load_truth(args.truth, args.level, args.data_dir)
        pred = clean_predictions(read_predictions(args.predictions), args.level, classes)
        result = evaluate(truth, pred, args.level, classes, allow_missing=args.allow_missing)
    except SubmissionError as err:
        hint = " Use --allow-missing to score them as wrong." if "no prediction" in str(err) else ""
        sys.exit(f"Error: {err}{hint}")

    print_report(result, args.truth)
    if args.out_dir:
        save_report(result, args.truth, args.out_dir)


if __name__ == "__main__":
    main()
