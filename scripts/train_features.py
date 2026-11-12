#!/usr/bin/env python
"""Baseline 1: a random forest on the 62 handcrafted UVPapp features.

No images or GPU needed: only ``python scripts/download_uvp6net.py --parts data``.
Trains on the UVP6Net train split, reports macro F1 on the validation split, and
writes predictions for the UVP6Net test split and, if downloaded, the NOC test set.

Examples
--------
    python scripts/train_features.py
    python scripts/train_features.py --class-weight balanced --out-dir outputs/rf_balanced
    python scripts/train_features.py --max-train 50000      # quicker, for trying things out

Then score it:
    python scripts/evaluate.py outputs/rf/predictions_uvp6net_test.csv --truth uvp6net-test --level 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from plankton.data import FEATURES, load_noc, load_uvp6net_features, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, choices=[1, 2], default=1,
                        help="taxonomic level to train on: 1 (54 classes, default) or 2 (32 classes)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--class-weight", choices=["none", "balanced", "balanced_subsample"], default="none",
                        help="scikit-learn's built-in reweighting: a first imbalance strategy to compare")
    parser.add_argument("--max-train", type=int, default=0,
                        help="randomly subsample the training set to this many rows (0 = use all)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rf"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target = f"level{args.level}"
    print("Loading features...")
    train = load_uvp6net_features("train")
    valid = load_uvp6net_features("valid")
    test = load_uvp6net_features("test")
    if args.max_train and args.max_train < len(train):
        train = train.sample(args.max_train, random_state=args.seed)
    print(f"  train {len(train):,}  valid {len(valid):,}  test {len(test):,}  ({train[target].nunique()} classes)")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1,
        class_weight=None if args.class_weight == "none" else args.class_weight, random_state=args.seed)
    t0 = time.time()
    print("Training random forest (uses all CPU cores)...")
    model.fit(train[FEATURES].to_numpy(np.float32), train[target])
    print(f"  done in {time.time() - t0:.0f} s")

    pred_valid = model.predict(valid[FEATURES].to_numpy(np.float32))
    metrics = {
        "valid_macro_f1": f1_score(valid[target], pred_valid, average="macro"),
        "valid_balanced_accuracy": balanced_accuracy_score(valid[target], pred_valid),
        "valid_accuracy": float(np.mean(pred_valid == valid[target])),
    }
    print(f"Validation ({target}): macro F1 {metrics['valid_macro_f1']:.4f}   "
          f"balanced acc {metrics['valid_balanced_accuracy']:.4f}   accuracy {metrics['valid_accuracy']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps({**metrics, **{k: str(v) for k, v in vars(args).items()}},
                                                          indent=2))
    out = write_predictions(test["object_id"], model.predict(test[FEATURES].to_numpy(np.float32)),
                            args.out_dir / "predictions_uvp6net_test.csv")
    print(f"Wrote {out}")

    try:
        noc = load_noc()
    except FileNotFoundError:
        print("NOC test set not downloaded yet (python scripts/download_noc_uvp6.py); skipping it.")
    else:
        out = write_predictions(noc["object_id"], model.predict(noc[FEATURES].to_numpy(np.float32)),
                                args.out_dir / "predictions_noc.csv")
        print(f"Wrote {out}  <- upload this to the leaderboard")

    print(f"\nScore it:  python scripts/evaluate.py {args.out_dir / 'predictions_uvp6net_test.csv'} "
          f"--truth uvp6net-test --level {args.level}")


if __name__ == "__main__":
    main()
