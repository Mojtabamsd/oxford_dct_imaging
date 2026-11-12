#!/usr/bin/env python
"""Write predictions from a trained CNN checkpoint (made by train_cnn.py).

Optionally labels low-confidence images ``unknown``: the simplest open-set
baseline (maximum softmax probability, Hendrycks & Gimpel 2017,
https://arxiv.org/abs/1610.02136). Better open-set methods are part of the challenge.

Examples
--------
    python scripts/predict_cnn.py outputs/cnn_ce/best.pt --on noc
    python scripts/predict_cnn.py outputs/cnn_ce/best.pt --on noc --unknown-threshold 0.5
    python scripts/predict_cnn.py outputs/cnn_ce/best.pt --on uvp6net-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plankton.data import UNKNOWN, load_noc, load_uvp6net, write_predictions
from plankton.inference import load_checkpoint, predict_proba
from plankton.models import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="best.pt from train_cnn.py")
    parser.add_argument("--on", choices=["noc", "uvp6net-test", "uvp6net-valid"], default="noc")
    parser.add_argument("--unknown-threshold", type=float, default=0.0,
                        help="label an image 'unknown' if its top softmax probability is below this "
                             "(0 = never; try 0.3-0.7)")
    parser.add_argument("--out", type=Path, help="output CSV (default: next to the checkpoint)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device(args.device)
    model, classes, cfg = load_checkpoint(args.checkpoint, device)
    df = load_noc() if args.on == "noc" else load_uvp6net(args.on.split("-", 1)[1])

    proba = predict_proba(model, df["path"].tolist(), cfg["img_size"], device, args.batch_size, args.workers)
    labels = np.array(classes, dtype=object)[proba.argmax(1)]
    if args.unknown_threshold > 0:
        uncertain = proba.max(1) < args.unknown_threshold
        labels[uncertain] = UNKNOWN
        print(f"{uncertain.sum():,} of {len(labels):,} images labelled '{UNKNOWN}' "
              f"(top probability < {args.unknown_threshold})")

    suffix = f"_unk{args.unknown_threshold:g}" if args.unknown_threshold > 0 else ""
    out = args.out or args.checkpoint.parent / f"predictions_{args.on.replace('-', '_')}{suffix}.csv"
    print(f"Wrote {write_predictions(df['object_id'], labels, out)}")


if __name__ == "__main__":
    main()
