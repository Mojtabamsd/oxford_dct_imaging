#!/usr/bin/env python
"""Baseline 2: fine-tune an ImageNet-pretrained CNN on the UVP6Net images.

Runs on a GPU if one is available (CUDA or Apple MPS), otherwise on the CPU.
A full epoch over the 444k training images takes a few minutes on a modern GPU
and hours on a CPU: on a CPU, use ``--max-per-class`` to train on a subset.

The model with the best validation macro F1 is saved, then used to write
predictions for the UVP6Net test split and (if downloaded) the NOC test set.

Examples
--------
    # Quick check that everything works on a CPU (~10 min, mostly predicting the 95k test images)
    python scripts/train_cnn.py --max-per-class 50 --max-valid 2000 --epochs 1 --img-size 64 \
        --out-dir outputs/cnn_smoke

    # Baseline on a GPU
    python scripts/train_cnn.py --epochs 10 --out-dir outputs/cnn_ce

    # Your imbalance strategies (implement them in src/plankton/imbalance.py first)
    python scripts/train_cnn.py --loss focal --out-dir outputs/cnn_focal
    python scripts/train_cnn.py --sampler balanced --out-dir outputs/cnn_balanced
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from plankton.data import class_names, load_noc, load_uvp6net, write_predictions
from plankton.images import PlanktonImages, build_transforms
from plankton.imbalance import LOSSES, SAMPLERS, class_counts, make_loss, make_sampler
from plankton.inference import predict_proba, save_checkpoint
from plankton.models import ARCHITECTURES, build_model, get_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, choices=[1, 2], default=1, help="1 = 54 classes (default), 2 = 32")
    p.add_argument("--arch", choices=ARCHITECTURES, default="resnet18")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch instead of ImageNet weights")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=LOSSES, default="ce")
    p.add_argument("--sampler", choices=SAMPLERS, default="none")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="cap training images per class, for quick runs (0 = use all). Note this also "
                        "changes the class balance!")
    p.add_argument("--max-valid", type=int, default=0,
                   help="validate on a random subset of this many images each epoch, for quick runs "
                        "(0 = all 95k)")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps or cpu")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/cnn"))
    return p.parse_args()


def evaluate(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
        for x, y in loader:
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = device.type == "cuda"  # mixed precision: faster on GPUs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------------
    classes = class_names(args.level)
    index = {c: i for i, c in enumerate(classes)}
    target = f"level{args.level}"
    train_df, valid_df = load_uvp6net("train"), load_uvp6net("valid")
    if args.max_per_class:
        shuffled = train_df.sample(frac=1, random_state=args.seed)
        train_df = shuffled[shuffled.groupby(target).cumcount() < args.max_per_class].reset_index(drop=True)
    if args.max_valid and args.max_valid < len(valid_df):
        valid_df = valid_df.sample(args.max_valid, random_state=args.seed).reset_index(drop=True)
    y_train = train_df[target].map(index).to_numpy()
    y_valid = valid_df[target].map(index).to_numpy()
    print(f"Device {device} | {len(classes)} classes | train {len(train_df):,} | valid {len(valid_df):,}")

    sampler = make_sampler(args.sampler, y_train)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        PlanktonImages(train_df["path"], y_train, build_transforms(args.img_size, train=True)),
        batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(
        PlanktonImages(valid_df["path"], y_valid, build_transforms(args.img_size, train=False)),
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    # ---- model -----------------------------------------------------------------
    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = make_loss(args.loss, class_counts(y_train, len(classes)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "device": str(device)}

    # ---- train -----------------------------------------------------------------
    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, n = total + loss.item() * len(y), n + len(y)

        yt, yp = evaluate(model, valid_loader, device, amp)
        row = {"epoch": epoch, "train_loss": total / n,
               "valid_macro_f1": f1_score(yt, yp, average="macro"),
               "valid_balanced_accuracy": balanced_accuracy_score(yt, yp),
               "valid_accuracy": float(np.mean(yt == yp)), "seconds": round(time.time() - t0)}
        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
        flag = ""
        if row["valid_macro_f1"] > best_f1:
            best_f1, flag = row["valid_macro_f1"], "  * best, saved"
            save_checkpoint(args.out_dir / "best.pt", model, classes, config)
        print(f"epoch {epoch:>2}: loss {row['train_loss']:.4f}  valid macro F1 {row['valid_macro_f1']:.4f}  "
              f"bal acc {row['valid_balanced_accuracy']:.4f}  acc {row['valid_accuracy']:.4f}  "
              f"({row['seconds']} s){flag}")

    # ---- predict with the best model ---------------------------------------------
    model.load_state_dict(torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)["state_dict"])
    (args.out_dir / "config.json").write_text(json.dumps({**config, "best_valid_macro_f1": best_f1}, indent=2))
    sets = {"uvp6net_test": load_uvp6net("test")}
    try:
        sets["noc"] = load_noc()
    except FileNotFoundError:
        print("NOC test set not downloaded yet (python scripts/download_noc_uvp6.py); skipping it.")
    for name, df in sets.items():
        proba = predict_proba(model, df["path"].tolist(), args.img_size, device, args.batch_size, args.workers)
        np.save(args.out_dir / f"proba_{name}.npy", proba.astype(np.float16))
        out = write_predictions(df["object_id"], np.array(classes)[proba.argmax(1)],
                                args.out_dir / f"predictions_{name}.csv")
        print(f"Wrote {out}" + ("  <- upload this to the leaderboard" if name == "noc" else ""))

    print(f"\nScore it:  python scripts/evaluate.py {args.out_dir / 'predictions_uvp6net_test.csv'} "
          f"--truth uvp6net-test --level {args.level}")


if __name__ == "__main__":
    main()
