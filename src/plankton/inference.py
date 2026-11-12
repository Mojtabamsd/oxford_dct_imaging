"""Run a trained CNN over a list of images, and save/load checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def predict_proba(model, paths, img_size: int, device, batch_size: int = 256, workers: int = 4) -> np.ndarray:
    """Softmax probabilities, shape (n_images, n_classes)."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from plankton.images import PlanktonImages, build_transforms

    loader = DataLoader(PlanktonImages(paths, transform=build_transforms(img_size, train=False)),
                        batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval().to(device)
    out = []
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc="predict", leave=False):
            out.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(path: Path, model, classes: list[str], config: dict) -> None:
    import torch

    torch.save({"state_dict": model.state_dict(), "classes": classes, "config": config}, path)


def load_checkpoint(path: Path, device="cpu"):
    """Returns (model, classes, config)."""
    import torch

    from plankton.models import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["classes"], cfg
