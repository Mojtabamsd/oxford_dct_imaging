"""Image loading, preprocessing and a PyTorch Dataset.

UVP images are small greyscale crops of very different sizes (from ~20 px to
~1000 px), dark objects on a light background. Preprocessing:

1. pad the crop to a square, filling with the image's own background colour
   (median of its border pixels), so the object keeps its shape;
2. resize to ``size`` x ``size``;
3. repeat the grey channel 3 times, so ImageNet-pretrained models can be used.

Note that resizing throws away the object's absolute size, which is a strong
cue for some classes. The handcrafted features (``area``, ``feret``, ...) keep it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

try:  # torch is optional: feature-based models don't need it
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str) -> Image.Image:
    """Open an image as 8-bit greyscale."""
    with Image.open(path) as img:
        return img.convert("L")


def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad to a square canvas centred on the image, filled with its background colour."""
    w, h = img.size
    if w == h:
        return img
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    side = max(w, h)
    canvas = Image.new("L", (side, side), int(np.median(border)))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class PadToSquare:
    """torchvision-style transform wrapper around :func:`pad_to_square`."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return pad_to_square(img)


def build_transforms(size: int = 128, train: bool = False):
    """Preprocessing (+ light augmentation when ``train``), returning a 3 x size x size tensor."""
    from torchvision import transforms as T

    steps = [PadToSquare(), T.Resize((size, size), antialias=True)]
    if train:
        # Plankton has no canonical orientation: flips and rotations are safe.
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                  T.RandomRotation(180, fill=255)]
    steps += [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class PlanktonImages(Dataset):
    """A torch Dataset yielding (image, target); target is -1 when ``targets`` is None.

    Defined at module level so DataLoader workers can use it on Windows and macOS.
    """

    def __init__(self, paths: Sequence[str], targets: Sequence[int] | None = None, transform=None):
        self.paths = list(paths)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, (-1 if self.targets is None else int(self.targets[i]))
