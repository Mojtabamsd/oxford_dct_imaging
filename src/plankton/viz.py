"""Small plotting helpers for looking at the data."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from plankton.images import load_image, pad_to_square


def show_images(paths: Sequence[str], titles: Sequence[str] | None = None, ncols: int = 8,
                size: float = 1.6, keep_scale: bool = False):
    """Show images in a grid.

    Args:
        paths: image files.
        titles: optional caption per image.
        ncols: images per row.
        size: inches per image.
        keep_scale: if True, draw every image at the same pixel scale (pasted on a
            canvas as big as the largest image), so you can compare object sizes.
            If False, each image is stretched to fill its cell.
    """
    imgs = [load_image(p) for p in paths]
    nrows = max(1, int(np.ceil(len(imgs) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size), squeeze=False)
    side = max(max(im.size) for im in imgs) if imgs and keep_scale else None
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, im) in enumerate(zip(axes.flat, imgs, strict=False)):
        im = pad_to_square(im)
        if side:
            canvas = np.full((side, side), np.asarray(im)[0, 0], dtype=np.uint8)
            w = im.size[0]
            o = (side - w) // 2
            canvas[o:o + w, o:o + w] = np.asarray(im)
            im = canvas
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=7)
    fig.tight_layout()
    return fig
