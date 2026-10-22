"""Small image helpers shared by the pipeline, the adapters and the scoring.

Masks are plain ``numpy`` boolean arrays of shape ``(H, W)``. On disk they are
stored as 8-bit grayscale PNGs where 255 means "inside the part".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.models import BBox


def load_rgba(path: Path) -> np.ndarray:
    """Load an image as ``(H, W, 4)`` uint8 RGBA."""

    with Image.open(path) as img:
        return np.array(img.convert("RGBA"), dtype=np.uint8)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as ``(H, W, 3)`` uint8 RGB."""

    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_mask(path: Path, mask: np.ndarray) -> Path:
    """Write a boolean mask as an 8-bit grayscale PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)
    return path


def load_mask(path: Path) -> np.ndarray:
    """Read a mask PNG back into a boolean array."""

    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8) > 127


def save_rgba(path: Path, rgba: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)
    return path


def mask_bounds(mask: np.ndarray) -> BBox:
    """Bounding box of the set pixels, or an empty box at the origin."""

    if not mask.any():
        return BBox(x0=0, y0=0, x1=0, y1=0)
    ys, xs = np.nonzero(mask)
    return BBox(x0=float(xs.min()), y0=float(ys.min()), x1=float(xs.max() + 1), y1=float(ys.max() + 1))


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by ``radius`` pixels using a box filter."""

    if radius <= 0:
        return mask
    size = radius * 2 + 1
    ys = np.ones((size, size), dtype=np.float32)
    # Separable box blur via cumulative sums keeps this O(n) and dependency free.
    padded = np.pad(mask.astype(np.float32), radius, mode="constant")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, w = mask.shape
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    del ys
    return total > 0


def label_components(mask: np.ndarray) -> np.ndarray:
    """Label the connected blobs of a mask.

    The blobs are found by labelling horizontal runs and joining the runs that
    overlap from one row to the next. Walking the pixels one at a time is
    simple but far too slow on a full size character.

    Returns an array of labels, where 0 means "not part of the mask" and every
    blob has its own number.
    """

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not mask.any():
        return labels

    parent: list[int] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    rows: list[list[tuple[int, int, int]]] = []
    previous: list[tuple[int, int, int]] = []
    zero = np.zeros(1, dtype=np.int8)
    for y in range(height):
        row = mask[y].view(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate((zero, row, zero))))
        starts = edges[0::2].tolist()
        ends = edges[1::2].tolist()
        current: list[tuple[int, int, int]] = []
        for start, end in zip(starts, ends, strict=True):
            label = len(parent)
            parent.append(label)
            for prev_start, prev_end, prev_label in previous:
                if start < prev_end and prev_start < end:
                    union(label, prev_label)
            current.append((start, end, label))
        rows.append(current)
        previous = current

    # Give every run the label of the blob it belongs to.
    for y, current in enumerate(rows):
        for start, end, label in current:
            labels[y, start:end] = find(label)
    return labels


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob (4-connectivity)."""

    if not mask.any():
        return mask
    labels = label_components(mask)
    if not labels.any():
        return mask
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    return labels == int(counts.argmax())


def border_connected(mask: np.ndarray) -> np.ndarray:
    """Keep the blobs of a mask that reach the edge of the image.

    This is how the background is found: the background is whatever background
    coloured area is connected to the edge of the picture. A pale head in the
    middle of a white background is not connected to the edge, so it stays part
    of the character.
    """

    if not mask.any():
        return mask
    labels = label_components(mask)
    if not labels.any():
        return mask
    touching = set(np.unique(labels[0, :]).tolist())
    touching.update(np.unique(labels[-1, :]).tolist())
    touching.update(np.unique(labels[:, 0]).tolist())
    touching.update(np.unique(labels[:, -1]).tolist())
    touching.discard(0)
    if not touching:
        return np.zeros_like(mask)
    return np.isin(labels, list(touching))


def gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of an RGB image, normalised to 0..1."""

    grey = rgb.astype(np.float32).mean(axis=2) / 255.0
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(grey, 1, mode="edge")
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    for dy in range(3):
        for dx in range(3):
            window = padded[dy : dy + grey.shape[0], dx : dx + grey.shape[1]]
            gx += kx[dy, dx] * window
            gy += ky[dy, dx] * window
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(mag.max())
    return mag / peak if peak > 0 else mag


def crop_to_bounds(rgba: np.ndarray, bounds: BBox) -> tuple[np.ndarray, int, int]:
    """Crop an RGBA image to a box, returning ``(crop, offset_x, offset_y)``."""

    h, w = rgba.shape[:2]
    x0 = max(0, int(bounds.x0))
    y0 = max(0, int(bounds.y0))
    x1 = min(w, int(bounds.x1))
    y1 = min(h, int(bounds.y1))
    if x1 <= x0 or y1 <= y0:
        return rgba[0:0, 0:0], 0, 0
    return rgba[y0:y1, x0:x1], x0, y0


def paste_into(canvas: np.ndarray, patch: np.ndarray, offset_x: int, offset_y: int) -> None:
    """Alpha-composite ``patch`` into ``canvas`` at the given offset."""

    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, offset_x), max(0, offset_y)
    x1, y1 = min(cw, offset_x + pw), min(ch, offset_y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    src = patch[y0 - offset_y : y1 - offset_y, x0 - offset_x : x1 - offset_x]
    dst = canvas[y0:y1, x0:x1]
    src_alpha = src[..., 3:4].astype(np.float32) / 255.0
    dst_alpha = dst[..., 3:4].astype(np.float32) / 255.0
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe = np.where(out_alpha > 0, out_alpha, 1.0)
    dst_rgb = dst[..., :3].astype(np.float32)
    src_rgb = src[..., :3].astype(np.float32)
    blended = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1.0 - src_alpha)) / safe
    dst[..., :3] = np.clip(blended, 0, 255).astype(np.uint8)
    dst[..., 3:4] = (np.clip(out_alpha, 0, 1) * 255).astype(np.uint8)


__all__ = [
    "crop_to_bounds",
    "dilate",
    "gradient_magnitude",
    "largest_component",
    "load_mask",
    "load_rgb",
    "load_rgba",
    "mask_area",
    "mask_bounds",
    "paste_into",
    "save_mask",
    "save_rgba",
]
