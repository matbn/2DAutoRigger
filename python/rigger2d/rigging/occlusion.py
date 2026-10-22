"""Slicing the character into layers and ordering them.

Two problems are solved here.

**Order.** The template gives every part a ``z``. The layers are emitted in
drawing order, back to front.

**Holes.** The parts of a character overlap in the drawing: the arm is drawn on
top of the torso. Once the character is cut apart, the area the arm used to
cover is missing from the torso, so rotating the arm reveals a hole.

The fix is a *fill* layer. It is the part's own mask grown a little, painted
with the colour of the part's edge, and placed **directly in front of the part
it belongs to**. The part below it therefore keeps a body under the limb that
covers it, and when the limb moves away, the body is still there.

The grown area is cut down to the parts that are drawn *above* this one. A
fill that grew into empty background would paint a halo around the character,
and a fill that grew over a part below would paint the wrong colour over it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.models import BBox, Layer, SkinMode, Vec2
from ..io import images
from .templates import RigTemplate


def _edge_colour(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edge = mask & ~images.dilate(mask, 2)
    if edge.any():
        return rgba[edge][:, :3].mean(axis=0)
    if mask.any():
        return rgba[mask][:, :3].mean(axis=0)
    return np.array([128.0, 128.0, 128.0], dtype=np.float64)


def _crop_and_compose(
    rgba: np.ndarray,
    mask: np.ndarray,
    bounds: BBox,
) -> tuple[np.ndarray, int, int]:
    height, width = rgba.shape[:2]
    x0 = max(0, int(bounds.x0))
    y0 = max(0, int(bounds.y0))
    x1 = min(width, int(bounds.x1))
    y1 = min(height, int(bounds.y1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 4), dtype=np.uint8), 0, 0

    patch = rgba[y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    patch[..., 3] = np.where(sub_mask, patch[..., 3], 0)
    return patch, x0, y0


def build_layers(
    rgba: np.ndarray,
    masks: dict[str, np.ndarray],
    template: RigTemplate,
    landmarks: dict[str, Vec2],
    out_dir: Path,
    with_fill: bool = True,
) -> list[Layer]:
    """Write one PNG per part and return the layers in drawing order."""

    out_dir.mkdir(parents=True, exist_ok=True)
    layers: list[Layer] = []

    ordered_parts = sorted(template.parts, key=lambda p: (p.z, p.id))

    # What is drawn above each part. A fill may only grow into this, or it
    # would paint a halo around the character or the wrong colour over the
    # part below it.
    above: dict[str, np.ndarray] = {}
    running = np.zeros(rgba.shape[:2], dtype=bool)
    for part in reversed(ordered_parts):
        above[part.id] = running.copy()
        mask = masks.get(part.id)
        if mask is not None:
            running |= mask

    # The character itself. A fill fills a hole in the character; it does not
    # grow the character, so it may not reach outside it.
    character = np.zeros(rgba.shape[:2], dtype=bool)
    for mask in masks.values():
        character |= mask

    for part in ordered_parts:
        mask = masks.get(part.id)
        if mask is None or not mask.any():
            continue

        bounds = images.mask_bounds(mask)

        # The fill goes *behind* the part. It covers the area a limb hides, so
        # at rest it is completely hidden by the limb itself; it only shows
        # when the limb moves away.
        if with_fill:
            radius = max(2, int(round(part.radius * bounds.height * 0.6))) if part.radius else 0
            if radius > 0:
                grown = images.dilate(mask, radius)
                fill_mask = (
                    grown
                    & above.get(part.id, np.zeros_like(mask))
                    & ~mask
                    & character
                )
                if fill_mask.any():
                    colour = _edge_colour(rgba, mask)
                    fill_bounds = images.mask_bounds(fill_mask)
                    fill_patch, fx, fy = _crop_and_compose(rgba, fill_mask, fill_bounds)
                    if fill_patch.size:
                        solid = np.zeros_like(fill_patch)
                        solid[..., :3] = colour.astype(np.uint8)
                        # Keep the shape of the fill: only the colour is flat.
                        solid[..., 3] = fill_patch[..., 3]
                        fill_path = out_dir / f"{part.id}_fill.png"
                        images.save_rgba(fill_path, solid)
                        layers.append(
                            Layer(
                                id=f"{part.id}_fill",
                                part_id=part.id,
                                image=fill_path.name,
                                z=part.z,
                                pivot=Vec2(x=fx, y=fy),
                                bounds=fill_bounds,
                                skin=SkinMode.RIGID,
                                kind="fill",
                            )
                        )

        patch, offset_x, offset_y = _crop_and_compose(rgba, mask, bounds)
        if patch.size:
            path = out_dir / f"{part.id}.png"
            images.save_rgba(path, patch)

            pivot = Vec2(x=0.0, y=0.0)
            spec = next((b for b in template.bones if b.part == part.id), None)
            if spec and spec.head and spec.head in landmarks:
                pivot = landmarks[spec.head]

            layers.append(
                Layer(
                    id=part.id,
                    part_id=part.id,
                    image=path.name,
                    z=part.z,
                    pivot=pivot,
                    bounds=bounds,
                    skin=part.skin,
                    kind="part",
                )
            )

    _write_preview(rgba, layers, out_dir)
    return layers


def _write_preview(rgba: np.ndarray, layers: list[Layer], out_dir: Path) -> Path:
    """Composite the layers back into one picture.

    The preview is what the editor shows before the character is sliced, so it
    has to look like the character did. If the layers do not add back up, the
    slicing is wrong, and this is where that shows up.
    """

    canvas = np.zeros_like(rgba)
    for layer in layers:
        path = out_dir / layer.image
        if not path.is_file():
            continue
        patch = images.load_rgba(path)
        if not patch.size:
            continue
        x = int(layer.bounds.x0)
        y = int(layer.bounds.y0)
        height = min(patch.shape[0], canvas.shape[0] - y)
        width = min(patch.shape[1], canvas.shape[1] - x)
        if height <= 0 or width <= 0:
            continue
        images.paste_into(canvas, patch[:height, :width], x, y)

    return images.save_rgba(out_dir / "preview.png", canvas)


__all__ = ["build_layers"]
