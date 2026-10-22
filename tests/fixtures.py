"""Test fixtures: a synthetic character to rig without needing a real drawing.

The character is built **from the rig template**, so the proportions are known
exactly and the pipeline can be checked against ground truth. Each body part is
a capsule along the bone that drives it, with the width the template asks for.

The character is drawn twice: once to find out how much room the drawing
actually needs, then again with the landmarks placed on the real bounding box.
Without that second pass the proportions would be measured against a box that
does not exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rigger2d.rigging.templates import TemplateRegistry

# A colour per part, so the layers are easy to tell apart by eye.
COLOURS = {
    "head": (245, 222, 190),
    "neck": (235, 205, 175),
    "torso": (70, 130, 200),
    "pelvis": (55, 110, 170),
    "upper_arm_l": (220, 90, 80),
    "lower_arm_l": (225, 130, 80),
    "hand_l": (230, 170, 120),
    "upper_arm_r": (220, 90, 80),
    "lower_arm_r": (225, 130, 80),
    "hand_r": (230, 170, 120),
    "upper_leg_l": (70, 160, 90),
    "lower_leg_l": (110, 190, 110),
    "foot_l": (60, 80, 60),
    "upper_leg_r": (70, 160, 90),
    "lower_leg_r": (110, 190, 110),
    "foot_r": (60, 80, 60),
}

_NOMINAL_BOX = (70.0, 80.0, 442.0, 960.0)


def _template():
    return TemplateRegistry().load("humanoid.biped.v1")


def _landmarks(template, box: tuple[float, float, float, float]):
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    return {
        name: (x0 + spec.x * width, y0 + spec.y * height)
        for name, spec in template.landmarks.items()
    }


def _bone_for_part(template, part_id: str):
    for spec in template.bones:
        if spec.part == part_id:
            return spec
    return None


def _draw_part_layer(template, landmarks, box, part_id) -> Image.Image:
    """Draw one part on its own transparent layer."""

    part = template.part(part_id)
    bone = _bone_for_part(template, part_id)
    if part is None or bone is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    height = box[3] - box[1]
    width = max(2, int(round(2 * part.radius * height)))
    colour = COLOURS.get(part_id, (150, 150, 150)) + (255,)

    layer = Image.new("RGBA", (512, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    start = landmarks.get(bone.head or "")
    end = landmarks.get(bone.tail or "")
    if start is None or end is None:
        return layer

    draw.line([start, end], fill=colour, width=width)
    radius = width // 2
    for point in (start, end):
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=colour,
        )

    # A capsule sticks out past the ends of its bone, so it would grow past
    # the character's box. The character is cut off at the box, which keeps the
    # drawing and the landmark placement in agreement -- and the fallback
    # places the landmarks over the character's box, so it has to.
    box_x0, box_y0, box_x1, box_y1 = (int(round(v)) for v in box)
    keep = Image.new("L", (512, 1024), 0)
    ImageDraw.Draw(keep).rectangle([box_x0, box_y0, box_x1, box_y1], fill=255)
    alpha = layer.getchannel("A")
    layer.putalpha(Image.composite(alpha, Image.new("L", (512, 1024), 0), keep))
    return layer


def _composite(layers: list[tuple[str, Image.Image]]) -> Image.Image:
    canvas = Image.new("RGBA", (512, 1024), (255, 255, 255, 255))
    for _part_id, layer in layers:
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


def _content_box(image: Image.Image) -> tuple[float, float, float, float]:
    array = np.array(image.convert("RGB"), dtype=np.int16)
    # The background is white; anything else is part of the character.
    non_background = np.any(np.abs(array - 255) > 12, axis=2)
    ys, xs = np.nonzero(non_background)
    if len(xs) == 0:
        return _NOMINAL_BOX
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def build_character() -> tuple[Image.Image, dict[str, Image.Image], tuple[float, float, float, float]]:
    """Build the character, its per-part layers and its real bounding box.

    The bounding box and the landmark placement have to agree, and each depends
    on the other: the capsules stick out past the outermost landmarks, so the
    box the landmarks are placed in is not the box the drawing occupies. The
    two are iterated until they agree.
    """

    template = _template()
    # The parts are drawn back to front, by the z the template gives them.
    # The ground truth and the pipeline both work in that order, so the
    # drawing has to as well.
    parts = [p.id for p in sorted(template.parts, key=lambda p: (p.z, p.id))]

    box = _NOMINAL_BOX
    layers: dict[str, Image.Image] = {}
    for _ in range(8):
        landmarks = _landmarks(template, box)
        layers = {pid: _draw_part_layer(template, landmarks, box, pid) for pid in parts}
        new_box = _content_box(_composite([(pid, layers[pid]) for pid in parts]))
        if new_box == box:
            break
        box = new_box

    image = _composite([(pid, layers[pid]) for pid in parts])
    return image, layers, box


def draw_character(path: Path, width: int = 512, height: int = 1024) -> Path:
    """Draw the synthetic character and return the path it was written to."""

    image, _layers, _box = build_character()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_character_transparent(path: Path, width: int = 512, height: int = 1024) -> Path:
    """The same character on a transparent background."""

    image, _layers, _box = build_character()
    array = np.array(image.convert("RGBA"))
    alpha = (np.abs(array[..., :3].astype(np.int16) - 255) > 12).any(axis=2)
    array[..., 3] = np.where(alpha, 255, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGBA").save(path)
    return path


def ground_truth_masks() -> dict[str, np.ndarray]:
    """The correct segmentation of the synthetic character.

    A segmentation gives every pixel to exactly one part, and where the parts
    overlap the one drawn on top wins. That is what the composite looks like,
    so the ground truth is built from the composite rather than from the
    individual layers.
    """

    template = _template()
    _image, layers, _box = build_character()
    owner = np.full((1024, 512), -1, dtype=np.int32)
    ordered = sorted(template.parts, key=lambda p: p.z)
    for index, part in enumerate(ordered):
        layer = layers.get(part.id)
        if layer is None:
            continue
        alpha = np.array(layer.convert("RGBA"))[..., 3] > 127
        owner[alpha] = index

    masks: dict[str, np.ndarray] = {}
    for index, part in enumerate(ordered):
        masks[part.id] = owner == index
    return masks


def character_box() -> tuple[float, float, float, float]:
    _image, _layers, box = build_character()
    return box


def noisy_mask_image(path: Path, width: int = 256, height: int = 256) -> Path:
    """A plain image used to exercise the matte code."""

    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "COLOURS",
    "build_character",
    "character_box",
    "draw_character",
    "draw_character_transparent",
    "ground_truth_masks",
    "noisy_mask_image",
]
