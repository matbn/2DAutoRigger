"""Building the skeleton.

The rule that makes this a *rigger* rather than a tracer: **bone length follows
the size of the part it drives.** A template says "the upper arm bone runs from
the shoulder to the elbow"; how long it actually is comes from how big the
upper arm is in this particular drawing.
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts.models import BBox, Bone, Vec2
from ..io import images
from .templates import BoneSpec, RigTemplate


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _mask_extent_along(
    mask: np.ndarray,
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float]:
    """Return ``(min_projection, max_projection)`` of the mask along a direction."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    dx, dy = direction
    projections = (xs - origin[0]) * dx + (ys - origin[1]) * dy
    return float(projections.min()), float(projections.max())


def build_skeleton(
    template: RigTemplate,
    landmarks: dict[str, Vec2],
    masks: dict[str, np.ndarray],
    bbox: BBox,
) -> list[Bone]:
    """Turn template bone specs plus landmarks plus masks into bones."""

    bones: list[Bone] = []
    for spec in template.bones:
        head_point = _landmark(landmarks, spec.head)
        tail_point = _landmark(landmarks, spec.tail)
        if head_point is None or tail_point is None:
            continue

        dx = tail_point.x - head_point.x
        dy = tail_point.y - head_point.y
        ux, uy = _unit(dx, dy)

        length = math.hypot(dx, dy)
        mask = masks.get(spec.part) if spec.part else None
        if spec.length == "from_mask" and mask is not None and mask.any():
            # Projections are measured from the head point, so a positive value
            # means "further along the bone" and a negative one "before it".
            lo, hi = _mask_extent_along(mask, (head_point.x, head_point.y), (ux, uy))
            if spec.direction == "tail_to_head":
                length = max(0.0, -lo)
            else:
                length = max(0.0, hi)
            if length < 1.0:
                length = math.hypot(dx, dy)

        tail = Vec2(x=head_point.x + ux * length, y=head_point.y + uy * length)
        bones.append(
            Bone(
                id=spec.id,
                parent=spec.parent,
                head=head_point,
                tail=tail,
                length=length,
                part_id=spec.part,
                layer=spec.part,
                rest_rotation=math.atan2(dy, dx),
            )
        )
    return bones


def _landmark(landmarks: dict[str, Vec2], name: str | None) -> Vec2 | None:
    if not name:
        return None
    return landmarks.get(name)


def bone_children(bones: list[Bone]) -> dict[str, list[Bone]]:
    """Map a bone id to its children."""

    children: dict[str, list[Bone]] = {b.id: [] for b in bones}
    for bone in bones:
        if bone.parent and bone.parent in children:
            children[bone.parent].append(bone)
    return children


def bone_chain(bones: list[Bone], bone_id: str) -> list[Bone]:
    """The bone and its ancestors, nearest first."""

    by_id = {b.id: b for b in bones}
    chain: list[Bone] = []
    current = by_id.get(bone_id)
    while current is not None:
        chain.append(current)
        current = by_id.get(current.parent) if current.parent else None
    return chain


__all__ = ["bone_chain", "bone_children", "build_skeleton"]
