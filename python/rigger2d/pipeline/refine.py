"""Applying the corrections a user makes during refinement.

Edits are stored as a log, not as re-rasterised masks. That keeps the edit list
compact, makes it undoable, and leaves the original segmentation intact as the
source of truth.
"""

from __future__ import annotations

import numpy as np

from ..contracts.models import Edit, JointEdit, MaskEdit, Vec2


def _stroke_mask(
    shape: tuple[int, int],
    points: list[Vec2],
    radius: float,
) -> np.ndarray:
    """Rasterise a polyline of points into a boolean mask."""

    out = np.zeros(shape, dtype=bool)
    if not points:
        return out
    height, width = shape
    r = max(1, int(round(radius)))
    radius_sq = float(r * r)

    for index, point in enumerate(points):
        cx = int(round(point.x))
        cy = int(round(point.y))
        x0, x1 = max(0, cx - r), min(width, cx + r + 1)
        y0, y1 = max(0, cy - r), min(height, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        ys = np.arange(y0, y1)[:, None] - cy
        xs = np.arange(x0, x1)[None, :] - cx
        out[y0:y1, x0:x1] |= (ys * ys + xs * xs) <= radius_sq

        if index + 1 < len(points):
            nxt = points[index + 1]
            out |= _segment_mask(shape, point, nxt, r)

    return out


def _segment_mask(
    shape: tuple[int, int],
    start: Vec2,
    end: Vec2,
    radius: int,
) -> np.ndarray:
    """Rasterise the capsule between two points."""

    out = np.zeros(shape, dtype=bool)
    height, width = shape
    steps = int(max(abs(end.x - start.x), abs(end.y - start.y)) * 2) + 1
    radius_sq = float(radius * radius)

    x0 = max(0, int(min(start.x, end.x)) - radius)
    x1 = min(width, int(max(start.x, end.x)) + radius + 1)
    y0 = max(0, int(min(start.y, end.y)) - radius)
    y1 = min(height, int(max(start.y, end.y)) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return out

    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        dist_sq = (xs - start.x) ** 2 + (ys - start.y) ** 2
    else:
        t = ((xs - start.x) * dx + (ys - start.y) * dy) / length_sq
        t = np.clip(t, 0.0, 1.0)
        dist_sq = (xs - (start.x + t * dx)) ** 2 + (ys - (start.y + t * dy)) ** 2
    out[y0:y1, x0:x1] = dist_sq <= radius_sq
    return out


def apply_mask_edits(
    masks: dict[str, np.ndarray],
    edits: list[Edit],
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Apply the mask part of an edit log to a set of part masks."""

    result = {part_id: mask.copy() for part_id, mask in masks.items()}

    for edit in edits:
        mask_edit: MaskEdit | None = edit.mask
        if mask_edit is None:
            continue

        if mask_edit.part_id not in result:
            blank = np.zeros(shape, dtype=bool)
            result[mask_edit.part_id] = blank

        stroke = _stroke_mask(shape, mask_edit.points, mask_edit.radius)
        if not stroke.any():
            continue

        if mask_edit.op == "add":
            result[mask_edit.part_id] |= stroke
        elif mask_edit.op == "erase":
            result[mask_edit.part_id] &= ~stroke
        elif mask_edit.op == "assign":
            for other_id, other_mask in result.items():
                if other_id == mask_edit.part_id:
                    continue
                result[other_id] = other_mask & ~stroke
            result[mask_edit.part_id] |= stroke

    return result


def apply_joint_edits(
    landmarks: dict[str, Vec2],
    edits: list[Edit],
) -> dict[str, Vec2]:
    """Apply the joint part of an edit log to a set of landmarks."""

    result = dict(landmarks)
    for edit in edits:
        joint_edit: JointEdit | None = edit.joint
        if joint_edit is None:
            continue
        current = result.get(joint_edit.joint_id)
        if current is None:
            continue
        result[joint_edit.joint_id] = Vec2(
            x=current.x + joint_edit.dx,
            y=current.y + joint_edit.dy,
        )
    return result


def summarize(edits: list[Edit]) -> dict[str, int]:
    """A small summary the editor can show without walking the whole log."""

    mask_add = mask_erase = mask_assign = joints = 0
    for edit in edits:
        if edit.mask is not None:
            if edit.mask.op == "add":
                mask_add += 1
            elif edit.mask.op == "erase":
                mask_erase += 1
            else:
                mask_assign += 1
        if edit.joint is not None:
            joints += 1
    return {
        "mask_add": mask_add,
        "mask_erase": mask_erase,
        "mask_assign": mask_assign,
        "joints": joints,
        "total": len(edits),
    }


__all__ = ["apply_joint_edits", "apply_mask_edits", "summarize"]
