"""Binding meshes to bones.

Engines do not bind pixels to bones, they bind vertices. This module turns the
skeleton plus the part meshes into per-vertex weights: for every vertex, which
bones influence it and by how much.

A vertex normally follows one bone, but near a joint it blends into the
neighbouring bone, which is what makes an elbow bend instead of tear.
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts.models import Bone, Mesh
from .templates import RigTemplate


def _distance_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def candidate_bones(
    part_id: str,
    template: RigTemplate,
    bones: list[Bone],
) -> list[Bone]:
    """The bones that may influence a part: its own, its parent and its children."""

    by_id = {b.id: b for b in bones}
    ids: set[str] = set()

    part = template.part(part_id)
    if part and part.bone:
        ids.add(part.bone)
    for spec in template.bones:
        if spec.part == part_id:
            ids.add(spec.id)

    base = set(ids)
    for bone_id in base:
        bone = by_id.get(bone_id)
        if bone is not None and bone.parent:
            ids.add(bone.parent)
    for bone in bones:
        if bone.parent in base:
            ids.add(bone.id)

    return [by_id[i] for i in ids if i in by_id]


def bind_mesh(
    mesh: Mesh,
    part_id: str,
    template: RigTemplate,
    bones: list[Bone],
    max_influences: int = 4,
) -> Mesh:
    """Fill ``mesh.bones`` and ``mesh.weights``."""

    if not mesh.vertices or not bones:
        return mesh

    candidates = candidate_bones(part_id, template, bones)
    if not candidates:
        candidates = bones

    bone_index = {bone.id: index for index, bone in enumerate(bones)}
    indices = [bone_index[b.id] for b in candidates]

    for vertex in mesh.vertices:
        distances = [
            _distance_to_segment(
                vertex.x,
                vertex.y,
                bone.head.x,
                bone.head.y,
                bone.tail.x,
                bone.tail.y,
            )
            for bone in candidates
        ]
        scores = [1.0 / (distance + 2.0) for distance in distances]
        order = np.argsort(scores)[::-1][:max_influences]

        chosen: list[int] = []
        weights: list[float] = []
        for position in order:
            score = float(scores[position])
            if score <= 0:
                continue
            chosen.append(indices[position])
            weights.append(score)
        if not chosen:
            chosen = [indices[0]]
            weights = [1.0]

        total = sum(weights) or 1.0
        weights = [w / total for w in weights]

        # Keep the arrays the same length as the influence limit so that
        # engines with a fixed-width format (Godot) can consume them directly.
        while len(chosen) < max_influences:
            chosen.append(chosen[0])
            weights.append(0.0)

        mesh.bones.append([int(c) for c in chosen])
        mesh.weights.append([float(w) for w in weights])

    return mesh


def bind_all(
    meshes: list[Mesh],
    template: RigTemplate,
    bones: list[Bone],
) -> list[Mesh]:
    for mesh in meshes:
        bind_mesh(mesh, mesh.part_id, template, bones)
    return meshes


__all__ = ["bind_all", "bind_mesh", "candidate_bones"]
