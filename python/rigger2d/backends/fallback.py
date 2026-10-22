"""The core's own backend: geometric, dependency-free, and deliberately honest.

This adapter implements every task the core knows about without any machine
learning. It exists for three reasons:

* the pipeline must always run end to end, even with no plugin installed;
* it is the last link of the fallback chain, so a real plugin failing still
  leaves a usable result;
* it is the baseline the evaluation harness compares real backends against.

It is not good at its job, and it says so: the pose it returns has a low
confidence on purpose, so the refinement step is treated as mandatory rather
than advisory.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.base import (
    AdapterContext,
    AdapterResult,
    InpaintPayload,
    MattePayload,
    PartsPayload,
    PosePayload,
)
from ..contracts.models import BBox, ProbeStatus, Task, Vec2
from ..io import images
from ..rigging.templates import RigTemplate


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a 0..1 signal, returned in the same units."""

    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_total = float((hist * centers).sum())
    sum_bg = np.cumsum(hist * centers)
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return 0.5
    mean_bg = np.zeros_like(centers)
    mean_fg = np.zeros_like(centers)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = np.zeros_like(centers)
    between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    return float(centers[int(np.argmax(between))])


def _border_ring(rgb: np.ndarray, width: int = 2) -> np.ndarray:
    h, w = rgb.shape[:2]
    ring = np.concatenate(
        [
            rgb[:width].reshape(-1, 3),
            rgb[-width:].reshape(-1, 3),
            rgb[:, :width].reshape(-1, 3),
            rgb[:, -width:].reshape(-1, 3),
        ],
        axis=0,
    )
    return ring


def _background_flood(rgba: np.ndarray, tolerance: int = 40) -> np.ndarray:
    """Find the background of the image.

    A global colour threshold is not good enough: a character with a pale head
    on a white background falls through it. What works is to ask which
    background coloured areas are *connected to the edge of the picture*. The
    head in the middle of the character is not, so it stays part of the
    character, while the sky around it is.
    """

    rgb = rgba[..., :3]
    ring = _border_ring(rgb, 2).astype(np.float32)
    model = np.median(ring, axis=0)
    looks_like_background = (
        np.abs(rgb.astype(np.float32) - model).max(axis=2) <= tolerance
    )
    background = images.border_connected(looks_like_background)
    return ~background


class _Reference:
    """How one part is measured against the character.

    A part is a capsule around the bone that drives it, with the radius the
    template asks for. The rules that make the split look like the drawing:

    * a pixel inside a capsule belongs to that part;
    * when several capsules overlap, the one drawn on top wins;
    * a pixel inside no capsule at all is left over and merged into the
      biggest part, so the sliced character has no holes.
    """

    __slots__ = ("part_id", "kind", "ax", "ay", "bx", "by", "radius", "z")

    def __init__(
        self,
        part_id: str,
        kind: str,
        a: tuple[float, float],
        b: tuple[float, float],
        radius: float,
        z: int,
    ) -> None:
        self.part_id = part_id
        self.kind = kind
        self.ax, self.ay = a
        self.bx, self.by = b
        self.radius = radius
        self.z = z

    def axis_distance(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        if self.kind == "point":
            return np.sqrt((px - self.ax) ** 2 + (py - self.ay) ** 2)
        dx = self.bx - self.ax
        dy = self.by - self.ay
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-9:
            return np.sqrt((px - self.ax) ** 2 + (py - self.ay) ** 2)
        t = np.clip(((px - self.ax) * dx + (py - self.ay) * dy) / length_sq, 0.0, 1.0)
        return np.sqrt((px - (self.ax + t * dx)) ** 2 + (py - (self.ay + t * dy)) ** 2)


def _part_references(
    template: RigTemplate,
    landmarks: dict[str, Vec2],
    bbox_height: float,
) -> list[_Reference]:
    """One reference per part: its bone if it has one, otherwise its anchor."""

    refs: list[_Reference] = []
    used: set[str] = set()

    for part in template.parts:
        bone = None
        for spec in template.bones:
            if spec.part == part.id and spec.id not in used:
                bone = spec
                break
        if bone is None and part.bone:
            for spec in template.bones:
                if spec.id == part.bone and spec.id not in used:
                    bone = spec
                    break

        radius = part.radius * bbox_height if part.radius else bbox_height * 0.05

        if bone is not None:
            head = landmarks.get(bone.head or "")
            tail = landmarks.get(bone.tail or "")
            if head is not None and tail is not None:
                if (head.x, head.y) != (tail.x, tail.y):
                    used.add(bone.id)
                    refs.append(
                        _Reference(
                            part.id,
                            "segment",
                            (head.x, head.y),
                            (tail.x, tail.y),
                            radius,
                            part.z,
                        )
                    )
                    continue

        for anchor_name in part.anchors or []:
            anchor = landmarks.get(anchor_name)
            if anchor is not None:
                refs.append(
                    _Reference(
                        part.id,
                        "point",
                        (anchor.x, anchor.y),
                        (anchor.x, anchor.y),
                        radius,
                        part.z,
                    )
                )
                break

    # Lowest z first, so that the part drawn on top is written last and wins.
    refs.sort(key=lambda r: r.z)
    return refs


class CoreFallbackAdapter:
    """Geometric stand-in for a real backend."""

    id = "core.fallback"
    version = "1"
    capabilities = frozenset({Task.MATTE, Task.POSE, Task.PARTS, Task.INPAINT})

    def __init__(self, template: RigTemplate | None = None) -> None:
        self._template = template

    # -- probing ---------------------------------------------------------

    def probe(self, ctx: AdapterContext) -> Any:
        from ..adapters.base import ok_probe

        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx="Apache-2.0",
            device="cpu",
            extra={"kind": "geometric baseline"},
        )

    # -- tasks -----------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task == Task.MATTE:
            return self._matte(payload, ctx)
        if task == Task.POSE:
            return self._pose(payload, ctx)
        if task == Task.PARTS:
            return self._parts(payload, ctx)
        if task == Task.INPAINT:
            return self._inpaint(payload, ctx)
        raise ValueError(f"core fallback does not implement {task}")

    def _matte(self, payload: MattePayload, ctx: AdapterContext) -> AdapterResult:
        rgba = images.load_rgba(payload.image_path)
        alpha = rgba[..., 3]
        if int(alpha.min()) < 250:
            mask = alpha > 127
            source = "alpha"
        else:
            mask = _background_flood(rgba)
            source = "background-flood"
        mask = images.largest_component(mask)
        out = payload.output_dir / "matte.png"
        images.save_mask(out, mask)
        bounds = images.mask_bounds(mask)
        return AdapterResult(
            task=Task.MATTE,
            mask_paths={"matte": out},
            detail={
                "coverage": float(mask.mean()),
                "source": source,
            },
        )

    def _pose(self, payload: PosePayload, ctx: AdapterContext) -> AdapterResult:
        template = self._template
        if template is None:
            raise RuntimeError("core fallback needs a template to estimate landmarks")
        bbox = payload.bbox
        landmarks: dict[str, Vec2] = {}
        confidence: dict[str, float] = {}
        for name, spec in template.landmarks.items():
            landmarks[name] = Vec2(
                x=bbox.x0 + spec.x * bbox.width,
                y=bbox.y0 + spec.y * bbox.height,
            )
            # The template says "this is where the landmark usually is". Saying so
            # with a high number would be a lie.
            confidence[name] = 0.25
        return AdapterResult(
            task=Task.POSE,
            landmarks=landmarks,
            confidence=confidence,
            detail={"source": "template-proportions", "confidence": 0.25},
        )

    def _parts(self, payload: PartsPayload, ctx: AdapterContext) -> AdapterResult:
        template = self._template
        if template is None:
            raise RuntimeError("core fallback needs a template to split parts")

        rgba = images.load_rgba(payload.image_path)
        if payload.matte_path is not None and payload.matte_path.exists():
            matte = images.load_mask(payload.matte_path)
        else:
            matte = np.ones(rgba.shape[:2], dtype=bool)

        ys, xs = np.nonzero(matte)
        if len(xs) == 0:
            raise RuntimeError("core fallback found no character pixels")

        px = xs.astype(np.float32)
        py = ys.astype(np.float32)

        # Each part is a capsule around the bone that drives it. A pixel inside
        # a capsule belongs to that part; when capsules overlap, the part drawn
        # on top wins, exactly like the original drawing.
        refs = _part_references(template, payload.landmarks, payload.bbox.height)
        if not refs:
            raise RuntimeError("core fallback found no part references")

        # The nearest capsule, measured as the distance to its axis less its
        # radius. This is what the leftover pixels are given.
        best_score = np.full(len(xs), np.inf, dtype=np.float32)
        best_index = np.zeros(len(xs), dtype=np.int32)
        for index, ref in enumerate(refs):
            signed = ref.axis_distance(px, py) - ref.radius
            closer = signed < best_score
            best_score = np.where(closer, signed, best_score)
            best_index = np.where(closer, index, best_index)

        # A pixel inside a capsule belongs to the part that is drawn on top of
        # it, which is what the drawing itself shows.
        assignment = np.full(len(xs), -1, dtype=np.int32)
        for index, ref in enumerate(refs):
            inside = ref.axis_distance(px, py) <= ref.radius
            assignment[inside] = index

        # A pixel inside no capsule at all -- the gap between two parts --
        # goes to the nearest one. Dumping those pixels into the biggest part
        # instead would leave a seam down the character, and the fill layers
        # would then grow that seam into a band.
        outside = assignment < 0
        assignment[outside] = best_index[outside]

        part_assignment = np.full(matte.shape, -1, dtype=np.int32)
        part_assignment[ys, xs] = assignment

        mask_paths: dict[str, Path] = {}
        for index, ref in enumerate(refs):
            mask = part_assignment == index
            if not mask.any():
                continue
            path = payload.output_dir / f"{ref.part_id}.png"
            images.save_mask(path, mask)
            mask_paths[ref.part_id] = path

        return AdapterResult(
            task=Task.PARTS,
            mask_paths=mask_paths,
            detail={"source": "bone-capsules", "parts": len(mask_paths)},
        )

    def _inpaint(self, payload: InpaintPayload, ctx: AdapterContext) -> AdapterResult:
        """Fill behind a part with a flat version of the part's own colour.

        Not a real inpainting model, but it removes the "hole in the torso"
        problem when a limb rotates, which is the whole point of the pass.
        """

        rgba = images.load_rgba(payload.image_path)
        mask = images.load_mask(payload.mask_path)
        if not mask.any():
            out = payload.output_dir / "inpaint.png"
            images.save_rgba(out, rgba)
            return AdapterResult(task=Task.INPAINT, mask_paths={"inpaint": out})

        filled = rgba.copy()
        edge = mask & ~images.dilate(mask, 2)
        if edge.any():
            colour = rgba[edge][:, :3].mean(axis=0)
        else:
            colour = rgba[mask][:, :3].mean(axis=0)
        filled[mask, :3] = colour.astype(np.uint8)
        filled[mask, 3] = 255

        out = payload.output_dir / "inpaint.png"
        images.save_rgba(out, filled)
        return AdapterResult(
            task=Task.INPAINT,
            mask_paths={"inpaint": out},
            detail={"source": "flat-fill"},
        )


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


__all__ = ["CoreFallbackAdapter", "ProbeStatus"]
