"""Choosing a backend, and falling back when one fails.

Two related jobs live here:

* :func:`run_chain` walks the candidate list for a task in priority order and
  returns the first result that is good enough. A backend that crashes, times
  out or returns nonsense does not end the pipeline -- the next candidate runs.
* :func:`compare_all` runs every candidate on the same image and returns the
  score table, which is what lets the editor show "BiRefNet scored 0.82, SAM 3
  scored 0.61 -- switch?" and lets the user override the automatic choice.

Scores are deliberately simple and explainable. A number nobody can reason
about is a number nobody can trust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.base import AdapterContext, AdapterResult
from ..adapters.registry import PluginEntry, Registry, probe_entry
from ..contracts.models import Probe, Task
from ..io import images
from ..rigging.templates import RigTemplate


@dataclass
class Attempt:
    """One backend, one try."""

    adapter_id: str
    ok: bool = False
    score: float | None = None
    error: str | None = None
    duration_ms: int = 0
    probe_status: str = ""
    probe_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "ok": self.ok,
            "score": self.score,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "probe_status": self.probe_status,
            "probe_reason": self.probe_reason,
        }


@dataclass
class Selection:
    """The outcome of running the fallback chain."""

    task: Task
    adapter_id: str
    result: AdapterResult
    score: float = 0.0
    attempts: list[Attempt] = field(default_factory=list)
    adapter_version: str = "0"
    license_spdx: str | None = None
    license_url: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "adapter_id": self.adapter_id,
            "score": self.score,
            "attempts": [a.to_json() for a in self.attempts],
        }


def _holes_ratio(mask: np.ndarray) -> float:
    """Fraction of the mask's bounding box that is background enclosed by it."""

    bounds = images.mask_bounds(mask)
    if bounds.area <= 0:
        return 1.0
    x0, y0 = int(bounds.x0), int(bounds.y0)
    x1, y1 = min(mask.shape[1], int(bounds.x1)), min(mask.shape[0], int(bounds.y1))
    sub = mask[y0:y1, x0:x1]
    if not sub.any():
        return 1.0
    h, w = sub.shape
    background = ~sub
    seen = np.zeros_like(background)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        for y in (0, h - 1):
            if background[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if background[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    enclosed = background & ~seen
    return float(enclosed.sum()) / float(max(1, sub.sum()))


def score_matte(result: AdapterResult, payload: Any) -> float:
    """Reward a matte that covers a believable amount of the frame and hugs edges."""

    path = result.mask_paths.get("matte")
    if path is None or not Path(path).exists():
        return 0.0
    mask = images.load_mask(Path(path))
    if not mask.any():
        return 0.0
    coverage = float(mask.mean())
    # A character usually covers a third of the frame; punish extremes.
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.35) / 0.55)

    edge_score = 0.0
    try:
        rgb = images.load_rgb(Path(payload.image_path))
        gradient = images.gradient_magnitude(rgb)
        boundary = mask & ~images.dilate(mask, 2)
        if boundary.any():
            edge_score = float(np.clip(gradient[boundary].mean() * 3.0, 0.0, 1.0))
    except Exception:  # noqa: BLE001 - scoring must never break the pipeline
        edge_score = 0.0

    hole_score = max(0.0, 1.0 - _holes_ratio(mask))
    return float(np.clip(0.5 * coverage_score + 0.3 * edge_score + 0.2 * hole_score, 0.0, 1.0))


def score_parts(result: AdapterResult, payload: Any, template: RigTemplate | None) -> float:
    """Reward parts that tile the character without gaps or overlaps."""

    if not result.mask_paths:
        return 0.0
    masks: dict[str, np.ndarray] = {}
    for part_id, path in result.mask_paths.items():
        p = Path(path)
        if p.exists():
            masks[part_id] = images.load_mask(p)
    if not masks:
        return 0.0

    shape = next(iter(masks.values())).shape
    union = np.zeros(shape, dtype=bool)
    overlap = np.zeros(shape, dtype=np.int32)
    for mask in masks.values():
        union |= mask
        overlap += mask.astype(np.int32)

    covered = float((overlap > 0).mean())
    overlap_ratio = float((overlap > 1).mean())
    coverage_score = max(0.0, 1.0 - abs(covered - 0.85) / 0.4)
    overlap_score = max(0.0, 1.0 - overlap_ratio * 12.0)

    proportion_score = 1.0
    if template and template.proportions:
        total = float(union.sum()) or 1.0
        expected = template.proportions
        errors: list[float] = []
        for part_id, mask in masks.items():
            want = expected.get(part_id)
            if want is None:
                continue
            got = float(mask.sum()) / total
            errors.append(min(1.0, abs(got - want) / max(want, 1e-6)))
        if errors:
            proportion_score = max(0.0, 1.0 - (sum(errors) / len(errors)))

    landmark_score = 1.0
    landmarks = getattr(payload, "landmarks", {}) or {}
    if landmarks:
        hits = 0
        total = 0
        for part_id, mask in masks.items():
            total += 1
            anchors = template.part(part_id).anchors if template and template.part(part_id) else []
            anchor = landmarks.get(anchors[0]) if anchors else None
            if anchor is None:
                continue
            x = int(np.clip(anchor.x, 0, mask.shape[1] - 1))
            y = int(np.clip(anchor.y, 0, mask.shape[0] - 1))
            if mask[y, x]:
                hits += 1
        landmark_score = hits / total if total else 1.0

    return float(
        np.clip(
            0.35 * coverage_score + 0.25 * overlap_score + 0.25 * proportion_score + 0.15 * landmark_score,
            0.0,
            1.0,
        )
    )


def score_pose(result: AdapterResult, payload: Any) -> float:
    """Reward landmarks that are both confident and inside the character."""

    if not result.landmarks:
        return 0.0
    confidence = result.confidence or {k: 0.5 for k in result.landmarks}
    mean_conf = float(np.mean(list(confidence.values()))) if confidence else 0.0

    bbox = getattr(payload, "bbox", None)
    inside = 1.0
    if bbox is not None:
        ok = 0
        for point in result.landmarks.values():
            if bbox.x0 - 1 <= point.x <= bbox.x1 + 1 and bbox.y0 - 1 <= point.y <= bbox.y1 + 1:
                ok += 1
        inside = ok / len(result.landmarks)

    return float(np.clip(0.6 * mean_conf + 0.4 * inside, 0.0, 1.0))


def score_inpaint(result: AdapterResult, payload: Any) -> float:
    path = result.mask_paths.get("inpaint")
    if path is None or not Path(path).exists():
        return 0.0
    return 0.6


def score_result(
    task: Task,
    result: AdapterResult,
    payload: Any,
    template: RigTemplate | None = None,
) -> float:
    if task == Task.MATTE:
        return score_matte(result, payload)
    if task == Task.PARTS:
        return score_parts(result, payload, template)
    if task == Task.POSE:
        return score_pose(result, payload)
    if task == Task.INPAINT:
        return score_inpaint(result, payload)
    if task == Task.CLASSIFY:
        return 1.0
    return 0.0


def _candidates(
    registry: Registry,
    task: Task,
    prefer: str | None = None,
) -> list[PluginEntry]:
    entries = list(registry.for_task(task))
    if prefer:
        entries.sort(key=lambda e: (0 if e.id == prefer else 1, e.priority, e.id))
    return entries


def run_chain(
    registry: Registry,
    task: Task,
    payload: Any,
    ctx: AdapterContext,
    template: RigTemplate | None = None,
    prefer: str | None = None,
    min_score: float = 0.25,
) -> Selection:
    """Run the candidates for a task in order and return the first good result."""

    attempts: list[Attempt] = []
    best: tuple[PluginEntry, AdapterResult, float] | None = None

    def _describe(entry: PluginEntry) -> dict[str, Any]:
        manifest = entry.manifest
        return {
            "version": manifest.version if manifest else "0",
            "spdx": manifest.license.spdx if manifest else None,
            "url": manifest.license.url if manifest else None,
        }

    for entry in _candidates(registry, task, prefer):
        probe = probe_entry(entry, ctx)
        attempt = Attempt(
            adapter_id=entry.id,
            probe_status=probe.status.value,
            probe_reason=probe.reason,
        )
        if not probe.available:
            attempt.error = probe.reason or probe.status.value
            attempts.append(attempt)
            continue

        started = time.perf_counter()
        try:
            result = entry.adapter.run(task, payload, ctx)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - a failing backend must not stop the run
            attempt.duration_ms = int((time.perf_counter() - started) * 1000)
            attempt.error = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            continue

        attempt.duration_ms = int((time.perf_counter() - started) * 1000)
        attempt.ok = True
        attempt.score = score_result(task, result, payload, template)
        attempts.append(attempt)

        if best is None or attempt.score > best[2]:
            best = (entry, result, attempt.score)
        if attempt.score >= min_score:
            described = _describe(entry)
            return Selection(
                task=task,
                adapter_id=entry.id,
                result=result,
                score=attempt.score,
                attempts=attempts,
                adapter_version=described["version"],
                license_spdx=described["spdx"],
                license_url=described["url"],
            )

    if best is not None:
        entry, result, score = best
        described = _describe(entry)
        return Selection(
            task=task,
            adapter_id=entry.id,
            result=result,
            score=score,
            attempts=attempts,
            adapter_version=described["version"],
            license_spdx=described["spdx"],
            license_url=described["url"],
        )

    errors = "; ".join(f"{a.adapter_id}: {a.error}" for a in attempts if a.error)
    raise RuntimeError(f"no backend could perform {task.value}: {errors or 'no candidates'}")


def compare_all(
    registry: Registry,
    task: Task,
    payload: Any,
    ctx: AdapterContext,
    template: RigTemplate | None = None,
) -> list[Attempt]:
    """Run every candidate and report how each of them did."""

    attempts: list[Attempt] = []
    for entry in _candidates(registry, task):
        probe = probe_entry(entry, ctx)
        attempt = Attempt(
            adapter_id=entry.id,
            probe_status=probe.status.value,
            probe_reason=probe.reason,
        )
        if not probe.available:
            attempt.error = probe.reason or probe.status.value
            attempts.append(attempt)
            continue
        started = time.perf_counter()
        try:
            result = entry.adapter.run(task, payload, ctx)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            attempt.duration_ms = int((time.perf_counter() - started) * 1000)
            attempt.error = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            continue
        attempt.duration_ms = int((time.perf_counter() - started) * 1000)
        attempt.ok = True
        attempt.score = score_result(task, result, payload, template)
        attempts.append(attempt)
    return attempts


__all__ = [
    "Attempt",
    "Selection",
    "compare_all",
    "run_chain",
    "score_matte",
    "score_parts",
    "score_pose",
    "score_result",
]
