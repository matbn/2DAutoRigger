"""The reference plugin.

This is what a backend adapter looks like. It performs no inference at all: it
copies the character matte into one mask per part, using the template's own
proportions. It exists so that the plugin contract can be exercised -- by the
tests, and by anyone writing a real adapter who wants a working example to
copy.

A real adapter replaces the ``run`` method and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rigger2d.adapters.base import (
    AdapterContext,
    AdapterResult,
    MattePayload,
    PartsPayload,
    PosePayload,
    ok_probe,
)
from rigger2d.contracts.models import Probe, Task, Vec2
from rigger2d.io import images

ADAPTER_ID = "example.mock"
VERSION = "0.1.0"


class MockAdapter:
    """A backend that does not need any weights."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({Task.MATTE, Task.POSE, Task.PARTS})

    def probe(self, ctx: AdapterContext) -> Probe:
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx="Apache-2.0",
            device="cpu",
            extra={"kind": "reference implementation"},
        )

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task == Task.MATTE:
            return self._matte(payload, ctx)
        if task == Task.POSE:
            return self._pose(payload, ctx)
        if task == Task.PARTS:
            return self._parts(payload, ctx)
        raise ValueError(f"the mock adapter does not implement {task}")

    def _matte(self, payload: MattePayload, ctx: AdapterContext) -> AdapterResult:
        rgba = images.load_rgba(payload.image_path)
        alpha = rgba[..., 3]
        mask = alpha > 127 if int(alpha.min()) < 250 else alpha > 0
        path = payload.output_dir / "matte.png"
        images.save_mask(path, mask)
        return AdapterResult(task=Task.MATTE, mask_paths={"matte": path})

    def _pose(self, payload: PosePayload, ctx: AdapterContext) -> AdapterResult:
        # The mock has no idea where the joints are, so it says so.
        landmarks: dict[str, Vec2] = {}
        confidence: dict[str, float] = {}
        for name, point in payload.landmarks.items():
            landmarks[name] = point
            confidence[name] = 0.05
        return AdapterResult(
            task=Task.POSE,
            landmarks=landmarks,
            confidence=confidence,
            detail={"source": "mock"},
        )

    def _parts(self, payload: PartsPayload, ctx: AdapterContext) -> AdapterResult:
        # The mock cannot segment, so it hands back an empty mask per part.
        paths: dict[str, Path] = {}
        for part_id in payload.part_ids:
            path = payload.output_dir / f"{part_id}.png"
            images.save_mask(path, images.load_mask(payload.matte_path) & False)
            paths[part_id] = path
        return AdapterResult(task=Task.PARTS, mask_paths=paths, detail={"source": "mock"})


adapter = MockAdapter()

__all__ = ["MockAdapter", "adapter"]
