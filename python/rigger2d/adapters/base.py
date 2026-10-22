"""The contract every segmentation/pose backend must implement.

The core of 2DAutoRigger contains no machine-learning code and no model
weights. A backend is an optional plugin that the user installs, and the user
is responsible for its license. Everything the core needs to know about a
backend is expressed by this module.

A plugin is expected to ship a ``manifest.json`` (see
``schemas/plugin-manifest.schema.json``) and an object implementing
:class:`Adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..contracts.models import BBox, Probe, ProbeStatus, Task, Vec2


@dataclass
class AdapterContext:
    """Everything an adapter is allowed to know about the job it is doing."""

    project_dir: Path
    work_dir: Path
    device: str = "cpu"


@dataclass
class MattePayload:
    """Ask for the whole-character alpha."""

    image_path: Path
    output_dir: Path


@dataclass
class PosePayload:
    """Ask for landmarks."""

    image_path: Path
    bbox: BBox
    output_dir: Path
    landmarks: dict[str, Vec2] = field(default_factory=dict)


@dataclass
class PartsPayload:
    """Ask for one mask per body part."""

    image_path: Path
    matte_path: Path | None
    bbox: BBox
    landmarks: dict[str, Vec2]
    output_dir: Path
    part_ids: list[str] = field(default_factory=list)


@dataclass
class InpaintPayload:
    """Ask for the hidden area behind a part, so a moving limb reveals no hole."""

    image_path: Path
    mask_path: Path
    bbox: BBox
    output_dir: Path


@dataclass
class AdapterResult:
    """What an adapter returns.

    Different tasks fill different fields; ``task`` says which ones matter.
    """

    task: Task
    mask_paths: dict[str, Path] = field(default_factory=dict)
    landmarks: dict[str, Vec2] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def mask(self, part_id: str) -> Path | None:
        return self.mask_paths.get(part_id)


@runtime_checkable
class Adapter(Protocol):
    """A backend that can do one or more of the core's tasks."""

    id: str
    version: str
    capabilities: frozenset[Task]

    def probe(self, ctx: AdapterContext) -> Probe:
        """Report whether this adapter can run right now.

        The core shows the result verbatim in the UI, so the ``reason`` should
        explain what the user has to do -- install a package, fetch weights,
        accept a license -- rather than a bare status code.
        """

    def run(
        self,
        task: Task,
        payload: Any,
        ctx: AdapterContext,
    ) -> AdapterResult:
        """Perform one task and return the result."""


def ok_probe(
    adapter_id: str,
    version: str,
    capabilities: frozenset[Task],
    *,
    license_spdx: str | None = None,
    license_url: str | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Probe:
    """Convenience builder for a healthy adapter."""

    return Probe(
        adapter_id=adapter_id,
        version=version,
        status=ProbeStatus.AVAILABLE,
        license_spdx=license_spdx,
        license_url=license_url,
        device=device,
        capabilities=sorted(capabilities, key=lambda t: t.value),
        extra=extra or {},
    )


def failed_probe(
    adapter_id: str,
    version: str,
    status: ProbeStatus,
    reason: str,
    capabilities: frozenset[Task] = frozenset(),
) -> Probe:
    """Convenience builder for an adapter that cannot run."""

    return Probe(
        adapter_id=adapter_id,
        version=version,
        status=status,
        reason=reason,
        capabilities=sorted(capabilities, key=lambda t: t.value),
    )


__all__ = [
    "Adapter",
    "AdapterContext",
    "AdapterResult",
    "InpaintPayload",
    "MattePayload",
    "PartsPayload",
    "PosePayload",
    "failed_probe",
    "ok_probe",
]
