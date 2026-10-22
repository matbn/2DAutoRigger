"""The pipeline stages.

Every stage is a pure-ish function: it reads the document, does one job, and
writes the document back. Nothing is global, nothing is cached in memory
between runs, and the document on disk is always the truth.

The stage order is::

    ingest -> matte -> pose -> parts  ==REFINE GATE==>  layers -> skeleton
            -> mesh -> weights -> emit

The refine gate is the boundary between "what the machine thinks" and "what the
user decides". The pipeline stops there by default and only continues when the
user asks it to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..adapters.base import MattePayload, PartsPayload, PosePayload
from ..adapters.selection import run_chain
from ..contracts.models import (
    BBox,
    Document,
    MaskInfo,
    MatteState,
    Mesh,
    PartState,
    ProvenanceEntry,
    Rig,
    SkinningMode,
    StageResult,
    Task,
    Vec2,
)
from ..io import images
from ..rigging import binding, mesh as meshlib, occlusion, skeleton as skeletonlib
from ..rigging.templates import RigTemplate
from .refine import apply_joint_edits, apply_mask_edits
from .session import Session, utcnow

STAGE_ORDER = [
    "ingest",
    "matte",
    "pose",
    "parts",
    "layers",
    "skeleton",
    "mesh",
    "weights",
    "emit",
]

REFINE_GATE = "parts"


@dataclass
class RunOptions:
    """Knobs for a pipeline run."""

    targets: list[str] = field(default_factory=lambda: ["native"])
    skinning: SkinningMode = SkinningMode.HYBRID
    prefer: str | None = None
    force: bool = False


def _provenance(selection: Any, doc: Document) -> None:
    """Record who did what, for the provenance sidecar."""

    entry = next((a for a in selection.attempts if a.adapter_id == selection.adapter_id), None)
    doc.provenance.append(
        ProvenanceEntry(
            adapter_id=selection.adapter_id,
            adapter_version=selection.adapter_version,
            task=selection.task,
            ok=True,
            duration_ms=entry.duration_ms if entry else 0,
            license_spdx=selection.license_spdx,
            license_url=selection.license_url,
            detail=f"score {selection.score:.2f}",
        )
    )


# -- stages ---------------------------------------------------------------


def stage_ingest(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    if not session.source_path.is_file():
        raise FileNotFoundError("the project has no source image")
    rgba = images.load_rgba(session.source_path)
    doc.image.width = int(rgba.shape[1])
    doc.image.height = int(rgba.shape[0])
    doc.image.has_alpha = bool(int(rgba[..., 3].min()) < 250)
    return StageResult(
        name="ingest",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"width": doc.image.width, "height": doc.image.height},
    )


def stage_matte(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    session.masks_dir.mkdir(parents=True, exist_ok=True)
    payload = MattePayload(image_path=session.source_path, output_dir=session.masks_dir)
    selection = run_chain(
        session.registry,
        Task.MATTE,
        payload,
        session.context,
        session.template,
        prefer=options.prefer,
    )
    path = selection.result.mask_paths.get("matte")
    if path is None:
        raise RuntimeError("the matte backend returned no mask")
    mask = images.load_mask(Path(path))
    doc.matte = MatteState(
        image=str(Path(path).relative_to(session.root)),
        bounds=images.mask_bounds(mask),
        coverage=float(mask.mean()),
    )
    _provenance(selection, doc)
    return StageResult(
        name="matte",
        duration_ms=int((time.perf_counter() - started) * 1000),
        adapter_id=selection.adapter_id,
        detail={"score": selection.score, "coverage": doc.matte.coverage},
    )


def stage_pose(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    bbox = doc.matte.bounds if doc.matte else BBox(
        x0=0, y0=0, x1=doc.image.width, y1=doc.image.height
    )
    payload = PosePayload(
        image_path=session.source_path,
        bbox=bbox,
        output_dir=session.masks_dir,
        landmarks=dict(doc.landmarks),
    )
    selection = run_chain(
        session.registry,
        Task.POSE,
        payload,
        session.context,
        session.template,
        prefer=options.prefer,
    )
    doc.landmarks = dict(selection.result.landmarks)
    doc.landmark_confidence = dict(selection.result.confidence)
    if not doc.landmarks:
        raise RuntimeError("the pose backend returned no landmarks")
    mean_confidence = (
        sum(doc.landmark_confidence.values()) / len(doc.landmark_confidence)
        if doc.landmark_confidence
        else 0.0
    )
    if mean_confidence < 0.5:
        doc.warnings.append(
            "The joints are a guess. Check them before the character is sliced."
        )
    _provenance(selection, doc)
    return StageResult(
        name="pose",
        duration_ms=int((time.perf_counter() - started) * 1000),
        adapter_id=selection.adapter_id,
        detail={"score": selection.score, "confidence": mean_confidence},
    )


def stage_parts(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    session.parts_dir.mkdir(parents=True, exist_ok=True)
    bbox = doc.matte.bounds if doc.matte else BBox(
        x0=0, y0=0, x1=doc.image.width, y1=doc.image.height
    )
    matte_path = session.root / doc.matte.image if doc.matte else None
    payload = PartsPayload(
        image_path=session.source_path,
        matte_path=matte_path,
        bbox=bbox,
        landmarks=dict(doc.landmarks),
        output_dir=session.parts_dir,
        part_ids=session.template.part_ids,
    )
    selection = run_chain(
        session.registry,
        Task.PARTS,
        payload,
        session.context,
        session.template,
        prefer=options.prefer,
    )
    masks = session.load_part_masks()
    if not masks:
        raise RuntimeError("the parts backend produced no masks")
    doc.parts = PartState(
        masks=[
            MaskInfo(
                part_id=part_id,
                image=str((session.parts_dir / f"{part_id}.png").relative_to(session.root)),
                bounds=images.mask_bounds(mask),
                area=int(mask.sum()),
                pixels=int(mask.sum()),
            )
            for part_id, mask in sorted(masks.items())
        ]
    )
    _provenance(selection, doc)
    return StageResult(
        name="parts",
        duration_ms=int((time.perf_counter() - started) * 1000),
        adapter_id=selection.adapter_id,
        detail={"score": selection.score, "parts": len(masks)},
    )


def _effective_masks(session: Session, doc: Document) -> dict[str, "np.ndarray"]:  # type: ignore[name-defined]
    """The part masks after the user's corrections."""

    import numpy as np

    masks = session.load_part_masks()
    if doc.edits:
        shape = (doc.image.height, doc.image.width)
        masks = apply_mask_edits(masks, doc.edits, shape)
    return masks


def _effective_landmarks(doc: Document) -> dict[str, Vec2]:
    if doc.edits:
        return apply_joint_edits(doc.landmarks, doc.edits)
    return dict(doc.landmarks)


def _ensure_rig(session: Session, doc: Document) -> Rig:
    """Create the rig document the later stages fill in."""

    if doc.rig is None:
        doc.rig = Rig(
            project_id=doc.project_id,
            template_id=doc.template_id,
            template_type=doc.template_type,
            width=doc.image.width,
            height=doc.image.height,
            skinning=SkinningMode.HYBRID,
            created_at=utcnow(),
        )
    return doc.rig


def stage_layers(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    rgba = images.load_rgba(session.source_path)
    masks = _effective_masks(session, doc)
    landmarks = _effective_landmarks(doc)
    layers = occlusion.build_layers(
        rgba,
        masks,
        session.template,
        landmarks,
        session.layers_dir,
    )
    if not layers:
        raise RuntimeError("the character could not be sliced into layers")
    rig = _ensure_rig(session, doc)
    rig.layers = layers
    rig.landmarks = landmarks
    return StageResult(
        name="layers",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"layers": len(layers)},
    )


def stage_skeleton(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    masks = _effective_masks(session, doc)
    landmarks = _effective_landmarks(doc)
    bbox = doc.matte.bounds if doc.matte else BBox(
        x0=0, y0=0, x1=doc.image.width, y1=doc.image.height
    )
    bones = skeletonlib.build_skeleton(session.template, landmarks, masks, bbox)
    if not bones:
        raise RuntimeError("no bones could be built")
    rig = _ensure_rig(session, doc)
    rig.bones = bones
    return StageResult(
        name="skeleton",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"bones": len(bones)},
    )


def stage_mesh(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    masks = _effective_masks(session, doc)
    meshes: list[Mesh] = []
    for part_id, mask in sorted(masks.items()):
        if not mask.any():
            continue
        bounds = images.mask_bounds(mask)
        built = meshlib.build_mesh(part_id, mask, bounds)
        if built.vertices:
            meshes.append(built)
    rig = _ensure_rig(session, doc)
    rig.meshes = meshes
    return StageResult(
        name="mesh",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"meshes": len(meshes)},
    )


def stage_weights(session: Session, doc: Document, options: RunOptions) -> StageResult:
    started = time.perf_counter()
    rig = _ensure_rig(session, doc)
    if not rig.bones:
        raise RuntimeError("no bones to bind the weights to")
    binding.bind_all(rig.meshes, session.template, rig.bones)
    rig.provenance = list(doc.provenance)
    return StageResult(
        name="weights",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"bones": len(rig.bones), "meshes": len(rig.meshes)},
    )


def stage_emit(session: Session, doc: Document, options: RunOptions) -> StageResult:
    from ..emitters import emit_all

    started = time.perf_counter()
    if doc.rig is None:
        raise RuntimeError("nothing to emit: the rig has not been built yet")
    doc.rig.skinning = options.skinning
    written = emit_all(session, doc.rig, options.targets)
    doc.stages["emit"] = StageResult(
        name="emit",
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail={"written": written},
    )
    return doc.stages["emit"]


STAGE_FUNCS: dict[str, Callable[[Session, Document, RunOptions], StageResult]] = {
    "ingest": stage_ingest,
    "matte": stage_matte,
    "pose": stage_pose,
    "parts": stage_parts,
    "layers": stage_layers,
    "skeleton": stage_skeleton,
    "mesh": stage_mesh,
    "weights": stage_weights,
    "emit": stage_emit,
}


def run_stages(
    session: Session,
    doc: Document,
    options: RunOptions | None = None,
    until: str | None = None,
    invalid_from: str | None = None,
) -> Document:
    """Run the pipeline.

    ``until`` stops after the named stage -- ``until="parts"`` is the refine
    gate. ``invalid_from`` forces every stage from that point on to re-run,
    which is what happens after the user edits the masks.
    """

    options = options or RunOptions()
    limit = STAGE_ORDER.index(until) + 1 if until in STAGE_ORDER else len(STAGE_ORDER)
    start = STAGE_ORDER.index(invalid_from) if invalid_from in STAGE_ORDER else 0

    for index, name in enumerate(STAGE_ORDER[:limit]):
        if index < start and name in doc.stages and not options.force:
            continue
        result = STAGE_FUNCS[name](session, doc, options)
        doc.stages[name] = result
        session.save(doc)

    return doc


__all__ = [
    "REFINE_GATE",
    "RunOptions",
    "STAGE_ORDER",
    "run_stages",
]
