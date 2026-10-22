"""The pipeline, end to end, against a character whose parts are known."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixtures import ground_truth_masks
from rigger2d.io import images
from rigger2d.pipeline.session import Session
from rigger2d.pipeline.stages import REFINE_GATE, RunOptions, run_stages


def test_the_pipeline_stops_at_the_refine_gate(project) -> None:
    _session, document = project
    assert "parts" in document.stages
    assert "layers" not in document.stages
    assert document.rig is None


def test_the_pipeline_finds_every_part(project) -> None:
    _session, document = project
    assert document.parts is not None
    found = {mask.part_id for mask in document.parts.masks}
    expected = set(ground_truth_masks().keys())
    assert expected <= found, f"missing: {expected - found}"


def test_the_masks_match_the_character(project) -> None:
    """The core fallback should recover a template shaped character."""

    session, document = project
    truth = ground_truth_masks()

    intersection = 0
    union = 0
    for part_id, expected in truth.items():
        path = session.parts_dir / f"{part_id}.png"
        if not path.is_file():
            continue
        actual = images.load_mask(path)
        intersection += int((actual & expected).sum())
        union += int((actual | expected).sum())

    assert union > 0
    iou = intersection / union
    assert iou > 0.85, f"the segmentation is worse than expected: IoU {iou:.3f}"


def test_the_pipeline_produces_a_rig(tmp_path: Path, character_image: Path) -> None:
    session = Session(tmp_path / "hero.rig")
    document = session.create(character_image)
    document = run_stages(session, document, RunOptions(targets=["native"]))

    assert document.rig is not None
    rig = document.rig
    assert len(rig.bones) >= 15
    assert rig.layers, "the character was not sliced"
    assert rig.meshes, "no mesh was built"
    assert rig.skinning.value == "hybrid"

    for layer in rig.layers:
        assert (session.layers_dir / layer.image).is_file()


def test_bone_length_follows_the_size_of_the_part(project) -> None:
    """A bone should be about as long as the part it drives."""

    session, document = project
    from rigger2d.rigging.skeleton import build_skeleton

    masks = session.load_part_masks()
    bones = build_skeleton(session.template, document.landmarks, masks, document.matte.bounds)
    by_id = {bone.id: bone for bone in bones}

    arm = by_id.get("upper_arm_l")
    assert arm is not None
    mask = masks["upper_arm_l"]
    bounds = images.mask_bounds(mask)
    assert arm.length > 0
    # The arm bone should be a sensible fraction of the arm's own height.
    assert 0.5 * bounds.height < arm.length < 2.0 * bounds.height


def test_the_document_is_the_source_of_truth(tmp_path: Path, character_image: Path) -> None:
    session = Session(tmp_path / "hero.rig")
    session.create(character_image)
    run_stages(session, session.load(), RunOptions())

    saved = json.loads(session.document_path.read_text(encoding="utf-8"))
    assert saved["project_id"]
    assert "parts" in saved["stages"]
    assert saved["stages"]["parts"]["adapter_id"]


def test_running_twice_does_not_change_the_result(tmp_path: Path, character_image: Path) -> None:
    session = Session(tmp_path / "hero.rig")
    document = session.create(character_image)
    document = run_stages(session, document, RunOptions())
    first = document.rig
    assert first is not None

    document = run_stages(session, session.load(), RunOptions())
    second = document.rig
    assert second is not None
    assert len(second.bones) == len(first.bones)
    assert len(second.layers) == len(first.layers)


def test_a_transparent_image_uses_its_alpha(tmp_path: Path) -> None:
    from fixtures import draw_character_transparent

    image = draw_character_transparent(tmp_path / "hero.png")
    session = Session(tmp_path / "hero.rig")
    document = session.create(image)
    document = run_stages(session, document, RunOptions(), until=REFINE_GATE)
    assert document.matte is not None
    assert document.matte.coverage > 0.1


def test_the_pipeline_reports_that_the_joints_are_a_guess(project) -> None:
    _session, document = project
    assert any("joints" in warning for warning in document.warnings)
