"""The refine step: correcting the masks and the joints before slicing."""

from __future__ import annotations

import numpy as np

from rigger2d.contracts.models import Edit, JointEdit, MaskEdit, Vec2
from rigger2d.pipeline.refine import apply_joint_edits, apply_mask_edits, summarize


def test_adding_a_stroke_grows_a_part() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    masks = {"torso": mask}

    edits = [
        Edit(
            mask=MaskEdit(
                op="add",
                part_id="torso",
                points=[Vec2(x=15, y=15)],
                radius=3.0,
            )
        )
    ]
    result = apply_mask_edits(masks, edits, (20, 20))
    assert result["torso"][15, 15]
    assert result["torso"][6, 6], "the original mask should still be there"


def test_erasing_a_stroke_shrinks_a_part() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    masks = {"torso": mask}

    edits = [
        Edit(
            mask=MaskEdit(
                op="erase",
                part_id="torso",
                points=[Vec2(x=7, y=7)],
                radius=2.0,
            )
        )
    ]
    result = apply_mask_edits(masks, edits, (20, 20))
    assert not result["torso"][7, 7]
    assert result["torso"][5, 5], "only the erased part should change"


def test_assigning_moves_a_pixel_from_one_part_to_another() -> None:
    torso = np.zeros((20, 20), dtype=bool)
    torso[5:10, 5:10] = True
    arm = np.zeros((20, 20), dtype=bool)
    arm[12:16, 5:8] = True
    masks = {"torso": torso, "upper_arm_l": arm}

    edits = [
        Edit(
            mask=MaskEdit(
                op="assign",
                part_id="upper_arm_l",
                points=[Vec2(x=7, y=7)],
                radius=2.0,
            )
        )
    ]
    result = apply_mask_edits(masks, edits, (20, 20))
    assert result["upper_arm_l"][7, 7]
    assert not result["torso"][7, 7], "the pixel should have left the torso"


def test_a_stroke_that_misses_does_nothing() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    edits = [Edit(mask=MaskEdit(op="add", part_id="torso", points=[], radius=3))]
    result = apply_mask_edits({"torso": mask}, edits, (20, 20))
    assert not result["torso"].any()


def test_moving_a_joint_moves_it() -> None:
    landmarks = {"neck": Vec2(x=10, y=20)}
    edits = [Edit(joint=JointEdit(joint_id="neck", dx=5, dy=-3))]
    result = apply_joint_edits(landmarks, edits)
    assert result["neck"].x == 15
    assert result["neck"].y == 17


def test_moving_a_joint_that_does_not_exist_is_ignored() -> None:
    landmarks = {"neck": Vec2(x=10, y=20)}
    edits = [Edit(joint=JointEdit(joint_id="elbow_l", dx=5, dy=-3))]
    result = apply_joint_edits(landmarks, edits)
    assert result == landmarks


def test_the_summary_counts_the_edits() -> None:
    edits = [
        Edit(mask=MaskEdit(op="add", part_id="a")),
        Edit(mask=MaskEdit(op="erase", part_id="a")),
        Edit(mask=MaskEdit(op="assign", part_id="a")),
        Edit(joint=JointEdit(joint_id="neck")),
    ]
    summary = summarize(edits)
    assert summary["mask_add"] == 1
    assert summary["mask_erase"] == 1
    assert summary["mask_assign"] == 1
    assert summary["joints"] == 1
    assert summary["total"] == 4


def test_the_refined_masks_reach_the_sliced_layers(project) -> None:
    """A correction made at the gate has to change what comes out."""

    from rigger2d.pipeline.session import Session
    from rigger2d.pipeline.stages import RunOptions, run_stages

    session, document = project
    mask = session.load_part_masks()["torso"]
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0

    # Erase a corner of the torso.
    corner = Vec2(x=float(xs.min()), y=float(ys.min()))
    document.edits.append(
        Edit(mask=MaskEdit(op="erase", part_id="torso", points=[corner], radius=6.0))
    )
    session.save(document)

    document = run_stages(session, session.load(), RunOptions())
    assert document.rig is not None
    torso = next(layer for layer in document.rig.layers if layer.id == "torso")
    assert torso.image == "torso.png"


def test_the_refine_gate_is_where_the_pipeline_pauses(project) -> None:
    session, document = project
    from rigger2d.pipeline.stages import REFINE_GATE, RunOptions, run_stages

    assert REFINE_GATE == "parts"
    document = run_stages(session, document, RunOptions(), until=REFINE_GATE)
    assert document.rig is None
    assert document.parts is not None
