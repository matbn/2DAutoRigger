"""The contracts: what goes over the wire has to survive a round trip."""

from __future__ import annotations

import json

from rigger2d.contracts.models import (
    Bone,
    Document,
    Edit,
    JointEdit,
    Layer,
    MaskEdit,
    Mesh,
    Rig,
    Task,
    Vec2,
)


def test_a_bone_measures_its_own_length() -> None:
    bone = Bone(id="arm", head=Vec2(x=0, y=0), tail=Vec2(x=3, y=4))
    assert bone.length == 5.0


def test_a_rig_survives_a_round_trip() -> None:
    rig = Rig(
        project_id="abc",
        template_id="humanoid.biped.v1",
        width=512,
        height=1024,
        bones=[Bone(id="root", head=Vec2(x=1, y=2), tail=Vec2(x=3, y=4))],
        layers=[Layer(id="torso", part_id="torso", image="torso.png", z=40)],
        meshes=[Mesh(part_id="torso", vertices=[Vec2(x=0, y=0)], uvs=[Vec2(x=0, y=0)])],
    )
    text = rig.model_dump_json()
    again = Rig.model_validate(json.loads(text))
    assert again.bones[0] is not None
    assert again.bones[0].id == "root"
    assert again.layers[0].image == "torso.png"


def test_a_document_holds_the_edits() -> None:
    document = Document(
        project_id="abc",
        image={"path": "source.png", "width": 10, "height": 10},
        edits=[
            Edit(mask=MaskEdit(op="add", part_id="torso", points=[Vec2(x=1, y=1)])),
            Edit(joint=JointEdit(joint_id="neck", dx=3, dy=-2)),
        ],
    )
    assert document.refined
    assert document.edits[0].mask is not None
    assert document.edits[0].mask.op == "add"
    assert document.edits[1].joint is not None
    assert document.edits[1].joint.dx == 3


def test_the_task_names_are_stable() -> None:
    # These strings travel over HTTP and appear in plugin manifests, so they
    # are part of the contract.
    assert [task.value for task in Task] == [
        "matte",
        "parts",
        "pose",
        "classify",
        "inpaint",
    ]
