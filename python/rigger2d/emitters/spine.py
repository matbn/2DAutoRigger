"""The Spine emitter.

Writes ``skeleton.json`` plus an atlas. Spine's own format is JSON, so nothing
else is needed: the user imports the file into Spine and the layers appear as
attachments.

Spine's coordinate system is y-up with the origin at the bottom-left of the
skeleton's setup area, which is the mirror of the image space the core works
in. Everything is converted here, at the edge.

The attachment format used:

* ``vertices`` -- the vertex positions, in the slot's bone space
* ``uvs``      -- the texture coordinates, 0..1 inside the atlas region
* ``triangles``-- the triangle indices
* ``weights``  -- optional, one ``"boneName:weight"`` string per vertex. When
  the array is present the mesh is weighted; when it is absent the mesh is
  rigid.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..contracts.models import Bone, Rig, SkinningMode
from ..io.atlas import build_atlas


@dataclass
class _SpineBone:
    bone: Bone
    parent: "_SpineBone | None" = None
    world_head: tuple[float, float] = (0.0, 0.0)
    world_rotation: float = 0.0  # degrees, spine space
    local_x: float = 0.0
    local_y: float = 0.0
    local_rotation: float = 0.0
    children: list["_SpineBone"] = field(default_factory=list)


def _num(value: float) -> float:
    return round(float(value), 4)


def _build_bones(rig: Rig, height: int) -> list[_SpineBone]:
    """Nest the bones and compute their local Spine transforms."""

    by_id = {bone.id: bone for bone in rig.bones}
    children: dict[str, list[Bone]] = {}
    roots: list[Bone] = []
    for bone in rig.bones:
        if bone.parent and bone.parent in by_id:
            children.setdefault(bone.parent, []).append(bone)
        else:
            roots.append(bone)

    def make(bone: Bone, parent: _SpineBone | None) -> _SpineBone:
        node = _SpineBone(bone=bone, parent=parent)
        image_angle = math.atan2(bone.tail.y - bone.head.y, bone.tail.x - bone.head.x)
        node.world_head = (bone.head.x, height - bone.head.y)
        node.world_rotation = math.degrees(-image_angle)

        if parent is None:
            node.local_x = node.world_head[0]
            node.local_y = node.world_head[1]
            node.local_rotation = node.world_rotation
        else:
            dx = node.world_head[0] - parent.world_head[0]
            dy = node.world_head[1] - parent.world_head[1]
            # The parent's frame is rotated by -world_rotation, and
            # world_rotation is the negated image angle.
            angle = math.radians(-parent.world_rotation)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            node.local_x = dx * cos_a - dy * sin_a
            node.local_y = dx * sin_a + dy * cos_a
            node.local_rotation = node.world_rotation - parent.world_rotation

        node.children = [make(child, node) for child in children.get(bone.id, [])]
        return node

    return [make(bone, None) for bone in roots]


def _flatten(nodes: list[_SpineBone]) -> list[_SpineBone]:
    out: list[_SpineBone] = []

    def walk(node: _SpineBone) -> None:
        out.append(node)
        for child in node.children:
            walk(child)

    for root in nodes:
        walk(root)
    return out


def _bone_local_point(
    point: tuple[float, float],
    bone: _SpineBone,
    height: int,
) -> tuple[float, float]:
    """Express an image-space point in a bone's local Spine frame."""

    dx = point[0] - bone.bone.head.x
    dy = bone.bone.head.y - point[1]
    angle = math.radians(-bone.world_rotation)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)


def _primary_bone(part_id: str, rig: Rig) -> Bone | None:
    for bone in rig.bones:
        if bone.part_id == part_id:
            return bone
    return rig.bones[0] if rig.bones else None


def emit(rig: Rig, out_dir: Path, layers_dir: Path) -> list[str]:
    """Write ``skeleton.json``, the atlas and the atlas description."""

    spine_dir = out_dir / "spine"
    spine_dir.mkdir(parents=True, exist_ok=True)

    roots = _build_bones(rig, rig.height)
    ordered = _flatten(roots)
    spine_bone = {node.bone.id: node for node in ordered}
    mesh_by_part = {mesh.part_id: mesh for mesh in rig.meshes}

    # Copy the layers next to the atlas so the export is self contained.
    spine_layers = spine_dir / "layers"
    spine_layers.mkdir(parents=True, exist_ok=True)
    atlas_inputs: list[tuple[str, Path]] = []
    for layer in rig.layers:
        source = layers_dir / layer.image
        if not source.is_file():
            continue
        shutil.copy2(source, spine_layers / layer.image)
        atlas_inputs.append((layer.id, spine_layers / layer.image))

    atlas_png, atlas_text, placed = build_atlas(atlas_inputs, spine_dir, "atlas.png")
    region_of = {entry.name: entry for entry in placed}

    bones_json = []
    for node in ordered:
        bone = node.bone
        bones_json.append(
            {
                "name": bone.id,
                "parent": node.parent.bone.id if node.parent else None,
                "rotation": _num(node.local_rotation),
                "x": _num(node.local_x),
                "y": _num(node.local_y),
                "scaleX": 1,
                "scaleY": 1,
                "shearX": 0,
                "shearY": 0,
                "length": _num(bone.length),
                "transformMode": "normal",
                "skin": "default",
            }
        )

    slots_json: list[dict] = []
    attachments: dict[str, dict] = {}

    for layer in rig.layers:
        if layer.id not in region_of:
            continue
        primary = _primary_bone(layer.part_id, rig)
        if primary is None:
            continue
        slot_bone = spine_bone.get(primary.id)
        if slot_bone is None:
            continue

        slots_json.append(
            {
                "name": layer.id,
                "bone": primary.id,
                "attachment": layer.id,
                "blend": "normal",
            }
        )

        mesh = mesh_by_part.get(layer.part_id)
        if layer.kind == "fill" or mesh is None or not mesh.vertices:
            # A fill (or a part with no mesh) becomes a plain region: the
            # texture is a flat shape, so a rectangle is enough.
            region = region_of[layer.id]
            centre_x = layer.bounds.x0 + region.width / 2.0
            centre_y = layer.bounds.y0 + region.height / 2.0
            local_x, local_y = _bone_local_point((centre_x, centre_y), slot_bone, rig.height)
            attachments.setdefault(layer.id, {})[layer.id] = {
                "x": _num(local_x),
                "y": _num(local_y),
                "rotation": _num(-slot_bone.world_rotation),
                "width": region.width,
                "height": region.height,
                "path": layer.id,
            }
            continue

        vertices: list[float] = []
        for vertex in mesh.vertices:
            lx, ly = _bone_local_point((vertex.x, vertex.y), slot_bone, rig.height)
            vertices.extend([_num(lx), _num(ly)])

        uvs = [value for uv in mesh.uvs for value in (_num(uv.x), _num(uv.y))]

        attachment: dict = {
            "type": "mesh",
            "path": layer.id,
            "vertices": vertices,
            "triangles": list(mesh.triangles),
            "uvs": uvs,
            "edges": [],
        }

        rigid = rig.skinning == SkinningMode.CUTOUT or (
            rig.skinning == SkinningMode.HYBRID and layer.skin.value == "rigid"
        )
        if not rigid and mesh.bones and mesh.weights:
            weights: list[str] = []
            for vertex_index in range(len(mesh.vertices)):
                indices = mesh.bones[vertex_index] if vertex_index < len(mesh.bones) else []
                values = mesh.weights[vertex_index] if vertex_index < len(mesh.weights) else []
                chosen = ""
                for slot, bone_id in enumerate(indices):
                    weight = values[slot] if slot < len(values) else 0.0
                    if weight > 0.0 and bone_id < len(rig.bones):
                        chosen = f"{rig.bones[bone_id].id}:{_num(weight)}"
                        break
                weights.append(chosen)
            if any(weights):
                attachment["weights"] = weights

        attachments.setdefault(layer.id, {})[layer.id] = attachment

    skeleton_json = {
        "skeleton": {
            "hash": "",
            "spine": "4.2.00",
            "x": 0,
            "y": 0,
            "width": rig.width,
            "height": rig.height,
            "images": "./",
            "audio": "",
            "animations": [],
        },
        "bones": bones_json,
        "slots": slots_json,
        "skins": [
            {
                "name": "default",
                "attachments": attachments,
            }
        ],
        "animations": {},
    }

    skeleton_path = spine_dir / "skeleton.json"
    with skeleton_path.open("w", encoding="utf-8") as fh:
        json.dump(skeleton_json, fh, indent=2)

    return [
        "spine/skeleton.json",
        f"spine/{atlas_png.name}",
        f"spine/{atlas_text.name}",
    ]


__all__ = ["emit"]
