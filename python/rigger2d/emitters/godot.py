"""The Godot emitter.

Writes a ``.tscn`` scene that Godot opens as an ordinary scene: a
``Skeleton2D`` with a ``Bone2D`` tree, and one ``Polygon2D`` per part with the
weights that bind it to the skeleton.

The format details that matter, and why:

* ``Transform2D(a, b, c, d, e, f)`` is parsed by Godot as
  ``columns[0] = (a, b)``, ``columns[1] = (c, d)``, ``columns[2] = (e, f)``.
  For a rotation of ``t`` that is ``(cos t, sin t, -sin t, cos t)``.
* A ``Bone2D``'s ``rest`` is **relative to its parent bone**; the skeleton's
  rest transform for a bone is the product of the chain above it.
* The skeleton's bone order is the depth-first order of the ``Bone2D`` nodes
  in the scene tree, so the bones have to be nested for them to inherit from
  each other.
* ``Polygon2D.bones`` is an ``Array`` of ``[bone path, weights]`` pairs. The
  path is relative to the ``Skeleton2D``, and the weights array holds one
  float per vertex of the polygon.

Godot's 2D axis points down, exactly like the image space the core works in,
so no flipping is needed.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..contracts.models import Bone, Rig, SkinningMode


@dataclass
class _BoneNode:
    bone: Bone
    children: list["_BoneNode"] = field(default_factory=list)
    path: str = ""
    world_rotation: float = 0.0
    rest_rotation: float = 0.0
    rest_origin: tuple[float, float] = (0.0, 0.0)


def _num(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _transform(rotation: float, origin: tuple[float, float]) -> str:
    cos_t = math.cos(rotation)
    sin_t = math.sin(rotation)
    return (
        f"Transform2D({_num(cos_t)}, {_num(sin_t)}, {_num(-sin_t)}, "
        f"{_num(cos_t)}, {_num(origin[0])}, {_num(origin[1])})"
    )


def _build_tree(rig: Rig) -> list[_BoneNode]:
    """Nest the bones by parent relationship."""

    by_id = {bone.id: bone for bone in rig.bones}
    children: dict[str, list[Bone]] = {}
    roots: list[Bone] = []
    for bone in rig.bones:
        if bone.parent and bone.parent in by_id:
            children.setdefault(bone.parent, []).append(bone)
        else:
            roots.append(bone)

    def build(bone: Bone, prefix: str) -> _BoneNode:
        path = f"{prefix}{bone.id}"
        node = _BoneNode(bone=bone, path=path)
        node.children = [build(child, f"{path}/") for child in children.get(bone.id, [])]
        return node

    return [build(bone, "") for bone in roots]


def _compute_rests(nodes: list[_BoneNode]) -> list[_BoneNode]:
    """Work out each bone's world rotation and its local rest transform."""

    ordered: list[_BoneNode] = []

    def walk(node: _BoneNode, parent: _BoneNode | None) -> None:
        bone = node.bone
        node.world_rotation = math.atan2(
            bone.tail.y - bone.head.y,
            bone.tail.x - bone.head.x,
        )
        if parent is None:
            node.rest_rotation = node.world_rotation
            node.rest_origin = (bone.head.x, bone.head.y)
        else:
            node.rest_rotation = node.world_rotation - parent.world_rotation
            dx = bone.head.x - parent.bone.head.x
            dy = bone.head.y - parent.bone.head.y
            cos_p = math.cos(parent.world_rotation)
            sin_p = math.sin(parent.world_rotation)
            node.rest_origin = (
                dx * cos_p + dy * sin_p,
                -dx * sin_p + dy * cos_p,
            )
        ordered.append(node)
        for child in node.children:
            walk(child, node)

    for root in nodes:
        walk(root, None)
    return ordered


def _vec_array(points: list[tuple[float, float]]) -> str:
    parts: list[str] = []
    for x, y in points:
        parts.append(_num(x))
        parts.append(_num(y))
    return ", ".join(parts)


def _primary_bone(part_id: str, rig: Rig) -> Bone | None:
    for bone in rig.bones:
        if bone.part_id == part_id:
            return bone
    return rig.bones[0] if rig.bones else None


def _weight_entries(
    layer,
    mesh,
    rig: Rig,
    bone_path: dict[str, str],
) -> list[str]:
    """Build the ``[path, weights]`` pairs for a Polygon2D."""

    vertex_count = len(mesh.vertices)
    if vertex_count == 0:
        return []

    mode = rig.skinning
    rigid = mode == SkinningMode.CUTOUT or (
        mode == SkinningMode.HYBRID and layer.skin.value == "rigid"
    )

    if rigid:
        primary = _primary_bone(layer.part_id, rig)
        if primary is None:
            return []
        path = bone_path.get(primary.id)
        if path is None:
            return []
        weights = ", ".join(_num(1.0) for _ in range(vertex_count))
        return [f'"{path}", PackedFloat32Array({weights})']

    if not mesh.bones or not mesh.weights:
        return []

    aggregated: dict[int, list[float]] = {}
    for vertex in range(vertex_count):
        indices = mesh.bones[vertex] if vertex < len(mesh.bones) else []
        values = mesh.weights[vertex] if vertex < len(mesh.weights) else []
        for slot, bone_id in enumerate(indices):
            weight = values[slot] if slot < len(values) else 0.0
            if weight <= 0.0:
                continue
            aggregated.setdefault(bone_id, [0.0] * vertex_count)[vertex] = weight

    entries: list[str] = []
    for bone_id, weights in aggregated.items():
        if bone_id >= len(rig.bones):
            continue
        path = bone_path.get(rig.bones[bone_id].id)
        if path is None:
            continue
        entries.append(
            f'"{path}", PackedFloat32Array({", ".join(_num(w) for w in weights)})'
        )
    return entries


def emit(rig: Rig, out_dir: Path, layers_dir: Path) -> list[str]:
    """Write the Godot scene and copy the layer images next to it."""

    godot_dir = out_dir / "godot"
    godot_layers = godot_dir / "layers"
    godot_layers.mkdir(parents=True, exist_ok=True)

    for layer in rig.layers:
        source = layers_dir / layer.image
        if source.is_file():
            shutil.copy2(source, godot_layers / layer.image)

    roots = _build_tree(rig)
    nodes = _compute_rests(roots)
    bone_path = {node.bone.id: node.path for node in nodes}

    textures: list[tuple[str, str]] = []
    for position, layer in enumerate(rig.layers, start=1):
        if (layers_dir / layer.image).is_file():
            textures.append((f"{position}_{layer.id}", layer.image))

    mesh_by_part = {mesh.part_id: mesh for mesh in rig.meshes}

    lines: list[str] = []
    lines.append(f"[gd_scene load_steps={1 + len(textures)} format=3]")
    lines.append("")
    for resource_id, image in textures:
        lines.append(
            f'[ext_resource type="Texture2D" path="res://layers/{image}" id="{resource_id}"]'
        )
    lines.append("")

    lines.append('[node name="Character" type="Skeleton2D"]')
    lines.append("")

    for node in nodes:
        parent_path = node.path.rsplit("/", 1)[0] if "/" in node.path else "."
        lines.append(f'[node name="{node.bone.id}" type="Bone2D" parent="{parent_path}"]')
        transform = _transform(node.rest_rotation, node.rest_origin)
        lines.append(f"transform = {transform}")
        lines.append(f"rest = {transform}")
        lines.append("")

    for (resource_id, image), layer in zip(textures, rig.layers, strict=False):
        if layer.kind == "fill":
            # A fill is a flat patch that hides the hole a part leaves behind.
            # It does not deform, so a sprite is the right node for it.
            lines.append(f'[node name="{layer.id}" type="Sprite2D" parent="Character"]')
            lines.append(
                f"position = Vector2({_num(layer.bounds.x0)}, {_num(layer.bounds.y0)})"
            )
            lines.append(f'texture = ExtResource("{resource_id}")')
            lines.append("centered = false")
            lines.append("")
            continue

        mesh = mesh_by_part.get(layer.part_id)
        if mesh is None or not mesh.vertices:
            continue
        origin_x = layer.bounds.x0
        origin_y = layer.bounds.y0

        polygon = [(v.x - origin_x, v.y - origin_y) for v in mesh.vertices]
        uv = [(u.x, u.y) for u in mesh.uvs]

        lines.append(f'[node name="{layer.id}" type="Polygon2D" parent="Character"]')
        lines.append(f"position = Vector2({_num(origin_x)}, {_num(origin_y)})")
        lines.append(f'texture = ExtResource("{resource_id}")')
        lines.append('skeleton = NodePath("..")')
        lines.append(f"polygon = PackedVector2Array({_vec_array(polygon)})")
        lines.append(f"uv = PackedVector2Array({_vec_array(uv)})")
        if mesh.triangles:
            chunks = [
                "PackedInt32Array("
                + ", ".join(str(i) for i in mesh.triangles[start : start + 3])
                + ")"
                for start in range(0, len(mesh.triangles), 3)
            ]
            lines.append(f"polygons = [{', '.join(chunks)}]")
        weights = _weight_entries(layer, mesh, rig, bone_path)
        if weights:
            lines.append(f"bones = [{', '.join(weights)}]")
        lines.append("")

    scene_path = godot_dir / "scene.tscn"
    scene_path.write_text("\n".join(lines), encoding="utf-8")
    return ["godot/scene.tscn"]


__all__ = ["emit"]
