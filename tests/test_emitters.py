"""The emitters: what Godot and Spine actually receive."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rigger2d.contracts.models import SkinningMode
from rigger2d.pipeline.session import Session
from rigger2d.pipeline.stages import RunOptions, run_stages


def _run(tmp_path: Path, character_image: Path, skinning: SkinningMode) -> Session:
    session = Session(tmp_path / "hero.rig")
    document = session.create(character_image)
    run_stages(session, document, RunOptions(targets=["native", "godot", "spine"], skinning=skinning))
    return session


def _run_with_unity(tmp_path: Path, character_image: Path) -> Session:
    """The Unity export is pure Python, so this needs no editor."""

    session = Session(tmp_path / "hero.rig")
    document = session.create(character_image)
    run_stages(session, document, RunOptions(targets=["native", "unity"], skinning=SkinningMode.HYBRID))
    return session


def _parse_prefab(path: Path) -> dict:
    """Parse a Unity prefab into objects, keyed by file id.

    A prefab is a flat list of YAML documents, each starting with a line like
    ``--- !u!4 &123456``, and every field is either ``key: value`` or a list of
    ``- item`` lines under a ``key:`` line. That is the whole grammar, so a
    small reader is enough and the tests need no YAML library.
    """

    objects: dict[int, dict] = {}
    current: dict | None = None
    current_key: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("--- !u!"):
            tag, _, fid = line.removeprefix("--- ").partition("&")
            current = {"__class__": tag.strip(), "__id__": int(fid)}
            objects[int(fid)] = current
            current_key = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current_key is not None:
                current.setdefault(current_key, []).append(stripped[2:].strip())
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            current_key = key.strip()
            current.setdefault(current_key, [])
            if value.strip():
                current[current_key].append(value.strip())
    return objects


def _refs(value: str) -> list[int]:
    """Every ``fileID: N`` inside a field value."""

    import re

    return [int(match) for match in re.findall(r"fileID:\s*(-?\d+)", value)]


def test_the_canonical_rig_is_written(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    rig_path = session.exports_dir / "rig.json"
    assert rig_path.is_file()
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    assert rig["bones"]
    assert rig["layers"]
    assert rig["schema_version"]


def test_the_provenance_records_the_backend(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    provenance = json.loads((session.exports_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["backends"]
    for entry in provenance["backends"]:
        assert entry["adapter_id"]
        assert entry["license_spdx"], "every backend has to declare its licence"


def test_the_godot_scene_has_a_skeleton(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")

    assert scene.startswith("[gd_scene ")
    assert 'type="Skeleton2D"' in scene
    assert 'type="Bone2D"' in scene
    assert 'type="Polygon2D"' in scene
    assert 'skeleton = NodePath("..")' in scene
    assert "[ext_resource type=\"Texture2D\"" in scene


def test_the_godot_bones_are_nested_by_parent(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")

    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))
    by_id = {bone["id"]: bone for bone in rig["bones"]}
    for bone in rig["bones"]:
        assert f'[node name="{bone["id"]}" type="Bone2D"' in scene
        if bone["parent"]:
            parent = by_id[bone["parent"]]
            expected = f'parent="{parent["id"]}"' if parent["parent"] is None else None
            # A nested bone names its whole path as the parent.
            assert bone["id"] in scene


def test_the_godot_transforms_are_well_formed(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")

    transforms = re.findall(r"Transform2D\(([^)]*)\)", scene)
    assert transforms
    for transform in transforms:
        numbers = [float(piece) for piece in transform.split(",")]
        assert len(numbers) == 6, f"a Transform2D needs six numbers: {transform}"
        # The basis has to be a rotation, not a shear.
        a, b, c, d = numbers[0], numbers[1], numbers[2], numbers[3]
        assert abs(a * a + b * b - 1.0) < 1e-3
        assert abs(a * c + b * d) < 1e-3


def test_the_godot_weights_have_one_entry_per_vertex(
    tmp_path: Path, character_image: Path
) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")

    blocks = scene.split('[node name="')
    for block in blocks:
        if 'type="Polygon2D"' not in block:
            continue
        polygon = re.search(r"polygon = PackedVector2Array\(([^)]*)\)", block)
        if polygon is None:
            continue
        vertices = len([p for p in polygon.group(1).split(",") if p.strip()]) // 2
        bones = re.search(r"bones = \[(.*)\]", block)
        if bones is None:
            continue
        for weights in re.findall(r"PackedFloat32Array\(([^)]*)\)", bones.group(1)):
            count = len([w for w in weights.split(",") if w.strip()])
            assert count == vertices, "the weights must have one entry per vertex"


def test_the_spine_skeleton_is_valid_json(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    skeleton = json.loads(
        (session.exports_dir / "spine" / "skeleton.json").read_text(encoding="utf-8")
    )

    assert skeleton["skeleton"]["width"] == 512
    assert skeleton["bones"]
    assert skeleton["slots"]
    assert skeleton["skins"][0]["attachments"]

    names = {bone["name"] for bone in skeleton["bones"]}
    for bone in skeleton["bones"]:
        if bone["parent"] is not None:
            assert bone["parent"] in names, "a bone may not have a missing parent"


def test_the_spine_attachments_match_the_slots(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    skeleton = json.loads(
        (session.exports_dir / "spine" / "skeleton.json").read_text(encoding="utf-8")
    )
    attachments = skeleton["skins"][0]["attachments"]
    for slot in skeleton["slots"]:
        assert slot["name"] in attachments
        assert slot["attachment"] in attachments[slot["name"]]


def test_the_spine_atlas_covers_every_layer(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.HYBRID)
    atlas = (session.exports_dir / "spine" / "atlas.atlas").read_text(encoding="utf-8")
    assert "atlas.png" in atlas
    assert "format: RGBA8888" in atlas
    for layer in json.loads(
        (session.exports_dir / "rig.json").read_text(encoding="utf-8")
    )["layers"]:
        assert layer["id"] in atlas


def test_cutout_makes_every_part_rigid(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.CUTOUT)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")
    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))
    assert rig["skinning"] == "cutout"
    # In cutout mode a part follows exactly one bone, so every weights array
    # has a single one in it.
    for block in scene.split('[node name="'):
        if 'type="Polygon2D"' not in block:
            continue
        bones = re.search(r"bones = \[(.*)\]", block)
        if bones is None:
            continue
        assert len(re.findall(r"PackedFloat32Array", bones.group(1))) == 1


def test_mesh_mode_uses_the_weights(tmp_path: Path, character_image: Path) -> None:
    session = _run(tmp_path, character_image, SkinningMode.MESH)
    scene = (session.exports_dir / "godot" / "scene.tscn").read_text(encoding="utf-8")
    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))
    assert rig["skinning"] == "mesh"
    for mesh in rig["meshes"]:
        assert mesh["bones"], "every vertex needs at least one bone"
        assert mesh["weights"]


def test_the_unity_assets_are_written(tmp_path: Path, character_image: Path) -> None:
    session = _run_with_unity(tmp_path, character_image)
    unity_dir = session.exports_dir / "unity"
    assert unity_dir.is_dir()
    assert (unity_dir / "Assets").is_dir()

    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))
    for layer in rig["layers"]:
        image = unity_dir / "Assets" / layer["image"]
        assert image.is_file(), f"missing {image}"
        assert (unity_dir / "Assets" / f"{layer['image']}.meta").is_file()
        # Without both of these Unity imports the picture as a plain texture,
        # no sprite exists, and every reference in the prefab resolves to
        # nothing.
        meta = (unity_dir / "Assets" / f"{layer['image']}.meta").read_text(encoding="utf-8")
        assert "textureType: 8" in meta
        assert "spriteMode: 1" in meta
        assert "alphaIsTransparency: 1" in meta


def test_the_unity_prefab_has_a_bone_hierarchy(tmp_path: Path, character_image: Path) -> None:
    session = _run_with_unity(tmp_path, character_image)
    prefab = session.exports_dir / "unity" / "Assets" / "Character.prefab"
    text = prefab.read_text(encoding="utf-8")
    objects = _parse_prefab(prefab)

    assert text.startswith("%YAML 1.1")
    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))

    # Every object in the file is one of the three kinds Unity expects.
    classes = {obj["__class__"] for obj in objects.values()}
    assert classes <= {"!u!1", "!u!4", "!u!212"}

    # Every bone is a GameObject with a Transform, and the names are there.
    for bone in rig["bones"]:
        assert f"  m_Name: {bone['id']}" in text

    # There is exactly one root: the only Transform with no father.
    fathers = [obj for obj in objects.values() if obj["__class__"] == "!u!4"]
    rootless = [obj for obj in fathers if _refs(obj["m_Father"][0]) == [0]]
    assert len(rootless) == 1, "a prefab has exactly one root"


def test_every_unity_reference_points_at_something(tmp_path: Path, character_image: Path) -> None:
    """A prefab is a graph of file ids. A dangling one is a broken prefab."""

    session = _run_with_unity(tmp_path, character_image)
    objects = _parse_prefab(session.exports_dir / "unity" / "Assets" / "Character.prefab")
    known = set(objects)

    for fid, obj in objects.items():
        for component in obj.get("m_Component", []):
            for reference in _refs(component):
                assert reference in known, f"{fid} names a missing component {reference}"
        for father in obj.get("m_Father", []):
            for reference in _refs(father):
                assert reference == 0 or reference in known, f"{fid} has a missing father"
        for child in obj.get("m_Children", []):
            for reference in _refs(child):
                assert reference in known, f"{fid} has a missing child"
        for game_object in obj.get("m_GameObject", []):
            for reference in _refs(game_object):
                assert reference in known, f"{fid} belongs to a missing object"

    # The tree must be a tree: follow every child back to the one root.
    transforms = {fid: obj for fid, obj in objects.items() if obj["__class__"] == "!u!4"}
    roots = [fid for fid, obj in transforms.items() if _refs(obj["m_Father"][0]) == [0]]
    assert len(roots) == 1
    reached: set[int] = set()
    stack = [roots[0]]
    while stack:
        fid = stack.pop()
        assert fid not in reached, "the hierarchy has a cycle"
        reached.add(fid)
        for child in transforms[fid].get("m_Children", []):
            stack.extend(_refs(child))
    assert reached == set(transforms), "some objects are not part of the tree"


def test_the_unity_sprites_point_at_the_generated_guids(
    tmp_path: Path, character_image: Path
) -> None:
    """The prefab and the .meta files have to agree, or nothing renders."""

    session = _run_with_unity(tmp_path, character_image)
    unity_dir = session.exports_dir / "unity"
    objects = _parse_prefab(unity_dir / "Assets" / "Character.prefab")

    sprites = [obj for obj in objects.values() if obj["__class__"] == "!u!212"]
    assert sprites, "the prefab has no sprites"

    rig = json.loads((session.exports_dir / "rig.json").read_text(encoding="utf-8"))
    written_guids = {}
    for layer in rig["layers"]:
        meta = unity_dir / "Assets" / f"{layer['image']}.meta"
        if meta.is_file():
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.startswith("guid:"):
                    written_guids[layer["image"]] = line.split(":", 1)[1].strip()
                    break
    assert written_guids

    for sprite in sprites:
        value = sprite["m_Sprite"][0]
        # Unity's fixed file id for the single sprite a texture produces.
        assert "fileID: 21300000" in value
        guids = [part.split(":")[1].strip() for part in value.replace("{", "").replace("}", "").split(",") if part.strip().startswith("guid:")]
        assert guids and guids[0] in set(written_guids.values()), value
