"""The Unity emitter.

Writes a Unity project folder:

    unity/
        Assets/
            <part>.png          the layer images
            <part>.png.meta     their import settings and GUIDs
        Character.prefab        the rig, as a prefab

A Unity prefab is a YAML file, so this is written from Python with no Unity
involved. Nothing in this repository depends on a Unity installation, and no
Unity-generated file is ever committed: the emitter *writes* Unity files, it
does not contain any.

The format was learned by saving a prefab from the editor once and reading it
back. The parts that matter:

* A prefab is a flat list of documents. Each object is a ``GameObject``
  (``!u!1``) followed by its components -- here a ``Transform`` (``!u!4``) and,
  for a body part, a ``SpriteRenderer`` (``!u!212``).
* ``&<id>`` after the class tag is the object's file id. Everything refers to
  everything else by that number, so the graph has to be built before it can be
  written.
* Only the root of the prefab has ``m_Father: {fileID: 0}``. There is no extra
  header block: a prefab that is not a variant of another prefab is just its
  objects.
* A ``SpriteRenderer`` points at its sprite as
  ``{fileID: 21300000, guid: <texture guid>, type: 3}``. That ``21300000`` is
  the fixed file id of the single sprite a texture importer in ``Sprite`` mode
  produces, which is why it can be written without asking Unity.
* ``m_LocalRotation`` is a quaternion. A 2D rotation about Z is
  ``(0, 0, sin(t/2), cos(t/2))``.

Unity's 2D axis points up, which is the mirror of the image space the core
works in, so everything is converted here, at the edge, exactly as in the Spine
emitter.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Callable

from ..contracts.models import Bone, Rig

# The fixed file id of the sprite a Sprite-mode texture importer produces.
SPRITE_FILE_ID = 21300000
# The built-in default sprite material and its GUID.
DEFAULT_MATERIAL = "{fileID: 10754, guid: 0000000000000000f000000000000000, type: 0}"
# Unity's default pixels per unit, which fixes the world size of a sprite.
PIXELS_PER_UNIT = 100.0


def _texture_meta(guid: str) -> str:
    """The ``.meta`` a texture needs to import as a single sprite.

    ``textureType: 8`` is Sprite and ``spriteMode: 1`` is Single. Without both,
    Unity imports the picture as a plain texture, no sprite is generated, and
    every ``m_Sprite`` reference in the prefab silently resolves to nothing.
    """

    return f"""fileFormatVersion: 2
guid: {guid}
TextureImporter:
  internalIDToNameTable: []
  externalObjects: {{}}
  serializedVersion: 13
  mipmaps:
    mipMapMode: 0
    enableMipMap: 0
    sRGBTexture: 1
    linearTexture: 0
    fadeOut: 0
    borderMipMap: 0
    mipMapsPreserveCoverage: 0
    alphaTestReferenceValue: 0.5
    mipMapFadeDistanceStart: 1
    mipMapFadeDistanceEnd: 3
  bumpmap:
    convertToNormalMap: 0
    externalNormalMap: 0
    heightScale: 0.25
    normalMapFilter: 0
    flipGreenChannel: 0
  isReadable: 0
  streamingMipmaps: 0
  streamingMipmapsPriority: 0
  vTOnly: 0
  ignoreMipmapLimit: 0
  grayScaleToAlpha: 0
  generateCubemap: 6
  cubemapConvolution: 0
  seamlessCubemap: 0
  textureFormat: 1
  maxTextureSize: 2048
  textureSettings:
    serializedVersion: 2
    filterMode: 1
    aniso: 1
    mipBias: 0
    wrapU: 0
    wrapV: 0
    wrapW: 0
  nPOTScale: 1
  lightmap: 0
  compressionQuality: 50
  spriteMode: 1
  spriteExtrude: 1
  spriteMeshType: 1
  alignment: 0
  spritePivot: {{x: 0.5, y: 0.5}}
  spritePixelsToUnits: {int(PIXELS_PER_UNIT)}
  spriteBorder: {{x: 0, y: 0, z: 0, w: 0}}
  spriteGenerateFallbackPhysicsShape: 1
  alphaUsage: 1
  alphaIsTransparency: 1
  spriteTessellationMethod: 0
  spriteTessellationDetail: -1
  spriteGeometrySubdivision: -1
  textureType: 8
  textureShape: 1
  singleChannelComponent: 0
  flipbookRows: 1
  flipbookColumns: 1
  maxTextureSizeSet: 0
  compressionQualitySet: 0
  textureFormatSet: 0
  ignorePngGamma: 0
  applyGammaDecoding: 0
  swizzle: 50462976
  cookieLightType: 0
  platformSettings: []
  spriteSheet:
    serializedVersion: 2
    sprites: []
    outline: []
    customData: 
    physicsShape: []
    bones: []
    spriteID: 
    internalID: 0
    vertices: []
    indices: 
    edges: []
    weights: []
    secondaryTextures: []
    spriteCustomMetadata:
      entries: []
    nameFileIdTable: {{}}
  mipmapLimitGroupName: 
  pSDRemoveMatte: 0
  userData: 
  assetBundleName: 
  assetBundleVariant: 
"""


def _id_maker(seed: str) -> Callable[[str, str], int]:
    """Stable, unique file ids.

    Unity's own ids look random but only have to be unique within the file.
    Deriving them from the project means the same rig always produces the same
    prefab, which makes the output diffable.
    """

    used: set[int] = set()
    counter = 0

    def make(kind: str, name: str) -> int:
        nonlocal counter
        while True:
            counter += 1
            digest = hashlib.sha1(f"{seed}|{kind}|{name}|{counter}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
            if value and value not in used:
                used.add(value)
                return value

    return make


def _num(value: float) -> str:
    """A number Unity will parse back to what we meant."""

    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _quaternion(angle: float) -> tuple[float, float, float, float]:
    """A rotation about Z, as Unity's quaternion."""

    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def _game_object(file_id: int, name: str, components: list[int], layer: int = 0) -> str:
    lines = [
        f"--- !u!1 &{file_id}",
        "GameObject:",
        "  m_ObjectHideFlags: 0",
        "  m_CorrespondingSourceObject: {fileID: 0}",
        "  m_PrefabInstance: {fileID: 0}",
        "  m_PrefabAsset: {fileID: 0}",
        "  serializedVersion: 6",
        "  m_Component:",
    ]
    for component in components:
        lines.append(f"  - component: {{fileID: {component}}}")
    lines += [
        f"  m_Layer: {layer}",
        f"  m_Name: {name}",
        "  m_TagString: Untagged",
        "  m_Icon: {fileID: 0}",
        "  m_NavMeshLayer: 0",
        "  m_StaticEditorFlags: 0",
        "  m_IsActive: 1",
    ]
    return "\n".join(lines) + "\n"


def _transform(
    file_id: int,
    owner: int,
    position: tuple[float, float],
    rotation: tuple[float, float, float, float],
    children: list[int],
    father: int,
) -> str:
    lines = [
        f"--- !u!4 &{file_id}",
        "Transform:",
        "  m_ObjectHideFlags: 0",
        "  m_CorrespondingSourceObject: {fileID: 0}",
        "  m_PrefabInstance: {fileID: 0}",
        "  m_PrefabAsset: {fileID: 0}",
        f"  m_GameObject: {{fileID: {owner}}}",
        "  serializedVersion: 2",
        "  m_LocalRotation: {x: %s, y: %s, z: %s, w: %s}"
        % tuple(_num(v) for v in rotation),
        f"  m_LocalPosition: {{x: {_num(position[0])}, y: {_num(position[1])}, z: 0}}",
        "  m_LocalScale: {x: 1, y: 1, z: 1}",
        "  m_ConstrainProportionsScale: 0",
        "  m_Children: []" if not children else "  m_Children:",
    ]
    for child in children:
        lines.append(f"  - {{fileID: {child}}}")
    lines += [
        f"  m_Father: {{fileID: {father}}}",
        "  m_LocalEulerAnglesHint: {x: 0, y: 0, z: 0}",
    ]
    return "\n".join(lines) + "\n"


def _sprite_renderer(
    file_id: int,
    owner: int,
    guid: str,
    sorting_order: int,
    size: tuple[float, float],
) -> str:
    lines = [
        f"--- !u!212 &{file_id}",
        "SpriteRenderer:",
        "  serializedVersion: 3",
        "  m_ObjectHideFlags: 0",
        "  m_CorrespondingSourceObject: {fileID: 0}",
        "  m_PrefabInstance: {fileID: 0}",
        "  m_PrefabAsset: {fileID: 0}",
        f"  m_GameObject: {{fileID: {owner}}}",
        "  m_Enabled: 1",
        "  m_CastShadows: 0",
        "  m_ReceiveShadows: 0",
        "  m_DynamicOccludee: 1",
        "  m_StaticShadowCaster: 0",
        "  m_MotionVectors: 1",
        "  m_LightProbeUsage: 0",
        "  m_ReflectionProbeUsage: 1",
        "  m_RayTracingMode: 0",
        "  m_RayTraceProcedural: 0",
        "  m_RayTracingAccelStructBuildFlagsOverride: 0",
        "  m_RayTracingAccelStructBuildFlags: 1",
        "  m_SmallMeshCulling: 1",
        "  m_ForceMeshLod: -1",
        "  m_MeshLodSelectionBias: 0",
        "  m_RenderingLayerMask: 1",
        "  m_RendererPriority: 0",
        "  m_Materials:",
        f"  - {DEFAULT_MATERIAL}",
        "  m_StaticBatchInfo:",
        "    firstSubMesh: 0",
        "    subMeshCount: 0",
        "  m_StaticBatchRoot: {fileID: 0}",
        "  m_ProbeAnchor: {fileID: 0}",
        "  m_LightProbeVolumeOverride: {fileID: 0}",
        "  m_ScaleInLightmap: 1",
        "  m_ReceiveGI: 1",
        "  m_PreserveUVs: 0",
        "  m_IgnoreNormalsForChartDetection: 0",
        "  m_ImportantGI: 0",
        "  m_StitchLightmapSeams: 1",
        "  m_SelectedEditorRenderState: 0",
        "  m_MinimumChartSize: 4",
        "  m_AutoUVMaxDistance: 0.5",
        "  m_AutoUVMaxAngle: 89",
        "  m_LightmapParameters: {fileID: 0}",
        "  m_GlobalIlluminationMeshLod: 0",
        "  m_SortingLayerID: 0",
        "  m_SortingLayer: 0",
        f"  m_SortingOrder: {sorting_order}",
        "  m_MaskInteraction: 0",
        f"  m_Sprite: {{fileID: {SPRITE_FILE_ID}, guid: {guid}, type: 3}}",
        "  m_Color: {r: 1, g: 1, b: 1, a: 1}",
        "  m_FlipX: 0",
        "  m_FlipY: 0",
        "  m_DrawMode: 0",
        f"  m_Size: {{x: {_num(size[0])}, y: {_num(size[1])}}}",
        "  m_AdaptiveModeThreshold: 0.5",
        "  m_SpriteTileMode: 0",
        "  m_WasSpriteAssigned: 1",
        "  m_SpriteSortPoint: 0",
        "  m_BlendShapeWeights: []",
    ]
    return "\n".join(lines) + "\n"


def emit(rig: Rig, out_dir: Path, layers_dir: Path) -> list[str]:
    """Write the Unity project folder: the textures and the prefab."""

    unity_dir = out_dir / "unity"
    assets = unity_dir / "Assets"
    assets.mkdir(parents=True, exist_ok=True)

    make_id = _id_maker(rig.project_id or rig.template_id)

    # -- the textures and their import settings ---------------------------

    guids: dict[str, str] = {}
    for layer in rig.layers:
        source = layers_dir / layer.image
        if not source.is_file():
            continue
        shutil.copy2(source, assets / layer.image)
        # A GUID Unity will not collide with: it is derived from the project and
        # the layer, so the same rig always produces the same GUID.
        guid = hashlib.md5(f"{rig.project_id}:{layer.id}".encode("utf-8")).hexdigest()
        guids[layer.id] = guid
        (assets / f"{layer.image}.meta").write_text(_texture_meta(guid), encoding="utf-8")

    # -- the bone tree ----------------------------------------------------

    by_id = {bone.id: bone for bone in rig.bones}
    bone_children: dict[str, list[Bone]] = {}
    roots: list[Bone] = []
    for bone in rig.bones:
        if bone.parent and bone.parent in by_id:
            bone_children.setdefault(bone.parent, []).append(bone)
        else:
            roots.append(bone)

    height = rig.height

    def world_head(bone: Bone) -> tuple[float, float]:
        """The bone's origin in Unity's space: y is flipped, nothing else."""

        return (bone.head.x, height - bone.head.y)

    def world_angle(bone: Bone) -> float:
        """The bone's direction in radians. Flipping y negates the angle."""

        return -math.atan2(bone.tail.y - bone.head.y, bone.tail.x - bone.head.x)

    def local_from_world(
        point: tuple[float, float],
        parent: Bone | None,
    ) -> tuple[tuple[float, float], float, float]:
        """Express a world point in a parent bone's frame.

        Returns ``(local_position, parent_angle, world_angle_of_parent)``.
        """

        if parent is None:
            return point, 0.0, 0.0
        px, py = world_head(parent)
        pa = world_angle(parent)
        dx = point[0] - px
        dy = point[1] - py
        # Rotate by -pa, which is what "in the parent's frame" means.
        return (
            (dx * math.cos(pa) + dy * math.sin(pa), -dx * math.sin(pa) + dy * math.cos(pa)),
            pa,
            0.0,
        )

    # Depth-first order, so a parent is always written before its children.
    bones_in_order: list[Bone] = []

    def walk(bone: Bone) -> None:
        bones_in_order.append(bone)
        for child in bone_children.get(bone.id, []):
            walk(child)

    for root in roots:
        walk(root)

    # -- the parts, and which bone drives each one ------------------------

    part_layers = [layer for layer in rig.layers if layer.kind != "fill"]
    part_bone: dict[str, Bone | None] = {}
    for layer in part_layers:
        bone = next((b for b in rig.bones if b.part_id == layer.part_id), None)
        if bone is None and rig.bones:
            bone = rig.bones[0]
        part_bone[layer.id] = bone

    # -- allocate the file ids -------------------------------------------

    # A bone and a part can share a name -- there is a "head" bone and a "head"
    # part -- so the two namespaces are kept apart.
    ids: dict[str, int] = {}
    ids["go:Character"] = make_id("go", "Character")
    ids["tr:Character"] = make_id("tr", "Character")
    for bone in bones_in_order:
        ids[f"bonego:{bone.id}"] = make_id("bonego", bone.id)
        ids[f"bonetr:{bone.id}"] = make_id("bonetr", bone.id)
    for layer in rig.layers:
        ids[f"partgo:{layer.id}"] = make_id("partgo", layer.id)
        ids[f"parttr:{layer.id}"] = make_id("parttr", layer.id)
        ids[f"partsr:{layer.id}"] = make_id("partsr", layer.id)

    # -- write the prefab -------------------------------------------------

    documents: list[str] = ["%YAML 1.1", "%TAG !u! tag:unity3d.com,2011:", ""]

    # The root.
    root_children = [ids[f"bonetr:{root.id}"] for root in roots]
    root_children += [
        ids[f"parttr:{layer.id}"] for layer in rig.layers if part_bone.get(layer.id) is None
    ]
    documents.append(_game_object(ids["go:Character"], "Character", [ids["tr:Character"]]))
    documents.append(
        _transform(
            ids["tr:Character"],
            ids["go:Character"],
            (0.0, 0.0),
            _quaternion(0.0),
            root_children,
            0,
        )
    )

    # The bones.
    for bone in bones_in_order:
        parent = by_id.get(bone.parent) if bone.parent else None
        position, parent_angle, _ = local_from_world(world_head(bone), parent)
        angle = world_angle(bone) - (world_angle(parent) if parent is not None else 0.0)
        children = [ids[f"bonetr:{child.id}"] for child in bone_children.get(bone.id, [])]
        children += [
            ids[f"parttr:{layer.id}"]
            for layer in rig.layers
            if part_bone.get(layer.id) is bone
        ]
        documents.append(
            _game_object(ids[f"bonego:{bone.id}"], bone.id, [ids[f"bonetr:{bone.id}"]])
        )
        documents.append(
            _transform(
                ids[f"bonetr:{bone.id}"],
                ids[f"bonego:{bone.id}"],
                position,
                _quaternion(angle),
                children,
                ids[f"bonetr:{parent.id}"] if parent is not None else ids["tr:Character"],
            )
        )

    # The parts, in the order they are drawn, so the sorting order is the
    # drawing order and the fills stay behind their own part.
    for index, layer in enumerate(rig.layers):
        bone = part_bone.get(layer.id)
        bounds = layer.bounds
        width = bounds.x1 - bounds.x0
        height_px = bounds.y1 - bounds.y0
        centre = (bounds.x0 + width / 2.0, height - (bounds.y0 + height_px / 2.0))
        parent = bone if bone is not None else None
        position, _pa, _wa = local_from_world(centre, parent)
        guid = guids.get(layer.id, "")

        documents.append(
            _game_object(
                ids[f"partgo:{layer.id}"],
                layer.id,
                [ids[f"parttr:{layer.id}"], ids[f"partsr:{layer.id}"]],
            )
        )
        documents.append(
            _transform(
                ids[f"parttr:{layer.id}"],
                ids[f"partgo:{layer.id}"],
                position,
                _quaternion(0.0),
                [],
                ids[f"bonetr:{parent.id}"] if parent is not None else ids["tr:Character"],
            )
        )
        documents.append(
            _sprite_renderer(
                ids[f"partsr:{layer.id}"],
                ids[f"partgo:{layer.id}"],
                guid,
                index,
                (width / PIXELS_PER_UNIT, height_px / PIXELS_PER_UNIT),
            )
        )

    # The prefab goes *inside* Assets/. Unity only imports what is under
    # Assets/, so a prefab next to it would never be seen.
    prefab_path = assets / "Character.prefab"
    prefab_path.write_text("\n".join(documents), encoding="utf-8")

    written = ["unity/Assets/Character.prefab"]
    written += [f"unity/Assets/{layer.image}" for layer in rig.layers if layer.id in guids]
    return written


__all__ = ["emit"]
