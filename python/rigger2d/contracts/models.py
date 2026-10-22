"""Data contracts shared by the pipeline, the emitters and the JSON Schema files.

Everything that crosses a boundary (disk, HTTP, a plugin) is described here, so
that the Python implementation, the TypeScript client and the schema files all
agree on one definition.

Coordinate convention used throughout the core:

* units are **image pixels**
* the origin is the **top-left** corner of the source image
* ``y`` grows **downwards**

Emitters that use a different convention (Spine is y-up) convert at the edge.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class Task(StrEnum):
    """A capability an adapter can provide."""

    MATTE = "matte"
    PARTS = "parts"
    POSE = "pose"
    CLASSIFY = "classify"
    INPAINT = "inpaint"


class ProbeStatus(StrEnum):
    """Why an adapter can or cannot be used right now."""

    AVAILABLE = "available"
    MISSING_DEPS = "missing_deps"
    MISSING_WEIGHTS = "missing_weights"
    LICENSE_UNACCEPTED = "license_unaccepted"
    DISABLED = "disabled"
    ERROR = "error"


class SkinMode(StrEnum):
    """How a part is bound to the skeleton."""

    RIGID = "rigid"
    WEIGHTED = "weighted"


class SkinningMode(StrEnum):
    """Whole-rig deformation mode, chosen at export time."""

    CUTOUT = "cutout"
    HYBRID = "hybrid"
    MESH = "mesh"


class Vec2(BaseModel):
    x: float
    y: float


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Vec2:
        return Vec2(x=(self.x0 + self.x1) / 2.0, y=(self.y0 + self.y1) / 2.0)


class ImageInfo(BaseModel):
    path: str
    width: int
    height: int
    has_alpha: bool = False


class ProvenanceEntry(BaseModel):
    """One backend call, recorded for the ``provenance.json`` sidecar."""

    adapter_id: str
    adapter_version: str = "0"
    task: Task
    ok: bool = True
    duration_ms: int = 0
    license_spdx: str | None = None
    license_url: str | None = None
    detail: str | None = None


class StageResult(BaseModel):
    name: str
    version: int = 1
    duration_ms: int = 0
    adapter_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Probe(BaseModel):
    """The result of asking an adapter whether it can run."""

    adapter_id: str
    version: str = "0"
    status: ProbeStatus = ProbeStatus.AVAILABLE
    reason: str | None = None
    license_spdx: str | None = None
    license_url: str | None = None
    device: str | None = None
    capabilities: list[Task] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == ProbeStatus.AVAILABLE


class MaskInfo(BaseModel):
    """A part mask, stored on disk as a PNG and described here."""

    part_id: str
    image: str
    bounds: BBox
    area: int
    pixels: int = 0


class MatteState(BaseModel):
    image: str
    bounds: BBox
    coverage: float = 0.0


class PartState(BaseModel):
    masks: list[MaskInfo] = Field(default_factory=list)


class Bone(BaseModel):
    """A bone. ``head``/``tail`` are in image pixels."""

    id: str
    parent: str | None = None
    head: Vec2
    tail: Vec2
    length: float = 0.0
    part_id: str | None = None
    layer: str | None = None
    rest_rotation: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        if self.length == 0.0:
            dx = self.tail.x - self.head.x
            dy = self.tail.y - self.head.y
            self.length = (dx * dx + dy * dy) ** 0.5


class Layer(BaseModel):
    """One RGBA image in the sliced character.

    ``kind`` is ``"part"`` for a cut out body part and ``"fill"`` for the
    solid colour patch that hides the hole a part leaves behind. A fill has no
    mesh of its own: it is a flat shape, not something to deform.
    """

    id: str
    part_id: str
    image: str
    z: int = 0
    pivot: Vec2 = Field(default_factory=lambda: Vec2(x=0.0, y=0.0))
    bounds: BBox = Field(default_factory=lambda: BBox(x0=0, y0=0, x1=0, y1=0))
    skin: SkinMode = SkinMode.RIGID
    kind: str = "part"


class Mesh(BaseModel):
    """A deformable mesh for one part."""

    part_id: str
    vertices: list[Vec2] = Field(default_factory=list)
    uvs: list[Vec2] = Field(default_factory=list)
    triangles: list[int] = Field(default_factory=list)
    bones: list[list[int]] = Field(default_factory=list)
    weights: list[list[float]] = Field(default_factory=list)


class Rig(BaseModel):
    """The canonical rig document. ``rig.json`` is this, serialised."""

    schema_version: str = SCHEMA_VERSION
    project_id: str
    template_id: str
    template_type: str = "humanoid"
    width: int
    height: int
    skinning: SkinningMode = SkinningMode.HYBRID
    landmarks: dict[str, Vec2] = Field(default_factory=dict)
    bones: list[Bone] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)
    meshes: list[Mesh] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    created_at: str = ""

    def bone(self, bone_id: str) -> Bone | None:
        for b in self.bones:
            if b.id == bone_id:
                return b
        return None

    def layer(self, layer_id: str) -> Layer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None


class MaskEdit(BaseModel):
    """A single mask correction made during refinement.

    Strokes are stored rather than re-rasterised masks so that the edit log is
    compact, undoable and replayable, and so the original segmentation stays the
    source of truth.
    """

    op: Literal["add", "erase", "assign"]
    part_id: str
    points: list[Vec2] = Field(default_factory=list)
    radius: float = 6.0


class JointEdit(BaseModel):
    """A landmark moved by the user during refinement."""

    joint_id: str
    dx: float = 0.0
    dy: float = 0.0


class Edit(BaseModel):
    mask: MaskEdit | None = None
    joint: JointEdit | None = None


class Document(BaseModel):
    """Everything the pipeline knows about one character."""

    schema_version: str = SCHEMA_VERSION
    project_id: str
    created_at: str = ""
    image: ImageInfo
    template_id: str = "humanoid.biped.v1"
    template_type: str = "humanoid"
    stages: dict[str, StageResult] = Field(default_factory=dict)
    matte: MatteState | None = None
    parts: PartState | None = None
    landmarks: dict[str, Vec2] = Field(default_factory=dict)
    landmark_confidence: dict[str, float] = Field(default_factory=dict)
    edits: list[Edit] = Field(default_factory=list)
    rig: Rig | None = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def refined(self) -> bool:
        return bool(self.edits)
