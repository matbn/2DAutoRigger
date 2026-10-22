"""Rig templates.

A template is data, not code: it says how a class of character is cut up, boned
and layered. Adding a quadruped, a bird or a wholly custom creature is a new
JSON file, which is why the core only ships a humanoid one and the mechanism to
discover others.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..contracts.models import SkinMode


class LandmarkSpec(BaseModel):
    x: float
    y: float
    confidence: float = 1.0
    optional: bool = False


class PartSpec(BaseModel):
    id: str
    z: int = 0
    skin: SkinMode = SkinMode.RIGID
    bone: str | None = None
    anchors: list[str] = Field(default_factory=list)
    radius: float = 0.0
    widen: float = 0.0


class BoneSpec(BaseModel):
    id: str
    parent: str | None = None
    head: str | None = None
    tail: str | None = None
    length: str = "landmark"
    part: str | None = None
    direction: str = "head_to_tail"


class RigTemplate(BaseModel):
    id: str
    type: str = "humanoid"
    version: int = 1
    description: str = ""
    landmarks: dict[str, LandmarkSpec] = Field(default_factory=dict)
    parts: list[PartSpec] = Field(default_factory=list)
    bones: list[BoneSpec] = Field(default_factory=list)
    proportions: dict[str, float] = Field(default_factory=dict)

    def part(self, part_id: str) -> PartSpec | None:
        for part in self.parts:
            if part.id == part_id:
                return part
        return None

    def bone(self, bone_id: str) -> BoneSpec | None:
        for bone in self.bones:
            if bone.id == bone_id:
                return bone
        return None

    @property
    def part_ids(self) -> list[str]:
        return [p.id for p in self.parts]

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _template_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("RIGGER2D_TEMPLATES")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    dirs.append(Path.home() / ".rigger2d" / "templates")
    dirs.append(Path.cwd() / "templates")
    return dirs


def _packaged_templates() -> Path:
    return Path(__file__).resolve().parent.parent / "templates"


class TemplateRegistry:
    """Finds templates on disk."""

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self._dirs: list[Path] = []
        self._extra = list(extra_dirs or [])
        self._cache: dict[str, RigTemplate] = {}

    def search_paths(self) -> list[Path]:
        seen: set[Path] = set()
        out: list[Path] = []
        candidates = [*self._extra, _packaged_templates(), *_template_dirs()]
        for candidate in candidates:
            resolved = candidate.expanduser()
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            out.append(resolved)
        return out

    def load(self, template_id: str) -> RigTemplate:
        if template_id in self._cache:
            return self._cache[template_id]
        for directory in self.search_paths():
            path = directory / f"{template_id}.json"
            if path.is_file():
                template = load_template_file(path)
                self._cache[template_id] = template
                return template
        # The file name does not have to match the template id, so fall back to
        # scanning every template for a matching id.
        for template in self.available():
            if template.id == template_id:
                self._cache[template_id] = template
                return template
        raise KeyError(f"no template named {template_id!r}")

    def available(self) -> list[RigTemplate]:
        found: dict[str, RigTemplate] = {}
        for directory in self.search_paths():
            for path in sorted(directory.glob("*.json")):
                if path.name.endswith(".schema.json"):
                    continue
                try:
                    template = load_template_file(path)
                except Exception:
                    continue
                found[template.id] = template
        return list(found.values())


def load_template_file(path: Path) -> RigTemplate:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RigTemplate.model_validate(data)


DEFAULT_TEMPLATE_ID = "humanoid.biped.v1"

__all__ = [
    "BoneSpec",
    "DEFAULT_TEMPLATE_ID",
    "LandmarkSpec",
    "PartSpec",
    "RigTemplate",
    "TemplateRegistry",
    "load_template_file",
]
