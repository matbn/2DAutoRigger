"""Loading and validating plugin manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..contracts.models import Task


class LicenseInfo(BaseModel):
    spdx: str
    url: str | None = None
    requires_acceptance: bool = False


class WeightsInfo(BaseModel):
    source: str | None = None
    gated: bool = False
    terms_url: str | None = None
    note: str | None = None


class RequiresInfo(BaseModel):
    python: str | None = None
    packages: list[str] = Field(default_factory=list)


class PluginManifest(BaseModel):
    """The manifest a plugin ships alongside its code."""

    id: str
    version: str
    name: str | None = None
    description: str | None = None
    tasks: list[Task]
    part_vocabulary: list[str] = Field(default_factory=list)
    license: LicenseInfo
    weights: WeightsInfo | None = None
    requires: RequiresInfo = Field(default_factory=RequiresInfo)
    entry_point: str
    priority: int = 100

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def parse_manifest(data: dict[str, Any]) -> PluginManifest:
    return PluginManifest.model_validate(data)


def load_manifest(path: Path) -> PluginManifest:
    with path.open("r", encoding="utf-8") as fh:
        return parse_manifest(json.load(fh))


def manifest_path(plugin_dir: Path) -> Path:
    return plugin_dir / "manifest.json"


__all__ = [
    "LicenseInfo",
    "PluginManifest",
    "RequiresInfo",
    "WeightsInfo",
    "load_manifest",
    "manifest_path",
    "parse_manifest",
]
