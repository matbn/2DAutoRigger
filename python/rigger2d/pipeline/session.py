"""A project on disk.

A project is a directory::

    hero.rig/
        document.json      the pipeline state
        source.png         the ingested image
        masks/matte.png    the character matte
        masks/parts/*.png  one mask per part
        layers/*.png       the sliced character
        exports/           rig.json, godot, spine, ...

The document is the single source of truth. Every stage reads it and writes it
back, which is what makes the pipeline resumable, cacheable and inspectable.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.base import AdapterContext
from ..adapters.registry import Registry
from ..contracts.models import Document, ImageInfo
from ..io import images
from ..rigging.templates import DEFAULT_TEMPLATE_ID, RigTemplate, TemplateRegistry

DOCUMENT_NAME = "document.json"
SOURCE_NAME = "source.png"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Session:
    """Everything the pipeline needs to work on one character."""

    def __init__(self, root: Path, template_id: str = DEFAULT_TEMPLATE_ID) -> None:
        self.root = Path(root)
        self.template_id = template_id
        self.templates = TemplateRegistry()
        self._registry: Registry | None = None

    # -- layout -----------------------------------------------------------

    @property
    def document_path(self) -> Path:
        return self.root / DOCUMENT_NAME

    @property
    def source_path(self) -> Path:
        return self.root / SOURCE_NAME

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def parts_dir(self) -> Path:
        return self.masks_dir / "parts"

    @property
    def layers_dir(self) -> Path:
        return self.root / "layers"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    # -- collaborators ----------------------------------------------------

    @property
    def template(self) -> RigTemplate:
        return self.templates.load(self.template_id)

    @property
    def registry(self) -> Registry:
        if self._registry is None:
            self._registry = Registry.discover(self.template)
        return self._registry

    def rebind(self) -> None:
        """Rebuild the adapter registry, e.g. after the template changed."""

        self._registry = None

    @property
    def context(self) -> AdapterContext:
        return AdapterContext(
            project_dir=self.root,
            work_dir=self.masks_dir,
            device="cpu",
        )

    # -- document ---------------------------------------------------------

    def load(self) -> Document:
        if not self.document_path.is_file():
            raise FileNotFoundError(f"no project at {self.root}")
        with self.document_path.open("r", encoding="utf-8") as fh:
            return Document.model_validate(json.load(fh))

    def save(self, document: Document) -> Document:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.document_path.open("w", encoding="utf-8") as fh:
            json.dump(document.model_dump(mode="json"), fh, indent=2)
        return document

    def create(self, image_path: Path, template_id: str | None = None) -> Document:
        """Start a project from an image."""

        self.root.mkdir(parents=True, exist_ok=True)
        if template_id:
            self.template_id = template_id
            self.rebind()

        rgba = images.load_rgba(Path(image_path))
        images.save_rgba(self.source_path, rgba)

        info = ImageInfo(
            path=SOURCE_NAME,
            width=int(rgba.shape[1]),
            height=int(rgba.shape[0]),
            has_alpha=bool(int(rgba[..., 3].min()) < 250),
        )
        document = Document(
            project_id=uuid.uuid4().hex[:12],
            created_at=utcnow(),
            image=info,
            template_id=self.template_id,
            template_type=self.template.type,
        )
        return self.save(document)

    # -- masks ------------------------------------------------------------

    def part_mask_paths(self) -> dict[str, Path]:
        if not self.parts_dir.is_dir():
            return {}
        return {path.stem: path for path in sorted(self.parts_dir.glob("*.png"))}

    def load_part_masks(self) -> dict[str, "np.ndarray"]:  # type: ignore[name-defined]
        import numpy as np  # local import keeps the module import cheap

        masks: dict[str, np.ndarray] = {}
        for part_id, path in self.part_mask_paths().items():
            masks[part_id] = images.load_mask(path)
        return masks

    def save_part_masks(self, masks: dict[str, "np.ndarray"]) -> None:  # type: ignore[name-defined]
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        for part_id, mask in masks.items():
            images.save_mask(self.parts_dir / f"{part_id}.png", mask)

    def copy_source_to(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source_path, destination)
        return destination


__all__ = ["Session", "utcnow"]
