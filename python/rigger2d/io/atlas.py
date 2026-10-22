"""A tiny atlas packer.

Some engines want every part of the character in one texture. A shelf packer
is not the best packer in the world, but it is simple, fast and produces tight
results for the roughly rectangular part images a sliced character produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .images import load_rgba, save_rgba


@dataclass
class AtlasEntry:
    """One image placed in the atlas."""

    name: str
    image: np.ndarray
    x: int = 0
    y: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def pack_atlas(
    entries: list[AtlasEntry],
    max_size: int = 2048,
    padding: int = 2,
) -> tuple[np.ndarray, list[AtlasEntry]]:
    """Pack images into one texture. Returns the atlas and the placed entries."""

    if not entries:
        return np.zeros((1, 1, 4), dtype=np.uint8), []

    ordered = sorted(entries, key=lambda e: (-e.height, e.width))
    total_area = sum(e.width * e.height for e in ordered)
    target = max(64, int((total_area * 1.4) ** 0.5))
    width = min(max_size, max(target, max(e.width for e in ordered) + padding * 2))

    x = padding
    y = padding
    shelf_height = 0
    for entry in ordered:
        if x + entry.width + padding > width:
            x = padding
            y += shelf_height + padding
            shelf_height = 0
        entry.x = x
        entry.y = y
        x += entry.width + padding
        shelf_height = max(shelf_height, entry.height)

    height = min(max_size, y + shelf_height + padding)
    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    for entry in ordered:
        h, w = entry.image.shape[:2]
        atlas[entry.y : entry.y + h, entry.x : entry.x + w] = entry.image
    return atlas, ordered


def write_atlas_text(
    path: Path,
    image_name: str,
    entries: list[AtlasEntry],
    filter_mode: str = "Linear,Linear",
) -> Path:
    """Write a Spine-style ``.atlas`` description next to the atlas image."""

    lines: list[str] = []
    lines.append(image_name)
    if entries:
        width = max(e.x + e.width for e in entries)
        height = max(e.y + e.height for e in entries)
        lines.append(f"size: {width}, {height}")
    lines.append("format: RGBA8888")
    lines.append(f"filter: {filter_mode}")
    lines.append("repeat: none")
    lines.append("")

    for entry in entries:
        lines.append(entry.name)
        lines.append("  rotate: false")
        lines.append(f"  xy: {entry.x}, {entry.y}")
        lines.append(f"  size: {entry.width}, {entry.height}")
        lines.append(f"  orig: {entry.width}, {entry.height}")
        lines.append("  offset: 0, 0")
        lines.append("  index: -1")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_atlas(
    layers: list[tuple[str, Path]],
    out_dir: Path,
    image_name: str = "atlas.png",
) -> tuple[Path, Path, list[AtlasEntry]]:
    """Build an atlas from layer images. Returns ``(png, atlas_text, entries)``."""

    entries = [
        AtlasEntry(name=name, image=load_rgba(path)) for name, path in layers
    ]
    atlas, placed = pack_atlas(entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_rgba(out_dir / image_name, atlas)
    text_path = write_atlas_text(out_dir / "atlas.atlas", image_name, placed)
    return png_path, text_path, placed


__all__ = ["AtlasEntry", "build_atlas", "pack_atlas", "write_atlas_text"]
