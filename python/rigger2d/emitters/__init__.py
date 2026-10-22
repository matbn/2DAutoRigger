"""Exporters.

Each emitter turns the canonical rig into the files one engine expects. The
canonical ``rig.json`` is always written; the engine exports are additions, not
substitutes.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.models import Rig
from . import godot, native, spine, unity

EMITTERS = {
    "native": native.emit,
    "godot": godot.emit,
    "spine": spine.emit,
    "unity": unity.emit,
}


def emit_all(session: Any, rig: Rig, targets: list[str]) -> dict[str, list[str]]:
    """Run the requested emitters and report what each of them wrote."""

    out_dir = session.exports_dir
    layers_dir = session.layers_dir
    written: dict[str, list[str]] = {}
    for target in targets:
        emitter = EMITTERS.get(target)
        if emitter is None:
            raise ValueError(
                f"unknown export target {target!r}; known: {', '.join(sorted(EMITTERS))}"
            )
        written[target] = emitter(rig, out_dir, layers_dir)
    return written


__all__ = ["EMITTERS", "emit_all"]
