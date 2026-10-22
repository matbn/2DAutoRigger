"""The canonical export: the rig document and its provenance sidecar."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.models import Rig


def write_rig_json(rig: Rig, path: Path) -> Path:
    """Write ``rig.json`` -- the canonical, engine-independent rig."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rig.model_dump(mode="json"), fh, indent=2)
    return path


def write_provenance(rig: Rig, path: Path) -> Path:
    """Write the sidecar recording which backend produced what, under which terms."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_id": rig.project_id,
        "template_id": rig.template_id,
        "skinning": rig.skinning.value,
        "backends": [entry.model_dump(mode="json") for entry in rig.provenance],
        "notice": (
            "The core of 2DAutoRigger is Apache-2.0. The backends listed above are "
            "separate packages with their own licenses, which the user installs and "
            "accepts. No model weights are redistributed here."
        ),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def emit(rig: Rig, out_dir: Path, layers_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    written.append(str(write_rig_json(rig, out_dir / "rig.json").name))
    written.append(str(write_provenance(rig, out_dir / "provenance.json").name))
    return written


__all__ = ["emit", "write_provenance", "write_rig_json"]
