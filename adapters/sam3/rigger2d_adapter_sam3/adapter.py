"""SAM 3 (promptable segmentation).

SAM 3 is Meta's promptable segmentation model: you ask it for a concept
such as "left arm" and it tells you where that is in the picture.

**This adapter has not been executed in the environment it was written in.** It
follows the documented API of its upstream project, but you should run it
against a character you know and check the result before trusting it. The
``probe`` method is honest: it reports exactly what is missing rather than
pretending to work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rigger2d.adapters.base import (
    AdapterContext,
    AdapterResult,
    PartsPayload,
    failed_probe,
    ok_probe,
)
from rigger2d.contracts.models import Probe, ProbeStatus, Task, Vec2
from rigger2d.io import images

ADAPTER_ID = "sam3.promptable"
VERSION = "0.1.0"

CONCEPTS = {
    "head": "head",
    "torso": "torso",
    "pelvis": "hips",
    "upper_arm_l": "left arm",
    "lower_arm_l": "left arm",
    "hand_l": "left hand",
    "upper_arm_r": "right arm",
    "lower_arm_r": "right arm",
    "hand_r": "right hand",
    "upper_leg_l": "left leg",
    "lower_leg_l": "left leg",
    "foot_l": "left foot",
    "upper_leg_r": "right leg",
    "lower_leg_r": "right leg",
    "foot_r": "right foot"
}


class Sam3Adapter:
    """SAM 3 (promptable segmentation)."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({Task.PARTS})

    def __init__(self, weights: str | os.PathLike[str] | None = None) -> None:
        self._weights = Path(weights) if weights else None
        self._model = None

    # -- probing ---------------------------------------------------------

    def probe(self, ctx: AdapterContext) -> Probe:
        missing = _missing_packages()
        if missing:
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_DEPS,
                "install the requirements first: " + ", ".join(missing),
                self.capabilities,
            )
        weights = self._weights or Path(os.environ.get("RIGGER2D_SAM3_WEIGHTS", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                'point the adapter at the weights with RIGGER2D_SAM3_WEIGHTS or by passing a path to the constructor. The weights are gated. Request access on Hugging Face and accept the Meta SAM License before downloading. This adapter never downloads them for you.',
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx='LicenseRef-Meta-SAM',
            license_url='https://github.com/facebookresearch/sam3',
            device='cuda',
            extra={"weights": str(weights)},
        )

    # -- inference -------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task not in self.capabilities:
            raise ValueError(f"{self.id} does not implement {task}")
        if task == Task.parts:
            return self._parts(payload, ctx)
        raise ValueError(f"{self.id} does not implement {task}")

    def _parts(self, payload: PartsPayload, ctx: AdapterContext) -> AdapterResult:
        """Segment the character by asking SAM 3 for concepts."""

        model, processor = self._load()
        rgba = images.load_rgba(payload.image_path)

        masks: dict[str, Path] = {}
        for part_id, concept in CONCEPTS.items():
            if payload.part_ids and part_id not in payload.part_ids:
                continue
            result = processor.set_image(rgba[..., :3], image_size=(1024, 1024))
            masks_out = processor.set_text_prompt(state=result, prompt=concept)
            if not masks_out:
                continue
            mask = _largest_mask(masks_out)
            if mask is None:
                continue
            path = payload.output_dir / f"{part_id}.png"
            images.save_mask(path, mask)
            masks[part_id] = path

        return AdapterResult(task=Task.PARTS, mask_paths=masks, detail={"source": "sam3"})

    def _load(self) -> tuple[Any, Any]:
        """Build the model once and keep it."""

        if self._model is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model()
            self._model = (model, Sam3Processor(model))
        return self._model


def _missing_packages() -> list[str]:
    """The requirements that are not installed."""

    import importlib.util

    missing = []
    for package in ['torch>=2.7', 'safetensors', 'sam3']:
        module = package.split(">")[0].split("=")[0].split("[")[0].strip()
        if module in ("sam3", "torch", "transformers", "timm", "safetensors", "onnxruntime"):
            if importlib.util.find_spec(module) is None:
                missing.append(package)
    return missing


adapter = Sam3Adapter()

__all__ = ["Sam3Adapter", "adapter"]


def _largest_mask(masks: Any) -> Any:
    """Pick the biggest of the masks SAM 3 returned for one concept."""

    import numpy as np

    best = None
    best_area = 0
    for candidate in masks:
        array = np.array(candidate)
        area = int((array > 0).sum())
        if area > best_area:
            best_area = area
            best = array > 0
    return best
