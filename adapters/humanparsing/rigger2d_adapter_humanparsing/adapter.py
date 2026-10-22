"""Human parsing (LIP / ATR labels).

Human parsing models label every pixel of a person with a body part
class. They are the strongest option for humanoids and the weakest option for
anything else, which is exactly the split the pipeline expects.

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
from rigger2d.contracts.models import Probe, ProbeStatus, Task
from rigger2d.io import images

ADAPTER_ID = "humanparsing.lip"
VERSION = "0.1.0"

LABELS = {
    "head": [
        11,
        12,
        13
    ],
    "torso": [
        5,
        6,
        7,
        8,
        10,
        15
    ],
    "pelvis": [
        9
    ],
    "upper_arm_l": [
        1
    ],
    "lower_arm_l": [
        2
    ],
    "hand_l": [
        3
    ],
    "upper_arm_r": [
        4
    ],
    "lower_arm_r": [
        14
    ],
    "hand_r": [
        16
    ],
    "upper_leg_l": [
        17
    ],
    "lower_leg_l": [
        18
    ],
    "foot_l": [
        19
    ],
    "upper_leg_r": [
        20
    ],
    "lower_leg_r": [
        21
    ],
    "foot_r": [
        22
    ]
}


class HumanparsingAdapter:
    """Human parsing (LIP / ATR labels)."""

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
        weights = self._weights or Path(os.environ.get("RIGGER2D_HUMANPARSING_WEIGHTS", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                'point the adapter at the weights with RIGGER2D_HUMANPARSING_WEIGHTS or by passing a path to the constructor. Parsing models are trained on human photographs and datasets such as LIP and ATR. Check the dataset licenses before commercial use.',
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx='MIT',
            license_url='https://github.com/GoGoBunny/HRNet-Semantic-Segmentation',
            device='cpu',
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
        """Parse the character into LIP style body part labels."""

        import onnxruntime

        if self._model is None:
            self._model = onnxruntime.InferenceSession(
                str(self._weights), providers=["CPUExecutionProvider"]
            )

        rgba = images.load_rgba(payload.image_path)
        size = (rgba.shape[1], rgba.shape[0])
        tensor = _preprocess(rgba[..., :3], size)
        input_name = self._model.get_inputs()[0].name
        logits = self._model.run(None, {input_name: tensor})[0]
        labels = np.array(logits[0].argmax(axis=0), dtype=np.int32)

        masks: dict[str, Path] = {}
        for part_id, label_ids in LABELS.items():
            if payload.part_ids and part_id not in payload.part_ids:
                continue
            mask = np.isin(labels, label_ids)
            if not mask.any():
                continue
            path = payload.output_dir / f"{part_id}.png"
            images.save_mask(path, mask)
            masks[part_id] = path

        return AdapterResult(task=Task.PARTS, mask_paths=masks, detail={"source": "humanparsing"})


def _missing_packages() -> list[str]:
    """The requirements that are not installed."""

    import importlib.util

    missing = []
    for package in ['onnxruntime', 'numpy', 'Pillow']:
        module = package.split(">")[0].split("=")[0].split("[")[0].strip()
        if module in ("sam3", "torch", "transformers", "timm", "safetensors", "onnxruntime"):
            if importlib.util.find_spec(module) is None:
                missing.append(package)
    return missing


adapter = HumanparsingAdapter()

__all__ = ["HumanparsingAdapter", "adapter"]


def _preprocess(image: Any, size: tuple[int, int]) -> Any:
    """Resize and normalise an image for the parsing model."""

    import numpy as np
    from PIL import Image

    resized = Image.fromarray(image.astype("uint8")).resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.485) / 0.229
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)
