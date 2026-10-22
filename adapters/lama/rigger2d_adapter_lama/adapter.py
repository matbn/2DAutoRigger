"""LaMa (inpainting, fills what a limb hides).

LaMa is an inpainting model. The pipeline uses it to fill in the area a limb
hides, so that rotating an arm does not reveal a hole in the torso.

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
    InpaintPayload,
    failed_probe,
    ok_probe,
)
from rigger2d.contracts.models import Probe, ProbeStatus, Task
from rigger2d.io import images

ADAPTER_ID = "lama.inpaint"
VERSION = "0.1.0"


class LamaAdapter:
    """LaMa (inpainting, fills what a limb hides)."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({Task.INPAINT})

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
        weights = self._weights or Path(os.environ.get("RIGGER2D_LAMA_WEIGHTS", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                'point the adapter at the weights with RIGGER2D_LAMA_WEIGHTS or by passing a path to the constructor. Apache-2.0. The model weights are published by the LaMa authors; check the repository for their terms.',
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx='Apache-2.0',
            license_url='https://github.com/advimman/lama',
            device='cpu',
            extra={"weights": str(weights)},
        )

    # -- inference -------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task not in self.capabilities:
            raise ValueError(f"{self.id} does not implement {task}")
        if task == Task.inpaint:
            return self._inpaint(payload, ctx)
        raise ValueError(f"{self.id} does not implement {task}")

    def _inpaint(self, payload: InpaintPayload, ctx: AdapterContext) -> AdapterResult:
        """Fill in the area behind a limb."""

        import onnxruntime

        if self._model is None:
            self._model = onnxruntime.InferenceSession(
                str(self._weights), providers=["CPUExecutionProvider"]
            )

        rgba = images.load_rgba(payload.image_path)
        mask = images.load_mask(payload.mask_path)
        tensor = _preprocess(rgba[..., :3], (rgba.shape[1], rgba.shape[0]))
        mask_tensor = _preprocess_mask(mask, (rgba.shape[1], rgba.shape[0]))
        input_names = [i.name for i in self._model.get_inputs()]
        feeds = {input_names[0]: tensor}
        if len(input_names) > 1:
            feeds[input_names[1]] = mask_tensor
        output = self._model.run(None, feeds)[0]
        filled = _postprocess(output, rgba)

        path = payload.output_dir / "inpaint.png"
        images.save_rgba(path, filled)
        return AdapterResult(task=Task.INPAINT, mask_paths={"inpaint": path}, detail={"source": "lama"})


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


adapter = LamaAdapter()

__all__ = ["LamaAdapter", "adapter"]


def _preprocess(image: Any, size: tuple[int, int]) -> Any:
    """Resize and normalise an image for the inpainting model."""

    import numpy as np
    from PIL import Image

    resized = Image.fromarray(image.astype("uint8")).resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _preprocess_mask(mask: Any, size: tuple[int, int]) -> Any:
    """Resize the mask for the inpainting model."""

    import numpy as np
    from PIL import Image

    resized = Image.fromarray((mask * 255).astype("uint8")).resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[None, None, ...].astype(np.float32)


def _postprocess(output: Any, rgba: Any) -> Any:
    """Put the inpainted image back into the original frame."""

    import numpy as np

    array = np.asarray(output, dtype=np.float32)
    while array.ndim > 3:
        array = array[0]
    if array.shape[0] == 1:
        array = np.transpose(array, (1, 2, 0))
    if array.max() > 1.5:
        array = array / 255.0
    height, width = rgba.shape[:2]
    from PIL import Image

    resized = Image.fromarray((np.clip(array, 0, 1) * 255).astype("uint8")).resize(
        (width, height), Image.BILINEAR
    )
    filled = rgba.copy()
    filled[..., :3] = np.asarray(resized, dtype=np.uint8)
    return filled
