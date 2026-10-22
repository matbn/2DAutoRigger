"""BiRefNet (high resolution matte).

BiRefNet is a high resolution dichotomous image segmentation model. It is
excellent at cutting a character out of its background, which is the first
thing the pipeline has to get right.

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
    MattePayload,
    failed_probe,
    ok_probe,
)
from rigger2d.contracts.models import Probe, ProbeStatus, Task
from rigger2d.io import images

ADAPTER_ID = "birefnet.matte"
VERSION = "0.1.0"

MODEL_ID = "ZhengPeng7/BiRefNet"


class BirefnetAdapter:
    """BiRefNet (high resolution matte)."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({Task.MATTE})

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
        weights = self._weights or Path(os.environ.get("RIGGER2D_BIREFNET_WEIGHTS", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                'point the adapter at the weights with RIGGER2D_BIREFNET_WEIGHTS or by passing a path to the constructor. The code is MIT. The weights are published by their authors under their own terms; read the model card before using them.',
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx='MIT',
            license_url='https://github.com/ZhengPeng7/BiRefNet',
            device='cuda',
            extra={"weights": str(weights)},
        )

    # -- inference -------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task not in self.capabilities:
            raise ValueError(f"{self.id} does not implement {task}")
        if task == Task.matte:
            return self._matte(payload, ctx)
        raise ValueError(f"{self.id} does not implement {task}")

    def _matte(self, payload: MattePayload, ctx: AdapterContext) -> AdapterResult:
        """Cut the character out of the background."""

        import numpy as np
        import torch
        from transformers import AutoModelForImageSegmentation

        if self._model is None:
            self._model = AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            self._model.eval()

        rgba = images.load_rgba(payload.image_path)
        tensor = torch.from_numpy(rgba[..., :3].copy()).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self._model(tensor)
        prediction = _sigmoid(outputs)
        mask = np.array(prediction[0, 0] > 0.5, dtype=bool)
        mask = images.largest_component(mask)

        path = payload.output_dir / "matte.png"
        images.save_mask(path, mask)
        return AdapterResult(
            task=Task.MATTE,
            mask_paths={"matte": path},
            detail={"source": "birefnet", "coverage": float(mask.mean())},
        )


def _missing_packages() -> list[str]:
    """The requirements that are not installed."""

    import importlib.util

    missing = []
    for package in ['torch>=2.2', 'transformers', 'timm', 'safetensors']:
        module = package.split(">")[0].split("=")[0].split("[")[0].strip()
        if module in ("sam3", "torch", "transformers", "timm", "safetensors", "onnxruntime"):
            if importlib.util.find_spec(module) is None:
                missing.append(package)
    return missing


adapter = BirefnetAdapter()

__all__ = ["BirefnetAdapter", "adapter"]


def _sigmoid(outputs: Any) -> Any:
    """Bring the model output to a 0..1 mask."""

    import numpy as np

    array = outputs
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] > 1:
        array = array[:, :1]
    return 1.0 / (1.0 + np.exp(-array))
