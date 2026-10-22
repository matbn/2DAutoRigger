"""RTMPose (pose estimation).

RTMPose is a fast pose estimator. It runs on the CPU through onnxruntime,
which makes it the practical choice on a machine without a GPU.

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
    PosePayload,
    failed_probe,
    ok_probe,
)
from rigger2d.contracts.models import Probe, ProbeStatus, Task, Vec2
from rigger2d.io import images

ADAPTER_ID = "rtmpose.pose"
VERSION = "0.1.0"


class RtmposeAdapter:
    """RTMPose (pose estimation)."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({Task.POSE})

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
        weights = self._weights or Path(os.environ.get("RIGGER2D_RTMPOSE_WEIGHTS", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                'point the adapter at the weights with RIGGER2D_RTMPOSE_WEIGHTS or by passing a path to the constructor. The code is Apache-2.0. The ONNX model you point this adapter at has its own terms; use a model you are allowed to use.',
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx='Apache-2.0',
            license_url='https://github.com/open-mmlab/mmpose',
            device='cpu',
            extra={"weights": str(weights)},
        )

    # -- inference -------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task not in self.capabilities:
            raise ValueError(f"{self.id} does not implement {task}")
        if task == Task.pose:
            return self._pose(payload, ctx)
        raise ValueError(f"{self.id} does not implement {task}")

    def _pose(self, payload: PosePayload, ctx: AdapterContext) -> AdapterResult:
        """Estimate the joints."""

        import onnxruntime

        if self._model is None:
            self._model = onnxruntime.InferenceSession(
                str(self._weights), providers=["CPUExecutionProvider"]
            )

        rgba = images.load_rgba(payload.image_path)
        tensor = _preprocess(rgba[..., :3], (rgba.shape[1], rgba.shape[0]))
        input_name = self._model.get_inputs()[0].name
        output = self._model.run(None, {input_name: tensor})[0]
        keypoints, scores = _decode(output)

        landmarks: dict[str, Vec2] = {}
        confidence: dict[str, float] = {}
        for index, name in enumerate(KEYPOINT_NAMES):
            if index >= len(keypoints):
                break
            point = keypoints[index]
            landmarks[name] = Vec2(x=float(point[0]), y=float(point[1]))
            confidence[name] = float(scores[index]) if index < len(scores) else 0.0

        return AdapterResult(
            task=Task.POSE,
            landmarks=landmarks,
            confidence=confidence,
            detail={"source": "rtmpose"},
        )


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


adapter = RtmposeAdapter()

__all__ = ["RtmposeAdapter", "adapter"]


KEYPOINT_NAMES = [
    "head_top",
    "head_center",
    "neck",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
]


def _preprocess(image: Any, size: tuple[int, int]) -> Any:
    """Resize and normalise an image for the pose model."""

    import numpy as np
    from PIL import Image

    resized = Image.fromarray(image.astype("uint8")).resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.485) / 0.229
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _decode(output: Any) -> tuple[Any, Any]:
    """Turn the raw model output into keypoints and scores."""

    import numpy as np

    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    heatmap = array[0] if array.shape[0] < 50 else array
    height, width = heatmap.shape[-2:]
    flat = heatmap.reshape(heatmap.shape[0], -1)
    indices = flat.argmax(axis=1)
    scores = flat.max(axis=1)
    ys, xs = np.divmod(indices, width)
    keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
    return keypoints, scores
