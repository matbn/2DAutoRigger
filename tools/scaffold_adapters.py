"""Generate the five adapter scaffolds, under ``adapters/``.

They live in this repository rather than beside it. The argument for splitting
them was licence separation, and it does not hold: the adapter *code* is
Apache-2.0, exactly like the core. The licences that differ belong to the model
weights, which are never shipped -- they live in each adapter's
``THIRD_PARTY.md`` and in its manifest. One repository is simpler to clone,
install and keep in step.

    python tools/scaffold_adapters.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE_LICENSE = ROOT / "LICENSE"
ADAPTERS_ROOT = ROOT / "adapters"

ADAPTERS = [
    {
        "slug": "sam3",
        "package": "rigger2d_adapter_sam3",
        "id": "sam3.promptable",
        "name": "SAM 3 (promptable segmentation)",
        "tasks": ["parts"],
        "spdx": "LicenseRef-Meta-SAM",
        "license_url": "https://github.com/facebookresearch/sam3",
        "weights_source": "https://huggingface.co/facebook/sam3",
        "gated": True,
        "terms_url": "https://huggingface.co/facebook/sam3/blob/main/LICENSE",
        "note": (
            "The weights are gated. Request access on Hugging Face and accept the "
            "Meta SAM License before downloading. This adapter never downloads "
            "them for you."
        ),
        "python": ">=3.12",
        "packages": ["torch>=2.7", "safetensors", "sam3"],
        "device": "cuda",
        "priority": 10,
        "concepts": {
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
            "foot_r": "right foot",
        },
    },
    {
        "slug": "birefnet",
        "package": "rigger2d_adapter_birefnet",
        "id": "birefnet.matte",
        "name": "BiRefNet (high resolution matte)",
        "tasks": ["matte"],
        "spdx": "MIT",
        "license_url": "https://github.com/ZhengPeng7/BiRefNet",
        "weights_source": "https://huggingface.co/ZhengPeng7/BiRefNet",
        "gated": False,
        "terms_url": "https://huggingface.co/ZhengPeng7/BiRefNet",
        "note": (
            "The code is MIT. The weights are published by their authors under "
            "their own terms; read the model card before using them."
        ),
        "python": ">=3.11",
        "packages": ["torch>=2.2", "transformers", "timm", "safetensors"],
        "device": "cuda",
        "priority": 10,
        "model_id": "ZhengPeng7/BiRefNet",
    },
    {
        "slug": "humanparsing",
        "package": "rigger2d_adapter_humanparsing",
        "id": "humanparsing.lip",
        "name": "Human parsing (LIP / ATR labels)",
        "tasks": ["parts"],
        "spdx": "MIT",
        "license_url": "https://github.com/GoGoBunny/HRNet-Semantic-Segmentation",
        "weights_source": "https://github.com/GoGoBunny/HRNet-Semantic-Segmentation",
        "gated": False,
        "terms_url": "https://github.com/GoGoBunny/HRNet-Semantic-Segmentation",
        "note": (
            "Parsing models are trained on human photographs and datasets such as "
            "LIP and ATR. Check the dataset licenses before commercial use."
        ),
        "python": ">=3.11",
        "packages": ["onnxruntime", "numpy", "Pillow"],
        "device": "cpu",
        "priority": 20,
        "labels": {
            "head": [11, 12, 13],
            "torso": [5, 6, 7, 8, 10, 15],
            "pelvis": [9],
            "upper_arm_l": [1],
            "lower_arm_l": [2],
            "hand_l": [3],
            "upper_arm_r": [4],
            "lower_arm_r": [14],
            "hand_r": [16],
            "upper_leg_l": [17],
            "lower_leg_l": [18],
            "foot_l": [19],
            "upper_leg_r": [20],
            "lower_leg_r": [21],
            "foot_r": [22],
        },
    },
    {
        "slug": "rtmpose",
        "package": "rigger2d_adapter_rtmpose",
        "id": "rtmpose.pose",
        "name": "RTMPose (pose estimation)",
        "tasks": ["pose"],
        "spdx": "Apache-2.0",
        "license_url": "https://github.com/open-mmlab/mmpose",
        "weights_source": "https://github.com/open-mmlab/mmpose",
        "gated": False,
        "terms_url": "https://github.com/open-mmlab/mmpose",
        "note": (
            "The code is Apache-2.0. The ONNX model you point this adapter at has "
            "its own terms; use a model you are allowed to use."
        ),
        "python": ">=3.11",
        "packages": ["onnxruntime", "numpy", "Pillow"],
        "device": "cpu",
        "priority": 10,
    },
    {
        "slug": "lama",
        "package": "rigger2d_adapter_lama",
        "id": "lama.inpaint",
        "name": "LaMa (inpainting, fills what a limb hides)",
        "tasks": ["inpaint"],
        "spdx": "Apache-2.0",
        "license_url": "https://github.com/advimman/lama",
        "weights_source": "https://github.com/advimman/lama",
        "gated": False,
        "terms_url": "https://github.com/advimman/lama",
        "note": (
            "Apache-2.0. The model weights are published by the LaMa authors; "
            "check the repository for their terms."
        ),
        "python": ">=3.11",
        "packages": ["onnxruntime", "numpy", "Pillow"],
        "device": "cpu",
        "priority": 20,
    },
]

ADAPTER_TEMPLATE = '''"""{name}.

{blurb}

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
    {payload_imports}
    failed_probe,
    ok_probe,
)
from rigger2d.contracts.models import Probe, ProbeStatus, Task{extra_imports}
from rigger2d.io import images

ADAPTER_ID = "{id}"
VERSION = "0.1.0"
{constants}

class {cls}:
    """{name}."""

    id = ADAPTER_ID
    version = VERSION
    capabilities = frozenset({{{capabilities}}})

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
        weights = self._weights or Path(os.environ.get("{env_var}", ""))
        if not weights or not weights.exists():
            return failed_probe(
                self.id,
                self.version,
                ProbeStatus.MISSING_WEIGHTS,
                {weights_message!r},
                self.capabilities,
            )
        return ok_probe(
            self.id,
            self.version,
            self.capabilities,
            license_spdx={spdx!r},
            license_url={license_url!r},
            device={device!r},
            extra={{"weights": str(weights)}},
        )

    # -- inference -------------------------------------------------------

    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult:
        if task not in self.capabilities:
            raise ValueError(f"{{self.id}} does not implement {{task}}")
        if task == Task.{entry_task}:
            return self._{entry_task}(payload, ctx)
        raise ValueError(f"{{self.id}} does not implement {{task}}")

{body}

def _missing_packages() -> list[str]:
    """The requirements that are not installed."""

    import importlib.util

    missing = []
    for package in {packages!r}:
        module = package.split(">")[0].split("=")[0].split("[")[0].strip()
        if module in ("sam3", "torch", "transformers", "timm", "safetensors", "onnxruntime"):
            if importlib.util.find_spec(module) is None:
                missing.append(package)
    return missing


adapter = {cls}()

__all__ = ["{cls}", "adapter"]
'''

INIT_TEMPLATE = '''"""The plugin package.

The manifest names this package as its entry point, so the adapter object has
to be importable as an attribute of it. ``adapter`` below is that object, which
is the whole contract: the core imports it and calls ``probe`` and ``run``.
"""

from __future__ import annotations

from .adapter import adapter

__all__ = ["adapter"]
'''

MANIFEST_TEMPLATE = """{{
  "id": "{id}",
  "version": "0.1.0",
  "name": "{name}",
  "description": "{description}",
  "tasks": {tasks},
  "part_vocabulary": {vocabulary},
  "license": {{
    "spdx": "{spdx}",
    "url": "{license_url}",
    "requires_acceptance": {requires_acceptance}
  }},
  "weights": {{
    "source": "{weights_source}",
    "gated": {gated},
    "terms_url": "{terms_url}",
    "note": "{note}"
  }},
  "requires": {{
    "python": "{python}",
    "packages": {packages}
  }},
  "entry_point": "{package}:adapter",
  "priority": {priority}
}}
"""

THIRD_PARTY_TEMPLATE = """# Third party material

This package is **adapter code**, not model weights. No weights are committed
to this repository and none are downloaded for you.

| What | Where | Terms |
| --- | --- | --- |
| {name} | [{weights_source}]({weights_source}) | {spdx} -- see {terms_url} |
| The adapter code in this repository | this repository | Apache-2.0 |

{note}

## What you have to do

1. Install the requirements: `pip install -r requirements.txt`.
2. Obtain the weights yourself, from the source above.
3. Read the terms linked above and make sure they are acceptable to you.
4. Point the adapter at the weights, either with the environment variable
   `{env_var}` or by constructing the adapter with an explicit path.

The core of 2DAutoRigger never sees, downloads or redistributes these weights.
It only asks the adapter whether it is ready.
"""

README_TEMPLATE = """# {name}

A {tasks_text} backend for [2DAutoRigger](https://github.com/matbn/2DAutoRigger).

{blurb}

This adapter lives in the core repository, at `adapters/{slug}/`, because its
own code is Apache-2.0 like the core. What differs is the licence of the
**model**, and no model is shipped either way.

## Licence of the model

This adapter's own code is Apache-2.0. The model it uses is **{spdx}** and is
**not** distributed here -- see [THIRD_PARTY.md](THIRD_PARTY.md).

{note}

## Install

```bash
pip install -r requirements.txt
```

Then make the adapter visible to the core, either by installing it:

```bash
pip install -e .
```

or by pointing the plugin path at this directory:

```bash
# Linux/macOS
export RIGGER2D_PLUGINS="$PWD"
# Windows
$env:RIGGER2D_PLUGINS = "$PWD"
```

or by adding it to your plugin folder:

```bash
# Linux/macOS
ln -s "$PWD" ~/.rigger2d/plugins/{slug}
# Windows
New-Item -ItemType Junction -Path "$HOME\\.rigger2d\\plugins\\{slug}" -Target "$PWD"
```

```bash
export RIGGER2D_PLUGINS="/path/to/this/directory"
```

## Weights

```bash
export {env_var}=/path/to/the/weights
```

`autorigger2d models` will then report `{id}` as available, and
`autorigger2d doctor` will tell you if anything is still missing.

## Note on verification

This adapter follows the documented API of its upstream project, but it was
written without the upstream project's dependencies installed. Run it against a
character whose parts you already know and check the result before you trust it.
The probe reports missing dependencies and missing weights honestly, so a
half-installed adapter never pretends to work.
"""

PYPROJECT_TEMPLATE = """[project]
name = "{package_dash}"
version = "0.1.0"
description = "{name} backend for 2DAutoRigger."
readme = "README.md"
requires-python = "{python}"
license = {{ text = "Apache-2.0" }}
authors = [{{ name = "2DAutoRigger contributors" }}]

dependencies = [
{dependencies}
]

[project.entry-points."rigger2d.adapters"]
"{id}" = "{package}:adapter"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{package}"]
"""

REQUIREMENTS_TEMPLATE = """{packages}
"""

BODIES = {
    "sam3": '''    def _parts(self, payload: PartsPayload, ctx: AdapterContext) -> AdapterResult:
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
''',
    "birefnet": '''    def _matte(self, payload: MattePayload, ctx: AdapterContext) -> AdapterResult:
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
''',
    "humanparsing": '''    def _parts(self, payload: PartsPayload, ctx: AdapterContext) -> AdapterResult:
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
''',
    "rtmpose": '''    def _pose(self, payload: PosePayload, ctx: AdapterContext) -> AdapterResult:
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
''',
    "lama": '''    def _inpaint(self, payload: InpaintPayload, ctx: AdapterContext) -> AdapterResult:
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
''',
}

HELPERS = {
    "sam3": '''

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
''',
    "birefnet": '''

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
''',
    "humanparsing": '''

def _preprocess(image: Any, size: tuple[int, int]) -> Any:
    """Resize and normalise an image for the parsing model."""

    import numpy as np
    from PIL import Image

    resized = Image.fromarray(image.astype("uint8")).resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.485) / 0.229
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)
''',
    "rtmpose": '''

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
''',
    "lama": '''

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
''',
}

BLURBS = {
    "sam3": "SAM 3 is Meta's promptable segmentation model: you ask it for a concept\nsuch as \"left arm\" and it tells you where that is in the picture.",
    "birefnet": "BiRefNet is a high resolution dichotomous image segmentation model. It is\nexcellent at cutting a character out of its background, which is the first\nthing the pipeline has to get right.",
    "humanparsing": "Human parsing models label every pixel of a person with a body part\nclass. They are the strongest option for humanoids and the weakest option for\nanything else, which is exactly the split the pipeline expects.",
    "rtmpose": "RTMPose is a fast pose estimator. It runs on the CPU through onnxruntime,\nwhich makes it the practical choice on a machine without a GPU.",
    "lama": "LaMa is an inpainting model. The pipeline uses it to fill in the area a limb\nhides, so that rotating an arm does not reveal a hole in the torso.",
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe(text: str) -> str:
    """Make a string safe to drop into a JSON string literal."""

    return text.replace("\\", "\\\\").replace('"', "'").replace("\n", " ")


def main() -> int:
    for spec in ADAPTERS:
        slug = spec["slug"]
        target = ADAPTERS_ROOT / slug
        if target.exists():
            print(f"skip {target} (exists)")
            continue

        entry_task = spec["tasks"][0]
        capabilities = ", ".join(f"Task.{task.upper()}" for task in spec["tasks"])

        payload_imports = {
            "parts": "PartsPayload,",
            "matte": "MattePayload,",
            "pose": "PosePayload,",
            "inpaint": "InpaintPayload,",
        }[entry_task]
        extra_imports = ""
        constants = ""
        if slug == "sam3":
            extra_imports = ", Vec2"
            constants = "\nCONCEPTS = " + json.dumps(spec["concepts"], indent=4) + "\n"
        elif slug == "birefnet":
            constants = f'\nMODEL_ID = "{spec["model_id"]}"\n'
        elif slug == "humanparsing":
            constants = "\nLABELS = " + json.dumps(spec["labels"], indent=4) + "\n"
        elif slug == "rtmpose":
            extra_imports = ", Vec2"

        body = BODIES[slug]
        helpers = HELPERS[slug]

        module = ADAPTER_TEMPLATE.format(
            name=spec["name"],
            id=spec["id"],
            blurb=BLURBS[slug].replace("\n", "\n"),
            payload_imports=payload_imports,
            extra_imports=extra_imports,
            cls="".join(part.capitalize() for part in slug.split("_")) + "Adapter",
            capabilities=capabilities,
            env_var=f"RIGGER2D_{slug.upper()}_WEIGHTS",
            weights_message=(
                f"point the adapter at the weights with RIGGER2D_{slug.upper()}_WEIGHTS "
                f"or by passing a path to the constructor. {spec['note']}"
            ),
            spdx=spec["spdx"],
            license_url=spec["license_url"],
            device=spec["device"],
            entry_task=entry_task,
            body=body,
            packages=spec["packages"],
            constants=constants,
        )
        module += helpers

        manifest = MANIFEST_TEMPLATE.format(
            id=spec["id"],
            name=spec["name"],
            description=safe(BLURBS[slug].replace("\n", " ").strip()),
            tasks=json.dumps(spec["tasks"]),
            vocabulary=json.dumps(sorted(spec.get("concepts", {}).keys())),
            spdx=spec["spdx"],
            license_url=spec["license_url"],
            requires_acceptance="true" if spec["gated"] else "false",
            weights_source=spec["weights_source"],
            gated=str(spec["gated"]).lower(),
            terms_url=spec["terms_url"],
            note=safe(spec["note"]),
            python=spec["python"],
            packages=json.dumps(spec["packages"]),
            package=spec["package"],
            priority=spec["priority"],
        )

        # A generator has to check what it generated. Nothing imports an
        # adapter during a normal test run -- its dependencies are not
        # installed -- so a syntax error or a broken manifest would sit here
        # unnoticed until someone tried to use the adapter.
        compile(module, f"adapters/{slug}/adapter.py", "exec")
        compile(INIT_TEMPLATE, f"adapters/{slug}/__init__.py", "exec")
        json.loads(manifest)

        write(target / f"{spec['package']}" / "__init__.py", INIT_TEMPLATE)
        write(target / f"{spec['package']}" / "adapter.py", module)
        write(target / "manifest.json", manifest)
        write(
            target / "LICENSE",
            CORE_LICENSE.read_text(encoding="utf-8")
            if CORE_LICENSE.is_file()
            else "Apache-2.0\n",
        )
        write(
            target / "THIRD_PARTY.md",
            THIRD_PARTY_TEMPLATE.format(
                name=spec["name"],
                weights_source=spec["weights_source"],
                spdx=spec["spdx"],
                terms_url=spec["terms_url"],
                note=spec["note"],
                env_var=f"RIGGER2D_{slug.upper()}_WEIGHTS",
            ),
        )
        write(
            target / "README.md",
            README_TEMPLATE.format(
                name=spec["name"],
                tasks_text=", ".join(spec["tasks"]),
                blurb=BLURBS[slug],
                spdx=spec["spdx"],
                note=spec["note"],
                slug=slug,
                id=spec["id"],
                env_var=f"RIGGER2D_{slug.upper()}_WEIGHTS",
            ),
        )
        write(
            target / "pyproject.toml",
            PYPROJECT_TEMPLATE.format(
                package=spec["package"],
                package_dash=spec["package"].replace("_", "-"),
                name=spec["name"],
                python=spec["python"],
                dependencies="\n".join(f'  "{p}",' for p in spec["packages"]),
                id=spec["id"],
            ),
        )
        write(target / "requirements.txt", REQUIREMENTS_TEMPLATE.format(packages="\n".join(spec["packages"])))
        print(f"created {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
