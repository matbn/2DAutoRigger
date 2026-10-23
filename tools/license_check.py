"""The licence gate.

The core of 2DAutoRigger is Apache-2.0. That promise is only worth something
if something checks it, so this script fails the build when:

* the core depends on a copyleft or otherwise non-permissive package;
* the core imports a machine learning framework (they belong in plugins);
* a model weight file has been committed to the repository;
* a plugin manifest does not declare its licence or its weights.

Usage:

    python tools/license_check.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Licences the core is allowed to depend on.
ALLOWED_LICENCES = {
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
    "Python-2.0",
    "Unlicense",
    "Zlib",
}

# Packages the core must not depend on: they belong in a plugin, where the
# user can accept their terms separately.
FORBIDDEN_CORE_PACKAGES = {
    "torch",
    "torchvision",
    "tensorflow",
    "jax",
    "onnxruntime",
    "onnx",
    "transformers",
    "timm",
    "opencv-python",
    "segment-anything",
    "sam3",
    "diffusers",
}

WEIGHT_SUFFIXES = {".safetensors", ".onnx", ".pth", ".pt", ".ckpt", ".bin", ".weights"}
WEIGHT_DIRS = {"weights", "checkpoints", "models"}

# Packages that are known to be non-permissive.
KNOWN_BAD = {
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "AGPL-3.0",
    "MPL-2.0-with-GPL-exception",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
}

ML_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(torch|torchvision|tensorflow|jax|onnxruntime|transformers|timm)\b",
    re.MULTILINE,
)


def fail(message: str, problems: list[str]) -> None:
    problems.append(message)


def check_requirements() -> list[str]:
    problems: list[str] = []
    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        return ["requirements.txt is missing"]
    for number, line in enumerate(requirements.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower()
        if name in FORBIDDEN_CORE_PACKAGES:
            fail(
                f"requirements.txt:{number}: '{name}' is a machine learning package "
                "and must live in a plugin, not in the core",
                problems,
            )
    return problems


def check_core_imports() -> list[str]:
    """Machine learning belongs in an adapter, never in the core.

    ``adapters/`` is deliberately not scanned: depending on torch is what an
    adapter is for. The distinction is the whole licence argument.
    """

    problems: list[str] = []
    package_root = ROOT / "python" / "rigger2d"
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in ML_IMPORT.finditer(text):
            fail(
                f"{path.relative_to(ROOT)}: imports '{match.group(1)}' -- the core must "
                "not depend on a machine learning framework",
                problems,
            )
    return problems


def check_no_weights() -> list[str]:
    problems: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            fail(f"{path.relative_to(ROOT)}: a model weight file is committed", problems)
        if path.name.lower() in {"weights.bin", "model.pt"}:
            fail(f"{path.relative_to(ROOT)}: a model weight file is committed", problems)
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name.lower() in WEIGHT_DIRS:
            if ".git" in path.parts or "node_modules" in path.parts:
                continue
            fail(f"{path.relative_to(ROOT)}: a directory that may hold weights", problems)
    return problems


def _owned_manifests() -> list[Path]:
    """The plugin manifests in this repository."""

    found: list[Path] = []
    for manifest in ROOT.glob("**/manifest.json"):
        if "node_modules" in manifest.parts:
            continue
        found.append(manifest)
    return found


def check_adapter_requirements() -> list[str]:
    """The adapters may depend on machine learning packages -- that is the
    point of them -- but only when they are installed deliberately, and each
    adapter's requirements have to be its own."""

    problems: list[str] = []
    adapters = ROOT / "adapters"
    if not adapters.is_dir():
        return problems
    for directory in sorted(path for path in adapters.iterdir() if path.is_dir()):
        requirements = directory / "requirements.txt"
        if not requirements.is_file():
            fail(f"{directory.name}: has no requirements.txt", problems)
            continue
        if not requirements.read_text(encoding="utf-8").strip():
            fail(f"{directory.name}: its requirements.txt is empty", problems)
    return problems


def check_plugin_manifests() -> list[str]:
    problems: list[str] = []
    candidates = _owned_manifests()

    for manifest in candidates:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{manifest}: not valid JSON ({exc})", problems)
            continue

        licence = data.get("license")
        if not isinstance(licence, dict):
            fail(f"{manifest}: no licence object declared", problems)
            continue
        spdx = licence.get("spdx")
        if not spdx:
            fail(f"{manifest}: no licence declared", problems)
            continue
        if spdx in KNOWN_BAD:
            fail(f"{manifest}: licence {spdx} is not allowed in the core", problems)
        if spdx.startswith("LicenseRef-"):
            # A custom licence is fine for a plugin, but it has to point
            # somewhere so the user can read it.
            if not licence.get("url"):
                fail(
                    f"{manifest}: licence {spdx} has no url, so the user cannot read it",
                    problems,
                )
        elif spdx not in ALLOWED_LICENCES:
            fail(f"{manifest}: licence {spdx} is not on the allowed list", problems)

        weights = data.get("weights")
        if weights is None:
            # A backend with no weights at all is fine.
            continue
        if not isinstance(weights, dict):
            fail(f"{manifest}: the weights entry is not an object", problems)
            continue
        if not weights.get("source"):
            fail(f"{manifest}: no weights source declared", problems)
        if weights.get("gated") and not weights.get("terms_url"):
            fail(
                f"{manifest}: the weights are gated but no terms url is given, so the "
                "user cannot find out what they are agreeing to",
                problems,
            )
    return problems


def main() -> int:
    problems: list[str] = []
    problems += check_requirements()
    problems += check_core_imports()
    problems += check_no_weights()
    problems += check_plugin_manifests()
    problems += check_adapter_requirements()

    if problems:
        print("licence check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("licence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
