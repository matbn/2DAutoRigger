"""The gates that keep the core honest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_the_licence_gate_passes() -> None:
    result = _run("license_check.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_schema_gate_passes() -> None:
    result = _run("schema_check.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_adapter_manifest_matches_the_schema() -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "schemas" / "plugin-manifest.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    manifests = list(ROOT.glob("examples/**/manifest.json"))
    manifests += list((ROOT / "adapters").glob("*/manifest.json"))
    assert len(manifests) >= 6, "the example and the five adapters should all be there"

    for manifest in manifests:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(document))
        assert not errors, f"{manifest}: {[error.message for error in errors]}"


def test_the_humanoid_template_matches_the_schema() -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "schemas" / "template.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    template = json.loads(
        (ROOT / "python" / "rigger2d" / "templates" / "humanoid_biped_v1.json").read_text(
            encoding="utf-8"
        )
    )
    errors = list(validator.iter_errors(template))
    assert not errors, [error.message for error in errors]


def test_the_core_does_not_import_a_ml_framework() -> None:
    forbidden = ("import torch", "import tensorflow", "import onnxruntime", "import transformers")
    for path in (ROOT / "python" / "rigger2d").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in forbidden:
            assert line not in text, f"{path} imports {line}"


def test_the_core_has_no_weights() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "node_modules" in path.parts or ".git" in path.parts:
            continue
        assert path.suffix.lower() not in {".safetensors", ".onnx", ".pth", ".ckpt"}, path
