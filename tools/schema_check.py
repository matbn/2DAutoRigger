"""The schema gate.

A manifest, a template or an emitted rig is a contract. This script checks
that they still match the JSON Schema that describes them, so that a field
the code reads but the schema forbids -- or a type the code never handles --
fails the build instead of failing in someone's game.

Usage:

    python tools/schema_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schemas"

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    print("jsonschema is not installed: pip install jsonschema", file=sys.stderr)
    sys.exit(2)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validator_for(name: str) -> Draft202012Validator:
    schema = load(SCHEMA_DIR / name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def check_documents(
    validator: Draft202012Validator,
    documents: list[tuple[Path, Any]],
    problems: list[str],
) -> None:
    for path, document in documents:
        for error in validator.iter_errors(document):
            location = "/".join(str(part) for part in error.absolute_path) or "<root>"
            problems.append(f"{path}: {location}: {error.message}")


def _owned_manifests() -> list[Path]:
    """The manifests that belong to this project.

    Only the core repository and the adapter repositories next to it. Other
    projects may live in the same parent directory and are none of our
    business.
    """

    found: list[Path] = []
    for manifest in ROOT.glob("**/manifest.json"):
        if "node_modules" in manifest.parts:
            continue
        found.append(manifest)
    return found


def main() -> int:
    problems: list[str] = []

    plugin_schema = validator_for("plugin-manifest.schema.json")
    template_schema = validator_for("template.schema.json")
    rig_schema = validator_for("rig.schema.json")

    # -- plugin manifests ------------------------------------------------
    manifests: list[tuple[Path, Any]] = []
    for manifest in _owned_manifests():
        try:
            manifests.append((manifest, load(manifest)))
        except json.JSONDecodeError as exc:
            problems.append(f"{manifest}: not valid JSON ({exc})")
    check_documents(plugin_schema, manifests, problems)

    # -- templates -------------------------------------------------------
    templates: list[tuple[Path, Any]] = []
    template_dirs = [
        ROOT / "python" / "rigger2d" / "templates",
        ROOT / "templates",
        Path.home() / ".rigger2d" / "templates",
    ]
    for directory in template_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".schema.json"):
                continue
            try:
                templates.append((path, load(path)))
            except json.JSONDecodeError as exc:
                problems.append(f"{path}: not valid JSON ({exc})")
    check_documents(template_schema, templates, problems)

    # -- emitted rigs ----------------------------------------------------
    rigs: list[tuple[Path, Any]] = []
    for path in sorted(ROOT.glob("**/exports/rig.json")):
        if "node_modules" in path.parts:
            continue
        try:
            rigs.append((path, load(path)))
        except json.JSONDecodeError as exc:
            problems.append(f"{path}: not valid JSON ({exc})")
    check_documents(rig_schema, rigs, problems)

    if problems:
        print("schema check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        f"schema check passed "
        f"({len(manifests)} manifests, {len(templates)} templates, {len(rigs)} rigs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
