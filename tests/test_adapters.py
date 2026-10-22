"""The adapter contract, the registry and the fallback chain."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from rigger2d.adapters.base import (
    AdapterContext,
    AdapterResult,
    MattePayload,
    PartsPayload,
    failed_probe,
    ok_probe,
)
from rigger2d.adapters.manifest import load_manifest
from rigger2d.adapters.registry import Registry, plugin_candidates, probe_entry
from rigger2d.adapters.selection import compare_all, run_chain
from rigger2d.contracts.models import ProbeStatus, Task
from rigger2d.io import images


class _Working:
    """A backend that always works."""

    id = "test.working"
    version = "1"
    capabilities = frozenset({Task.MATTE})

    def probe(self, ctx: AdapterContext):
        return ok_probe(self.id, self.version, self.capabilities)

    def run(self, task, payload, ctx) -> AdapterResult:
        mask = np.ones(payload.image_path and images.load_rgba(payload.image_path).shape[:2], bool)
        path = payload.output_dir / "matte.png"
        images.save_mask(path, mask)
        return AdapterResult(task=Task.MATTE, mask_paths={"matte": path})


class _Broken:
    """A backend that always fails."""

    id = "test.broken"
    version = "1"
    capabilities = frozenset({Task.MATTE})

    def probe(self, ctx: AdapterContext):
        return ok_probe(self.id, self.version, self.capabilities)

    def run(self, task, payload, ctx) -> AdapterResult:
        raise RuntimeError("the model exploded")


class _Missing:
    """A backend that is not installed."""

    id = "test.missing"
    version = "1"
    capabilities = frozenset({Task.MATTE})

    def probe(self, ctx: AdapterContext):
        return failed_probe(
            self.id,
            self.version,
            ProbeStatus.MISSING_WEIGHTS,
            "the weights are not downloaded",
            self.capabilities,
        )

    def run(self, task, payload, ctx) -> AdapterResult:  # pragma: no cover
        raise AssertionError("a missing backend must not be run")


def _entry(adapter, priority: int):
    from rigger2d.adapters.manifest import PluginManifest
    from rigger2d.adapters.registry import PluginEntry

    manifest = PluginManifest(
        id=adapter.id,
        version="1",
        tasks=[Task.MATTE],
        license={"spdx": "Apache-2.0"},
        entry_point="test:adapter",
        priority=priority,
    )
    return PluginEntry(manifest, adapter, "test", None, None)


def test_a_broken_backend_falls_through_to_the_next(tmp_path: Path, character_image: Path) -> None:
    from rigger2d.adapters.registry import Registry

    registry = Registry(entries=[_entry(_Broken(), 1), _entry(_Working(), 2)])
    payload = MattePayload(image_path=character_image, output_dir=tmp_path)
    selection = run_chain(registry, Task.MATTE, payload, AdapterContext(tmp_path, tmp_path))

    assert selection.adapter_id == "test.working"
    assert selection.attempts[0].adapter_id == "test.broken"
    assert selection.attempts[0].ok is False
    assert "exploded" in (selection.attempts[0].error or "")


def test_a_missing_backend_is_not_run(tmp_path: Path, character_image: Path) -> None:
    from rigger2d.adapters.registry import Registry

    registry = Registry(entries=[_entry(_Missing(), 1), _entry(_Working(), 2)])
    payload = MattePayload(image_path=character_image, output_dir=tmp_path)
    selection = run_chain(registry, Task.MATTE, payload, AdapterContext(tmp_path, tmp_path))

    assert selection.adapter_id == "test.working"
    assert selection.attempts[0].probe_status == "missing_weights"
    assert selection.attempts[0].ok is False


def test_comparing_reports_every_backend(tmp_path: Path, character_image: Path) -> None:
    from rigger2d.adapters.registry import Registry

    registry = Registry(entries=[_entry(_Working(), 1), _entry(_Broken(), 2)])
    payload = MattePayload(image_path=character_image, output_dir=tmp_path)
    attempts = compare_all(registry, Task.MATTE, payload, AdapterContext(tmp_path, tmp_path))

    assert len(attempts) == 2
    working = next(a for a in attempts if a.adapter_id == "test.working")
    broken = next(a for a in attempts if a.adapter_id == "test.broken")
    assert working.ok
    assert working.score is not None
    assert not broken.ok


def test_the_core_fallback_is_always_there() -> None:
    registry = Registry.discover()
    assert any(entry.id == "core.fallback" for entry in registry.entries)


def test_a_plugin_directory_is_discovered(tmp_path: Path, monkeypatch) -> None:
    source = Path(__file__).resolve().parent.parent / "examples" / "mock_adapter"
    target = tmp_path / "plugins" / "mock_adapter"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    monkeypatch.setenv("RIGGER2D_PLUGINS", str(target))
    registry = Registry.discover()
    ids = {entry.id for entry in registry.entries}
    assert "example.mock" in ids


def test_a_broken_plugin_does_not_break_the_registry(tmp_path: Path, monkeypatch) -> None:
    plugin = tmp_path / "plugins" / "broken"
    plugin.mkdir(parents=True)
    (plugin / "manifest.json").write_text(
        json.dumps(
            {
                "id": "broken.plugin",
                "version": "1",
                "tasks": ["matte"],
                "license": {"spdx": "Apache-2.0"},
                "entry_point": "nothing:here",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RIGGER2D_PLUGINS", str(tmp_path / "plugins"))

    registry = Registry.discover()
    broken = registry.by_id("broken.plugin")
    assert broken is not None
    assert broken.adapter is None
    assert broken.error


def test_a_plugin_may_be_a_directory_of_plugins(tmp_path: Path) -> None:
    container = tmp_path / "plugins"
    (container / "one").mkdir(parents=True)
    (container / "two").mkdir(parents=True)
    candidates = plugin_candidates(container)
    assert {path.name for path in candidates} == {"one", "two"}

    (container / "one" / "manifest.json").write_text("{}", encoding="utf-8")
    assert plugin_candidates(container / "one") == [container / "one"]


def test_the_manifest_is_validated(tmp_path: Path) -> None:
    manifest = load_manifest(
        Path(__file__).resolve().parent.parent / "examples" / "mock_adapter" / "manifest.json"
    )
    assert manifest.id == "example.mock"
    assert Task.PARTS in manifest.tasks


def test_a_plugin_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    manifest = {
        "id": "nope",
        "version": "1",
        "tasks": ["matte"],
        "license": {"spdx": "Apache-2.0"},
        "entry_point": "nope:adapter",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded.entry_point == "nope:adapter"


def test_every_shipped_adapter_loads(monkeypatch, tmp_path: Path) -> None:
    """The five adapters in adapters/ have to import and answer honestly.

    Their dependencies are deliberately not installed, so they cannot be run
    here -- but they must still load, and their probe must say what is missing
    instead of pretending to be ready. A syntax error in a generated adapter
    used to slip through because nothing ever imported one.
    """

    adapters = Path(__file__).resolve().parent.parent / "adapters"
    directories = sorted(path for path in adapters.iterdir() if path.is_dir())
    assert len(directories) == 5, f"expected five adapters, found {len(directories)}"

    monkeypatch.setenv("RIGGER2D_PLUGINS", os.pathsep.join(str(d) for d in directories))
    registry = Registry.discover()
    ctx = AdapterContext(project_dir=tmp_path, work_dir=tmp_path)

    expected = {
        "birefnet.matte",
        "humanparsing.lip",
        "lama.inpaint",
        "rtmpose.pose",
        "sam3.promptable",
    }
    loaded = {entry.id for entry in registry.usable()}
    assert expected <= loaded, f"not loaded: {sorted(expected - loaded)}"

    for entry in registry.usable():
        if entry.id not in expected:
            continue
        probe = probe_entry(entry, ctx)
        # No machine learning packages here, so none of them can be ready.
        assert probe.status == ProbeStatus.MISSING_DEPS, (entry.id, probe.status)
        assert probe.reason, f"{entry.id} gave no reason"
        assert "requirement" in probe.reason.lower(), probe.reason
        assert probe.capabilities, f"{entry.id} claims no capabilities"


def test_every_shipped_adapter_declares_its_terms() -> None:
    """An adapter has to say where its weights come from and under what terms."""

    adapters = Path(__file__).resolve().parent.parent / "adapters"
    for directory in sorted(path for path in adapters.iterdir() if path.is_dir()):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["license"]["spdx"], directory.name
        assert manifest["weights"]["source"], directory.name
        note = (directory / "THIRD_PARTY.md").read_text(encoding="utf-8")
        assert manifest["weights"]["source"] in note, directory.name
        if manifest["weights"].get("gated"):
            assert manifest["weights"]["terms_url"], directory.name
