"""Finding and loading backend plugins.

Discovery order:

1. Python entry points in the ``rigger2d.adapters`` group (a pip-installed plugin)
2. ``$RIGGER2D_PLUGINS`` -- a list of directories
3. ``~/.rigger2d/plugins`` -- the per-user plugin directory
4. ``./plugins`` -- plugins that live with the project

A plugin that cannot be imported is reported, never fatal: the core must keep
working with whatever plugins happen to be installed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.models import ProbeStatus, Task
from .base import Adapter, failed_probe
from .manifest import PluginManifest, load_manifest, manifest_path

ENTRY_POINT_GROUP = "rigger2d.adapters"


@dataclass
class PluginEntry:
    """One discovered plugin, whether or not it could be loaded."""

    manifest: PluginManifest | None
    adapter: Adapter | None
    origin: str
    path: Path | None = None
    error: str | None = None

    @property
    def id(self) -> str:
        if self.manifest:
            return self.manifest.id
        if self.adapter is not None:
            return getattr(self.adapter, "id", "unknown")
        return "unknown"

    @property
    def priority(self) -> int:
        return self.manifest.priority if self.manifest else 100


def _user_plugin_dir() -> Path:
    return Path.home() / ".rigger2d" / "plugins"


def _project_plugin_dir() -> Path:
    return Path.cwd() / "plugins"


def _env_plugin_dirs() -> list[Path]:
    env = os.environ.get("RIGGER2D_PLUGINS")
    if not env:
        return []
    return [Path(p).expanduser() for p in env.split(os.pathsep) if p]


def plugin_dirs() -> list[Path]:
    """The directories that may hold plugins.

    A directory may be a plugin itself (it has a ``manifest.json``) or a
    folder of plugins. Both are accepted, because ``$RIGGER2D_PLUGINS`` is
    usually a list of plugin directories while ``~/.rigger2d/plugins`` is
    usually a folder of them.
    """

    dirs: list[Path] = []
    for path in (*_env_plugin_dirs(), _user_plugin_dir(), _project_plugin_dir()):
        if path.is_dir():
            dirs.append(path)
    return dirs


def plugin_candidates(directory: Path) -> list[Path]:
    """The plugin directories inside ``directory``."""

    if (directory / "manifest.json").is_file():
        return [directory]
    return [child for child in sorted(directory.iterdir()) if child.is_dir()]


def _import_object(dotted: str) -> Any:
    """Import ``package.module:attribute`` or ``package.module.attribute``."""

    if ":" in dotted:
        module_name, _, attr = dotted.partition(":")
    else:
        module_name, _, attr = dotted.rpartition(".")
    if not module_name:
        raise ImportError(f"cannot resolve {dotted!r}")
    module = importlib.import_module(module_name)
    try:
        obj = module
        for piece in attr.split("."):
            obj = getattr(obj, piece)
    except AttributeError as exc:
        raise ImportError(f"{dotted!r} does not exist: {exc}") from exc
    return obj


def _coerce_adapter(obj: Any) -> Adapter:
    if isinstance(obj, type):
        obj = obj()
    for attr in ("probe", "run"):
        if not hasattr(obj, attr):
            raise TypeError(f"plugin object {obj!r} is missing {attr}()")
    return obj  # type: ignore[return-value]


def _manifest_from_adapter(adapter: Adapter, entry_point: str) -> PluginManifest:
    """Build a manifest for a plugin that did not ship one."""

    capabilities = getattr(adapter, "capabilities", frozenset())
    return PluginManifest(
        id=getattr(adapter, "id", "unknown"),
        version=str(getattr(adapter, "version", "0")),
        tasks=list(capabilities) if capabilities else [Task.PARTS],
        license={"spdx": "LicenseRef-Unknown"},
        entry_point=entry_point,
    )


def _load_from_entry_point(entry_point: str) -> tuple[Adapter | None, str | None]:
    try:
        obj = _import_object(entry_point)
        return _coerce_adapter(obj), None
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill the core
        return None, f"{type(exc).__name__}: {exc}"


def _load_from_directory(plugin_dir: Path) -> PluginEntry:
    mpath = manifest_path(plugin_dir)
    if not mpath.is_file():
        return PluginEntry(None, None, str(plugin_dir), plugin_dir, "no manifest.json")
    try:
        manifest = load_manifest(mpath)
    except Exception as exc:  # noqa: BLE001
        return PluginEntry(None, None, str(plugin_dir), plugin_dir, f"bad manifest: {exc}")

    entry_point = manifest.entry_point
    added_to_path = False
    if str(plugin_dir) not in sys.path:
        sys.path.insert(0, str(plugin_dir))
        added_to_path = True
    try:
        adapter, error = _load_from_entry_point(entry_point)
    finally:
        if added_to_path:
            try:
                sys.path.remove(str(plugin_dir))
            except ValueError:
                pass
    if adapter is None:
        return PluginEntry(manifest, None, str(plugin_dir), plugin_dir, error)
    if not hasattr(adapter, "manifest"):
        try:
            adapter.manifest = manifest  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - some objects disallow attributes
            pass
    return PluginEntry(manifest, adapter, str(plugin_dir), plugin_dir, None)


def _load_entry_points() -> list[PluginEntry]:
    entries: list[PluginEntry] = []
    try:
        found = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 shape
        found = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in found:
        adapter, error = _load_from_entry_point(ep.value)
        manifest = getattr(adapter, "manifest", None) if adapter is not None else None
        if manifest is None and adapter is not None:
            manifest = _manifest_from_adapter(adapter, ep.value)
        entries.append(PluginEntry(manifest, adapter, f"entry_point:{ep.name}", None, error))
    return entries


@dataclass
class Registry:
    """The set of plugins visible to this process."""

    entries: list[PluginEntry] = field(default_factory=list)

    @classmethod
    def discover(cls, template: Any = None) -> Registry:
        entries: list[PluginEntry] = []
        entries.extend(_load_entry_points())
        seen_dirs: set[Path] = set()
        for directory in plugin_dirs():
            for child in plugin_candidates(directory):
                if child in seen_dirs:
                    continue
                seen_dirs.add(child)
                entries.append(_load_from_directory(child))
        entries.append(core_entry(template))
        return cls(entries=entries)

    def bind_template(self, template: Any) -> Registry:
        """Return a copy whose core fallback is bound to ``template``.

        The core fallback needs the template to estimate landmarks and to know
        which parts to cut, so it is rebuilt whenever the project changes.
        """

        entries = [
            core_entry(template) if e.origin == "core" else e for e in self.entries
        ]
        return Registry(entries=entries)

    def usable(self) -> list[PluginEntry]:
        return [e for e in self.entries if e.adapter is not None]

    def broken(self) -> list[PluginEntry]:
        return [e for e in self.entries if e.adapter is None]

    def by_id(self, adapter_id: str) -> PluginEntry | None:
        for entry in self.entries:
            if entry.id == adapter_id:
                return entry
        return None

    def for_task(self, task: Task) -> list[PluginEntry]:
        out = []
        for entry in self.usable():
            caps = getattr(entry.adapter, "capabilities", frozenset())
            if task in caps:
                out.append(entry)
        out.sort(key=lambda e: (e.priority, e.id))
        return out


def probe_entry(entry: PluginEntry, ctx: Any) -> Any:
    """Probe a plugin, converting a crash into a failed probe."""

    if entry.adapter is None:
        return failed_probe(
            entry.id,
            "0",
            ProbeStatus.ERROR,
            entry.error or "plugin could not be loaded",
        )
    try:
        return entry.adapter.probe(ctx)
    except Exception as exc:  # noqa: BLE001
        return failed_probe(entry.id, "0", ProbeStatus.ERROR, f"{type(exc).__name__}: {exc}")


def core_entry(template: Any = None) -> PluginEntry:
    """The built-in geometric backend, always present and always last."""

    from ..backends.fallback import CoreFallbackAdapter

    adapter = CoreFallbackAdapter(template)
    manifest = PluginManifest(
        id=adapter.id,
        version=adapter.version,
        name="Core geometric fallback",
        description="Dependency-free geometric baseline. Always available.",
        tasks=sorted(adapter.capabilities, key=lambda t: t.value),
        license={"spdx": "Apache-2.0"},
        entry_point="rigger2d.backends.fallback:CoreFallbackAdapter",
        priority=1000,
    )
    return PluginEntry(manifest, adapter, "core", None, None)


__all__ = [
    "ENTRY_POINT_GROUP",
    "PluginEntry",
    "Registry",
    "plugin_dirs",
    "probe_entry",
]
