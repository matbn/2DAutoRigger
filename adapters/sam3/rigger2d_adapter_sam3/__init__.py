"""The plugin package.

The manifest names this package as its entry point, so the adapter object has
to be importable as an attribute of it. ``adapter`` below is that object, which
is the whole contract: the core imports it and calls ``probe`` and ``run``.
"""

from __future__ import annotations

from .adapter import adapter

__all__ = ["adapter"]
