"""Entry point for the plugin loader.

The core finds this directory, reads ``manifest.json`` and imports the object
named by ``entry_point``. That is the whole contract.
"""

from __future__ import annotations

from adapter import adapter  # noqa: F401

__all__ = ["adapter"]
