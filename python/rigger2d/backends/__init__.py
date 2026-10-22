"""Backends that ship with the core.

The only one is the geometric fallback. Real backends live in their own
repositories so that their licenses stay separate from the Apache-2.0 core.
"""

from __future__ import annotations

from .fallback import CoreFallbackAdapter

__all__ = ["CoreFallbackAdapter"]
