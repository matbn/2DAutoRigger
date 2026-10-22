"""The HTTP service that the JavaScript side talks to."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
