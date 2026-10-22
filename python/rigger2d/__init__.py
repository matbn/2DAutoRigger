"""2DAutoRigger -- automatic 2D character segmentation, slicing and rigging.

The core package is deliberately free of machine-learning dependencies: every
AI backend is an optional plugin installed by the user, who is then responsible
for its license. See ``rigger2d.adapters`` for the contract those plugins
implement.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
