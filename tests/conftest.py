"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT / "python", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixtures import draw_character  # noqa: E402


@pytest.fixture(scope="session")
def character_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic character to rig."""

    return draw_character(tmp_path_factory.mktemp("art") / "hero.png")


@pytest.fixture()
def project(tmp_path: Path, character_image: Path):
    """A project with the pipeline already run up to the refine gate."""

    from rigger2d.pipeline.session import Session
    from rigger2d.pipeline.stages import REFINE_GATE, RunOptions, run_stages

    session = Session(tmp_path / "hero.rig")
    document = session.create(character_image)
    document = run_stages(session, document, RunOptions(), until=REFINE_GATE)
    return session, document
