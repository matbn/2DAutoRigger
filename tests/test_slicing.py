"""The sliced character has to add back up.

This is the check that catches the mistakes a unit test on one stage cannot:
the layers are cut out of the picture, and putting them back together has to
give the picture again. If it does not, the slicing lost or invented pixels.
"""

from __future__ import annotations

import numpy as np

from rigger2d.io import images
from rigger2d.pipeline.session import Session
from rigger2d.pipeline.stages import RunOptions, run_stages


def _composite(session: Session, rig) -> np.ndarray:
    canvas = np.zeros_like(images.load_rgba(session.source_path))
    for layer in rig.layers:
        patch = images.load_rgba(session.layers_dir / layer.image)
        if not patch.size:
            continue
        images.paste_into(canvas, patch, int(layer.bounds.x0), int(layer.bounds.y0))
    return canvas


def test_the_layers_add_back_up_to_the_character(tmp_path, character_image) -> None:
    session = Session(tmp_path / "hero.rig")
    document = run_stages(session, session.create(character_image), RunOptions())
    assert document.rig is not None

    source = images.load_rgba(session.source_path)
    preview = _composite(session, document.rig)

    character = np.zeros(source.shape[:2], dtype=bool)
    for path in sorted(session.parts_dir.glob("*.png")):
        character |= images.load_mask(path)

    # The alpha has to match exactly.
    assert np.array_equal(preview[..., 3] > 127, character)

    # And so does the colour, everywhere the character is.
    both = character
    difference = np.abs(
        source[..., :3][both].astype(float) - preview[..., :3][both].astype(float)
    )
    assert difference.max() == 0, f"the layers changed the picture by {difference.max()}"


def test_the_preview_is_the_same_picture(tmp_path, character_image) -> None:
    session = Session(tmp_path / "hero.rig")
    document = run_stages(session, session.create(character_image), RunOptions())
    assert document.rig is not None

    preview = images.load_rgba(session.layers_dir / "preview.png")
    rebuilt = _composite(session, document.rig)
    assert np.array_equal(preview[..., 3], rebuilt[..., 3])


def test_the_fill_layers_are_hidden_at_rest(tmp_path, character_image) -> None:
    """A fill covers the hole a limb leaves, so at rest it is invisible."""

    session = Session(tmp_path / "hero.rig")
    document = run_stages(session, session.create(character_image), RunOptions())
    assert document.rig is not None

    source = images.load_rgba(session.source_path)

    parts_only = np.zeros_like(source)
    for layer in document.rig.layers:
        if layer.kind != "part":
            continue
        patch = images.load_rgba(session.layers_dir / layer.image)
        images.paste_into(parts_only, patch, int(layer.bounds.x0), int(layer.bounds.y0))

    with_fills = _composite(session, document.rig)
    assert np.array_equal(parts_only[..., 3], with_fills[..., 3])
