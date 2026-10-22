"""Turning a mask into a mesh."""

from __future__ import annotations

import numpy as np

from rigger2d.contracts.models import BBox
from rigger2d.io import images
from rigger2d.rigging.mesh import (
    build_mesh,
    ear_clip,
    polygon_area,
    simplify_polygon,
    trace_contours,
)


def _capsule(height: int = 100, width: int = 20) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[10:90, 5:15] = True
    mask[5:95, 8:12] = True
    return mask


def test_the_outline_covers_the_mask_exactly() -> None:
    mask = _capsule()
    loops = trace_contours(mask)
    assert loops, "no outline was found"
    area = polygon_area(loops[0])
    assert abs(area - int(mask.sum())) < mask.sum() * 0.02


def test_a_thin_shape_still_has_an_outline() -> None:
    """A pixel walker closes a loop around the tip of an arm. Edge following does not."""

    mask = np.zeros((200, 6), dtype=bool)
    mask[0:200, 2:4] = True
    loops = trace_contours(mask)
    assert loops
    assert len(loops[0]) > 50
    assert polygon_area(loops[0]) > 300


def test_every_blob_gets_its_own_outline() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:10, 2:10] = True
    mask[20:30, 20:30] = True
    loops = trace_contours(mask)
    assert len(loops) == 2


def test_simplifying_keeps_the_shape() -> None:
    mask = _capsule()
    loops = trace_contours(mask)
    simplified = simplify_polygon(loops[0], epsilon=1.0)
    assert 3 <= len(simplified) <= len(loops[0])
    assert polygon_area(simplified) > 0


def test_a_mesh_has_vertices_and_triangles() -> None:
    mask = _capsule()
    bounds = images.mask_bounds(mask)
    mesh = build_mesh("arm", mask, bounds)
    assert mesh.vertices
    assert mesh.triangles
    assert len(mesh.triangles) % 3 == 0
    assert max(mesh.triangles) < len(mesh.vertices)


def test_the_mesh_uvs_stay_inside_the_texture() -> None:
    mask = _capsule()
    bounds = images.mask_bounds(mask)
    mesh = build_mesh("arm", mask, bounds)
    for uv in mesh.uvs:
        assert 0.0 <= uv.x <= 1.0
        assert 0.0 <= uv.y <= 1.0


def test_an_empty_mask_gives_an_empty_mesh() -> None:
    mesh = build_mesh("nothing", np.zeros((10, 10), dtype=bool), BBox(x0=0, y0=0, x1=0, y1=0))
    assert not mesh.vertices


def test_ear_clipping_covers_a_square() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    triangles = ear_clip(square)
    assert len(triangles) == 6
    area = 0.0
    for index in range(0, len(triangles), 3):
        a, b, c = (square[triangles[index]], square[triangles[index + 1]], square[triangles[index + 2]])
        area += abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    assert abs(area - 100) < 1e-6
