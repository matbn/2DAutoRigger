"""Turning a part mask into a deformable mesh.

Engines bind **vertices** to bones, not pixels, so every part needs a mesh.
The pipeline is:

    mask -> contour -> simplify -> interior points -> triangulate

``scipy`` is used for Delaunay triangulation when it is installed; without it
the contour is triangulated by ear clipping, which still yields a valid mesh but
deforms less smoothly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..contracts.models import BBox, Mesh, Vec2


@dataclass
class Contour:
    points: list[tuple[float, float]]
    area: float


def trace_contours(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Every outline of a mask, as closed polygons.

    A mask can have more than one blob, and each blob has its own outline, so
    all of them are returned. The biggest one comes first.

    The outline is built from the boundary *edges* of the mask rather than by
    walking the boundary pixels. Following pixels is fragile on thin shapes --
    an arm is only a few pixels wide, and a pixel walker happily closes a loop
    around the tip of it. Edge following cannot: every set pixel contributes
    the edges that separate it from the background, and those edges link up
    into exactly one loop per blob.

    The points are the corners of the pixels, so the polygon covers the mask
    exactly.
    """

    height, width = mask.shape
    padded = np.pad(mask, 1)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(start: tuple[int, int], end: tuple[int, int]) -> None:
        edges.setdefault(start, []).append(end)

    for y, x in zip(ys, xs, strict=True):
        py, px = y + 1, x + 1
        if not padded[py - 1, px]:
            add((x, y), (x + 1, y))
        if not padded[py, px + 1]:
            add((x + 1, y), (x + 1, y + 1))
        if not padded[py + 1, px]:
            add((x + 1, y + 1), (x, y + 1))
        if not padded[py, px - 1]:
            add((x, y + 1), (x, y))

    loops: list[list[tuple[float, float]]] = []
    limit = 8 * int(mask.sum()) + 64

    while True:
        start = None
        for point, targets in edges.items():
            if targets:
                start = point
                break
        if start is None:
            break

        polygon: list[tuple[float, float]] = []
        current = start
        guard = 0
        while guard < limit:
            guard += 1
            targets = edges.get(current)
            if not targets:
                break
            nxt = targets.pop(0)
            if not targets:
                del edges[current]
            polygon.append((float(current[0]), float(current[1])))
            if nxt == start:
                break
            current = nxt

        if len(polygon) >= 3:
            loops.append(polygon)

    loops.sort(key=polygon_area, reverse=True)
    return loops


def trace_contour(mask: np.ndarray) -> list[tuple[float, float]]:
    """The biggest outline of a mask, as a closed polygon."""

    loops = trace_contours(mask)
    return loops[0] if loops else []


def simplify_polygon(
    points: list[tuple[float, float]],
    epsilon: float = 1.5,
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification of a closed polygon."""

    if len(points) < 4:
        return list(points)

    points = list(points)
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 4:
        return points

    def rdp(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(seq) < 3:
            return seq
        x1, y1 = seq[0]
        x2, y2 = seq[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        worst_index = 0
        worst = 0.0
        for index in range(1, len(seq) - 1):
            px, py = seq[index]
            if norm < 1e-9:
                distance = math.hypot(px - x1, py - y1)
            else:
                distance = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if distance > worst:
                worst = distance
                worst_index = index
        if worst <= epsilon:
            return [seq[0], seq[-1]]
        left = rdp(seq[: worst_index + 1])
        right = rdp(seq[worst_index:])
        return left[:-1] + right

    simplified = rdp(points)
    if simplified and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    return simplified


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray casting test."""

    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def ear_clip(polygon: list[tuple[float, float]]) -> list[int]:
    """Triangulate a simple polygon by ear clipping."""

    count = len(polygon)
    if count < 3:
        return []
    indices = list(range(count))
    if _signed_area(polygon) < 0:
        indices.reverse()

    triangles: list[int] = []
    guard = 0
    while len(indices) > 3 and guard < count * count + 64:
        guard += 1
        clipped = False
        for position in range(len(indices)):
            i_prev = indices[position - 1]
            i_curr = indices[position]
            i_next = indices[(position + 1) % len(indices)]
            a, b, c = polygon[i_prev], polygon[i_curr], polygon[i_next]
            if not _is_convex(a, b, c):
                continue
            if _has_point_inside(polygon, indices, a, b, c):
                continue
            triangles.extend([i_prev, i_curr, i_next])
            indices.pop(position)
            clipped = True
            break
        if not clipped:
            break
    if len(indices) == 3:
        triangles.extend(indices)
    return triangles


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _is_convex(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def _has_point_inside(
    polygon: list[tuple[float, float]],
    indices: list[int],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    for index in indices:
        point = polygon[index]
        if point in (a, b, c):
            continue
        if _point_in_triangle(point, a, b, c):
            return True
    return False


def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    has_negative = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_positive = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_negative and has_positive)


def _delaunay(points: list[tuple[float, float]]) -> list[int] | None:
    """Delaunay triangulation via scipy, when it is installed."""

    try:
        from scipy.spatial import Delaunay  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - scipy is optional
        return None
    if len(points) < 3:
        return None
    array = np.asarray(points, dtype=np.float64)
    try:
        triangulation = Delaunay(array)
    except Exception:  # noqa: BLE001 - degenerate input (collinear points)
        return None
    return [int(v) for v in triangulation.simplices.reshape(-1)]


def interior_points(
    polygon: list[tuple[float, float]],
    bounds: BBox,
    spacing: float,
) -> list[tuple[float, float]]:
    """Grid points strictly inside the polygon."""

    if len(polygon) < 3 or spacing <= 0:
        return []
    points: list[tuple[float, float]] = []
    y = bounds.y0 + spacing * 0.5
    while y < bounds.y1:
        x = bounds.x0 + spacing * 0.5
        while x < bounds.x1:
            if point_in_polygon(x, y, polygon):
                points.append((x, y))
            x += spacing
        y += spacing
    return points


def build_mesh(
    part_id: str,
    mask: np.ndarray,
    bounds: BBox,
    max_vertices: int = 1200,
) -> Mesh:
    """Build a deformable mesh for one part."""

    mesh = Mesh(part_id=part_id)
    if not mask.any():
        return mesh

    loops = trace_contours(mask)
    if not loops:
        return mesh

    epsilon = max(1.0, min(bounds.width, bounds.height) * 0.004)
    polygons: list[list[tuple[float, float]]] = []
    for loop in loops:
        simplified = simplify_polygon(loop, epsilon=epsilon)
        if len(simplified) >= 3:
            polygons.append(simplified)
    if not polygons:
        return mesh

    # Interior points let the mesh bend instead of just stretching.
    area = polygon_area(polygons[0])
    spacing = max(3.0, math.sqrt(max(area, 1.0) / 120.0))
    inner = interior_points(polygons[0], bounds, spacing)
    if len(polygons[0]) + len(inner) > max_vertices:
        inner = interior_points(polygons[0], bounds, spacing * 2.0)

    points: list[tuple[float, float]] = []
    triangles: list[int] = []

    for index, polygon in enumerate(polygons):
        if index == 0 and inner:
            candidate = list(polygon) + inner
            delaunay = _delaunay(candidate)
            if delaunay:
                base = len(points)
                points.extend(candidate)
                triangles.extend(base + t for t in delaunay)
                continue
        clipped = ear_clip(polygon)
        if not clipped:
            continue
        base = len(points)
        points.extend(polygon)
        triangles.extend(base + t for t in clipped)

    if not points or not triangles:
        return mesh

    width = max(bounds.width, 1e-6)
    height = max(bounds.height, 1e-6)

    mesh.vertices = [Vec2(x=p[0], y=p[1]) for p in points]
    mesh.uvs = [
        Vec2(
            x=float(np.clip((p[0] - bounds.x0) / width, 0.0, 1.0)),
            y=float(np.clip((p[1] - bounds.y0) / height, 0.0, 1.0)),
        )
        for p in points
    ]
    mesh.triangles = [int(t) for t in triangles]
    return mesh


__all__ = [
    "Contour",
    "build_mesh",
    "ear_clip",
    "interior_points",
    "point_in_polygon",
    "polygon_area",
    "simplify_polygon",
    "trace_contour",
]
