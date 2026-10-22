/**
 * Rig maths and a canvas preview.
 *
 * The editor draws the skeleton and the weights so that the user can see what
 * the pipeline believes before they accept it. The maths here mirrors what
 * the Godot emitter writes, so what the preview shows is what the game will
 * get.
 */

import type { Bone, Mesh, Rig, Vec2 } from "@rigger2d/schema";

export interface Matrix2D {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

export function identity(): Matrix2D {
  return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
}

export function multiply(m: Matrix2D, n: Matrix2D): Matrix2D {
  return {
    a: m.a * n.a + m.c * n.b,
    b: m.b * n.a + m.d * n.b,
    c: m.a * n.c + m.c * n.d,
    d: m.b * n.c + m.d * n.d,
    e: m.a * n.e + m.c * n.f + m.e,
    f: m.b * n.e + m.d * n.f + m.f,
  };
}

export function translation(x: number, y: number): Matrix2D {
  return { a: 1, b: 0, c: 0, d: 1, e: x, f: y };
}

export function rotation(angle: number): Matrix2D {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return { a: cos, b: sin, c: -sin, d: cos, e: 0, f: 0 };
}

export function apply(m: Matrix2D, p: Vec2): Vec2 {
  return { x: m.a * p.x + m.c * p.y + m.e, y: m.b * p.x + m.d * p.y + m.f };
}

export function invert(m: Matrix2D): Matrix2D {
  const determinant = m.a * m.d - m.b * m.c;
  if (Math.abs(determinant) < 1e-12) {
    return identity();
  }
  const inverse = 1 / determinant;
  return {
    a: m.d * inverse,
    b: -m.b * inverse,
    c: -m.c * inverse,
    d: m.a * inverse,
    e: (m.c * m.f - m.d * m.e) * inverse,
    f: (m.b * m.e - m.a * m.f) * inverse,
  };
}

/**
 * The rest transform of a bone in skeleton space.
 *
 * A bone's rest maps its own space -- the origin at the joint, the x axis
 * along the bone -- into the skeleton. That is a rotation to the bone's
 * direction and a translation to its head, and it does not depend on the
 * parent: the parent only matters for the *local* rest, which is what a
 * scene file stores.
 */
export function boneWorld(bone: Bone): Matrix2D {
  const angle = Math.atan2(bone.tail.y - bone.head.y, bone.tail.x - bone.head.x);
  // The translation comes first: a point is rotated about the joint, not
  // about the origin.
  return multiply(translation(bone.head.x, bone.head.y), rotation(angle));
}

/** The rest transform of every bone in skeleton space, keyed by bone id. */
export function restTransforms(rig: Rig): Record<string, Matrix2D> {
  const out: Record<string, Matrix2D> = {};
  for (const bone of rig.bones) {
    out[bone.id] = boneWorld(bone);
  }
  return out;
}

/**
 * The rest transform of every bone relative to its parent, keyed by bone id.
 *
 * This is what a scene file stores: a child bone is placed inside its
 * parent, not in the skeleton's space.
 */
export function localRestTransforms(rig: Rig): Record<string, Matrix2D> {
  const byId = new Map(rig.bones.map((bone) => [bone.id, bone]));
  const world = restTransforms(rig);
  const out: Record<string, Matrix2D> = {};
  for (const bone of rig.bones) {
    const parent = bone.parent ? byId.get(bone.parent) : undefined;
    if (!parent) {
      out[bone.id] = world[bone.id]!;
      continue;
    }
    out[bone.id] = multiply(invert(world[parent.id]!), world[bone.id]!);
  }
  return out;
}

/** Where a mesh vertex ends up once the bones are posed. */
export function skinVertex(
  mesh: Mesh,
  vertexIndex: number,
  rig: Rig,
  poses: Record<string, Matrix2D>,
  rest: Record<string, Matrix2D>,
): Vec2 {
  const vertex = mesh.vertices[vertexIndex];
  if (!vertex) return { x: 0, y: 0 };
  const indices = mesh.bones[vertexIndex] ?? [];
  const weights = mesh.weights[vertexIndex] ?? [];

  let x = 0;
  let y = 0;
  let total = 0;
  for (let slot = 0; slot < indices.length; slot += 1) {
    const weight = weights[slot] ?? 0;
    if (weight <= 0) continue;
    const bone = rig.bones[indices[slot] ?? -1];
    if (!bone) continue;
    const restMatrix = rest[bone.id];
    if (!restMatrix) continue;
    const poseMatrix = poses[bone.id] ?? restMatrix;
    const local = apply(invert(restMatrix), vertex);
    const moved = apply(poseMatrix, local);
    x += moved.x * weight;
    y += moved.y * weight;
    total += weight;
  }
  if (total <= 0) {
    return vertex;
  }
  return { x: x / total, y: y / total };
}

/** The bones, in the order the pipeline produced them, as a flat list. */
export function boneChain(rig: Rig): Bone[] {
  const byId = new Map(rig.bones.map((bone) => [bone.id, bone]));
  const children = new Map<string | null, Bone[]>();
  for (const bone of rig.bones) {
    const parent = bone.parent && byId.has(bone.parent) ? bone.parent : null;
    const list = children.get(parent) ?? [];
    list.push(bone);
    children.set(parent, list);
  }
  const ordered: Bone[] = [];
  const walk = (bone: Bone): void => {
    ordered.push(bone);
    for (const child of children.get(bone.id) ?? []) {
      walk(child);
    }
  };
  for (const root of children.get(null) ?? []) {
    walk(root);
  }
  return ordered;
}

export interface DrawOptions {
  /** Extra rotation, in radians, applied to a bone id. */
  poses?: Record<string, number>;
  showWeights?: boolean;
  boneColor?: string;
  jointColor?: string;
}

/** Draw the skeleton onto a 2D canvas context. */
export function drawSkeleton(
  context: CanvasRenderingContext2D,
  rig: Rig,
  options: DrawOptions = {},
): void {
  const rest = restTransforms(rig);
  const poses: Record<string, Matrix2D> = {};

  for (const [id, matrix] of Object.entries(rest)) {
    const extra = options.poses?.[id];
    if (!extra) {
      poses[id] = matrix;
      continue;
    }
    const bone = rig.bones.find((candidate) => candidate.id === id);
    const head = bone?.head ?? { x: 0, y: 0 };
    // The bone is turned about its own head: the head stays where it is and
    // the rest of the bone swings around it.
    poses[id] = multiply(
      multiply(multiply(translation(head.x, head.y), rotation(extra)), translation(-head.x, -head.y)),
      matrix,
    );
  }

  context.save();
  context.lineWidth = Math.max(1, rig.width / 400);
  context.strokeStyle = options.boneColor ?? "#ffcc44";
  context.fillStyle = options.jointColor ?? "#ff5577";

  for (const bone of boneChain(rig)) {
    const matrix = poses[bone.id] ?? rest[bone.id];
    if (!matrix) continue;
    const start = apply(matrix, { x: 0, y: 0 });
    const end = apply(matrix, { x: bone.length, y: 0 });
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    context.beginPath();
    context.arc(start.x, start.y, context.lineWidth * 2.5, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

/** The bounding box of a set of points. */
export function boundsOf(points: Vec2[]): { x0: number; y0: number; x1: number; y1: number } {
  if (points.length === 0) {
    return { x0: 0, y0: 0, x1: 0, y1: 0 };
  }
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const point of points) {
    x0 = Math.min(x0, point.x);
    y0 = Math.min(y0, point.y);
    x1 = Math.max(x1, point.x);
    y1 = Math.max(y1, point.y);
  }
  return { x0, y0, x1, y1 };
}
