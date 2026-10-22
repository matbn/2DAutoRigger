import { describe, expect, it } from "vitest";
import type { Bone, Rig } from "@rigger2d/schema";
import {
  apply,
  boneChain,
  identity,
  invert,
  localRestTransforms,
  multiply,
  restTransforms,
  rotation,
  translation,
} from "../src/index";

function bone(id: string, parent: string | null, head: Bone["head"], tail: Bone["tail"]): Bone {
  return {
    id,
    parent,
    head,
    tail,
    length: Math.hypot(tail.x - head.x, tail.y - head.y),
    part_id: null,
    layer: null,
    rest_rotation: 0,
  };
}

const rig: Rig = {
  schema_version: "1.0",
  project_id: "test",
  template_id: "humanoid.biped.v1",
  template_type: "humanoid",
  width: 100,
  height: 200,
  skinning: "hybrid",
  landmarks: {},
  bones: [
    bone("root", null, { x: 50, y: 100 }, { x: 50, y: 100 }),
    bone("spine", "root", { x: 50, y: 100 }, { x: 50, y: 50 }),
    bone("arm", "spine", { x: 50, y: 60 }, { x: 80, y: 60 }),
  ],
  layers: [],
  meshes: [],
  provenance: [],
  created_at: "",
};

describe("the rig maths", () => {
  it("multiplies and inverts", () => {
    const matrix = multiply(translation(10, 20), rotation(Math.PI / 2));
    const point = apply(matrix, { x: 1, y: 0 });
    expect(point.x).toBeCloseTo(10);
    expect(point.y).toBeCloseTo(21);

    const back = apply(invert(matrix), point);
    expect(back.x).toBeCloseTo(1);
    expect(back.y).toBeCloseTo(0);
  });

  it("leaves a point alone under the identity", () => {
    expect(apply(identity(), { x: 3, y: 4 })).toEqual({ x: 3, y: 4 });
  });

  it("puts the rest of a bone where the bone is", () => {
    const rest = restTransforms(rig);
    const start = apply(rest["spine"]!, { x: 0, y: 0 });
    expect(start.x).toBeCloseTo(50);
    expect(start.y).toBeCloseTo(100);
  });

  it("points the bone along its own direction", () => {
    const rest = restTransforms(rig);
    const tip = apply(rest["spine"]!, { x: 50, y: 0 });
    expect(tip.x).toBeCloseTo(50);
    expect(tip.y).toBeCloseTo(50);
  });

  it("orders the bones so that a parent comes first", () => {
    const chain = boneChain(rig).map((candidate) => candidate.id);
    expect(chain.indexOf("root")).toBeLessThan(chain.indexOf("spine"));
    expect(chain.indexOf("spine")).toBeLessThan(chain.indexOf("arm"));
  });

  it("keeps a child bone relative to its parent", () => {
    const local = localRestTransforms(rig);
    // The arm sits 40 units along the spine, which points up the image.
    const armStart = apply(local["arm"]!, { x: 0, y: 0 });
    expect(armStart.x).toBeCloseTo(40);
    expect(armStart.y).toBeCloseTo(0);
  });

  it("puts a root bone in the skeleton's space", () => {
    const local = localRestTransforms(rig);
    const rootStart = apply(local["root"]!, { x: 0, y: 0 });
    expect(rootStart.x).toBeCloseTo(50);
    expect(rootStart.y).toBeCloseTo(100);
  });
});
