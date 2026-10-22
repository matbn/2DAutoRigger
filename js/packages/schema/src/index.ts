/**
 * The types the Python core and the JavaScript side agree on.
 *
 * These mirror `schemas/rig.schema.json` in the core repository. The schema is
 * the single definition; this file is its TypeScript face, and
 * `tools/schema_check.py` keeps the two honest.
 */

export type Task = "matte" | "parts" | "pose" | "classify" | "inpaint";

export type SkinMode = "rigid" | "weighted";

export type SkinningMode = "cutout" | "hybrid" | "mesh";

export type ProbeStatus =
  | "available"
  | "missing_deps"
  | "missing_weights"
  | "license_unaccepted"
  | "disabled"
  | "error";

export interface Vec2 {
  x: number;
  y: number;
}

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Bone {
  id: string;
  parent: string | null;
  head: Vec2;
  tail: Vec2;
  length: number;
  part_id: string | null;
  layer: string | null;
  rest_rotation: number;
}

export interface Layer {
  id: string;
  part_id: string;
  image: string;
  z: number;
  pivot: Vec2;
  bounds: BBox;
  skin: SkinMode;
  kind: string;
}

export interface Mesh {
  part_id: string;
  vertices: Vec2[];
  uvs: Vec2[];
  triangles: number[];
  bones: number[][];
  weights: number[][];
}

export interface ProvenanceEntry {
  adapter_id: string;
  adapter_version: string;
  task: Task;
  ok: boolean;
  duration_ms: number;
  license_spdx: string | null;
  license_url: string | null;
  detail: string | null;
}

export interface Rig {
  schema_version: string;
  project_id: string;
  template_id: string;
  template_type: string;
  width: number;
  height: number;
  skinning: SkinningMode;
  landmarks: Record<string, Vec2>;
  bones: Bone[];
  layers: Layer[];
  meshes: Mesh[];
  provenance: ProvenanceEntry[];
  created_at: string;
}

export interface StageResult {
  name: string;
  version: number;
  duration_ms: number;
  adapter_id: string | null;
  detail: Record<string, unknown>;
}

export interface MaskInfo {
  part_id: string;
  image: string;
  bounds: BBox;
  area: number;
  pixels: number;
}

export interface MatteState {
  image: string;
  bounds: BBox;
  coverage: number;
}

export interface PartState {
  masks: MaskInfo[];
}

export interface MaskEdit {
  op: "add" | "erase" | "assign";
  part_id: string;
  points: Vec2[];
  radius: number;
}

export interface JointEdit {
  joint_id: string;
  dx: number;
  dy: number;
}

export interface Edit {
  mask?: MaskEdit | null;
  joint?: JointEdit | null;
}

export interface Document {
  schema_version: string;
  project_id: string;
  created_at: string;
  image: {
    path: string;
    width: number;
    height: number;
    has_alpha: boolean;
  };
  template_id: string;
  template_type: string;
  stages: Record<string, StageResult>;
  matte: MatteState | null;
  parts: PartState | null;
  landmarks: Record<string, Vec2>;
  landmark_confidence: Record<string, number>;
  edits: Edit[];
  rig: Rig | null;
  provenance: ProvenanceEntry[];
  warnings: string[];
}

export interface BackendInfo {
  id: string;
  origin: string;
  tasks: Task[];
  status: ProbeStatus;
  reason: string | null;
  license: string | null;
  license_url: string | null;
  device: string | null;
  error: string | null;
}

export interface Attempt {
  adapter_id: string;
  ok: boolean;
  score: number | null;
  error: string | null;
  duration_ms: number;
  probe_status: string;
  probe_reason: string | null;
}

export interface ExportFile {
  name: string;
  size: number;
  url: string;
}

export interface TemplateInfo {
  id: string;
  type: string;
  description: string;
  parts: string[];
}

export interface ProjectSummary {
  id: string;
  created_at: string;
  template_id: string;
  width: number;
  height: number;
  stages: string[];
  refined: boolean;
}
