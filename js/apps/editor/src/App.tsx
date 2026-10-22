/**
 * The editor shell.
 *
 * The flow is the pipeline's flow: load a character, run the pipeline up to
 * the refine gate, fix the masks and the joints by hand, then slice and export.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { createClient } from "@rigger2d/client";
import type {
  Attempt,
  BackendInfo,
  Document,
  Edit,
  ExportFile,
  ProjectSummary,
  SkinningMode,
  Task,
  Vec2,
} from "@rigger2d/schema";
import { Editor, type BrushMode, type PartColour } from "./Editor";

const PART_COLOURS: PartColour[] = [
  { id: "head", colour: "#ffcc44" },
  { id: "neck", colour: "#ffb347" },
  { id: "torso", colour: "#4aa3ff" },
  { id: "pelvis", colour: "#2f7fd6" },
  { id: "upper_arm_l", colour: "#ff6b6b" },
  { id: "lower_arm_l", colour: "#ff9f6b" },
  { id: "hand_l", colour: "#ffc48a" },
  { id: "upper_arm_r", colour: "#ff6b6b" },
  { id: "lower_arm_r", colour: "#ff9f6b" },
  { id: "hand_r", colour: "#ffc48a" },
  { id: "upper_leg_l", colour: "#6bd97a" },
  { id: "lower_leg_l", colour: "#9fe0a8" },
  { id: "foot_l", colour: "#4f9e5e" },
  { id: "upper_leg_r", colour: "#6bd97a" },
  { id: "lower_leg_r", colour: "#9fe0a8" },
  { id: "foot_r", colour: "#4f9e5e" },
];

const client = createClient();

export function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [document, setDocument] = useState<Document | null>(null);
  const [landmarks, setLandmarks] = useState<Record<string, Vec2>>({});
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [exports, setExports] = useState<ExportFile[]>([]);
  const [activePart, setActivePart] = useState("torso");
  const [brush, setBrush] = useState<BrushMode>("add");
  const [brushRadius, setBrushRadius] = useState(10);
  const [skinning, setSkinning] = useState<SkinningMode>("hybrid");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [reloadToken, setReloadToken] = useState(0);

  const refreshProjects = useCallback(async () => {
    const result = await client.listProjects();
    setProjects(result.projects);
    if (!projectId && result.projects.length > 0) {
      setProjectId(result.projects[0]!.id);
    }
  }, [projectId]);

  useEffect(() => {
    void refreshProjects();
    void client.backends().then((result) => setBackends(result.backends));
  }, [refreshProjects]);

  useEffect(() => {
    if (!projectId) return;
    void client.getProject(projectId).then((result) => {
      setDocument(result.document);
      setLandmarks(result.document.landmarks);
    });
  }, [projectId]);

  const parts = useMemo(() => {
    const ids = document?.parts?.masks.map((mask) => mask.part_id) ?? [];
    const known = PART_COLOURS.filter((part) => ids.includes(part.id));
    const extra = ids
      .filter((id) => !known.some((part) => part.id === id))
      .map((id, index) => ({ id, colour: PALETTE[index % PALETTE.length]! }));
    return [...known, ...extra];
  }, [document]);

  const upload = async (file: File): Promise<void> => {
    setBusy(true);
    setMessage("");
    try {
      const base64 = await fileToBase64(file);
      const result = await client.createProject({ imageBase64: base64 });
      setProjectId(result.id);
      setDocument(result.document);
      setLandmarks(result.document.landmarks);
      await refreshProjects();
      setMessage("Loaded. Run the pipeline to segment the character.");
    } catch (error) {
      setMessage(`Could not load the image: ${describe(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const run = async (): Promise<void> => {
    if (!projectId) return;
    setBusy(true);
    setMessage("Segmenting...");
    try {
      const result = await client.run(projectId, { until: "refine" });
      setDocument(result.document);
      setLandmarks(result.document.landmarks);
      setReloadToken((token) => token + 1);
      setMessage("Segmented. Check the masks, then slice.");
    } catch (error) {
      setMessage(`The pipeline failed: ${describe(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const slice = async (): Promise<void> => {
    if (!projectId) return;
    setBusy(true);
    setMessage("Slicing...");
    try {
      const result = await client.slice(projectId, { targets: ["native", "godot", "spine"], skinning });
      setDocument(result.document);
      setExports(result.exports);
      setMessage(`Done. ${result.exports.length} files written.`);
    } catch (error) {
      setMessage(`Could not slice the character: ${describe(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const onMaskEdit = async (edit: Edit): Promise<void> => {
    if (!projectId) return;
    try {
      const result = await client.refine(projectId, [edit]);
      setDocument(result.document);
      setReloadToken((token) => token + 1);
    } catch (error) {
      setMessage(`The correction could not be saved: ${describe(error)}`);
    }
  };

  const onJointEdit = async (edit: Edit): Promise<void> => {
    if (!projectId) return;
    try {
      const result = await client.moveJoints(projectId, [edit]);
      setDocument(result.document);
      setLandmarks(result.document.landmarks);
    } catch (error) {
      setMessage(`The joint could not be moved: ${describe(error)}`);
    }
  };

  const compare = async (task: Task): Promise<void> => {
    if (!projectId) return;
    setBusy(true);
    try {
      const result = await client.compare(projectId, task);
      setAttempts(result.attempts);
    } catch (error) {
      setMessage(`The comparison failed: ${describe(error)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>2DAutoRigger</h1>
        <div className="app-header-actions">
          <label className="button">
            Load a character
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
              }}
            />
          </label>
          <button onClick={() => void run()} disabled={busy || !projectId}>
            Segment
          </button>
          <button onClick={() => void slice()} disabled={busy || !projectId}>
            Slice and export
          </button>
        </div>
      </header>

      <div className="app-body">
        <aside className="panel panel-left">
          <h2>Projects</h2>
          <ul className="list">
            {projects.map((project) => (
              <li key={project.id}>
                <button
                  className={project.id === projectId ? "selected" : ""}
                  onClick={() => setProjectId(project.id)}
                >
                  {project.id.slice(0, 8)} · {project.width}x{project.height}
                  {project.refined ? " · edited" : ""}
                </button>
              </li>
            ))}
          </ul>

          <h2>Parts</h2>
          <ul className="list">
            {parts.map((part) => (
              <li key={part.id}>
                <button
                  className={part.id === activePart ? "selected" : ""}
                  onClick={() => setActivePart(part.id)}
                >
                  <span className="swatch" style={{ background: part.colour }} />
                  {part.id}
                </button>
              </li>
            ))}
          </ul>

          <h2>Brush</h2>
          <div className="row">
            {(["add", "erase", "assign"] as BrushMode[]).map((mode) => (
              <button
                key={mode}
                className={brush === mode ? "selected" : ""}
                onClick={() => setBrush(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
          <label className="row">
            radius
            <input
              type="range"
              min={2}
              max={60}
              value={brushRadius}
              onChange={(event) => setBrushRadius(Number(event.target.value))}
            />
          </label>

          <h2>Deformation</h2>
          <div className="row">
            {(["cutout", "hybrid", "mesh"] as SkinningMode[]).map((mode) => (
              <button
                key={mode}
                className={skinning === mode ? "selected" : ""}
                onClick={() => setSkinning(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
          <p className="hint">
            cutout: every part is rigid. hybrid: the torso and head are rigid, the limbs
            bend. mesh: everything deforms.
          </p>
        </aside>

        <main className="stage">
          {document ? (
            <Editor
              document={document}
              landmarks={landmarks}
              client={client}
              parts={parts}
              activePart={activePart}
              brush={brush}
              brushRadius={brushRadius}
              onMaskEdit={(edit) => void onMaskEdit(edit)}
              onJointEdit={(edit) => void onJointEdit(edit)}
              reloadToken={reloadToken}
            />
          ) : (
            <p className="hint">Load a character to begin.</p>
          )}
          {message ? <p className="message">{message}</p> : null}
          {document?.warnings.map((warning) => (
            <p className="warning" key={warning}>
              {warning}
            </p>
          ))}
        </main>

        <aside className="panel panel-right">
          <h2>Backends</h2>
          <ul className="list">
            {backends.map((backend) => (
              <li key={backend.id}>
                <div className="backend">
                  <strong>{backend.id}</strong>
                  <span className={backend.status === "available" ? "ok" : "bad"}>
                    {backend.status}
                  </span>
                  {backend.license ? <em>{backend.license}</em> : null}
                  {backend.reason ? <p className="hint">{backend.reason}</p> : null}
                </div>
              </li>
            ))}
          </ul>
          <div className="row">
            <button onClick={() => void compare("parts")} disabled={!projectId}>
              Compare part backends
            </button>
          </div>
          {attempts.length > 0 ? (
            <ul className="list">
              {attempts.map((attempt) => (
                <li key={attempt.adapter_id}>
                  {attempt.adapter_id}:{" "}
                  {attempt.ok ? (attempt.score ?? 0).toFixed(2) : attempt.error ?? "failed"}
                </li>
              ))}
            </ul>
          ) : null}

          <h2>Stages</h2>
          <ul className="list">
            {Object.values(document?.stages ?? {}).map((stage) => (
              <li key={stage.name}>
                {stage.name} · {stage.duration_ms} ms {stage.adapter_id ?? ""}
              </li>
            ))}
          </ul>

          <h2>Exports</h2>
          <ul className="list">
            {exports.map((file) => (
              <li key={file.name}>
                <a href={client.exportUrl(projectId ?? "", file.name)} download>
                  {file.name}
                </a>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}

const PALETTE = [
  "#8f6bff",
  "#4ad6c8",
  "#ff7ab8",
  "#a8d84a",
  "#ff9a3c",
  "#5ac8fa",
];

function describe(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

async function fileToBase64(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}
