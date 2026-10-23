/**
 * The typed client for the 2DAutoRigger service.
 *
 * The service is the only thing that knows how to segment, slice and rig a
 * character. This client is the whole JavaScript surface: it works in a
 * browser, in Node, and in an editor built with Electron.
 */

import type {
  Attempt,
  BackendInfo,
  Document,
  Edit,
  ExportFile,
  ProjectSummary,
  Rig,
  SkinningMode,
  Task,
  TemplateInfo,
} from "@rigger2d/schema";

export interface ClientOptions {
  /** Where the service is listening. Empty means "the page I was served from". */
  baseUrl?: string;
  /** Injected for testing. Defaults to the global fetch. */
  fetch?: typeof globalThis.fetch;
}

export interface CreateProjectOptions {
  imagePath?: string;
  imageBase64?: string;
  template?: string;
  name?: string;
}

export interface RunOptions {
  until?: "all" | "refine";
  prefer?: string;
  force?: boolean;
}

export interface SliceOptions {
  targets?: string[];
  skinning?: SkinningMode;
  prefer?: string;
}

export class RiggerError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `request failed with ${status}`);
    this.name = "RiggerError";
    this.status = status;
    this.detail = detail;
  }
}

export class RiggerClient {
  private readonly baseUrl: string;
  private readonly doFetch: typeof globalThis.fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "").replace(/\/$/, "");
    this.doFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.doFetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "content-type": "application/json" } : {}),
        ...(init?.headers ?? {}),
      },
    });
    if (!response.ok) {
      throw new RiggerError(response.status, await readDetail(response));
    }
    if (response.status === 204) {
      return undefined as T;
    }
    const type = response.headers.get("content-type") ?? "";
    if (type.includes("application/json")) {
      return (await response.json()) as T;
    }
    return (await response.text()) as unknown as T;
  }

  // -- meta -------------------------------------------------------------

  health(): Promise<{ status: string; stages: string[]; refine_gate: string }> {
    return this.request("/api/health");
  }

  backends(): Promise<{ backends: BackendInfo[] }> {
    return this.request("/api/backends");
  }

  templates(): Promise<{ templates: TemplateInfo[] }> {
    return this.request("/api/templates");
  }

  // -- projects ---------------------------------------------------------

  listProjects(): Promise<{ projects: ProjectSummary[] }> {
    return this.request("/api/projects");
  }

  createProject(options: CreateProjectOptions): Promise<{ id: string; document: Document }> {
    return this.request("/api/projects", {
      method: "POST",
      body: JSON.stringify(options),
    });
  }

  getProject(id: string): Promise<{ document: Document; edits: Record<string, number> }> {
    return this.request(`/api/projects/${id}`);
  }

  deleteProject(id: string): Promise<{ deleted: string }> {
    return this.request(`/api/projects/${id}`, { method: "DELETE" });
  }

  run(id: string, options: RunOptions = {}): Promise<{ document: Document }> {
    return this.request(`/api/projects/${id}/run`, {
      method: "POST",
      body: JSON.stringify({ until: options.until ?? "refine", ...options }),
    });
  }

  refine(id: string, edits: Edit[]): Promise<{ document: Document; edits: Record<string, number> }> {
    return this.request(`/api/projects/${id}/refine`, {
      method: "POST",
      body: JSON.stringify({ edits }),
    });
  }

  moveJoints(id: string, edits: Edit[]): Promise<{ document: Document }> {
    return this.request(`/api/projects/${id}/joints`, {
      method: "POST",
      body: JSON.stringify({ edits }),
    });
  }

  compare(id: string, task: Task): Promise<{ task: Task; attempts: Attempt[] }> {
    return this.request(`/api/projects/${id}/compare`, {
      method: "POST",
      body: JSON.stringify({ task }),
    });
  }

  slice(id: string, options: SliceOptions = {}): Promise<{ document: Document; exports: ExportFile[] }> {
    return this.request(`/api/projects/${id}/slice`, {
      method: "POST",
      body: JSON.stringify({
        targets: options.targets ?? ["native"],
        skinning: options.skinning ?? "hybrid",
        ...options,
      }),
    });
  }

  exports(id: string): Promise<{ exports: ExportFile[] }> {
    return this.request(`/api/projects/${id}/exports`);
  }

  // -- images -----------------------------------------------------------

  sourceUrl(id: string): string {
    return `${this.baseUrl}/api/projects/${id}/source`;
  }

  maskUrl(id: string, partId: string): string {
    return `${this.baseUrl}/api/projects/${id}/mask/${partId}`;
  }

  previewUrl(id: string): string {
    return `${this.baseUrl}/api/projects/${id}/preview`;
  }

  exportUrl(id: string, name: string): string {
    return `${this.baseUrl}/api/projects/${id}/exports/${name}`;
  }

  /** The rig document, as it was written to disk. */
  async rig(id: string): Promise<Rig> {
    const response = await this.doFetch(this.exportUrl(id, "rig.json"));
    if (!response.ok) {
      throw new RiggerError(response.status, await response.text());
    }
    return (await response.json()) as Rig;
  }
}

export function createClient(options: ClientOptions = {}): RiggerClient {
  return new RiggerClient(options);
}

/**
 * The reason a request failed, as the service stated it.
 *
 * FastAPI wraps every error in a `{"detail": ...}` envelope, and validation
 * errors come back as `{"detail": [{...}]}`. Both would otherwise surface as an
 * unhelpful "request failed with 400", so unwrap them here.
 */
async function readDetail(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed: unknown = JSON.parse(text);
    if (Array.isArray(parsed)) {
      const messages = parsed
        .map((item) => (typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : ""))
        .filter(Boolean);
      if (messages.length > 0) return messages.join("; ");
    }
    if (parsed && typeof parsed === "object" && "detail" in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      return typeof detail === "string" ? detail : JSON.stringify(detail);
    }
    return String(parsed);
  } catch {
    return text || `request failed with ${response.status}`;
  }
}
