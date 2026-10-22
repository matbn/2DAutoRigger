import { describe, expect, it } from "vitest";
import { RiggerClient, RiggerError } from "../src/index";

type Handler = (url: string, init?: RequestInit) => Response;

function fakeFetch(handler: Handler): typeof globalThis.fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    return handler(String(input), init);
  }) as typeof globalThis.fetch;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("the client", () => {
  it("asks the health endpoint", async () => {
    const client = new RiggerClient({
      baseUrl: "http://example.test",
      fetch: fakeFetch(() => json({ status: "ok", stages: ["parts"], refine_gate: "parts" })),
    });
    const health = await client.health();
    expect(health.status).toBe("ok");
    expect(health.refine_gate).toBe("parts");
  });

  it("joins the base url and the path", async () => {
    let seen = "";
    const client = new RiggerClient({
      baseUrl: "http://example.test/",
      fetch: fakeFetch((url) => {
        seen = url;
        return json({ backends: [] });
      }),
    });
    await client.backends();
    expect(seen).toBe("http://example.test/api/backends");
  });

  it("sends json when there is a body", async () => {
    let body: unknown = null;
    const client = new RiggerClient({
      fetch: fakeFetch((_url, init) => {
        body = init?.body;
        return json({ document: {} });
      }),
    });
    await client.run("abc", { until: "refine" });
    expect(body).toEqual(JSON.stringify({ until: "refine" }));
  });

  it("raises a helpful error when the service says no", async () => {
    const client = new RiggerClient({
      fetch: fakeFetch(() => json({ detail: "no project" }, 404)),
    });
    await expect(client.getProject("nope")).rejects.toBeInstanceOf(RiggerError);
  });

  it("builds the image urls", () => {
    const client = new RiggerClient({ baseUrl: "http://example.test" });
    expect(client.sourceUrl("abc")).toBe("http://example.test/api/projects/abc/source");
    expect(client.maskUrl("abc", "torso")).toBe("http://example.test/api/projects/abc/mask/torso");
    expect(client.exportUrl("abc", "rig.json")).toBe(
      "http://example.test/api/projects/abc/exports/rig.json",
    );
  });

  it("reads the rig document", async () => {
    const client = new RiggerClient({
      fetch: fakeFetch(() => json({ bones: [], layers: [] })),
    });
    const rig = await client.rig("abc");
    expect(Array.isArray(rig.bones)).toBe(true);
  });
});
