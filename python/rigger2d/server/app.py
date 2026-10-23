"""The HTTP service.

This is the bridge between JavaScript and Python. The editor is a browser app;
the pipeline is Python. They talk over JSON.

The service also serves the built editor from its own origin, so there is no
CORS configuration and no token to copy around.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field

from ..adapters.base import AdapterContext
from ..adapters.registry import Registry, probe_entry
from ..adapters.selection import compare_all
from ..contracts.models import Edit, SkinningMode, Task
from ..pipeline.refine import apply_joint_edits, apply_mask_edits, summarize
from ..pipeline.session import Session
from ..pipeline.stages import REFINE_GATE, RunOptions, STAGE_ORDER, run_stages
from ..rigging.templates import TemplateRegistry

STATIC_DIR = Path(__file__).resolve().parent / "static"


def projects_root() -> Path:
    env = os.environ.get("RIGGER2D_PROJECTS")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "projects"


class CreateProject(BaseModel):
    """Create a project.

    Both spellings of the image fields are accepted: the Python client and the
    tests use snake_case, the browser client sends camelCase.
    """

    model_config = {"populate_by_name": True}

    image_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_path", "imagePath"),
    )
    image_base64: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_base64", "imageBase64"),
    )
    template: str = "humanoid.biped.v1"
    name: str | None = None


class RunRequest(BaseModel):
    until: Literal["all", "refine"] = "refine"
    prefer: str | None = None
    force: bool = False


class RefineRequest(BaseModel):
    edits: list[Edit] = Field(default_factory=list)


class SliceRequest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["native"])
    skinning: SkinningMode = SkinningMode.HYBRID
    prefer: str | None = None


class CompareRequest(BaseModel):
    task: Task = Task.PARTS


def create_app(projects: Path | None = None) -> FastAPI:
    root = Path(projects) if projects else projects_root()
    root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="2DAutoRigger",
        version="0.1.0",
        description="Automatic 2D character segmentation, slicing and rigging.",
    )
    app.state.projects_root = root

    # -- helpers ---------------------------------------------------------

    def session_for(project_id: str) -> Session:
        path = root / project_id
        if not path.is_dir():
            raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
        return Session(path)

    def load_session(project_id: str) -> tuple[Session, Any]:
        session = session_for(project_id)
        try:
            return session, session.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def context_for(session: Session) -> AdapterContext:
        return session.context

    # -- meta ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "stages": STAGE_ORDER,
            "refine_gate": REFINE_GATE,
            "projects": str(root),
        }

    @app.get("/api/backends")
    def backends() -> dict[str, Any]:
        registry = Registry.discover()
        ctx = AdapterContext(project_dir=root, work_dir=root, device="cpu")
        payload = []
        for entry in registry.entries:
            probe = probe_entry(entry, ctx)
            payload.append(
                {
                    "id": entry.id,
                    "origin": entry.origin,
                    "tasks": [t.value for t in probe.capabilities],
                    "status": probe.status.value,
                    "reason": probe.reason,
                    "license": probe.license_spdx,
                    "license_url": probe.license_url,
                    "device": probe.device,
                    "error": entry.error,
                }
            )
        return {"backends": payload}

    @app.get("/api/templates")
    def templates() -> dict[str, Any]:
        found = TemplateRegistry().available()
        return {
            "templates": [
                {
                    "id": t.id,
                    "type": t.type,
                    "description": t.description,
                    "parts": t.part_ids,
                }
                for t in found
            ]
        }

    # -- projects --------------------------------------------------------

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        items = []
        for path in sorted(root.iterdir()):
            document = path / "document.json"
            if not document.is_file():
                continue
            import json

            data = json.loads(document.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": path.name,
                    "created_at": data.get("created_at", ""),
                    "template_id": data.get("template_id", ""),
                    "width": data.get("image", {}).get("width", 0),
                    "height": data.get("image", {}).get("height", 0),
                    "stages": sorted(data.get("stages", {}).keys()),
                    "refined": bool(data.get("edits")),
                }
            )
        return {"projects": items}

    @app.post("/api/projects")
    def create_project(payload: CreateProject) -> dict[str, Any]:
        import base64

        project_id = payload.name or uuid.uuid4().hex[:12]
        target = root / project_id
        if target.exists():
            raise HTTPException(status_code=409, detail=f"project {project_id!r} exists")

        if payload.image_path:
            source = Path(payload.image_path).expanduser()
            if not source.is_file():
                raise HTTPException(status_code=400, detail=f"no such image: {source}")
        elif payload.image_base64:
            import binascii

            try:
                blob = base64.b64decode(payload.image_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"bad base64: {exc}") from exc
            target.mkdir(parents=True, exist_ok=True)
            source = target / "upload"
            source.write_bytes(blob)
        else:
            raise HTTPException(status_code=400, detail="give an image_path or image_base64")

        session = Session(target, payload.template)
        try:
            document = session.create(source, payload.template)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(target, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": project_id, "document": document.model_dump(mode="json")}

    @app.post("/api/projects/upload")
    async def upload_project(file: UploadFile) -> dict[str, Any]:
        project_id = uuid.uuid4().hex[:12]
        target = root / project_id
        target.mkdir(parents=True, exist_ok=True)
        source = target / "upload"
        source.write_bytes(await file.read())
        session = Session(target)
        document = session.create(source)
        return {"id": project_id, "document": document.model_dump(mode="json")}

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        _session, document = load_session(project_id)
        return {
            "document": document.model_dump(mode="json"),
            "edits": summarize(document.edits),
        }

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        path = root / project_id
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="no such project")
        shutil.rmtree(path)
        return {"deleted": project_id}

    # -- pipeline --------------------------------------------------------

    @app.post("/api/projects/{project_id}/run")
    def run_project(project_id: str, request: RunRequest = Body(default=RunRequest())):
        session, document = load_session(project_id)
        options = RunOptions(prefer=request.prefer, force=request.force)
        until = REFINE_GATE if request.until == "refine" else None
        try:
            document = run_stages(session, document, options, until=until)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"document": document.model_dump(mode="json")}

    @app.post("/api/projects/{project_id}/refine")
    def refine_project(project_id: str, request: RefineRequest) -> dict[str, Any]:
        session, document = load_session(project_id)
        document.edits.extend(request.edits)
        session.save(document)
        return {
            "edits": summarize(document.edits),
            "document": document.model_dump(mode="json"),
        }

    @app.post("/api/projects/{project_id}/joints")
    def move_joints(project_id: str, request: RefineRequest) -> dict[str, Any]:
        session, document = load_session(project_id)
        document.edits.extend(request.edits)
        document.landmarks = apply_joint_edits(document.landmarks, request.edits)
        session.save(document)
        return {
            "landmarks": {k: v.model_dump() for k, v in document.landmarks.items()},
            "edits": summarize(document.edits),
        }

    @app.post("/api/projects/{project_id}/compare")
    def compare(project_id: str, request: CompareRequest) -> dict[str, Any]:
        session, document = load_session(project_id)
        registry = session.registry
        ctx = context_for(session)

        if request.task == Task.PARTS:
            from ..adapters.base import PartsPayload

            matte = session.root / document.matte.image if document.matte else None
            payload = PartsPayload(
                image_path=session.source_path,
                matte_path=matte,
                bbox=document.matte.bounds if document.matte else None,
                landmarks=dict(document.landmarks),
                output_dir=session.parts_dir / "_compare",
                part_ids=session.template.part_ids,
            )
        elif request.task == Task.POSE:
            from ..adapters.base import PosePayload

            payload = PosePayload(
                image_path=session.source_path,
                bbox=document.matte.bounds if document.matte else None,
                output_dir=session.masks_dir,
                landmarks=dict(document.landmarks),
            )
        else:
            from ..adapters.base import MattePayload

            payload = MattePayload(
                image_path=session.source_path,
                output_dir=session.masks_dir / "_compare",
            )

        attempts = compare_all(registry, request.task, payload, ctx, session.template)
        return {"task": request.task.value, "attempts": [a.to_json() for a in attempts]}

    @app.post("/api/projects/{project_id}/slice")
    def slice_project(project_id: str, request: SliceRequest) -> dict[str, Any]:
        session, document = load_session(project_id)
        options = RunOptions(
            targets=request.targets,
            skinning=request.skinning,
            prefer=request.prefer,
        )
        try:
            document = run_stages(session, document, options, invalid_from="layers")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "document": document.model_dump(mode="json"),
            "exports": _export_list(session),
        }

    # -- artefacts -------------------------------------------------------

    @app.get("/api/projects/{project_id}/preview")
    def preview(project_id: str) -> FileResponse:
        session, _document = load_session(project_id)
        path = session.layers_dir / "preview.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no preview yet")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/projects/{project_id}/source")
    def source_image(project_id: str) -> FileResponse:
        session, _document = load_session(project_id)
        return FileResponse(session.source_path, media_type="image/png")

    @app.get("/api/projects/{project_id}/masks")
    def masks(project_id: str) -> FileResponse:
        session, _document = load_session(project_id)
        path = session.parts_dir / "index.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no masks yet")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/projects/{project_id}/mask/{part_id}")
    def mask(project_id: str, part_id: str) -> FileResponse:
        session, _document = load_session(project_id)
        path = session.parts_dir / f"{part_id}.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no mask for {part_id!r}")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/projects/{project_id}/exports")
    def exports(project_id: str) -> dict[str, Any]:
        session, _document = load_session(project_id)
        return {"exports": _export_list(session)}

    @app.get("/api/projects/{project_id}/exports/{name}")
    def download_export(project_id: str, name: str) -> FileResponse:
        session, _document = load_session(project_id)
        path = (session.exports_dir / name).resolve()
        if not str(path).startswith(str(session.exports_dir.resolve())):
            raise HTTPException(status_code=400, detail="bad path")
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no export named {name!r}")
        return FileResponse(path, filename=path.name)

    # -- editor ----------------------------------------------------------

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="editor")
    else:

        @app.get("/")
        def no_editor() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": (
                        "The editor has not been built. Run `npm install && npm run build` "
                        "in js/, or use the API directly."
                    )
                }
            )

    return app


def _export_list(session: Session) -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = []
    base = session.exports_dir
    if not base.is_dir():
        return exports
    for path in sorted(base.rglob("*")):
        if path.is_file():
            exports.append(
                {
                    "name": path.relative_to(base).as_posix(),
                    "size": path.stat().st_size,
                    "url": f"/api/projects/{session.root.name}/exports/"
                    + path.relative_to(base).as_posix(),
                }
            )
    return exports


__all__ = ["create_app"]
