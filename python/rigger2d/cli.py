"""The ``autorigger2d`` command line.

    autorigger2d run hero.png --project hero.rig
    autorigger2d run hero.png --project hero.rig --until refine
    autorigger2d serve
    autorigger2d models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.registry import Registry, probe_entry
from .contracts.models import SkinningMode
from .pipeline.session import Session
from .pipeline.stages import REFINE_GATE, RunOptions, STAGE_ORDER, run_stages
from .rigging.templates import DEFAULT_TEMPLATE_ID, TemplateRegistry


def _project_dir(project: str | None, image: Path) -> Path:
    if project:
        return Path(project)
    return image.with_suffix("").with_suffix(".rig") if image.suffix else image.with_suffix(".rig")


def cmd_run(args: argparse.Namespace) -> int:
    image = Path(args.image).expanduser()
    if not image.is_file():
        print(f"error: no such image: {image}", file=sys.stderr)
        return 2

    session = Session(_project_dir(args.project, image), args.template)
    if not session.document_path.is_file():
        document = session.create(image, args.template)
        print(f"[s0]  ingest            {image.name} -> {document.image.width}x{document.image.height}")
    else:
        document = session.load()
        print(f"[s0]  ingest            reusing {session.root}")

    options = RunOptions(
        targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        skinning=SkinningMode(args.skinning),
        prefer=args.prefer,
        force=args.force,
    )

    until = None if args.until == "all" else args.until
    document = run_stages(session, document, options, until=until)

    for name in STAGE_ORDER:
        result = document.stages.get(name)
        if result is None:
            continue
        marker = "!" if name == REFINE_GATE else " "
        print(
            f"[{name[:2]}]  {result.duration_ms:>6} ms  {name:<12} "
            f"{result.adapter_id or ''} {marker}"
        )

    for warning in document.warnings:
        print(f"  ! {warning}")

    if document.rig:
        print(
            f"\n{len(document.rig.bones)} bones, {len(document.rig.layers)} layers, "
            f"{len(document.rig.meshes)} meshes, skinning={document.rig.skinning.value}"
        )
    if until == REFINE_GATE:
        print(
            "\nThe pipeline stopped at the refine gate. Review the masks, then run:\n"
            f"  autorigger2d run {image} --project {session.root} --until all"
        )
    print(f"\nProject: {session.root}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    registry = Registry.discover()
    payload = []
    for entry in registry.entries:
        probe = probe_entry(entry, registry_ctx())
        payload.append(
            {
                "id": entry.id,
                "origin": entry.origin,
                "tasks": [t.value for t in probe.capabilities],
                "status": probe.status.value,
                "reason": probe.reason,
                "license": probe.license_spdx,
                "license_url": probe.license_url,
                "error": entry.error,
            }
        )
    print(json.dumps(payload, indent=2))
    return 0


def registry_ctx():
    from .adapters.base import AdapterContext

    return AdapterContext(
        project_dir=Path.cwd(),
        work_dir=Path.cwd() / ".rigger2d",
        device="cpu",
    )


def cmd_templates(args: argparse.Namespace) -> int:
    registry = TemplateRegistry()
    for template in registry.available():
        print(f"{template.id:<28} {template.type:<12} {template.description}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import numpy
    import PIL
    import pydantic

    print("2DAutoRigger doctor")
    print(f"  python     {sys.version.split()[0]}")
    print(f"  numpy      {numpy.__version__}")
    print(f"  pillow     {PIL.__version__}")
    print(f"  pydantic   {pydantic.VERSION}")
    try:
        import scipy

        print(f"  scipy      {scipy.__version__} (mesh triangulation enabled)")
    except ImportError:
        print("  scipy      not installed (mesh falls back to ear clipping)")

    registry = Registry.discover()
    print("\n  backends:")
    for entry in registry.entries:
        probe = probe_entry(entry, registry_ctx())
        state = probe.status.value
        if probe.reason:
            state = f"{state} ({probe.reason})"
        print(f"    {entry.id:<28} {state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autorigger2d",
        description="Automatic 2D character segmentation, slicing and rigging.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="rig a character image")
    run.add_argument("image", help="the character image")
    run.add_argument("--project", help="project directory (default: <image>.rig)")
    run.add_argument("--template", default=DEFAULT_TEMPLATE_ID, help="rig template id")
    run.add_argument(
        "--skinning",
        choices=[mode.value for mode in SkinningMode],
        default=SkinningMode.HYBRID.value,
        help="deformation mode",
    )
    run.add_argument(
        "--targets",
        default="native",
        help="comma separated: native,godot,spine,unity",
    )
    run.add_argument(
        "--until",
        choices=["all", REFINE_GATE],
        default="all",
        help="stop at the refine gate instead of slicing",
    )
    run.add_argument("--prefer", help="prefer one backend id")
    run.add_argument("--force", action="store_true", help="re-run every stage")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="run the service and the editor")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    models = sub.add_parser("models", help="list the installed backends")
    models.set_defaults(func=cmd_models)

    templates = sub.add_parser("templates", help="list the available templates")
    templates.set_defaults(func=cmd_templates)

    doctor = sub.add_parser("doctor", help="check the environment")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
