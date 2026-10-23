# 2DAutoRigger

Automatic 2D character segmentation, slicing, and rigging. Give it an image; get back sliced layers, a skeleton sized from the part extents, and exports for Godot, Spine, and Unity.

It runs entirely on your machine. No account, no cloud, no telemetry. The core is Apache-2.0; backend plugins carry their own licenses.

## Quick start

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev]"

# Rig a character (stops at the refine gate for human correction)
.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig --until refine

# Then slice and export
.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig --targets native,godot,spine
```

Or run everything in one shot:

```bash
.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig
```

### Browser editor

```bash
cd js && npm install && npm run build
.venv/bin/python -m rigger2d.cli serve
```
Then open `http://127.0.0.1:8765`.

## Deformation modes

| Mode | What it does |
|------|-------------|
| `cutout` | Every part is rigid — a sprite on a bone. Simple, robust, never tears. |
| `hybrid` | Torso and head are rigid; limbs bend. Default for humanoids. |
| `mesh` | Everything deforms. |

The mode is per-part in the template and chosen at export time:

```bash
.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig --skinning hybrid --targets native,godot,spine
```

## Backends (plugin system)

The core contains no machine learning — only `numpy` and `Pillow`. Real backends are plugins installed separately:

| Adapter | Task | Model license | Weights |
|---------|------|---------------|---------|
| `sam3` | parts | Meta SAM License | gated |
| `birefnet` | matte | MIT | open |
| `humanparsing` | parts | MIT | open |
| `rtmpose` | pose | Apache-2.0 | open |
| `lama` | inpaint | Apache-2.0 | open |

They live in `adapters/`. Make one available:

```bash
export RIGGER2D_PLUGINS="$(ls -d adapters/*)"
autorigger2d models
```

No model weights are shipped. Each adapter's `THIRD_PARTY.md` and manifest record whose terms apply and where to obtain the weights. See `docs/PLUGINS.md` to write your own.

## The refine step

The pipeline stops after segmentation so you can correct the masks and joints before slicing:

```
ingest -> matte -> pose -> parts  ==REFINE GATE==>  layers -> skeleton -> mesh -> weights -> emit
```

Edits are stored as a stroke/joint log, not re-rasterised masks, so they're compact, undoable, and the original segmentation stays the source of truth.

## Outputs

| File | What it is |
|------|------------|
| `rig.json` | Canonical rig: bones, layers, meshes, weights, provenance |
| `layers/*.png` | Character cut into one image per part |
| `exports/godot/scene.tscn` | Godot scene (`Skeleton2D` + `Polygon2D` bones) |
| `exports/spine/skeleton.json` + `atlas.atlas` / `atlas.png` | Spine skeleton and atlas |
| `exports/unity/Character.prefab` + `Assets/` | Unity prefab with textures and import settings |
| `provenance.json` | Which backend produced what, under which license |

## Checks

```bash
.venv/bin/python -m pytest tests -q            # Python tests
cd js && npm run test                          # JavaScript tests
.venv/bin/python tools/license_check.py        # Apache-2.0 core check
.venv/bin/python tools/schema_check.py         # contract check
```

## Licence

The core is **Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Backend models carry their own terms and are never bundled. Godot is a trademark of the Godot Engine contributors; Spine is a trademark of Esoteric Software LLC. This project writes files in their formats and contains no engine code.