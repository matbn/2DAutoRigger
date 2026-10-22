# 2DAutoRigger

Give it a picture of a 2D character. Get back the character's body parts as
masks, the character sliced into layers, a skeleton whose bones are the size of
the parts they drive, and exports for Godot and Spine.

It runs entirely on your machine. No account, no cloud, no telemetry.

```
[matte]    378 ms  core.fallback
[pose]       0 ms  core.fallback
[parts]    323 ms  core.fallback  !
[la]      372 ms  layers
[sk]       29 ms  skeleton
[me]       83 ms  mesh
[we]        0 ms  weights
[em]      320 ms  emit

18 bones, 30 layers, 16 meshes, skinning=hybrid
```

## The idea

The interesting part is not the segmentation, it is what happens after it.

Segmentation is the part everyone gets wrong. The machine will put a bit of
the shoulder in the arm, and the arm will be a bit too fat. So the pipeline
**stops** after the parts are found, and waits for a human:

```
ingest -> matte -> pose -> parts  ==REFINE GATE==>  layers -> skeleton
        -> mesh -> weights -> emit
```

At the gate you can paint the parts back into shape and move the joints
around. Nothing is sliced until you are happy. The corrections are stored as an
edit log -- strokes and joint moves -- rather than as re-rasterised masks, so
they are compact, undoable, and the original segmentation stays the source of
truth.

## Getting started

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig
```

To stop at the refine gate and look at the masks first:

```bash
.venv/bin/python -m rigger2d.cli run hero.png --project hero.rig --until refine
```

To get the editor in a browser:

```bash
cd js && npm install && npm run build
.venv/bin/python -m rigger2d.cli serve
```

Then open <http://127.0.0.1:8765>. The service serves the editor from its own
origin, so there is nothing to configure.

## What you get

| File | What it is |
| --- | --- |
| `rig.json` | the canonical rig: bones, layers, meshes, weights, provenance |
| `layers/*.png` | the character, cut into one image per part |
| `exports/godot/scene.tscn` | a Godot scene with a `Skeleton2D` and weighted `Polygon2D`s |
| `exports/spine/skeleton.json` | a Spine skeleton, with the atlas next to it |
| `exports/unity/` | a Unity project folder: `Character.prefab`, the layer images and their `.meta` files |
| `provenance.json` | which backend produced what, under which license |

## Deformation is the user's choice

Three modes, and they are all built:

| Mode | What it does |
| --- | --- |
| `cutout` | every part is rigid -- a sprite on a bone. Simple, robust, never tears. |
| `hybrid` | the torso and head are rigid, the limbs bend. The default for a humanoid. |
| `mesh` | everything deforms. |

The mode is in the template, per part, and can be changed at export time.

## The core has no machine learning in it

The core depends on `numpy` and `Pillow`. That is all. Every real backend --
SAM 3, BiRefNet, a human parsing model, a pose estimator -- is a **plugin** in
its own package, with its own license, installed by you.

This is not fussiness about packaging. It means:

* the core can be Apache-2.0 without dragging a copyleft model into it;
* you choose which models you are willing to accept the terms of, and which
  you are not;
* the pipeline always runs, even with no plugin installed at all, because the
  core has a geometric fallback that says out loud how much it is guessing.

```bash
autorigger2d models     # what is installed, and what is missing
autorigger2d doctor     # what the environment can do
```

Five adapters ship as scaffolds, under `adapters/`:

```
adapters/sam3           Meta SAM License, gated weights
adapters/birefnet       MIT code, own terms on the weights
adapters/humanparsing   the strongest option for humanoids
adapters/rtmpose        pose estimation, runs on the CPU
adapters/lama           fills in what a moving limb hides
```

They are in this repository because their own code is Apache-2.0, like the
core. What differs is the licence of the *model*, and no model is shipped
either way -- each adapter's `THIRD_PARTY.md` and manifest record whose terms
apply and where to get the weights. Splitting them into separate repositories
would only have spread one Apache-2.0 project across six of them.

See [`docs/PLUGINS.md`](docs/PLUGINS.md) to write your own.

## Other character types

A rig template is data, not code. It says which landmarks a character has,
which parts they are cut into, how the parts are layered, and how long each
bone is. Adding an animal or a creature of your own is a new JSON file in
`~/.rigger2d/templates/`:

```bash
autorigger2d templates
```

The humanoid one ships by default. A quadruped is a template away.

## Using it from JavaScript

The service is the whole JavaScript surface:

```typescript
import { createClient } from "@rigger2d/client";

const client = createClient();
const { id } = await client.createProject({ imageBase64 });
await client.run(id, { until: "refine" });
await client.refine(id, [
  { mask: { op: "add", part_id: "upper_arm_l", points: [{ x: 10, y: 20 }], radius: 6 } },
]);
await client.slice(id, { targets: ["native", "godot", "spine"], skinning: "hybrid" });
```

It works in a browser, in Node, and in an Electron app.

## Checks

```bash
.venv/bin/python -m pytest tests -q     # the Python side
cd js && npm run test                   # the JavaScript side
.venv/bin/python tools/license_check.py # the Apache-2.0 promise, enforced
.venv/bin/python tools/schema_check.py  # the contracts, enforced
```

## Licence

The core is **Apache-2.0** -- see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The backends are not part of this repository and carry their own terms. The
`provenance.json` next to every export records which backend produced which
part and the license that backend declared, so you always know what you are
shipping.

Godot is a trademark of the Godot Engine contributors. Spine is a trademark of
Esoteric Software LLC. This project writes files in their formats; it contains
no code from either, and grants no rights in either.
