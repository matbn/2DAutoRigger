# Architecture

This is the design, and the reasons behind it. For how to *use* it, read the
[README](README.md); for how to *extend* it, read [docs/PLUGINS.md](docs/PLUGINS.md).

## The shape of the thing

```
        JavaScript                             Python
   ┌────────────────────┐            ┌───────────────────────────┐
   │  the editor        │  HTTP/JSON │  the service              │
   │  @rigger2d/client  │ ─────────► │  the pipeline             │
   │                    │ ◄───────── │  the adapter registry     │
   └────────────────────┘            │  the emitters             │
                                     └────────────┬──────────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │  installed plugins          │
                                    │  (your machine, your terms)  │
                                    └────────────────────────────┘
```

Two languages, one boundary. The boundary is JSON over HTTP, and it is the
same boundary the plugins sit behind.

## Why the core has no machine learning

The requirement was a core that is Apache-2.0, with the models installable and
their terms owned by the user. That is not only a licensing decision; it
changes the design.

If the core depended on a model, then the core's license would depend on that
model's license, and "installable" would become "vendored". So the core knows
about models only as an *interface*: five tasks, a probe, a run. Everything
above that line -- the pipeline, the scoring, the fallback chain -- is written
against the interface and has never heard of SAM.

The five adapters that ship with the core live in `adapters/`, and that is not
a contradiction. Their own code is Apache-2.0, exactly like the core; what
differs is the licence of the *model*, and no model is shipped either way. Each
adapter's `THIRD_PARTY.md` and manifest say whose terms apply and where to
fetch the weights. The rule the licence gate enforces is narrower and sharper
than "no machine learning in the repository": **the core must not import it.**
`tools/license_check.py` scans `python/rigger2d` for `torch`, `onnxruntime` and
friends and fails the build; it deliberately does not scan `adapters/`, because
depending on torch is what an adapter is for.

The cost of that decision is that the core has to be useful with no model at
all. That is why there is a geometric fallback, and why the fallback reports a
confidence of 0.25 rather than pretending to have found the joints.

## The pipeline

```
s0 ingest    s1 matte    s2 type    s3 pose    s4 parts
                                              │
                                    ╔═════════▼═════════╗
                                    ║   REFINE GATE      ║
                                    ╚═════════╤═════════╝
                                              ▼
s5 layers    s6 skeleton    s7 mesh    s8 weights    s9 emit
```

Every stage is `Document -> Document`. Nothing is global, nothing is cached in
memory between runs, and `document.json` on disk is always the truth. That is
what makes the pipeline resumable (`run --until refine`, then `run --until
all`), what makes the cache meaningful, and what makes the tests possible
without a service.

### The refine gate

The gate is the whole point of the design. Segmentation is the step most likely
to be slightly wrong, and a slightly wrong mask becomes a visibly wrong
character once the parts are cut apart and animated.

So the pipeline stops after `parts` and waits. The corrections are an **edit
log**, not re-rasterised masks:

```json
{"mask": {"op": "add", "part_id": "upper_arm_l",
          "points": [{"x": 12, "y": 30}], "radius": 6.0}}
```

An edit log is compact, undoable, replayable, and -- the important part -- it
leaves the original segmentation intact. The user can always throw their
corrections away and go back.

Three operations, because three are enough:

| op | what it does |
| --- | --- |
| `add` | paint pixels into a part |
| `erase` | take pixels out of a part |
| `assign` | move pixels from whatever had them to this part |

`assign` is the one that matters in practice: parts overlap, and the pixel
that belongs to the arm rather than the torso has to be *moved*, not copied.

## The data model

`python/rigger2d/contracts/models.py` is the single definition, and
`schemas/` is its JSON Schema face. The TypeScript in
`js/packages/schema` is the third face of the same thing. `tools/schema_check.py`
fails the build when they drift.

Coordinate convention, everywhere in the core:

* units are **image pixels**;
* the origin is the **top-left** of the image;
* **y grows downwards**.

Godot's 2D axis points down, so the Godot emitter needs no conversion. Spine's
points up, so the Spine emitter converts at the edge. Converting in one place,
at the boundary, is much easier to reason about than a convention nobody
remembers.

## Bones follow the size of the part

This is the rule that makes the output a *rig* rather than a traced outline.
The template says:

```json
{"id": "upper_arm_l", "parent": "shoulder_l",
 "head": "shoulder_l", "tail": "elbow_l",
 "length": "from_mask", "part": "upper_arm_l"}
```

`"length": "from_mask"` means the bone runs from the shoulder towards the
elbow, but how far it goes is decided by how big the upper arm is *in this
drawing*. A character with a long arm gets a long upper arm bone; a chunky
character gets a chunky one. `"length": "landmark"` is the other option, for
the bones whose length is structural rather than anatomical.

## Pixels become vertices

Every engine the core targets binds **vertices** to bones, not pixels. So the
mask has to become a mesh:

```
mask -> contour -> simplify -> interior points -> triangulate
```

The contour is built by following the *edges* of the mask, not the pixels.
This is not a detail: an arm is a few pixels wide, and a pixel walker happily
closes a loop around the tip of it and reports a two pixel outline. Edge
following cannot, because every set pixel contributes the edges that separate
it from the background, and those edges link into exactly one loop per blob.

` scipy` is used for the triangulation when it is installed, and ear clipping
is the fallback when it is not. The difference is how smoothly a limb bends,
not whether the rig is valid.

## Occlusion

The parts of a character overlap in the drawing: the arm is drawn on top of the
torso. Cut them apart and the area the arm used to cover is missing from the
torso, so rotating the arm reveals a hole.

The fix is a **fill layer**: the part's own mask grown a little and painted
with the colour of its edge, placed directly behind the part it belongs to.
It is flat, it does not deform, and it is emitted as a sprite rather than a
polygon.

Two rules keep it honest. It may only grow into the parts drawn *above* it,
and only within the character -- otherwise it would paint a halo around the
character, or the wrong colour over the part below it. And because it is
drawn *behind* the part it belongs to, it is completely hidden at rest: the
picture is unchanged, and it only shows when the limb moves away and the hole
appears.

That is what makes the result checkable. The layers have to composite back
into the original picture, pixel for pixel, and `tests/test_slicing.py` says
exactly that.

## Choosing a backend

```
probe -> run -> score -> accept, or fall through to the next
```

The scoring is deliberately simple and explainable, because a number nobody
can reason about is a number nobody can trust:

| task | what the score measures |
| --- | --- |
| matte | how much of the frame is covered, how well the edge follows the drawing, how many holes |
| parts | how much of the character the parts cover, how much they overlap, how the sizes compare with the template, whether the joints are inside the right part |
| pose | the mean confidence, and whether the landmarks are inside the character |

And the score is **weighted by the character type**: a quadruped template
expects four limbs and a tail, a humanoid expects two arms and two legs. A
backend that returns five blobs is right for one and wrong for the other.

`POST /api/projects/{id}/compare` runs every candidate on the same image and
returns the score table, so the user can see why one was chosen and override
it.

## The emitters

The core never ships engine code. It writes files in the engines' own formats,
and the engines' own importers do the rest.

### Godot

A `.tscn` scene with a `Skeleton2D`, a tree of `Bone2D` nodes, and one
`Polygon2D` per part.

The details that matter, and why:

* `Transform2D(a, b, c, d, e, f)` is parsed by Godot as `columns[0] = (a, b)`,
  `columns[1] = (c, d)`, `columns[2] = (e, f)`. For a rotation of `t` that is
  `(cos t, sin t, -sin t, cos t)`.
* A `Bone2D`'s `rest` is **relative to its parent bone**, so the bones have to
  be nested for them to inherit from each other.
* The skeleton's bone order is the depth-first order of the `Bone2D` nodes in
  the scene tree, which is why the emitter nests them rather than flattening
  them.
* `Polygon2D.bones` is an `Array` of `[bone path, weights]` pairs. The path is
  relative to the `Skeleton2D`, and the weights array holds one float per
  vertex.

### Spine

`skeleton.json` plus an atlas.

### Unity

A `Character.prefab` inside `Assets/`, plus the layer images and their `.meta`
files.

A prefab is a YAML file, so this is written from Python like every other
emitter. Nothing in this repository depends on a Unity installation and no
Unity-generated file is ever committed: the emitter *writes* Unity files, it
does not contain any. The format was learned once, by saving a prefab from the
editor and reading it back.

The details that matter:

* A prefab is a flat list of documents. Each object is a `GameObject` (`!u!1`)
  followed by its components -- a `Transform` (`!u!4`) and, for a body part, a
  `SpriteRenderer` (`!u!212`).
* `&<id>` after the class tag is the object's file id, and everything refers to
  everything else by that number. The graph is therefore built before it is
  written, and a test walks it back to the single root to prove it is a tree.
* Only the root has `m_Father: {fileID: 0}`.
* A sprite is referenced as `{fileID: 21300000, guid: <texture guid>, type: 3}`.
  That `21300000` is the fixed file id of the single sprite a texture importer
  in Sprite mode produces -- which is the whole reason this can be written
  without asking Unity anything. The GUIDs are ours, and the `.meta` files that
  carry them are written alongside the textures. `textureType: 8` (Sprite) and
  `spriteMode: 1` (Single) both have to be set, or Unity imports the picture as
  a plain texture, no sprite exists, and every reference silently resolves to
  nothing.
* `m_LocalRotation` is a quaternion, so a 2D rotation about Z is
  `(0, 0, sin(t/2), cos(t/2))`.

The result is a cutout rig: each part is a sprite on a bone, and animating the
bones moves the parts. Mesh deformation is not exported, because it would need
`com.unity.2d.animation`, which is not part of a default Unity install.
 Spine's own format is JSON, so nothing else is
needed: the user imports the file into Spine and the layers appear as
attachments.

Spine is y-up, so the emitter converts. The attachment format used is:

* `vertices` -- the positions, in the slot's bone space;
* `uvs` -- the texture coordinates inside the atlas region;
* `triangles` -- the triangle indices;
* `weights` -- optional, one `boneName:weight` per vertex. Present means the
  mesh is weighted; absent means it is rigid.

## What is not in v1

Honest scope, because pretending otherwise wastes everyone's time:

| | status |
| --- | --- |
| The pipeline, the refine gate, the layers, the bones, the mesh, the weights | done |
| Godot and Spine exports | done |
| The core fallback, the plugin contract, the registry, the fallback chain | done |
| The editor, the client, the service | done |
| Real model backends | scaffolds only -- they need your machine and your terms |
| Unity (cutout) | done, written from Python like the others |
| Animation and clips | not started |
| Quadruped and other templates | the mechanism is there; the content is not |
| Live2D | out of scope: it needs a real runtime and a different data model |

### Why Unity is not in v1

The three ways to get a rig into Unity all cost something:

* **A `.prefab` written by hand.** The format is YAML, and while it is
  stable enough in practice, it is not a published contract, and it changes
  between Unity versions. The layer images also need `.meta` files with GUIDs
  we would have to forge, because the prefab has to reference the textures
  before Unity has imported them.
* **A C# Editor script.** The most robust option by a wide margin, because it
  uses the public API -- but it is C#, and this project is Python and
  JavaScript.
* **A runtime script.** Also C#.

The decision was made with the "Python and JavaScript only" constraint in
force, and the honest consequence is that the `.prefab` route is the only one
left, and it is the one most likely to rot. It is better to find out what
Unity path the user actually wants -- `com.unity.2d.animation`, built-in
`SkinnedMeshRenderer`, or a plain sprite hierarchy -- than to build a fragile
generator for the wrong one.

The canonical `rig.json` and the layer PNGs are engine independent, so nothing
is lost by waiting: the Unity exporter only has to read a file that already
exists.
