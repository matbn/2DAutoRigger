# Contributing

Thanks for wanting to help. The short version:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev,mesh]"
.venv/bin/python -m pytest tests -q
```

And on the JavaScript side:

```bash
cd js
npm install
npm run test
npm run build
```

## Before you push: check it in a clean environment

A working directory that has been developed in is a superset of what the
project declares. Something installed by hand during a session will pass
locally and fail in CI — that is exactly how `jsonschema`, used by
`tools/schema_check.py`, once went undeclared and broke the build.

The two commands below are the ones CI runs. If they pass in a fresh
environment, CI will pass:

```bash
uv venv /tmp/clean
uv pip install --python /tmp/clean/bin/python -r requirements.txt
uv pip install --python /tmp/clean/bin/python -e ".[dev,mesh]"
/tmp/clean/bin/python -m pytest tests -q
/tmp/clean/bin/python tools/license_check.py
/tmp/clean/bin/python tools/schema_check.py
```

If you add an import anywhere outside `adapters/`, add its distribution to
`requirements.txt` (runtime) or to the `dev` extra (tests and tools).

## The rules that matter

**The core stays free of machine learning.** No `torch`, no `transformers`, no
`onnxruntime` in `python/rigger2d`. Backends are plugins, in their own
repositories, with their own licenses. `tools/license_check.py` fails the build
if this is broken, and it also fails the build if a weight file is committed or
a plugin manifest does not declare its license.

**Every stage is a pure function.** Read the document, do one job, write the
document back. No globals, no state that lives outside `document.json`. That is
what makes the pipeline resumable and the tests possible.

**The schema is the contract.** `schemas/` describes the manifests, the
templates and the rig. `tools/schema_check.py` fails the build when the code
and the schema drift apart -- a field the code reads but the schema forbids, or
a type the code never handles.

**Say when you are guessing.** The core fallback estimates the joints from the
template's proportions and reports a confidence of 0.25, because that is what it
is doing. A backend that knows it is guessing is more useful than one that
does not.

## The layout

| Path | What lives there |
| --- | --- |
| `python/rigger2d/contracts/` | the data model, shared with the TypeScript side |
| `python/rigger2d/adapters/` | the plugin contract, the registry, the fallback chain |
| `python/rigger2d/pipeline/` | the stages, the session, the refine step |
| `python/rigger2d/rigging/` | the skeleton, the mesh, the weights, the layers |
| `python/rigger2d/emitters/` | the Godot and Spine writers |
| `python/rigger2d/server/` | the HTTP service |
| `js/packages/` | the client, the shared types, the rig maths |
| `js/apps/editor/` | the refine editor |
| `adapters/` | the five shipped backend scaffolds, one directory each |
| `examples/mock_adapter/` | a reference plugin, with no dependencies at all |
| `tools/` | the licence gate, the schema gate, the adapter scaffolder |

## Adding a backend

Follow [`docs/PLUGINS.md`](docs/PLUGINS.md). A third party backend belongs in
its own repository, with its own licence, and is installed by the user.
`tools/scaffold_adapters.py` writes the five that ship with the core, into
`adapters/`; it is there to be edited, not to be run blindly.

The one rule that holds for every backend: **the core must not depend on it.**
`tools/license_check.py` scans `python/rigger2d` for imports of `torch`,
`onnxruntime` and friends and fails the build if it finds one. It does not scan
`adapters/`, because depending on torch is what an adapter is for.

## Tests

`pytest` for the Python side, `vitest` for the JavaScript side. The synthetic
character in `tests/fixtures.py` is built from the rig template, so the parts
of the character are known exactly and the pipeline can be checked against them
rather than against a picture of a picture.

If you change the template, the fixture follows it automatically -- which also
means the segmentation test tells you whether the change made the template
better or worse.
