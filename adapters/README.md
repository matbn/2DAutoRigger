# The shipped adapters

Five backend scaffolds. Each one is a complete implementation of the adapter
contract in `python/rigger2d/adapters/base.py`.

| Directory | Task | Model licence | Weights |
| --- | --- | --- | --- |
| [`sam3/`](sam3) | parts | Meta SAM License | **gated** — request access |
| [`birefnet/`](birefnet) | matte | MIT (code), own terms on the weights | open |
| [`humanparsing/`](humanparsing) | parts | MIT (code) | open |
| [`rtmpose/`](rtmpose) | pose | Apache-2.0 | open |
| [`lama/`](lama) | inpaint | Apache-2.0 | open |

## Why they are here and not in their own repositories

They were separate repositories to begin with, on the argument that a single
repository would force one LICENSE onto backends with very different terms.

That argument does not survive contact with what is actually in them. The
adapter *code* is Apache-2.0 -- all five, the same as the core. The terms that
differ belong to the **model weights**, and no weights are shipped at all: they
are downloaded by the user, from the upstream source, after the user has read
the terms. Each adapter's `THIRD_PARTY.md` and `manifest.json` record whose
terms apply and where to get them.

So there was nothing to separate, and six repositories to clone, install and
keep in step was the only thing the split actually produced.

## The rule that is enforced

The core must not depend on machine learning. `tools/license_check.py` scans
`python/rigger2d` for `torch`, `onnxruntime`, `transformers` and friends, and
fails the build if the core imports one. It does **not** scan this directory,
because depending on torch is exactly what an adapter is for.

The point is that installing the core cannot pull a model's licence onto it,
and uninstalling an adapter cannot break the core.

## They are scaffolds, and they say so

Each adapter follows the documented API of its upstream project, but it was
written **without that dependency installed**, so it could not be run when it
was written. Their `probe` reports missing dependencies and missing weights
honestly, which means a half-installed adapter never pretends to be ready.

Before relying on one, run it against a character whose parts you already know
and check the result.

## Trying them

Nothing here is installed by default, so none of them can run in a fresh
checkout. You can still see them being honest about it:

```bash
export RIGGER2D_PLUGINS="$(ls -d adapters/*)"     # bash
autorigger2d models
```

Each one answers `missing_deps` with the exact packages to install.

## Writing your own

Follow [`../docs/PLUGINS.md`](../docs/PLUGINS.md). A third party backend belongs
in its own repository, with its own licence; the contract is identical either
way. `../tools/scaffold_adapters.py` is the generator that produced these five,
and it is meant to be read and edited rather than trusted.
