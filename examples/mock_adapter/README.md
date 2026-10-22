# The reference plugin

This is what a backend adapter looks like. It performs no inference at all: it
copies the character matte into one mask per part, using the template's own
proportions. It exists so that the plugin contract can be exercised -- by the
tests, and by anyone writing a real adapter who wants a working example to
copy.

A real adapter replaces the `run` method and nothing else.

## Try it

```bash
export RIGGER2D_PLUGINS="$PWD/examples/mock_adapter"
autorigger2d models
```

The mock shows up as `example.mock`. Its probe reports that it is available,
because it genuinely is: it needs nothing installed and no weights.

## The contract in full

An adapter is any object with:

| member | what it is |
| --- | --- |
| `id` | a unique string, e.g. `"sam3.promptable"` |
| `version` | a version string |
| `capabilities` | a set of `Task` values it can perform |
| `probe(ctx)` | returns a `Probe` saying whether it can run right now |
| `run(task, payload, ctx)` | does the work and returns an `AdapterResult` |

`probe` is the important one. It is what the core shows the user, so it should
say what the user has to do -- install a package, fetch weights, accept a
licence -- rather than a bare status code.

See [`python/rigger2d/adapters/base.py`](../../python/rigger2d/adapters/base.py)
for the full contract, and
[`docs/PLUGINS.md`](../../docs/PLUGINS.md) for how to write a real one.
