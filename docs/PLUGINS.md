# Writing a backend

A backend is a plugin. It has its own package, its own license and, when it
needs one, its own model; the user installs it and accepts its terms. The core
never downloads a weight, never names a model's license, and never ships a
model.

Five adapters ship with the core, under `adapters/`. They are there because
their own code is Apache-2.0 like the core -- what differs is the licence of
the *model*, which is not shipped either way. A third party backend is welcome
to live in its own repository instead; the contract is identical.

## The contract

Any object with these members is a backend:

```python
class Adapter(Protocol):
    id: str                              # "sam3.promptable"
    version: str
    capabilities: frozenset[Task]        # MATTE | PARTS | POSE | CLASSIFY | INPAINT

    def probe(self, ctx: AdapterContext) -> Probe: ...
    def run(self, task: Task, payload: Any, ctx: AdapterContext) -> AdapterResult: ...
```

`probe` is the important one. The core shows its result to the user verbatim,
so it should say what the user has to do -- install a package, fetch weights,
accept a license -- rather than a bare status code:

```python
def probe(self, ctx):
    if importlib.util.find_spec("torch") is None:
        return failed_probe(
            self.id, self.version, ProbeStatus.MISSING_DEPS,
            "install the requirements first: pip install -r requirements.txt",
            self.capabilities,
        )
    if not self.weights.exists():
        return failed_probe(
            self.id, self.version, ProbeStatus.MISSING_WEIGHTS,
            f"point the adapter at the weights with {self.env_var}",
            self.capabilities,
        )
    return ok_probe(self.id, self.version, self.capabilities,
                    license_spdx="LicenseRef-Meta-SAM", license_url="...")
```

A backend that is not ready must never pretend to be. The core falls through to
the next candidate, and the user sees exactly why the one they wanted was
skipped.

## The tasks

| Task | You get | You return |
| --- | --- | --- |
| `MATTE` | the image | one mask: the character against the background |
| `PARTS` | the image, the matte, the landmarks, the template | one mask per part |
| `POSE` | the image, the character's bounding box | the landmarks, with a confidence each |
| `CLASSIFY` | the image | what kind of character it is |
| `INPAINT` | the image, a mask | the image with the area behind the mask filled in |

Masks are written as 8-bit grayscale PNGs, 255 inside the part. Landmarks are
in image pixels.

## The part names

The part ids have to match the template. The humanoid template uses:

```
head  neck  torso  pelvis
upper_arm_l  lower_arm_l  hand_l
upper_arm_r  lower_arm_r  hand_r
upper_leg_l  lower_leg_l  foot_l
upper_leg_r  lower_leg_r  foot_r
```

A backend that can only produce some of them is fine: it returns the masks it
can and the pipeline works with what it gets. A backend that produces *more*
than the template asks for -- five legs, say -- is also fine, and this is
exactly where the character type matters: the fallback chain scores a result
against the template it is being used with, so five legs is right for a
quadruped and wrong for a humanoid.

## The manifest

Every plugin ships a `manifest.json`:

```json
{
  "id": "sam3.promptable",
  "version": "0.1.0",
  "tasks": ["parts"],
  "part_vocabulary": ["head", "torso"],
  "license": {
    "spdx": "LicenseRef-Meta-SAM",
    "url": "https://github.com/facebookresearch/sam3",
    "requires_acceptance": true
  },
  "weights": {
    "source": "https://huggingface.co/facebook/sam3",
    "gated": true,
    "terms_url": "https://huggingface.co/facebook/sam3/blob/main/LICENSE",
    "note": "Request access and accept the Meta SAM License before downloading."
  },
  "requires": { "python": ">=3.12", "packages": ["torch>=2.7"] },
  "entry_point": "rigger2d_adapter_sam3:adapter",
  "priority": 10
}
```

`priority` decides the order of the fallback chain: lower runs first. A
backend that is usually the best answer for a character type should have a low
number; one that is a specialist should have a high one.

The manifest is checked against `schemas/plugin-manifest.schema.json` in CI.

## Installing it

Three ways, in the order the core looks:

1. A pip-installed package that registers an entry point in the
   `rigger2d.adapters` group.
2. `$RIGGER2D_PLUGINS` -- a list of directories, separated by the platform's
   path separator.
3. `~/.rigger2d/plugins/` -- a directory of plugins.
4. `./plugins/` -- plugins that live with the project.

```bash
export RIGGER2D_PLUGINS="/path/to/your-adapter"
autorigger2d models
```

## The shipped adapters

`adapters/` holds five scaffolds: `sam3`, `birefnet`, `humanparsing`,
`rtmpose`, `lama`. Each is a complete contract implementation whose `probe`
reports missing dependencies and missing weights honestly, so a half-installed
adapter never pretends to work. They were written without their upstream
dependencies installed and could not be executed here -- run one against a
character you know before you trust it.

## The reference implementation

`examples/mock_adapter/` is a working plugin that performs no inference and
needs nothing installed. It is the shortest complete example, and the test
suite exercises it, so if your plugin is discovered and loaded the same way,
the rest is inference and plumbing.

Copy it, rename it, and replace the `run` method.

## A note on honesty

An adapter's `probe` is what the user sees before anything runs. It should say
what to do -- install a package, fetch weights, accept a licence -- not just
that something is wrong. An adapter that reports `missing_weights` with a
sentence the user can act on is worth more than one that reports `error`.

The same goes for results. A backend that estimates rather than measures should
say so in the confidence it returns; the core treats a low confidence as a
reason to ask for the refinement step, and a high one as a reason to trust the
automatic result.
