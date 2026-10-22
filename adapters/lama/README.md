# LaMa (inpainting, fills what a limb hides)

A inpaint backend for [2DAutoRigger](https://github.com/matbn/2DAutoRigger).

LaMa is an inpainting model. The pipeline uses it to fill in the area a limb
hides, so that rotating an arm does not reveal a hole in the torso.

This adapter lives in the core repository, at `adapters/lama/`, because its
own code is Apache-2.0 like the core. What differs is the licence of the
**model**, and no model is shipped either way.

## Licence of the model

This adapter's own code is Apache-2.0. The model it uses is **Apache-2.0** and is
**not** distributed here -- see [THIRD_PARTY.md](THIRD_PARTY.md).

Apache-2.0. The model weights are published by the LaMa authors; check the repository for their terms.

## Install

```bash
pip install -r requirements.txt
```

Then make the adapter visible to the core, either by installing it:

```bash
pip install -e .
```

or by pointing the plugin path at this directory:

```bash
# Linux/macOS
export RIGGER2D_PLUGINS="$PWD"
# Windows
$env:RIGGER2D_PLUGINS = "$PWD"
```

or by adding it to your plugin folder:

```bash
# Linux/macOS
ln -s "$PWD" ~/.rigger2d/plugins/lama
# Windows
New-Item -ItemType Junction -Path "$HOME\.rigger2d\plugins\lama" -Target "$PWD"
```

```bash
export RIGGER2D_PLUGINS="/path/to/this/directory"
```

## Weights

```bash
export RIGGER2D_LAMA_WEIGHTS=/path/to/the/weights
```

`autorigger2d models` will then report `lama.inpaint` as available, and
`autorigger2d doctor` will tell you if anything is still missing.

## Note on verification

This adapter follows the documented API of its upstream project, but it was
written without the upstream project's dependencies installed. Run it against a
character whose parts you already know and check the result before you trust it.
The probe reports missing dependencies and missing weights honestly, so a
half-installed adapter never pretends to work.
