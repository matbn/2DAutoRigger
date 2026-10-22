# SAM 3 (promptable segmentation)

A parts backend for [2DAutoRigger](https://github.com/matbn/2DAutoRigger).

SAM 3 is Meta's promptable segmentation model: you ask it for a concept
such as "left arm" and it tells you where that is in the picture.

This adapter lives in the core repository, at `adapters/sam3/`, because its
own code is Apache-2.0 like the core. What differs is the licence of the
**model**, and no model is shipped either way.

## Licence of the model

This adapter's own code is Apache-2.0. The model it uses is **LicenseRef-Meta-SAM** and is
**not** distributed here -- see [THIRD_PARTY.md](THIRD_PARTY.md).

The weights are gated. Request access on Hugging Face and accept the Meta SAM License before downloading. This adapter never downloads them for you.

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
ln -s "$PWD" ~/.rigger2d/plugins/sam3
# Windows
New-Item -ItemType Junction -Path "$HOME\.rigger2d\plugins\sam3" -Target "$PWD"
```

```bash
export RIGGER2D_PLUGINS="/path/to/this/directory"
```

## Weights

```bash
export RIGGER2D_SAM3_WEIGHTS=/path/to/the/weights
```

`autorigger2d models` will then report `sam3.promptable` as available, and
`autorigger2d doctor` will tell you if anything is still missing.

## Note on verification

This adapter follows the documented API of its upstream project, but it was
written without the upstream project's dependencies installed. Run it against a
character whose parts you already know and check the result before you trust it.
The probe reports missing dependencies and missing weights honestly, so a
half-installed adapter never pretends to work.
