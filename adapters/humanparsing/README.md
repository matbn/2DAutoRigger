# Human parsing (LIP / ATR labels)

A parts backend for [2DAutoRigger](https://github.com/matbn/2DAutoRigger).

Human parsing models label every pixel of a person with a body part
class. They are the strongest option for humanoids and the weakest option for
anything else, which is exactly the split the pipeline expects.

This adapter lives in the core repository, at `adapters/humanparsing/`, because its
own code is Apache-2.0 like the core. What differs is the licence of the
**model**, and no model is shipped either way.

## Licence of the model

This adapter's own code is Apache-2.0. The model it uses is **MIT** and is
**not** distributed here -- see [THIRD_PARTY.md](THIRD_PARTY.md).

Parsing models are trained on human photographs and datasets such as LIP and ATR. Check the dataset licenses before commercial use.

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
ln -s "$PWD" ~/.rigger2d/plugins/humanparsing
# Windows
New-Item -ItemType Junction -Path "$HOME\.rigger2d\plugins\humanparsing" -Target "$PWD"
```

```bash
export RIGGER2D_PLUGINS="/path/to/this/directory"
```

## Weights

```bash
export RIGGER2D_HUMANPARSING_WEIGHTS=/path/to/the/weights
```

`autorigger2d models` will then report `humanparsing.lip` as available, and
`autorigger2d doctor` will tell you if anything is still missing.

## Note on verification

This adapter follows the documented API of its upstream project, but it was
written without the upstream project's dependencies installed. Run it against a
character whose parts you already know and check the result before you trust it.
The probe reports missing dependencies and missing weights honestly, so a
half-installed adapter never pretends to work.
