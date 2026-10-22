# RTMPose (pose estimation)

A pose backend for [2DAutoRigger](https://github.com/matbn/2DAutoRigger).

RTMPose is a fast pose estimator. It runs on the CPU through onnxruntime,
which makes it the practical choice on a machine without a GPU.

This adapter lives in the core repository, at `adapters/rtmpose/`, because its
own code is Apache-2.0 like the core. What differs is the licence of the
**model**, and no model is shipped either way.

## Licence of the model

This adapter's own code is Apache-2.0. The model it uses is **Apache-2.0** and is
**not** distributed here -- see [THIRD_PARTY.md](THIRD_PARTY.md).

The code is Apache-2.0. The ONNX model you point this adapter at has its own terms; use a model you are allowed to use.

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
ln -s "$PWD" ~/.rigger2d/plugins/rtmpose
# Windows
New-Item -ItemType Junction -Path "$HOME\.rigger2d\plugins\rtmpose" -Target "$PWD"
```

```bash
export RIGGER2D_PLUGINS="/path/to/this/directory"
```

## Weights

```bash
export RIGGER2D_RTMPOSE_WEIGHTS=/path/to/the/weights
```

`autorigger2d models` will then report `rtmpose.pose` as available, and
`autorigger2d doctor` will tell you if anything is still missing.

## Note on verification

This adapter follows the documented API of its upstream project, but it was
written without the upstream project's dependencies installed. Run it against a
character whose parts you already know and check the result before you trust it.
The probe reports missing dependencies and missing weights honestly, so a
half-installed adapter never pretends to work.
