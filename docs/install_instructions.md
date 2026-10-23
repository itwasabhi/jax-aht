# Instructions
These instructions target Python 3.11.

Step 1 sets up an environment with the project installed; pick one of the two
options. Steps 2-4 are the same either way.

## 1a. Set up the environment with uv (recommended)

From the repository root:
```bash
uv sync                  # CPU-only Linux and macOS
uv sync --extra cuda12   # Linux with NVIDIA GPUs and CUDA 12
uv sync --extra test     # add pytest, ruff, and the human-data Flask server
```

[uv](https://docs.astral.sh/uv/) creates a `.venv/` with Python 3.11 (pinned in
`.python-version`, downloaded by uv if it is not already present) and installs
the project in editable mode with the dependency set from `pyproject.toml`.

Run project commands with `uv run`, which uses that environment without
activating it:
```bash
uv run scripts/verify_install.py
```
Or activate it once with `source .venv/bin/activate` and use `python` as usual.
The remaining steps are written with a plain `python`; prefix them with `uv run`
if you prefer not to activate.

`uv sync` writes a `uv.lock` with the exact resolved versions. It is gitignored
by default; commit it if you want everyone on the project to get identical
versions.

## 1b. Set up the environment with pip

Create and activate a Python 3.11 environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Then install the project in editable mode from the repository root:
```bash
python -m pip install -e .          # CPU-only Linux and macOS
python -m pip install -e ".[cuda12]" # Linux with NVIDIA GPUs and CUDA 12
```

This installs the dependency set from `pyproject.toml` and puts the project on
the import path, so no `PYTHONPATH` setup is needed.

## 2. Verify the installation

```bash
python scripts/verify_install.py
```

This reports the devices JAX can see and runs the array operations the training
code relies on. The output should end with `All checks passed.`. A default CPU
installation looks like:
```
jax 0.5.3, backend cpu, [CpuDevice(id=0)]
[ok] devices
[ok] matrix multiplication
[ok] QR decomposition
[ok] orthogonal init (as used by agents/)
All checks passed.
```
With the CUDA extra installed, the backend should be `gpu` and the devices should
be listed as `CudaDevice` entries.

## 3. Download evaluation data to get the evaluation agents

```bash
python download_eval_data.py
```

Re-running the script only downloads files that are missing locally; pass `--force` to re-download everything.

## 4. Test the installation by running our IPPO implementation

```bash
python marl/run.py task=lbf/lbf_7x7_nolevels algorithm=ippo/lbf/lbf_7x7_nolevels
```

# Logging to Weights & Biases

Training entrypoints log through `common/wandb_visualizations.py`, which calls
`wandb.init` once per process with the `logger` block of the Hydra config.

Log in once per machine:
```bash
wandb login
```

The configs leave `logger.entity: null`, so wandb resolves the entity the
standard way: `$WANDB_ENTITY` if it is set, otherwise the default entity of the
logged-in user. Nothing is needed to log to your own account.

To log to a shared team instead, set the environment variable:
```bash
export WANDB_ENTITY=aht-project
```

Both the entity and the project can also be overridden per run on the Hydra
command line, which takes precedence over the environment:
```bash
python marl/run.py task=... algorithm=... logger.entity=my-team logger.project=my-project
```

`logger.mode` is set per entrypoint (`online` for `marl`, `offline` for the
others) and is passed to `wandb.init` explicitly, so it takes precedence over
`$WANDB_MODE`. Use `logger.mode=online` to upload, or `logger.mode=disabled` to
turn logging off for a run. Offline runs are uploaded later with
`wandb sync <run-dir>`, and are assigned to the resolved entity at sync time.

# Alternative Manual Installation

If you encounter issues with the editable install, install the pinned
dependencies directly into an activated environment:
```bash
python -m pip install -r requirements.txt
```
On Linux machines with NVIDIA GPUs and CUDA 12, also run
`python -m pip install "jax[cuda12]==0.5.3"`.

This does not install the project itself, so run all commands from the
repository root, or set `PYTHONPATH=/path/to/jax-aht`. Then follow steps 2-4
above.

# Troubleshooting

We provide some basic troubleshooting guidance.

## If JAX is not importable after installation

Confirm that `pip` and `python` point to the same environment:
```
which python
which pip
python -m pip --version
```

If they do not point to the same environment, reinstall using:
```
python -m pip install -e .
```

If `jumanji` fails with `ModuleNotFoundError: No module named 'pkg_resources'`, your environment likely has `setuptools>=81`, which removed `pkg_resources`. Reinstall a compatible version:
```
python -m pip install "setuptools<81" --force-reinstall
python -m pip install -e .
```

## CUDA library conflicts

If you installed the `cuda12` extra, pip installed its own CUDA 12 libraries. If a
system CUDA toolkit is also on the library search path, the two can be mixed at
runtime, which causes segfaults or cuBLAS/cuSolver errors after `jax.devices()`
already reports a GPU. Check with `echo $LD_LIBRARY_PATH`.
If the output is not empty, use:
```bash
export LD_LIBRARY_PATH=""  # so that it defaults to the pip-installed CUDA
```
To make that persistent, add the `export` to your shell profile or to your
environment's activation script (`.venv/bin/activate`), so it applies whenever
the environment is activated.

On macOS, CUDA is not used. If you see CPU devices from `jax.devices()`, that is the expected behavior.

## If video export fails

Some evaluation and test scripts save `.mp4` files through the `ffmpeg` executable. Install it separately if needed.

On macOS:
```
brew install ffmpeg
```

On Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Verify it is available with:
```
ffmpeg -version
```
