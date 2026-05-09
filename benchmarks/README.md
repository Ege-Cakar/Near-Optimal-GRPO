# Auxiliary Procgen and MiniGrid Benchmarks

This folder is separate from the Mario platformer suite. It gives cleaner, actively used procedural benchmarks for the same warm-start amplification question.

## Setup

```bash
uv sync --extra benchmarks
```

For Procgen on Linux/x86_64 or an x86_64/Rosetta Python:

```bash
uv sync --extra benchmarks --extra procgen
```

`procgen==0.10.7` does not provide a native macOS arm64 distribution, so this checkout can still run MiniGrid locally on Apple Silicon while Procgen is better run on a Linux/x86_64 cluster.

## Procgen Through Rosetta

The current local `.venv` is arm64, so it cannot install the x86_64 Procgen wheel. Use a separate Rosetta shell and a separate venv:

```bash
arch -x86_64 zsh
arch
python3 -m venv .venv-procgen-x86
source .venv-procgen-x86/bin/activate
pip install -U pip uv
uv sync --active --extra benchmarks --extra procgen
uv run --active python benchmarks/run_benchmarks.py --quick --env procgen_coinrun
```

`arch` should print `i386` in the Rosetta shell. Keep this venv separate from `.venv`; mixing arm64 and x86_64 packages will break binary wheels.

## Run

```bash
uv run python benchmarks/run_benchmarks.py --quick --env minigrid_doorkey
uv run python benchmarks/run_benchmarks.py --quick --env procgen_coinrun
uv run python benchmarks/run_benchmarks.py --quick --env procgen_jumper
```

Omit `--quick` for the longer config in `benchmarks/config.yaml`.

Outputs are under `results/benchmarks/<env>/csv`, `results/benchmarks/<env>/figures`, and `results/benchmarks/<env>/checkpoints`.

## What This Runs

- `minigrid_doorkey`: exact BFS planner supplies imitation data on fully observable DoorKey grids by default, then GRPO freezes the CNN/MLP representation and updates only the final linear head.
- `procgen_coinrun` and `procgen_jumper`: RGB Procgen observations with a small previous-action auxiliary vector. The warm-start stage uses policy-gradient updates plus a simple right/jump scripted controller when enabled in config.

The Procgen controller is not an optimal expert. If a run does not reach the requested measured held-out `p0`, the script raises instead of pretending GRPO has a success signal to amplify.
