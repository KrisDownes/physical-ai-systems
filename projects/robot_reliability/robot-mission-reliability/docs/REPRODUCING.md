# Reproducing the Robot Mission Reliability Benchmark

## Requirements

- Python 3.11 or newer
- A shell environment with `make`
- Internet access during dependency installation

Docker is not required.

## Installation

From the `robot-mission-reliability` project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The project dependencies are declared in `pyproject.toml`.

## Run the Tests

```bash
make test
```

The test suite uses Python's built-in `unittest` framework.

## Run the Complete Benchmark

```bash
make benchmark
```

The benchmark executes:

- Four scenarios
- Two software versions
- Five deterministic seeds
- Forty total mission runs

Generated outputs:

- `artifacts/runs/`
- `artifacts/results.csv`
- `docs/scenario_comparison.md`
- `docs/assets/success_rate.svg`
- `docs/assets/cross_track_rmse.svg`
- `docs/assets/severe_slip_v2_seed_0.gif`

## Reproducibility Scope

For the same software version, scenario configuration, and seed, the mission outcomes and calculated metrics are deterministic.

Generated run IDs are UUIDs and intentionally differ between executions. Matplotlib may also generate different SVG metadata or internal element identifiers. Therefore, reproduced files are not expected to be byte-for-byte identical even when their benchmark metrics and rendered plots are equivalent.