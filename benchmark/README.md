# Benchmark Scripts

This directory contains small benchmark utilities for TVM/Relax models and workloads used in this repository.

## Contents

- `export_mobilenet_v3_small.py`: Exports a MobileNetV3-Small model for use with TVM/Relax.
- `memory_usage_rss.py`: Measures the resident set size (RSS) memory usage of running workloads.
- `relax_estimate_memory_usage.py`: Estimates memory usage of Relax programs statically.
- `relax_vm_profile.py`: Profiles Relax programs using the Relax VM.
- `relax_vm_time_evaluator.py`: Measures execution time of Relax VM functions.

## Prerequisites

- A working TVM build under `3rdparty/tvm` (see its `README.md` for setup).
- Python environment with the same dependencies as the main project (e.g., use `uv` or `pip` as in the top-level instructions).

## Basic Usage

From this directory, you can run any script with your configured Python or `uv`:

```bash
cd benchmark

# Example: profile a Relax workload
uv run relax_vm_profile.py

# Example: estimate memory usage
uv run relax_estimate_memory_usage.py
```
