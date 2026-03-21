# GPU Kernel Development & Benchmarking

This repository contains implementations, benchmarks, and profiling scripts for RMS Normalization and Flash Attention kernels across multiple frameworks: PyTorch, Triton, Helion, CUTE, and Gluon.

## Repository Structure

- `[framework]_kernels/`: Source code for each framework's kernel implementations.
- `ptx_ir/`: Generated intermediate representations (PTX, LLIR, TTIR) for A100 and T4 architectures.
- `setup-nsys-ncu.sh`: Installs and configures Nsight Systems and Nsight Compute.
- `run_from_top.sh`: Coordinates config generation, benchmarking, and profiling.
- `rms_benchmark_driver.py` / `attn_benchmark_driver.py`: Standalone drivers for specific benchmarks.
- `generate_dumps.sh`: Runs `nsys` or `ncu` profiling on specific kernels.
- `generate_ncu_sweep.py`: Automated NCU profiling sweep over varying M and N dimensions.
- `plot_ncu_results.py`: Generates visualizations from `ncu-reports/` data.

## Run

### 1. Main Script - `run_from_top.sh`

```bash
# Run everything (RMS Norm + Flash Attention)
bash run_from_top.sh all

# Run only RMS Norm or only Flash Attention
bash run_from_top.sh rms
bash run_from_top.sh attn

# Skip benchmarks and only generate the nsys/ncu profiles
bash run_from_top.sh all --profile-only
```

### 2. Benchmark Drivers

```bash
# Run all benchmarks
python3 bench-driver.py

# Run specific benchmark suites (comma-separated)
python3 bench-driver.py --run-benchmark=rms_bench
python3 bench-driver.py --run-benchmark=rms_bench,attn_bench

# Generate a kernel dump (format: <bench-name>-<kernel-name>)
python3 bench-driver.py --generate-kernel-dump rms_bench-torch-rmsnorm --M 1024 --N 1024
python3 bench-driver.py --generate-kernel-dump attn_bench-torch-flash_attn
```

### 3. NCU Profiling Sweep

`generate_ncu_sweep.py` profiles over varying `M` (fixed N=4096) and `N` (fixed M=4096). Reports are saved to `ncu-reports/varying_m/` and `ncu-reports/varying_n/`.

```bash
# Run with default RMS kernels
python3 generate_ncu_sweep.py

# Run for specific kernels
python3 generate_ncu_sweep.py rms_bench-torch-rmsnorm rms_bench-triton-rmsnorm_with_loops
```
