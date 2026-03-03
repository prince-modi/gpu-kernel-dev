#!/usr/bin/env python3
"""
Run NCU profiling sweep over M (fixed N) and N (fixed M), matching benchmark configs.
Saves to ncu-reports/varying_m/ and ncu-reports/varying_n/.

Usage:
  python3 generate_ncu_sweep.py [kernel1 kernel2 ...]
  python3 generate_ncu_sweep.py   # uses default rms kernels

Requires: ncu in PATH, run with sudo for ncu if needed
"""

import os
import subprocess
import sys

# Match benchmark: 256*i for i in range(2, 100) - use subset for reasonable NCU time
# 10 points: 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576
SWEEP_VALS = [256 * i for i in [2, 4, 8, 16, 24, 32, 48, 64, 80, 96]]
FIXED_M = 4096
FIXED_N = 4096

OUT_DIR = "ncu-reports"
HELION_SKIP = "1"


def get_rms_kernels():
    """Default RMS kernels, excluding CUTE on Ada."""
    try:
        import torch
        if torch.cuda.get_device_capability() == (8, 9):
            return [
                "rms_bench-helion-helion_rms_kernel",
                "rms_bench-torch-rmsnorm",
                "rms_bench-triton-rmsnorm_with_loops",
                "rms_bench-gluon-rms_norm",
            ]
    except Exception:
        pass
    return [
        "rms_bench-helion-helion_rms_kernel",
        "rms_bench-torch-rmsnorm",
        "rms_bench-triton-rmsnorm_with_loops",
        "rms_bench-cute-cute_rms_norm",
        "rms_bench-gluon-rms_norm",
    ]


def run_ncu(kernel: str, M: int, N: int, out_path: str) -> bool:
    """Run ncu profile for one (M,N) and save to out_path."""
    env = os.environ.copy()
    env["HELION_SKIP_AUTOTUNE"] = HELION_SKIP
    cmd = [
        "ncu", "-f", "-o", out_path,
        "python3", "bench-driver.py",
        "--generate-kernel-dump", kernel,
        "--M", str(M), "--N", str(N),
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {out_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return False


def main():
    kernels = sys.argv[1:] if len(sys.argv) > 1 else get_rms_kernels()
    total = len(kernels) * 2 * len(SWEEP_VALS)
    done = 0

    os.makedirs(f"{OUT_DIR}/varying_m", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/varying_n", exist_ok=True)

    print(f"NCU sweep: {len(kernels)} kernels × 2 sweeps × {len(SWEEP_VALS)} points = {total} runs")
    print(f"Saving to {OUT_DIR}/")

    for kernel in kernels:
        # Varying N (M fixed)
        for n in SWEEP_VALS:
            out_path = f"{OUT_DIR}/varying_n/{kernel}-N{n}-ncu.rep"
            if os.path.exists(out_path + ".ncu-rep"):
                print(f"[{done+1}/{total}] Skip (exists): {kernel} N={n}")
            else:
                print(f"[{done+1}/{total}] {kernel} M={FIXED_M} N={n}")
                run_ncu(kernel, FIXED_M, n, out_path)
            done += 1

        # Varying M (N fixed)
        for m in SWEEP_VALS:
            out_path = f"{OUT_DIR}/varying_m/{kernel}-M{m}-ncu.rep"
            if os.path.exists(out_path + ".ncu-rep"):
                print(f"[{done+1}/{total}] Skip (exists): {kernel} M={m}")
            else:
                print(f"[{done+1}/{total}] {kernel} M={m} N={FIXED_N}")
                run_ncu(kernel, m, FIXED_N, out_path)
            done += 1

    print(f"Done. Reports in {OUT_DIR}/varying_m/ and {OUT_DIR}/varying_n/")


if __name__ == "__main__":
    main()
