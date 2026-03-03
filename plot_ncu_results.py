#!/usr/bin/env python3
"""
Plot NCU (Nsight Compute) profiling results from .ncu-rep files.

If ncu-reports/varying_m/ and ncu-reports/varying_n/ exist, plots metrics over M and N
(one line per implementation). Otherwise falls back to single-file bar charts.

Usage:
  python3 plot_ncu_results.py              # use ncu-reports/ if present, else cwd
  python3 plot_ncu_results.py [dir]        # specify directory with ncu-rep files

Requires: ncu in PATH, matplotlib
"""

import csv
import subprocess
import re
import glob
import os
import sys
from io import StringIO
from collections import defaultdict

import matplotlib.pyplot as plt


# Setup/randn kernels to exclude (not the RMS workload)
SETUP_KERNEL_PATTERNS = ("distribution", "normal_kernel", "normal_and_transform")

# Implementation name mapping from filename
IMPL_NAMES = {
    "helion": "Helion",
    "torch": "Torch",
    "triton": "Triton",
    "gluon": "Gluon",
    "cute": "CUTE",
}
EXCLUDE_IMPLS = {"Torch"}

# (metric_name, unit, ylabel, subfolder)
METRIC_CONFIG = [
    # occupancy
    ("Achieved Occupancy", "%", "Achieved Occupancy (%)", "occupancy"),
    ("Theoretical Occupancy", "%", "Theoretical Occupancy (%)", "occupancy"),
    ("Achieved Active Warps Per SM", "warp", "Active Warps/SM", "occupancy"),
    # throughput
    ("Memory Throughput", "%", "Memory Throughput (%)", "throughput"),
    ("Compute (SM) Throughput", "%", "Compute (SM) Throughput (%)", "throughput"),
    ("DRAM Throughput", "%", "DRAM Throughput (%)", "throughput"),
    ("L1/TEX Cache Throughput", "%", "L1/TEX Cache Throughput (%)", "throughput"),
    ("L2 Cache Throughput", "%", "L2 Cache Throughput (%)", "throughput"),
    # duration
    ("Duration", "us", "Duration (µs)", "duration"),
    ("Elapsed Cycles", "cycle", "Elapsed Cycles", "duration"),
    # launch config
    ("Block Size", "", "Block Size", "launch"),
    ("Grid Size", "", "Grid Size", "launch"),
    ("Registers Per Thread", "register/thread", "Registers/Thread", "launch"),
    ("Shared Memory Configuration Size", "Kbyte", "Shared Mem (KB)", "launch"),
]
KEY_METRICS = [(m[0], m[1], m[2]) for m in METRIC_CONFIG]  # for extraction
# Occupancy is bounded [0, 100]; NCU can report >100% due to measurement quirks
OCCUPANCY_CAP = 100.0


def parse_ncu_csv(csv_text: str) -> list[dict]:
    """Parse ncu --csv output into list of row dicts."""
    reader = csv.DictReader(StringIO(csv_text), quoting=csv.QUOTE_MINIMAL)
    return list(reader)


def metric_to_safe_name(metric_name: str) -> str:
    """Convert metric name to filesystem-safe string (no /, spaces, parens)."""
    return metric_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()


def is_setup_kernel(kernel_name: str) -> bool:
    return any(p in kernel_name for p in SETUP_KERNEL_PATTERNS)


def get_impl_name(filepath: str) -> str:
    """Extract implementation label from filename."""
    basename = os.path.basename(filepath)
    for key, label in IMPL_NAMES.items():
        if f"-{key}-" in basename:
            return label
    m = re.search(r"rms_bench-(\w+)-", basename)
    return m.group(1).capitalize() if m else basename[:20]


def extract_metrics_for_kernel(rows: list[dict], kernel_name: str) -> dict:
    """Extract key metrics for a kernel from the details CSV rows."""
    metrics = {}
    for r in rows:
        if r.get("Kernel Name") != kernel_name:
            continue
        metric_name = r.get("Metric Name", "")
        if metric_name not in [m[0] for m in KEY_METRICS]:
            continue
        val = r.get("Metric Value", "")
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        try:
            metrics[metric_name] = float(val)
        except (ValueError, TypeError):
            pass
    return metrics


def load_ncu_report(filepath: str) -> list[dict]:
    """Run ncu --import --csv and return parsed rows."""
    result = subprocess.run(
        ["ncu", "--import", filepath, "--csv", "--page", "details"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ncu failed for {filepath}: {result.stderr}")
    return parse_ncu_csv(result.stdout)


def extract_rms_metrics(rows: list[dict]) -> dict:
    """Extract metrics for primary RMS kernel (excludes setup, picks max Duration)."""
    kernels = list({r["Kernel Name"] for r in rows if r.get("Kernel Name")})
    rms_kernels = [k for k in kernels if not is_setup_kernel(k)]
    if not rms_kernels:
        return {}

    best_metrics = {}
    best_duration = -1
    for k in rms_kernels:
        m = extract_metrics_for_kernel(rows, k)
        if m and m.get("Duration", 0) > best_duration:
            best_duration = m["Duration"]
            best_metrics = m
    return best_metrics


def collect_sweep_data(report_dir: str, dim: str) -> dict:
    """
    Collect metrics from sweep reports. dim is 'M' or 'N'.
    Returns {impl: {dim_value: {metric: value}}}
    """
    pattern = f"{report_dir}/*-{dim}*-ncu.rep.ncu-rep"
    files = sorted(glob.glob(pattern))
    if not files:
        return {}

    data = defaultdict(dict)
    for fp in files:
        basename = os.path.basename(fp)
        m = re.search(rf"-{dim}(\d+)-", basename)
        if not m:
            continue
        dim_val = int(m.group(1))
        impl = get_impl_name(fp)
        if impl in EXCLUDE_IMPLS:
            continue
        try:
            rows = load_ncu_report(fp)
            metrics = extract_rms_metrics(rows)
            if metrics:
                data[impl][dim_val] = metrics
        except Exception as e:
            print(f"Warning: skipped {fp}: {e}", file=sys.stderr)
    return dict(data)


def plot_sweep(data: dict, dim: str, out_dir: str):
    """Plot metrics over M or N, one line per implementation. Saves to subfolders by category."""
    impls = sorted(data.keys())
    if not impls:
        return

    colors = plt.cm.Set2.colors
    for metric_name, _unit, ylabel, subfolder in METRIC_CONFIG:
        safe_name = metric_to_safe_name(metric_name)
        subdir = os.path.join(out_dir, subfolder, safe_name)
        os.makedirs(subdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, impl in enumerate(impls):
            points = data[impl]
            if not points:
                continue
            xs = sorted(points.keys())
            ys = [points[x].get(metric_name) for x in xs]
            if all(y is None for y in ys):
                continue
            ys = [y if y is not None else float("nan") for y in ys]
            if "Occupancy" in metric_name:
                ys = [min(OCCUPANCY_CAP, y) if y == y else y for y in ys]  # cap, skip nan
            ax.plot(xs, ys, label=impl, marker="o", markersize=4, color=colors[i % len(colors)])

        ax.set_xlabel(dim)
        ax.set_ylabel(ylabel)
        ax.set_title(f"RMS Norm: {metric_name} vs {dim}")
        if "Occupancy" in metric_name:
            ax.set_ylim(0, OCCUPANCY_CAP)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        file_name = f"ncu_{safe_name}_vs_{dim.lower()}.png"
        out_path = os.path.join(subdir, file_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def plot_bar_charts(data: dict, out_dir: str):
    """Single-point bar charts (legacy mode). Saves to subfolders by category."""
    impls = list(data.keys())
    if not impls:
        return

    colors = plt.cm.Set2.colors[: len(impls)]
    for metric_name, _unit, ylabel, subfolder in METRIC_CONFIG:
        values = [data[impl].get(metric_name) for impl in impls]
        valid = [(impl, v) for impl, v in zip(impls, values) if v is not None]
        if not valid:
            continue
        impls_v, vals = zip(*valid)
        if "Occupancy" in metric_name:
            vals = tuple(min(OCCUPANCY_CAP, v) for v in vals)
        safe_name = metric_to_safe_name(metric_name)
        subdir = os.path.join(out_dir, subfolder, safe_name)
        os.makedirs(subdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(impls_v, vals, color=colors[: len(impls_v)])
        ax.set_ylabel(ylabel)
        ax.set_title(f"RMS Norm: {metric_name}")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, f"ncu_{safe_name}.png"), dpi=150)
        plt.close()


def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."

    out_dir = "ncu-plots"
    varying_m_dir = os.path.join(base_dir, "ncu-reports", "varying_m")
    varying_n_dir = os.path.join(base_dir, "ncu-reports", "varying_n")

    # Prefer sweep data from ncu-reports
    if os.path.isdir(varying_m_dir) and os.path.isdir(varying_n_dir):
        print("Using ncu-reports/ sweep data")
        data_m = collect_sweep_data(varying_m_dir, "M")
        data_n = collect_sweep_data(varying_n_dir, "N")
        if data_m:
            plot_sweep(data_m, "M", out_dir)
        if data_n:
            plot_sweep(data_n, "N", out_dir)
        if not data_m and not data_n:
            print("No sweep data found in ncu-reports/")
    else:
        # Fallback: single-point bar charts from cwd or given dir
        pattern = os.path.join(base_dir, "*ncu*.ncu-rep")
        files = sorted(glob.glob(pattern))
        if not files:
            files = sorted(glob.glob(os.path.join(base_dir, "rms_bench-*-ncu.rep.ncu-rep")))
        if not files:
            print("No .ncu-rep files found. Run NCU sweep or pass path to ncu-reports/")
            sys.exit(1)

        data = {}
        for fp in files:
            impl = get_impl_name(fp)
            if impl in EXCLUDE_IMPLS:
                continue
            try:
                rows = load_ncu_report(fp)
                metrics = extract_rms_metrics(rows)
                if metrics:
                    data[impl] = metrics
            except Exception as e:
                print(f"Warning: skipped {fp}: {e}", file=sys.stderr)
        if data:
            plot_bar_charts(data, out_dir)
        else:
            print("No metrics extracted.")
            sys.exit(1)

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
