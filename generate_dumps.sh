#!/bin/bash
# Profile kernels with nsys or ncu. Note: profiling takes 1-3 minutes per kernel.
# For ncu: if you see ERR_NVGPUCTRPERM, run with: sudo bash generate_dumps.sh ncu ...
#   Or enable permanently: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u && reboot

if (( $# < 2 )); then
    echo "Expected: bash generate_dumps.sh (nsys | ncu) <kernel>[ <kernel>]*"
    exit 1
fi
profiler_option=$1

if [[ "$profiler_option" != "nsys" && "$profiler_option" != "ncu" ]]; then
    echo "Expected profiler option to be nsys or ncu. Received '$profiler_option' instead "
    exit 1
fi

if [ "$profiler_option" == "ncu" ] && ! command -v ncu &>/dev/null; then
    echo "Skipping ncu: command not found (install Nsight Compute for GPU profiling)"
    exit 0
fi

if [ "$profiler_option" == "nsys" ] && ! command -v nsys &>/dev/null; then
    echo "Error: nsys not found (install Nsight Systems)"
    exit 1
fi

kernels=("${@:2}")
total=${#kernels[@]}
bar_width=20
echo "Profiling $total kernel(s) - may take 1-3 min each..."

for i in "${!kernels[@]}"; do
    kernel="${kernels[$i]}"
    current=$((i + 1))
    filled=$((current * bar_width / total))
    empty=$((bar_width - filled))
    bar_filled=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar_empty=$(printf '%*s' "$empty" '' | tr ' ' '-')
    echo "[${bar_filled}${bar_empty}] $current/$total --- $profiler_option: $kernel ---"
    # HELION_SKIP_AUTOTUNE=1 avoids minutes of autotuning during profiling
    export HELION_SKIP_AUTOTUNE=1
    if [ "$profiler_option" == "nsys" ]; then
        rm -f "$kernel-nsys.nsys-rep"
        nsys profile --force-overwrite=true --stats=false --duration=60 --trace=cuda,nvtx --wait=all -o "$kernel-nsys" python3 bench-driver.py "--generate-kernel-dump=$kernel"
    else
        ncu -f -o "$kernel-ncu.rep" python3 bench-driver.py "--generate-kernel-dump=$kernel"
    fi
done