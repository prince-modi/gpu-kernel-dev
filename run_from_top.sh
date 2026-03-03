#!/bin/bash
# Usage: bash run_from_top.sh [rms|attn|all] [--profile-only]
#   rms  - RMS norm configs, benchmark, and dumps only
#   attn - Attention configs, benchmark, and dumps only
#   all  - both (default)
#   --profile-only - skip config gen and benchmark, run nsys/ncu dumps only

mode="${1:-all}"
profile_only=false
for arg in "$@"; do
    if [[ "$arg" == "--profile-only" ]]; then
        profile_only=true
        break
    fi
done

if [[ "$mode" != "rms" && "$mode" != "attn" && "$mode" != "all" ]]; then
    echo "Usage: bash run_from_top.sh [rms|attn|all] [--profile-only]"
    echo "  rms  - RMS norm only"
    echo "  attn - Attention only"
    echo "  all  - both (default)"
    echo "  --profile-only - skip config gen and benchmark, run nsys/ncu only"
    exit 1
fi

if [[ "$profile_only" != true ]]; then
    if [[ "$mode" == "rms" || "$mode" == "all" ]]; then
        python3 helion_kernels/rmsnorm/generate_config.py
    fi
    if [[ "$mode" == "attn" || "$mode" == "all" ]]; then
        python3 helion_kernels/flash_attn/generate_config.py
    fi

    if [[ "$mode" == "rms" ]]; then
        python3 bench-driver.py --run-benchmark=rms_bench
    elif [[ "$mode" == "attn" ]]; then
        python3 bench-driver.py --run-benchmark=attn_bench
    else
        python3 bench-driver.py
    fi
fi

# Exclude cute on Ada (sm_89) - CUTLASS fails there
if python3 -c "import torch; exit(0 if torch.cuda.get_device_capability() != (8, 9) else 1)" 2>/dev/null; then
    rms_kernels=("rms_bench-helion-helion_rms_kernel" "rms_bench-torch-rmsnorm" "rms_bench-triton-rmsnorm_with_loops" "rms_bench-cute-cute_rms_norm" "rms_bench-gluon-rms_norm")
else
    rms_kernels=("rms_bench-helion-helion_rms_kernel" "rms_bench-torch-rmsnorm" "rms_bench-triton-rmsnorm_with_loops" "rms_bench-gluon-rms_norm")
fi
attn_kernels=("attn_bench-torch-flash_attn" "attn_bench-helion-flashatt_fwd" "attn_bench-triton-forward" "attn_bench-cute-flashatt_fwd")

if [[ "$mode" == "rms" || "$mode" == "all" ]]; then
    bash generate_dumps.sh nsys "${rms_kernels[@]}"
    # NCU sweep over M (fixed N) and N (fixed M), saves to ncu-reports/
    echo "Running NCU sweep (varying M and N) - saves to ncu-reports/"
    sudo env "PATH=$PATH" python3 generate_ncu_sweep.py "${rms_kernels[@]}"
fi
if [[ "$mode" == "attn" || "$mode" == "all" ]]; then
    bash generate_dumps.sh nsys "${attn_kernels[@]}"
    sudo env "PATH=$PATH" bash generate_dumps.sh ncu "${attn_kernels[@]}"
fi

