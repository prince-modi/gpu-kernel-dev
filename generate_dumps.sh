if (( $# < 2 )); then
    echo "Expected: bash generate_dumps.sh <profiler-mode> <kernel>[ <kernel>]*"
    exit 1
fi
profiler_option=$1
kernels=${@:2}

for kernel in $kernels; do
    #good for high-level kernels like pytorch and maybe even helion?
    nsys profile --trace=cuda,nvtx,osrt,cudnn -o $kernel-nsys.rep python3 bench-driver.py --generate_kernel_dump=${kernel}
    ncu -o $kernel-ncu.rep python3 bench-driver.py --generate_kernel_dump=${kernel}
done