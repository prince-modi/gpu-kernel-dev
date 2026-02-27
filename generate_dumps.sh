if [[ $(nsys --verion) -ne "0" || $(ncu --version) -ne "0" ]]; then
    echo "Need nsys and/or ncu in order to run. Exiting..."
    exit 1
fi

if (( $# < 2 )); then
    echo "Expected: bash generate_dumps.sh (nsys | ncu) <kernel>[ <kernel>]*"
    exit 1
fi
profiler_option=$1

if [[ "$profiler_option" != "nsys" && "$profiler_option" != "ncu" ]]; then
    echo "Expected profiler option to be nsys or ncu. Received '$profiler_option' instead "
    exit 1
fi

kernels=${@:2}

for kernel in $kernels; do
    #good for high-level kernels like pytorch and maybe even helion?
    if [ "$profiler_option" == "nsys" ]; then
        nsys profile --trace=cuda,nvtx,osrt,cudnn -o $kernel-nsys.rep python3 bench-driver.py --generate-kernel-dump=${kernel}
    else
        ncu -o $kernel-ncu.rep python3 bench-driver.py --generate_kernel_dump=${kernel}
    fi
done