#consider adding check to see if ncu / nsys are installed
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
        nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cudnn -o $kernel-nsys.rep python3 bench-driver.py --generate-kernel-dump=${kernel}
    else
        ncu -f -o $kernel-ncu.rep python3 bench-driver.py --generate-kernel-dump=${kernel}
    fi
done