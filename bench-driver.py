import triton
import torch
import os
import argparse
import sys
from rms_benchmark_driver import rms_benchmark, get_rms_benchmark
from attn_benchmark_driver import attn_benchmark, get_attn_benchmark, is_hopper


    
if __name__ == "__main__":
    if os.path.exists("rms-results") == False:
        os.mkdir("rms-results")
    if os.path.exists("attn-results") == False:
        os.mkdir("attn-results")

    if len(sys.argv) == 1:
        rms_benchmark.run(show_plots=True, print_data=True,save_path = "rms-results")
        attn_benchmark.run(show_plots=True,print_data = True, save_path = "attn-results")
        # print('Running rms_benchmark')
        #add other benchmarks here
    else:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--generate-kernel-dump", help="Specific kernel name to generate dump for (write as <bench-name>-<kernel-name>)",nargs=1,type=str)
        group.add_argument("--run-benchmark", help="Specify list of benchmarks to run - comma-separated list",nargs=1,type=str)
        parser.add_argument("--M",help="Specify M dimensions - write multiple if there are multiple tensors(e.g., rmsnorm only has 1 M while attention has 3 Ms)",nargs='*',type=int)
        parser.add_argument("--N",help="Specify N dimensions - write multiple if there are multiple tensors(e.g., rmsnorm only has 1 N while attention has 3 Ns)",nargs='*',type=int)
        args = parser.parse_args()

        if args.generate_kernel_dump is not None:
            kernel_name_string = args.generate_kernel_dump[0]
            split_string = kernel_name_string.split('-',maxsplit = 1)
            benchmark_name = split_string[0].lower()
            kernel_test_name = split_string[1].lower()
            if benchmark_name == 'rms_bench':
                # if len(args.M) != 1 or len(args.N) != 1:
                #     raise Exception(f'Require 1 M and N value, not len(M) = {len(args.M)} and len(N) = {len(args.N)}')
                #for sake of completion, just hardcoding M, N; either manually adjust or find another way to automate (another python script instead of bash)
                method = get_rms_benchmark(1024,1024,kernel_test_name)
                method()
                # print(f'Running {kernel_test_name} from {benchmark_name}; M = {args.M[0]} ; N= {args.N[0]}')
            elif benchmark_name == 'attn_bench':
                method = get_attn_benchmark(4096,32,4,128,is_hopper(),kernel_test_name)
                method()
            else:
                print(f'Benchmark {benchmark_name} has not been identified. Exiting...')
                sys.exit(1)
        else:
            benchmark_names = args.run_benchmark[0].split(',')
            print(benchmark_names)
            for bench in benchmark_names:
                if bench == 'rms_bench':
                    rms_benchmark.run(show_plots=True, print_data=True,save_path = "rms-results")
                elif bench == 'attn_bench':
                    attn_benchmark.run(show_plots=True,print_data = True, save_path = "attn-results")
                else:
                    raise Exception(f'Benchmark {bench} has no corresponding function. Please Add...')