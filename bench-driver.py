import triton_kernels.benchmark as tlb
import helion_kernels.benchmark as hlb
import torch_kernels.benchmark as torlb
import cutlass.cute as cute
import cute_kernels.benchmark as curlb
import triton
import torch
import os
import random
import argparse
import sys

#these are in ms
warmup_count = 25 # this is anyways the default for triton, can adjust if feel it helps
repetitions = 100 #default from triton - can increase if want - number of times kernel runs for measurements


if torch.cuda.is_available():
    # Use GPU
    DEVICE = torch.device("cuda:0")
    print("CUDA is available. Using GPU.")
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
else:
    # Fallback to CPU
    DEVICE = torch.device("cpu")
    print("CUDA not available. Using CPU.")
    os.environ["TRITON_INTERPRET"] = "1"
    
@triton.testing.perf_report(
    [triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch-rmsnorm','triton-rmsnorm_with_loops','helion-helion_rms_kernel','cute-cute_rms_norm'],  #,'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=["Torch", "Triton","Helion","Cute"],  #,"Helion" label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow','-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-N",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ),
     triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch-rmsnorm','triton-rmsnorm_with_loops','helion-helion_rms_kernel','cute-cute_rms_norm'],  # 'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=["Torch", "Triton","Helion","Cute"],  #,"Helion" label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'),('yellow','-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-M",  # name for the plot. Used also as a file name for saving the plot.
        args={'N': 4096},  # values for function arguments not in `x_names` and `y_name`
    )])
def rms_benchmark(M: int, N: int, provider: str):
    method = get_rms_benchmark(M,N,provider)
    ms = triton.testing.do_bench(lambda: method(), warmup = warmup_count, rep = repetitions)
    gbps = lambda ms: 2 * M * N * 4 * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def get_rms_benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    eps = random.random()
    split_name = provider.split('-')
    dsl_type = split_name[0]
    bench_name = split_name[1]
    if dsl_type == 'triton':
        return lambda: tlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps)
    elif dsl_type == 'helion':
        compiled_code = hlb.compile_rms_benchmark(bench_name,X=x,w=w,eps=eps)
        return lambda: hlb.rms_benchmarks(compiled_code,X=x,w=w,eps=eps)
    elif dsl_type == 'torch':
        return lambda: torlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps)
    elif dsl_type == 'cute':
        compiled_code = curlb.compile_rms_benchmark(bench_name,X=x,w=w,eps=eps)
        return lambda: curlb.rms_benchmarks(compiled_code,X=x,w=w,eps=eps)
    else:
        raise Exception(f'{provider} is not yet supported')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        if os.path.exists("results") == False:
            os.mkdir("results")
        rms_benchmark.run(show_plots=True, print_data=True,save_path = "results")
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
            elif benchmark_name == 'flashattn_bench':
              #can be defined here
              pass
            else:
                print(f'Benchmark {benchmark_name} has not been identified. Exiting...')
                sys.exit(1)
        else:
            benchmark_names = args.run_benchmark[0].split(',')
            print(benchmark_names)
            for bench in benchmark_names:
                if bench == 'rms_bench':
                    rms_benchmark.run(show_plots=True, print_data=True,save_path = "results")
                elif bench == 'flashattn_bench':
                    #can be defined here
                    raise Exception(f'Need to add benchmark flashattn_bench here. Exiting...')
                else:
                    raise Exception(f'Benchmark {bench} has no corresponding function. Please Add...')