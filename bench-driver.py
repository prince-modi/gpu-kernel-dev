import triton_kernels.benchmark as tlb
import helion_kernels.benchmark as hlb
import torch_kernels.benchmark as torlb
import triton
import torch
import os
import random

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
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch-rmsnorm','triton-rmsnorm_with_loops','helion-helion_rms_kernel'],  #,'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=["Torch", "Triton w Loops","Helion"],  #,"Helion" label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-N",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ),
     triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch-rmsnorm','triton-rmsnorm_with_loops','helion-helion_rms_kernel'],  # 'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=["Torch", "Triton w Loops","Helion"],  #,"Helion" label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-M",  # name for the plot. Used also as a file name for saving the plot.
        args={'N': 4096},  # values for function arguments not in `x_names` and `y_name`
    )])
def rms_benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    eps = random.random()
    split_name = provider.split('-')
    dsl_type = split_name[0]
    bench_name = split_name[1]
    if dsl_type == 'triton':
        ms = triton.testing.do_bench(lambda: tlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps))
    elif dsl_type == 'helion':
        ms = triton.testing.do_bench(lambda: hlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps))
    elif dsl_type == 'torch':
        ms = triton.testing.do_bench(lambda: torlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps))
    else:
        raise Exception(f'{provider} is not yet supported')
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    if os.path.exists("results") == False:
        os.mkdir("results")
    rms_benchmark.run(show_plots=True, print_data=True,save_path = "results")
