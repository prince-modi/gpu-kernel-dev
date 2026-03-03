import triton_kernels.benchmark as tlb
import helion_kernels.benchmark as hlb
import torch_kernels.benchmark as torlb
import gluon_kernels.benchmark as gln
import cutlass.cute as cute
import cute_kernels.benchmark as curlb
import triton
import torch
import os
import random


#these are in ms
warmup_count = 25 # this is anyways the default for triton, can adjust if feel it helps
repetitions = 100 #default from triton - can increase if want - number of times kernel runs for measurements

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"
def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


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


line_vals=['torch-rmsnorm','triton-rmsnorm_with_loops','helion-helion_rms_kernel','cute-cute_rms_norm','gluon-rms_norm']  # 'helion-helion_rms_kernel' possible values for `line_arg``
line_names=["Torch", "Triton","Helion","Cute","Gluon"]  #,"Helion" label name for the lines
styles=[('blue', '-'), ('green', '-'), ('red', '-'),('yellow','-'),('cyan','-')]  # line styles
if is_hopper():
    line_vals.append('tk-tk_rms_norm')
    line_names.append('thunkit')
    styles.append(('purple','-'))


rms_benchmark_configs = [triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=line_vals,  #,'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=line_names,  #,"Helion" label name for the lines
        styles=styles,  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-N",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ),
     triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=line_vals,  #,'helion-helion_rms_kernel' possible values for `line_arg``
        line_names=line_names,  #,"Helion" label name for the lines
        styles=styles,  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="RMS-Norm-With-Varying-M",  # name for the plot. Used also as a file name for saving the plot.
        args={'N': 4096},  # values for function arguments not in `x_names` and `y_name`
    )
]

@triton.testing.perf_report(rms_benchmark_configs)
def rms_benchmark(M: int, N: int, provider: str):
    method = get_rms_benchmark(M,N,provider)
    ms = triton.testing.do_bench(lambda: method(), warmup = warmup_count, rep = repetitions)
    #2 for float16 = 2 bytes
    gbps = lambda ms: 2 * M * N * 2 * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def get_rms_benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    eps = random.random()
    split_name = provider.split('-')
    dsl_type = split_name[0]
    bench_name = split_name[1]
    if dsl_type == 'triton':
        if is_cuda():
          properties = torch.cuda.get_device_properties(0)
          SMS_COUNT = properties.multi_processor_count
        else:
          SMS_COUNT = 1
        return lambda: tlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps,SMS_COUNT=SMS_COUNT)
    elif dsl_type == 'helion':
        compiled_code = hlb.compile_rms_benchmark(bench_name,X=x,w=w,eps=eps)
        return lambda: hlb.rms_benchmarks(compiled_code,X=x,w=w,eps=eps)
    elif dsl_type == 'torch':
        return lambda: torlb.rms_benchmarks(bench_name,X=x,w=w,eps=eps)
    elif dsl_type == 'cute':
        compiled_code = curlb.compile_rms_benchmark(bench_name,X=x,w=w,eps=eps)
        return lambda: curlb.rms_benchmarks(compiled_code,X=x,w=w,eps=eps)
    elif dsl_type == 'gluon':
        return lambda: gln.rms_benchmarks(bench_name, X=x, w=w, eps=eps)
    elif dsl_type == 'thunderkittens' and is_hopper():
        raise Exception(f'Thunderkittens needs to be placed here!!!')
    else:
        raise Exception(f'{provider} is not yet supported')
