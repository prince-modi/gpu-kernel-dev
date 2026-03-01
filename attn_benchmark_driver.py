import triton_kernels.benchmark as tlb
import helion_kernels.benchmark as hlb
import torch_kernels.benchmark as torlb
import cutlass.cute as cute
import cute_kernels.benchmark as curlb
import triton
import torch
import os
import random

#based on https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"
def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9

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

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
#assume causal is false
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

BATCH, N_HEADS = 4, 32
dtype = torch.float16
# vary seq length for fixed head and batch=4

line_vals=["torch-flash_attn", "helion-flashatt_fwd", "triton-forward","cute-flashatt_fwd"]
line_names=["Torch", "Triton","Helion","Cute"]  #,"Helion" label name for the lines
styles=[('blue', '-'), ('green', '-'), ('red', '-'),('yellow','-')]  # line styles
if is_hopper():
    line_vals.append('tk-tk_rms_norm')
    line_names.append('Thunderkittens')
    styles.append(('purple','-'))


flash_attn_bench_configs = []


for HEAD_DIM in [64, 128]:
    enable_ws = is_hopper()
    for warp_specialize in [False, True] if enable_ws else [False]:
        flash_attn_bench_configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                ylabel="TFLOPS",
                plot_name = f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-warp_specialize={warp_specialize}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "warp_specialize": warp_specialize,
                },
            ))


@triton.testing.perf_report(flash_attn_bench_configs)
def attn_benchmark(N_CTX: int,H,BATCH: int,HEAD_DIM: int ,warp_specialize: bool, provider: str):
    method = get_attn_benchmark(N_CTX,H,BATCH,HEAD_DIM,warp_specialize, provider)
    ms = triton.testing.do_bench(lambda: method(), warmup = warmup_count, rep = repetitions)
    gbps = lambda ms: 2 * M * N * 2 * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def get_attn_benchmark(N_CTX: int,H,BATCH: int,HEAD_DIM: int ,warp_specialize: bool, provider: str):
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    eps = random.random()
    split_name = provider.split('-')
    dsl_type = split_name[0]
    bench_name = split_name[1]
    if dsl_type == 'triton':
        return lambda: tlb.rms_benchmarks(bench_name,Q=q,K=k,V=v)
    elif dsl_type == 'helion':
        compiled_code = hlb.compile_attn_benchmark(bench_name,Q=q,K=k,V=v)
        return lambda: hlb.attn_benchmarks(compiled_code,Q=q,K=k,V=v)
    elif dsl_type == 'torch':
        return lambda: torlb.attn_benchmarks(bench_name,Q=q,K=k,V=v)
    elif dsl_type == 'cute':
        compiled_code = curlb.compile_rms_benchmark(bench_name,Q=q,K=k,V=v)
        return lambda: curlb.rms_benchmarks(compiled_code,Q=q,K=k,V=v)
    elif dsl_type == 'thunderkittens' and is_hopper():
        raise Exception(f'Thunderkittens needs to be placed here!!!')
    else:
        raise Exception(f'{provider} is not yet supported')

