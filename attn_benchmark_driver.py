import os
import random

import cutlass.cute as cute
import torch
import triton

import cute_kernels.benchmark as curlb
import helion_kernels.benchmark as hlb
import torch_kernels.benchmark as torlb
import triton_kernels.benchmark as tlb

# based on https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

properties = torch.cuda.get_device_properties()
major_minor_version = properties.major * 10 + properties.minor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


# these are in ms
warmup_count = 25  # this is anyways the default for triton, can adjust if feel it helps
repetitions = 100  # default from triton - can increase if want - number of times kernel runs for measurements

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

# assume causal is false
headdim_vals = [64, 128]
BATCH, N_HEADS = 4, 32
dtype = torch.float16
# vary seq length for fixed head and batch=4


line_vals = [
    "torch-flash_attn",
    "helion-flashatt_fwd",
    "triton-forward",
    "cute-flashatt_fwd",
]
line_names = ["Torch", "Helion", "Triton", "Cute"]
styles = [("blue", "-"), ("green", "-"), ("red", "-"), ("yellow", "-")]
if is_hopper():
    line_vals.append("tk-tk_rms_norm")
    line_names.append("Thunderkittens")
    styles.append(("purple", "-"))


flash_attn_bench_configs = []


for HEAD_DIM in headdim_vals:
    enable_ws = is_hopper()
    for warp_specialize in [False, True] if enable_ws else [False]:
        flash_attn_bench_configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                ylabel="TFLOPS",
                line_vals=line_vals,  #,'helion-helion_rms_kernel' possible values for `line_arg``
                line_names=line_names,  #,"Helion" label name for the lines
                styles=styles,  # line styles
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-warp_specialize={warp_specialize}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "warp_specialize": warp_specialize,
                },
            )
        )


@triton.testing.perf_report(flash_attn_bench_configs)
def attn_benchmark(
    N_CTX: int, H, BATCH: int, HEAD_DIM: int, warp_specialize: bool, provider: str
):
    method = get_attn_benchmark(BATCH, H, N_CTX, HEAD_DIM, warp_specialize, provider)
    ms = triton.testing.do_bench(lambda: method(), warmup=warmup_count, rep=repetitions)
    # as per test_daoAI_lab file
    tflops = (
        lambda ms: (4.0 * BATCH * H * (N_CTX * N_CTX) * HEAD_DIM) / (ms * 1e-3) / 1e12
    )
    return tflops(ms)


def get_attn_benchmark(
    BATCH: int, H: int, N_CTX: int, HEAD_DIM: int, warp_specialize: bool, provider: str
):
    split_name = provider.split("-")
    dsl_type = split_name[0]
    bench_name = split_name[1]
    if dsl_type not in ["triton", "helion", "torch"]:
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM))
    else:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    if dsl_type == "triton":
        return lambda: tlb.attn_benchmarks(bench_name, Q=q, K=k, V=v)
    elif dsl_type == "helion":
        compiled_code = hlb.compile_attn_benchmark(bench_name, Q=q, K=k, V=v)
        return lambda: hlb.attn_benchmarks(compiled_code, Q=q, K=k, V=v)
    elif dsl_type == "torch":
        if major_minor_version not in range(80,122):
            raise Exception(f'Version {major_minor_version} will not run Flash Attention. Exiting...')
        return lambda: torlb.attn_benchmarks(bench_name, Q=q, K=k, V=v)
    elif dsl_type == "cute":
        # based on helion code - pls confirm
        compiled_code = curlb.compile_fa2_benchmark(
            bench_name, qkv=qkv, softmax_scale=1 / HEAD_DIM**0.5
        )
        return lambda: compiled_code(qkv=qkv)
    elif dsl_type == "thunderkittens" and is_hopper():
        raise Exception(f"Thunderkittens needs to be placed here!!!")
    else:
        raise Exception(f"{provider} is not yet supported")
