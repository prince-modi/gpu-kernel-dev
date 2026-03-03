from triton_kernels.rmsnorm.another_rmsnorm_with_loops import rmsnorm_kernel_3
# from triton_kernels.flashattn.flash_attn_og import og_flash_attention
from triton_kernels.flashattn.flash_attn_v2 import flash_attention_v2_wrapper
import triton
import torch


def check_args_rms(**kwargs):
    if "X" not in kwargs or "w" not in kwargs or "eps" not in kwargs:
        raise Exception(f"Expected arguments (X,w,eps) are not in given arguments")
    return (kwargs["X"], kwargs["w"], kwargs["eps"])


def check_args_attn(**kwargs):
    if "Q" not in kwargs or "K" not in kwargs or "V" not in kwargs:
        raise Exception(f"Expected arguments (Q,K,V) are not in given arguments")
    return (kwargs["Q"], kwargs["K"], kwargs["V"])


def rms_benchmarks(benchmark_name: str, **kwargs):
    X, w, eps = check_args_rms(**kwargs)
    MAX_FUSED_SIZE = 65536
    if "BLOCK_SIZE" in kwargs:
        BLOCK_SIZE = kwargs["BLOCK_SIZE"]
    else:
        BLOCK_SIZE = MAX_FUSED_SIZE

    num_stages = 2
    if num_stages in kwargs:
        num_stages = kwargs["num_stages"]

    DEVICE = X.device
    if "DEVICE" in kwargs:
        DEVICE = kwargs["DEVICE"]

    output = torch.empty_like(X, device=DEVICE)
    if benchmark_name == "rmsnorm_with_loops":
        # BLOCK_SIZE = triton.next_power_of_2(X.shape[1])
        BLOCK_SIZE = 2048
        NUMBER_OF_RUN_WARPS_PER_SM = 4 #know that abt 4 warps run in one instance
        ROW_INTERVAL = triton.cdiv(X.shape[0],kwargs["SMS_COUNT"] * NUMBER_OF_RUN_WARPS_PER_SM)
        # can increase denominator to improve performance for larger sizes
        # rmsnorm_kernel_1[(X.shape[0],1,1)](output,X,w,eps,X.stride(0),BLOCK_SIZE,num_stages)
        rmsnorm_kernel_3[(ROW_INTERVAL, 1, 1)](
            output, X, w, eps, X.shape[0], X.stride(0), ROW_INTERVAL,BLOCK_SIZE, num_stages, False
        )
    else:
        raise Exception(f"No kernel with name {benchmark_name}")


def attn_benchmarks(benchmark_name: str, **kwargs):
    q, k, v = check_args_attn(**kwargs)
    BLOCK = 128
    if "BLOCK" in kwargs:
        BLOCK = kwargs["BLOCK"]
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    num_warps = 4 if Lk <= 64 else 8

    if benchmark_name == "forward":
        return flash_attention_v2_wrapper(q, k, v, causal=False)
    else:
        raise Exception(f"No kernel with name {benchmark_name}")


def triton_provide_benchmark(benchmark_name: str, **kwargs):
    if "rms" in benchmark_name:
        rms_benchmarks(benchmark_name, **kwargs)
    elif (
        "flashattn" in benchmark_name
        or "attn" in benchmark_name
        or "forward" in benchmark_name
    ):
        attn_benchmarks(benchmark_name, **kwargs)
    elif "load" in benchmark_name:
        raise Exception(f"Received unsupported kernel {benchmark_name}")
    else:
        raise Exception(f"Received unsupported kernel {benchmark_name}")
