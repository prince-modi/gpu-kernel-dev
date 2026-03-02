from triton_kernels.rmsnorm.another_rmsnorm_with_loops import rmsnorm_kernel_3
from triton_kernels.flashattn.flash_attn_og import og_flash_attention
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
        # can increase denominator to improve performance for larger sizes
        # rmsnorm_kernel_1[(X.shape[0],1,1)](output,X,w,eps,X.stride(0),BLOCK_SIZE,num_stages)
        rmsnorm_kernel_3[(triton.cdiv(X.shape[0], 128), 1, 1)](
            output, X, w, eps, X.shape[0], X.stride(0), BLOCK_SIZE, num_stages, False
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

    # elif benchmark_name == 'og_flash_attention':
    #     # shape constraints
    #     assert Lq == Lk and Lk == Lv
    #     assert Lk in {16, 32, 64, 128}
    #     o = torch.empty_like(q)
    #     grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
    #     tuple_list = (q.shape[0] * q.shape[1], q.shape[2])
    #     tmp = torch.empty(
    #         tuple_list, device=q.device, dtype=torch.float32
    #     )
    #     L = torch.empty(tuple_list, device=q.device, dtype=torch.float32)
    #     m = torch.empty(tuple_list, device=q.device, dtype=torch.float32)
    #     og_flash_attention[grid](
    #     q,
    #     k,
    #     v,
    #     1 / q.stride(3) ** 2, # based on qk_scale from helion kernel
    #     tmp,
    #     L,
    #     m,
    #     o,
    #     q.stride(0),
    #     q.stride(1),
    #     q.stride(2),
    #     q.stride(3),
    #     k.stride(0),
    #     k.stride(1),
    #     k.stride(2),
    #     k.stride(3),
    #     v.stride(0),
    #     v.stride(1),
    #     v.stride(2),
    #     v.stride(3),
    #     o.stride(0),
    #     o.stride(1),
    #     o.stride(2),
    #     o.stride(3),
    #     q.shape[0],
    #     q.shape[1],
    #     q.shape[2],
    #     BLOCK_M=BLOCK,
    #     BLOCK_N=BLOCK,
    #     BLOCK_DMODEL=Lk,
    #     num_warps=num_warps,
    #     num_stages=1)
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
