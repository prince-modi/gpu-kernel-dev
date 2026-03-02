from torch_kernels.rmsnorm.torch_rmsnorm import rmsnorm_kernel_basic
from torch_kernels.flash_attn.flash_attn_torch import sdpa_reference


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
    if benchmark_name == "rmsnorm":
        rmsnorm_kernel_basic(X, w, eps)
    else:
        raise Exception(f"No kernel with name {benchmark_name}")


def attn_benchmarks(benchmark_name: str, **kwargs):
    q, k, v = check_args_attn(**kwargs)  # FIXED
    if benchmark_name in [
        "sdpa_reference",
        "flash_attn",
    ]:  # Catches driver's "torch-flash_attn"
        sdpa_reference(q, k, v, None)
    else:
        raise Exception(f"No kernel with name {benchmark_name}")


def torch_provide_benchmark(benchmark_name: str, **kwargs):
    if "rms" in benchmark_name:
        rms_benchmarks(benchmark_name, **kwargs)
    elif "flashattn" in benchmark_name or "attn" in benchmark_name:
        attn_benchmarks(benchmark_name, **kwargs)
    elif "load" in benchmark_name:
        raise Exception(f"Received unsupported kernel {benchmark_name}")
    else:
        raise Exception(f"Received unsupported kernel {benchmark_name}")
