# NOTE: Flash Attention benchmarks require the flash_attn package.
# Use the wheel finder at https://flashattn.dev/#finder to install the correct
# version for your CUDA toolkit and PyTorch build, e.g.:
#   import torch
#   import sys
#   print(torch.version.cuda)
#   print(torch.__version__)
#   print(torch.cuda.get_device_name(0))
#   print(sys.version)
# Then select the appropriate wheel from https://flashattn.dev/#finder and install it with the following command:
#   !uv pip install <wheel_name>
# You can use pip itself, although this may be slower.

from cute_kernels.rmsnorm.A100_optimized_RMS_norm import cute_rms_norm as cute_rms_norm_a100
from cute_kernels.rmsnorm.H100_optimized_RMS_norm import cute_rms_norm as cute_rms_norm_h100
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32

_RMS_KERNELS = {
    'A100': cute_rms_norm_a100,
    'H100': cute_rms_norm_h100,
}

def _resolve_gpu(kwargs):
    gpu = kwargs.get('gpu', 'A100')
    if gpu not in _RMS_KERNELS:
        raise ValueError(f"Unsupported gpu={gpu!r}. Choose from {list(_RMS_KERNELS)}")
    return _RMS_KERNELS[gpu]


def _get_gpu_arch():
    """Get target SM arch for CUTLASS. Use sm_86 on Ada (sm_89) for compatibility."""
    if not torch.cuda.is_available():
        return "sm_80"  # fallback
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    return "sm_86" if arch == "sm_89" else arch  # Ada can run sm_86 binaries

def compile_rms_benchmark(benchmark_name: str, **kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']

    DEVICE = X.device
    if 'DEVICE' in kwargs:
        DEVICE = kwargs['DEVICE']
    y = torch.zeros(X.shape, device=DEVICE, dtype=X.dtype)

    mX = from_dlpack(X, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)

    kernel_fn = _resolve_gpu(kwargs)
    if benchmark_name in ('optimized_RMS_norm', 'cute_rms_norm'):
        gpu_arch = kwargs.get('gpu_arch') or _get_gpu_arch()
        options = f"--gpu-arch {gpu_arch}"
        return cute.compile(
            kernel_fn, mX, mW, mY, X.shape[0], w.shape[0], Float32(eps),
            options=options,
        )
    else:
        raise Exception(f'No kernel with name {benchmark_name}')

def rms_benchmarks(compiled_code, **kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']

    DEVICE = X.device
    if 'DEVICE' in kwargs:
        DEVICE = kwargs['DEVICE']
    y = torch.zeros(X.shape, device=DEVICE, dtype=X.dtype)

    mX = from_dlpack(X, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)
    compiled_code(mX, mW, mY, Float32(eps))

# ---------------------------------------------------------------------------
# Flash Attention 2 — CuTe DSL SM80 kernel (self-contained, no flash_attn pip pkg)
# ---------------------------------------------------------------------------

from cute_kernels.flash_attn.SM80_flash_attn_kernel import flash_attn_sm80_fwd

def _get_sm_version():
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def compile_fa2_benchmark(benchmark_name: str, **kwargs):
    """Compile/bind the CuTe DSL SM80 Flash Attention forward kernel."""
    sm = _get_sm_version()
    if sm >= 90:
        raise NotImplementedError(
            f"CuTe DSL Flash Attention forward for SM{sm} (H100/Hopper) is not implemented yet. "
            f"Only SM80 (A100/Ampere) is currently supported."
        )

    if 'qkv' not in kwargs:
        raise Exception("Expected argument 'qkv' in kwargs")
    qkv = kwargs['qkv'].contiguous()
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    causal = bool(kwargs.get('causal', False))
    softmax_scale = kwargs.get('softmax_scale', None)

    # Warm-compile on first call; flash_attn_sm80_fwd caches internally.
    flash_attn_sm80_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal)

    def compiled_code(**call_kwargs):
        qkv_in = call_kwargs.get('qkv', qkv).contiguous()
        qi, ki, vi = qkv_in[:, :, 0], qkv_in[:, :, 1], qkv_in[:, :, 2]
        return flash_attn_sm80_fwd(qi, ki, vi, softmax_scale=softmax_scale, causal=causal)

    return compiled_code


def fa2_benchmark(compiled_code, **kwargs):
    qkv = kwargs.get('qkv')
    if qkv is None:
        raise Exception("Expected argument 'qkv' in kwargs")
    compiled_code(qkv=qkv)


# ---------------------------------------------------------------------------
# Flash Attention 2 — pre-compiled C++/CUDA pip package (flash_attn)
# This is NOT the CuTe DSL kernel; it wraps the pip-installable flash_attn
# library which ships its own CUDA binaries.
# ---------------------------------------------------------------------------

def compile_fa2_benchmark_precompiled(benchmark_name: str, **kwargs):
    """Bind flash_attn_qkvpacked_func (pip package) — not a CuTe kernel."""
    if 'qkv' not in kwargs:
        raise Exception("Expected argument 'qkv' in kwargs")
    qkv = kwargs['qkv']

    causal = bool(kwargs.get('causal', False))
    dropout_p = float(kwargs.get('dropout_p', 0.0))
    softmax_scale = kwargs.get('softmax_scale', None)

    from flash_attn import flash_attn_qkvpacked_func

    qkv = qkv.contiguous()

    def compiled_code(**call_kwargs):
        q = call_kwargs.get('qkv', qkv).contiguous()
        return flash_attn_qkvpacked_func(
            q,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return compiled_code

# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def cute_provide_benchmark(benchmark_name: str, **kwargs):
    if 'rms' in benchmark_name:
        compiled_code = compile_rms_benchmark(benchmark_name, **kwargs)
        rms_benchmarks(compiled_code, **kwargs)
    elif 'flashattn' in benchmark_name or 'attn' in benchmark_name:
        compiled_code = compile_fa2_benchmark(benchmark_name, **kwargs)
        fa2_benchmark(compiled_code, **kwargs)
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')