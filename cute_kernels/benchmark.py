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
    if benchmark_name == 'optimized_RMS_norm':
        return cute.compile(kernel_fn, mX, mW, mY, X.shape[0], w.shape[0], Float32(eps))
    else:
        raise Exception(f'No kernel with name {benchmark_name}')

def rms_benchmark(compiled_code, **kwargs):
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
    compiled_code(mX, mW, mY, X.shape[0], w.shape[0], Float32(eps))

# ---------------------------------------------------------------------------
# Flash Attention 2 (via flash_attn library)
# ---------------------------------------------------------------------------

def compile_fa2_benchmark(benchmark_name: str, **kwargs):
    """Bind flash_attn_qkvpacked_func with the given config and return a callable."""
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

def fa2_benchmark(compiled_code, **kwargs):
    qkv = kwargs.get('qkv')
    if qkv is None:
        raise Exception("Expected argument 'qkv' in kwargs")
    compiled_code(qkv=qkv)

# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def cute_provide_benchmark(benchmark_name: str, **kwargs):
    if 'rms' in benchmark_name:
        compiled_code = compile_rms_benchmark(benchmark_name, **kwargs)
        rms_benchmark(compiled_code, **kwargs)
    elif 'flashattn' in benchmark_name or 'attn' in benchmark_name:
        compiled_code = compile_fa2_benchmark(benchmark_name, **kwargs)
        fa2_benchmark(compiled_code, **kwargs)
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')