from cute_kernels.rmsnorm.optimized_RMS_norm import cute_rms_norm
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32

def compile_rms_benchmark(benchmark_name: str,**kwargs):
    global compiled_code
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

    if benchmark_name == 'optimized_RMS_norm':
        return cute.compile(cute_rms_norm, mX, mW, mY, X.shape[0], w.shape[0], Float32(eps))
    else:
        raise Exception(f'No kernel with name {benchmark_name}')

#to maintain parity with other benchmarks, redoing set up of arguments
def rms_benchmark(compiled_code,**kwargs):
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
    compiled_code(mX,mW,mY,X.shape[0],w.shape[0],Float32(eps))

def cute_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        #can handle similar to helion + autotune -> for now, thinking of keeping simple and not doing AOT compilation
        compiled_code = compile_rms_benchmark(benchmark_name,kwargs)
        rms_benchmark(compiled_code,kwargs)
    elif 'flashattn' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')