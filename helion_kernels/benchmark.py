from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel
import os
import helion
from helion.autotuner import FiniteSearch

def compile_rms_benchmark(benchmark_name: str, **kwargs):
    global compiled_code
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']
    bound_kernel = None
    if benchmark_name == 'helion_rms_kernel':
        bound_kernel = helion_rms_kernel.bind((X,w,eps))
    else:
        raise Exception(f'No kernel with name {benchmark_name}')
    best_config = bound_kernel.autotune()
    return bound_kernel.compile_config(best_config)

def rms_benchmarks(compiled_code, **kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']
    compiled_code(X,w,eps)

def helion_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        bounded_kernel = compile_rms_benchmark(benchmark_name,kwargs)
        rms_benchmarks(bounded_kernel,kwargs)
    elif 'flashattn' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')