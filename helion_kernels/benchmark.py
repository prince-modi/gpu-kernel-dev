from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel
from helion_kernels.rmsnorm.example_rmsnorm import rms_norm_fwd

def rms_benchmarks(benchmark_name: str,**kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']
    
    if benchmark_name == 'helion_rms_kernel':
        helion_rms_kernel(X,w,eps)
    elif benchmark_name == 'example_helion_rms_kernel':
        rms_norm_fwd(X,w,eps)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')


def helion_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        rms_benchmarks(benchmark_name,kwargs)
    elif 'flashattn' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')