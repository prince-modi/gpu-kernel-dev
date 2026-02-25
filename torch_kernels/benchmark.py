def rms_benchmarks(benchmark_name: str,**kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']
    
    if benchmark_name == 'helion_rms_kernel':
        rmsnorm_kernel_basic(X,w,eps)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')


def torch_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        rms_benchmarks(benchmark_name,kwargs)
    elif 'flashattn' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')