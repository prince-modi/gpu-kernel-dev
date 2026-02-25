from rmsnorm.torch_rmsnorm import rmsnorm

def rms_benchmarks(benchmark_name: str, args: dict):
    if 'X' not in args or 'w' not in args or 'eps' not in args:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = args['X']
    w = args['w']
    eps = args['eps']
    
    if benchmark_name == 'helion_rms_kernel':
        rmsnorm(X,w,eps)
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