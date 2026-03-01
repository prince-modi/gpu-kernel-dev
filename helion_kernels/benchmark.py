from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel
import os
import helion
from helion.autotuner import FiniteSearch

def retrieve_configs(benchmark_name: str):
    if os.path.exists('configs') == False:
        print('Need configs directory to run these tests...')
    assert os.path.exists('configs')
    all_configs = os.listdir('configs')
    filtered_configs = []
    for conf in all_configs:
        if benchmark_name in conf and 'json' in conf:
            filtered_configs.append(helion.Config.load(os.path.join('configs',conf)))
    return filtered_configs


def compile_rms_benchmark(benchmark_name: str, **kwargs):
    global compiled_code
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']
    bound_kernel = None
    args = (X,w,eps)
    configs = None
    if benchmark_name == 'helion_rms_kernel':
        bound_kernel = helion_rms_kernel.bind(args)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')
    configs = retrieve_configs(benchmark_name)
    tuner = FiniteSearch(bound_kernel,args,configs)
    best_config = tuner.autotune()
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