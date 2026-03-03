from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel
from helion_kernels.flash_attn.flash_attn_helion import flashatt_fwd
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

def _compile_code(benchmark_name: str, bound_kernel, args):
    configs = retrieve_configs(benchmark_name)
    tuner = FiniteSearch(bound_kernel,args,configs)
    best_config = tuner.autotune()
    return bound_kernel.compile_config(best_config)

def check_args_rms(**kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    return (kwargs['X'],kwargs['w'],kwargs['eps'])

def check_args_attn(**kwargs):
    if 'Q' not in kwargs or 'K' not in kwargs or 'V' not in kwargs:
        raise Exception(f'Expected arguments (Q,K,V) are not in given arguments')
    return (kwargs['Q'],kwargs['K'],kwargs['V'])


def compile_rms_benchmark(benchmark_name: str, **kwargs):
    args = check_args_rms(**kwargs)
    bound_kernel = None
    if benchmark_name == 'helion_rms_kernel':
        bound_kernel = helion_rms_kernel.bind(args)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')
    return _compile_code(benchmark_name,bound_kernel,args)

def compile_attn_benchmark(benchmark_name:str, **kwargs):
    args = check_args_attn(**kwargs)
    bound_kernel = None
    if benchmark_name == 'flashatt_fwd':
        bound_kernel = flashatt_fwd.bind(args)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')
    return _compile_code(benchmark_name,bound_kernel,args)


def rms_benchmarks(compiled_code, **kwargs):
    X,w,eps = check_args_rms(**kwargs)
    compiled_code(X,w,eps)

def attn_benchmarks(compiled_code,**kwargs):
    Q,K,V = check_args_attn(**kwargs)
    compiled_code(Q,K,V)

def helion_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        bounded_kernel = compile_rms_benchmark(benchmark_name,**kwargs)
        rms_benchmarks(bounded_kernel,**kwargs)
    elif 'flashattn' in benchmark_name or 'attn' in benchmark_name:
        bounded_kernel = compile_attn_benchmark(benchmark_name,**kwargs)
        attn_benchmarks(bounded_kernel,**kwargs)
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')