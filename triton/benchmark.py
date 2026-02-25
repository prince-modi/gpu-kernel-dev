from rmsnorm.rmsnorm_with_loops import rmsnorm_with_loops

def rms_benchmarks(benchmark_name: str, args: dict):
    if 'X' not in args or 'w' not in args or 'eps' not in args:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = args['X']
    w = args['w']
    eps = args['eps']

    BLOCK_SIZE = meta['BLOCK_SIZE']
    if 'BLOCK_SIZE' in args:
        BLOCK_SIZE = args['BLOCK_SIZE']
    
    num_stages = 2
    if num_stages in args:
        num_stages = args['num_stages']
    
    output = torch.empty_like(X, device = DEVICE)
    if benchmark_name == 'rmsnorm_with_loops':
        rmsnorm_with_loops[(X.shape[0],1,1)](output,X,w,X.stride(0),X.shape[0],eps,BLOCK_SIZE,num_stages)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')


def triton_provide_benchmark(benchmark_name: str,**kwargs):
    if 'rms' in benchmark_name:
        rms_benchmarks(benchmark_name,kwargs)
    elif 'flashattn' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    elif 'load' in benchmark_name:
        raise Exception(f'Received unsupported kernel {benchmark_name}')
    else:
        raise Exception(f'Received unsupported kernel {benchmark_name}')