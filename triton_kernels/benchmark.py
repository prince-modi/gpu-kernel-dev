def rms_benchmarks(benchmark_name: str,**kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']

    BLOCK_SIZE = meta['BLOCK_SIZE']
    if 'BLOCK_SIZE' in kwargs:
        BLOCK_SIZE = kwargs['BLOCK_SIZE']
    
    num_stages = 2
    if num_stages in kwargs:
        num_stages = kwargs['num_stages']
    
    output = torch.empty_like(X, device = DEVICE)
    if benchmark_name == 'rmsnorm_with_loops':
        rmsnorm_kernel_1[(X.shape[0],1,1)](output,X,w,X.stride(0),X.shape[0],eps,BLOCK_SIZE,num_stages)
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