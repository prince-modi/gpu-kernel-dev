from gluon_kernels.rmsnorm.rms_norm import rms_norm

def rms_benchmarks(benchmark_name: str, **kwargs):
    if 'X' not in kwargs or 'w' not in kwargs or 'eps' not in kwargs:
        raise Exception(f'Expected arguments (X,w,eps) are not in given arguments')
    X = kwargs['X']
    w = kwargs['w']
    eps = kwargs['eps']

    if benchmark_name == 'rms_norm':
        rms_norm(X, eps=eps, gamma=w)
    else:
        raise Exception(f'No kernel with name {benchmark_name}')
