import os
import torch
import helion
from basic_rmsnorm import helion_rms_kernel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def retrieve_configs(benchmark_name: str):
    all_configs = os.listdir('configs')
    filtered_configs = []
    for conf in all_configs:
        if benchmark_name in conf and 'json' in conf:
            filtered_configs.append(helion.Config.load(os.path.join('configs',conf)))
    return filtered_configs


def get_signature(args: list) -> str:
    sig = []
    for arg in args:
        if type(arg) is torch.Tensor:
            sig.append('x'.join(list(arg.shape)))
    return 'x'.join(sig)

#give attention_configs

def generate_tensors(M: list[int], N: list[int], dtype=torch.float16, device=None):
    tensors = []

    if device is None:
        device = DEVICE

    for m in M:
        for n in N:
            tensors.append(torch.randn((m, n)), dtype=dtype, device=device)

    return tensors


def autotune(kernel, kernel_args=None, M=None, N=None, force_autotune=False):
    """
    Runs autotuning for a Helion kernel and saves the optimal configurations to disk.

    This function iterates through a set of input tensors and utilizes the
    Helion kernel's autotuning feature to find the best hardware-specific
    configurations for the provided kernel. Results are cached in the 'configs/' directory
    to avoid redundant tuning in future runs.

    Args:
        kernel: The Helion kernel instance to be autotuned.
        kernel_args: A nested iterable (e.g., list of lists, tuple of tuples) where
            each inner iterable contains arguments to tune against. If None, tensors will
            be generated using `generate_tensors(M, N)`.
        M: A sequence of tensor batch dimensions to tune against. Only used if kernel_args is None.
        N: A sequence of tensor feature dimensions to tune against. Only used if kernel_args is None.
        force_autotune: If True, ignores existing configuration files
            and re-runs the tuning process. Defaults to False.
    """
    if kernel_args is None:
        if M is None or N is None:
            raise RuntimeError(
                "Either a list of arguments or a list of tensor dimensions must be provided."
            )
        kernel_args = generate_tensors(M, N)

    for args in kernel_args:
        signature = get_signature(args)
        config_name = f"configs/{kernel.__name__}-{signature}.json"
        if os.path.exists(config_name) and not force_autotune:
            print(f"Config already exists for argument sizes {signature}.")
            continue
        config = kernel.autotune(args,force=force_autotune)
        config.save(config_name)
        compiled_code = kernel.bind(args).to_triton_code(config)
        #store compiled triton code according to generated config
        with open(f"configs/{kernel.__name__}-{signature}-triton.txt","w") as f:
            f.write(compiled_code)

rmsnorm_configs = [ 
        (
            #small config
            torch.randn((256, 256), dtype= torch.float16, device=DEVICE),
            torch.randn((256,), dtype= torch.float16, device=DEVICE),
            1e-5,
        )
        (
            #medium config
            torch.randn((1024,1024), dtype= torch.float16,device=DEVICE),
            torch.randn((1024,), dtype= torch.float16,device = DEVICE),
            1e-5
        ),
        (
            #large config
            torch.randn((8192,8192), dtype= torch.float16, device=DEVICE),
            torch.randn((8192,), dtype= torch.float16,device = DEVICE),
            1e-5
        )
]


if __name__ == "__main__":
    if(os.path.exists('configs') == False):
        os.mkdir('configs')
    #rmsnorm_config_block
    rmsnorm_functions = [helion_rms_kernel]
    for func in rmsnorm_functions:
        autotune(func, rmsnorm_configs,force_autotune=True)