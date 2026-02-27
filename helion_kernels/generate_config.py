import os

import torch

from helion_kernels.rmsnorm.basic_rmsnorm import helion_rms_kernel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def generate_tensors(M: list[int], N: list[int], dtype=torch.float32, device=None):
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
        signature = str(tensor.size() for tensor in args)
        config_name = f"configs/{kernel.__name__}-{signature}.json"
        if os.path.exists(config_name) and not force_autotune:
            print(f"Config already exists for argument sizes {signature}.")
            continue
        config = kernel.autotune(args)
        config.save(config_name)


if __name__ == "__main__":
    kernel_args = [
        (
            torch.randn((400, 800), device="cuda"),
            torch.randn((800,), device="cuda"),
            1e-5,
        )
    ]
    autotune(helion_rms_kernel, kernel_args)
