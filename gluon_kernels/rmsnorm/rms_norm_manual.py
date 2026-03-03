import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
THREADS_PER_WARP = properties["warpSize"]

# This is an auto tunable property using @triton.autotune
MAX_WARP_PER_CTA_COUNT = 8

@gluon.jit
def rms_norm_kernel(
    inp_ptr,
    out_ptr,
    epsilon,
    gamma_ptr,
    n_rows,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: gl.constexpr,
    layout: gl.constexpr,
    ):

    ## Identify which program are you
    row_start = gl.program_id(0)
    row_step = gl.num_programs(0)

    for row_id in range(row_start, n_rows, row_step):
        input_start_ptr = inp_ptr + row_id * input_stride
        offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
        
        inp_ptrs = input_start_ptr + offsets

        mask = inp_ptrs < n_cols

        inp = gl.load(inp_ptrs, mask=mask, other=0.0)
        gamma = gl.load(gamma_ptr + offsets, mask=mask, other=1.0)

        ## This makes the assumption that the entire array length has been computed within this program
        rms = gl.sqrt(gl.sum(inp * inp) / n_cols + epsilon)
        rms_norm = inp / rms * gamma

        output_start_ptr = out_ptr + row_id * output_stride
        output_ptrs = output_start_ptr + offsets
        gl.store(output_ptrs, rms_norm, mask=mask)


def rms_norm(x, epsilon=None, gamma=None):

    n_rows, n_cols = x.shape

    if epsilon is None:
        epsilon = 1e-6
    
    if gamma is None:
        gamma = torch.ones((n_cols, ), device=x.device, dtype=x.dtype)

    block_size = triton.next_power_of_2(n_cols)
    warp_per_cta = gl.max(1, gl.min(MAX_WARP_PER_CTA_COUNT, block_size // THREADS_PER_WARP))
    size_per_thread = gl.max(1, block_size // (THREADS_PER_WARP * warp_per_cta))

    layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread],
        threads_per_warp=[THREADS_PER_WARP],
        warps_per_cta=[warp_per_cta],
        order=[0]
    )

    y = torch.empty_like(x)

    # Compile the kernel to get info about register assignments
    kernel = rms_norm_kernel(
        x,
        y,
        epsilon,
        gamma,
        n_rows,
        n_cols,
        x.stride(0),
        y.stride(0),
        block_size,
        layout
    )
    kernel.__init_handles()
    n_regs = kernel.n_regs
    size_smem_per_CTA = kernel.shared

    operations = gl.min(NUM_REGS // (n_regs * THREADS_PER_WARP * warp_per_cta), 
        SIZE_SMEM // size_smem_per_CTA)
    hardware_bottleneck = NUM_SM * operations

    grid = (gl.min(hardware_bottleneck, n_rows),)

    rms_norm[grid](
        x,
        y,
        epsilon,
        gamma,
        n_rows,
        n_cols,
        x.stride(0),
        y.stride(0),
        block_size,
        layout
    )

    return y

    


