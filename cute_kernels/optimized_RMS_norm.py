# Bhrugu Bharathi, A16641798, 02/25/2026

# The objective here is to implement an optimized RMS norm kernel forward using the CuTe library. This implementation is heavily derived from the quack implementation in https://github.com/Dao-AILab/quack/blob/main/quack/rmsnorm.py as well as the tutorial in https://veitner.bearblog.dev/simple-reduction-in-cutedsl/.

import math
import operator
from functools import partial
from typing import Callable, Optional, Type

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync


################################################################################
################################################################################
# Below are borrowed utilities from the quack library. Thread Block Cluster handling has been removed.
################################################################################
################################################################################

@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Inter-warp reduction within a single CTA (no cluster).
    reduction_buffer shape: (rows_per_cta, warps_per_row)
    """
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = reduction_buffer.shape[1]
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    reduced_val = init_val
    if lane_idx < warps_per_row:
        reduced_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(reduced_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Per-row reduction across threads_per_row threads.
    When threads_per_row <= WARP_SIZE: intra-warp reduction only.
    When threads_per_row > WARP_SIZE: intra-warp then inter-warp via reduction_buffer.
    reduction_buffer shape: (rows_per_cta, warps_per_row), required when threads_per_row > WARP_SIZE.
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(reduction_buffer is not None):
        if const_expr(reduction_buffer.shape[1] > 1):
            val = block_reduce(val, warp_op, reduction_buffer, init_val=init_val)
    return val

################################################################################
################################################################################
# End of borrowed utilities from the quack library.
################################################################################
################################################################################


def _threads_per_row(N):
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            return threads
    return 256


def _num_threads(N):
    return 128 if N <= 4096 else 256


def _get_tiled_copy(dtype, N, vecsize=1):
    assert N % vecsize == 0, f"N={N} is not divisible by vecsize={vecsize}"
    threads_per_row = _threads_per_row(N)
    num_threads = _num_threads(N)
    assert num_threads % 32 == 0
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
    tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
    tc = tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize)
    return tc, tiler_mn, threads_per_row, num_threads


def benchmark(compiled, mX, mW, mY, eps):
    avg_time_us = cute.testing.benchmark(
        compiled,
        kernel_arguments=cute.testing.JitArguments(mX, mW, mY, Float32(eps)),
        warmup_iterations=100,
        iterations=1000,
    )
    print(f"Kernel execution time: {avg_time_us:.4f} us")

@cute.kernel
def rms_norm_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    eps: Float32,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()

    # Shared memory for X: row-major (N dim is stride-1, contiguous)
    sX = smem.allocate_tensor(
        mX.element_type,
        cute.make_ordered_layout(tiler_mn, order=(1, 0)),
        byte_alignment=16,
    )

    # Reduction buffer for inter-warp reduction (only needed when threads_per_row > WARP_SIZE)
    warps_per_row = const_expr(max(1, threads_per_row // cute.arch.WARP_SIZE))
    rows_per_cta = const_expr(tiler_mn[0])
    if const_expr(warps_per_row > 1):
        reduction_buffer = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((rows_per_cta, warps_per_row)),
            byte_alignment=16,
        )
    else:
        reduction_buffer = None

    shape = mX.shape

    # idX is a coordinate tensor where element [i,j] holds the coordinate (i,j), used for bounds checking
    idX = cute.make_identity_tensor(shape)
    # gX is this CTA's tile of the input tensor X in gmem: rows [bidx*rows_per_cta, (bidx+1)*rows_per_cta), all N columns
    gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
    # gW is the weight tile in gmem: stride-0 in row dim, so all rows see the same weight vector
    gW = cute.local_tile(mW, tiler_mn, (0, 0))
    # gO is this CTA's tile of the output tensor Y in gmem, same row range as gX
    gO = cute.local_tile(mY, tiler_mn, (bidx, 0))
    # cX is this CTA's tile of the coordinate tensor, used to recover global row/col indices per thread
    cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

    # thr_copy is this thread's view of the tiled copy, mapping thread index to data elements
    thr_copy = tiled_copy.get_slice(tidx)

    # tXgX is this thread's partition of gX in gmem (source for async copy to smem)
    tXgX = thr_copy.partition_S(gX)
    # tXsX is this thread's partition of sX in smem (destination for async copy from gmem)
    tXsX = thr_copy.partition_D(sX)
    # tXgW is this thread's partition of gW in gmem (source for sync copy to registers)
    tXgW = thr_copy.partition_S(gW)
    # tXgO is this thread's partition of gO in gmem (destination for final output write)
    tXgO = thr_copy.partition_D(gO)
    # tXcX is this thread's partition of the coordinate tensor, sliced to extract the row index
    tXcX = thr_copy.partition_S(cX)[(0, None), None, None]

    # tXrX holds this thread's X elements in registers after loading from smem
    tXrX = cute.make_fragment_like(tXgX)
    # tXrW holds this thread's weight elements in registers after loading from gmem
    tXrW = cute.make_fragment_like(tXgW)
    # tXrO holds this thread's output elements in registers before writing to gmem
    tXrO = cute.make_fragment_like(tXgO)

    # Predication for uneven N
    is_even_N = const_expr(shape[1] == tiler_mn[1])
    tXpX = (
        predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        if not is_even_N
        else None
    )
    pred_copy = partial(copy, pred=tXpX)

    # Bounds check: which row does this thread's first element belong to?
    row = tXcX[0][0]

    # --- Async copy: gmem X -> smem ---
    if row < shape[0]:
        pred_copy(tXgX, tXsX, is_async=True)
    cute.arch.cp_async_commit_group()

    # --- Sync copy: gmem W -> registers ---
    pred_copy(tXgW, tXrW)

    # --- Wait for async X copy, then smem -> registers ---
    cute.arch.cp_async_wait_group(0)
    cute.autovec_copy(tXsX, tXrX)
    x = tXrX.load().to(Float32)

    # --- Reduction: sum(x^2) across threads_per_row threads ---
    sum_sq = row_reduce(
        x * x,
        cute.ReductionOp.ADD,
        threads_per_row,
        reduction_buffer=reduction_buffer,
        init_val=0.0,
    )
    rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

    # --- Epilogue: normalize, scale, write output ---
    y = x * rstd * tXrW.load().to(Float32)
    tXrO.store(y.to(tXrO.element_type))
    if row < shape[0]:
        pred_copy(tXrO, tXgO)


@cute.jit
def cute_rms_norm(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    eps: Float32,
):
    dtype = mX.element_type
    vecsize = math.gcd(N, 128 // dtype.width)
    tiled_copy, tiler_mn, threads_per_row, num_threads = _get_tiled_copy(dtype, N, vecsize)

    mW_expanded = expand(mW, dim=0, size=tiler_mn[0])

    rms_norm_kernel(mX, mW_expanded, mY, eps, tiler_mn, tiled_copy, threads_per_row).launch(
        grid=[cute.ceil_div(M, tiler_mn[0]), 1, 1],
        block=[num_threads, 1, 1],
    )


if __name__ == "__main__":
    device = "cuda"
    eps = 1e-5
    USE_FP16 = True  # Toggle: True for fp16, False for fp32

    dtype = torch.float16 if USE_FP16 else torch.float32
    atol = 1e-2 if USE_FP16 else 1e-5
    rtol = 1e-2 if USE_FP16 else 1.3e-6
    print(f"Running with dtype={dtype}")

    test_configs = [
        (256, 64),      # threads_per_row=8
        (256, 128),     # threads_per_row=16
        (1024, 1024),   # threads_per_row=32
        (1024, 2048),   # threads_per_row=32
        (4096, 4096),   # threads_per_row=64
        (4096, 8192),   # threads_per_row=128
        (65536, 1024),  # threads_per_row=32, large M
    ]

    print("=== Correctness Tests ===")
    for M, N in test_configs:
        x = torch.randn(M, N, device=device, dtype=dtype)
        w = torch.randn(N, device=device, dtype=dtype)
        y = torch.zeros(M, N, device=device, dtype=dtype)

        ref_rms_norm = nn.RMSNorm(N, eps=eps, device=device, dtype=dtype)
        with torch.no_grad():
            ref_rms_norm.weight.copy_(w)
        y_ref = ref_rms_norm(x)

        mX = from_dlpack(x, assumed_align=16)
        mW = from_dlpack(w, assumed_align=16)
        mY = from_dlpack(y, assumed_align=16)

        cute_rms_norm(mX, mW, mY, M, N, eps)

        try:
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
            print(f"  PASSED: M={M:>6}, N={N:>6}  (threads_per_row={_threads_per_row(N)})")
        except AssertionError as e:
            print(f"  FAILED: M={M:>6}, N={N:>6}  (threads_per_row={_threads_per_row(N)})")
            print(f"          {e}")

    print("\n=== Benchmark ===")
    M_bench, N_bench = 20000, 3072
    x = torch.randn(M_bench, N_bench, device=device, dtype=dtype)
    w = torch.randn(N_bench, device=device, dtype=dtype)
    y = torch.zeros(M_bench, N_bench, device=device, dtype=dtype)

    mX = from_dlpack(x, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)

    compiled = cute.compile(cute_rms_norm, mX, mW, mY, M_bench, N_bench, Float32(0))
    benchmark(compiled, mX, mW, mY, eps)
