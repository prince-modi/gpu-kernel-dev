# Bhrugu Bharathi, A16641798, 02/25/2026

# Optimized RMS norm kernel forward using CuTe library for Hopper architecture.
# Takes advantage of thread-block clusters for cross-CTA reduction via DSMEM.
# Derived from quack (https://github.com/Dao-AILab/quack/blob/main/quack/rmsnorm.py)
# and the CuTe DSL tutorial (https://veitner.bearblog.dev/simple-reduction-in-cutedsl/).

import math
import operator
from functools import partial
from typing import Callable, Optional, Type

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

# =====================================================================================
# DSMEM Utilities (Hopper-specific, from quack.utils)
# =====================================================================================

@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =====================================================================================
# Copy Utilities
# =====================================================================================

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


# =====================================================================================
# Layout Utilities
# =====================================================================================

@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
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


# =====================================================================================
# Reduction Utilities
# =====================================================================================

@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """Inter-warp reduction within a single CTA.
    reduction_buffer shape: (rows_per_cta, (warps_per_row, 1)) when cluster_n == 1.
    """
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
) -> cute.Numeric:
    """Cross-CTA reduction within a thread-block cluster via DSMEM and mbarrier.
    reduction_buffer shape: (rows_per_cta, (warps_per_row, cluster_n))
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    if const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """Per-row reduction with optional cross-cluster synchronization.
    reduction_buffer shape: (rows_per_cta, (warps_per_row, cluster_n))
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
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val, warp_op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val
            )
    return val


# =====================================================================================
# Configuration
# =====================================================================================

def _threads_per_row(N):
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            return threads
    return 256


def _num_threads(N):
    return 128 if N <= 4096 else 256


def _set_cluster_n(N, dtype_width):
    """Determine thread-block cluster size based on N and dtype bit-width."""
    if dtype_width == 16:
        thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
    else:
        thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
    for limit, cluster in thresholds:
        if N <= limit:
            return cluster
    return 16


def _get_tiled_copy(dtype, N, cluster_n=1, vecsize=1):
    """Build tiled copy and per-CTA tile shape. N is divided by cluster_n."""
    assert N % vecsize == 0, f"N={N} is not divisible by vecsize={vecsize}"
    threads_per_row = _threads_per_row(N)
    num_threads = _num_threads(N)
    assert num_threads % 32 == 0
    num_blocks_N = cute.ceil_div(N // (cluster_n * vecsize), threads_per_row)
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


# =====================================================================================
# Kernel
# =====================================================================================

@cute.kernel
def rms_norm_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    eps: Float32,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    cluster_y = const_expr(0) if const_expr(cluster_n == 1) else cute.arch.block_idx()[1]

    smem = cutlass.utils.SmemAllocator()

    sX = smem.allocate_tensor(
        mX.element_type,
        cute.make_ordered_layout(tiler_mn, order=(1, 0)),
        byte_alignment=16,
    )

    # Reduction buffer: hierarchical (warps_per_row, cluster_n) second mode.
    # When cluster_n == 1 this degrades to the A100-equivalent layout.
    warps_per_row = const_expr(max(1, threads_per_row // cute.arch.WARP_SIZE))
    rows_per_cta = const_expr(tiler_mn[0])
    if const_expr(warps_per_row > 1 or cluster_n > 1):
        reduction_buffer = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((rows_per_cta, (warps_per_row, cluster_n))),
            byte_alignment=8,
        )
    else:
        reduction_buffer = None

    # Mbarrier for cross-cluster synchronization (1 stage for RMS norm forward)
    if const_expr(cluster_n > 1):
        mbar_ptr = smem.allocate_array(Int64, num_elems=1)
    else:
        mbar_ptr = None

    shape = mX.shape

    idX = cute.make_identity_tensor(shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
    gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
    gO = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
    cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

    thr_copy = tiled_copy.get_slice(tidx)

    tXgX = thr_copy.partition_S(gX)
    tXsX = thr_copy.partition_D(sX)
    tXgW = thr_copy.partition_S(gW)
    tXgO = thr_copy.partition_D(gO)
    tXcX = thr_copy.partition_S(cX)[(0, None), None, None]

    tXrX = cute.make_fragment_like(tXgX)
    tXrW = cute.make_fragment_like(tXgW)
    tXrO = cute.make_fragment_like(tXgO)

    # Initialize cluster: mbarrier init + non-blocking cluster arrive.
    # The matching cluster_wait is deferred to hook_fn inside row_reduce,
    # ensuring all CTAs have finished mbarrier_init before any reduction begins.
    if const_expr(cluster_n > 1):
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

    is_even_N = const_expr(shape[1] == tiler_mn[1] * cluster_n)
    tXpX = (
        predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        if not is_even_N
        else None
    )
    pred_copy = partial(copy, pred=tXpX)

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

    # --- Reduction: sum(x^2) across threads_per_row threads + across cluster ---
    sum_sq = row_reduce(
        x * x,
        cute.ReductionOp.ADD,
        threads_per_row,
        reduction_buffer,
        mbar_ptr,
        init_val=0.0,
        hook_fn=cute.arch.cluster_wait if const_expr(cluster_n > 1) else None,
    )
    rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

    # --- Epilogue: normalize, scale, write output ---
    y = x * rstd * tXrW.load().to(Float32)
    tXrO.store(y.to(tXrO.element_type))
    if row < shape[0]:
        pred_copy(tXrO, tXgO)


# =====================================================================================
# Launch
# =====================================================================================

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
    cluster_n = _set_cluster_n(N, dtype.width)
    vecsize = math.gcd(N, 128 // dtype.width)
    tiled_copy, tiler_mn, threads_per_row, num_threads = _get_tiled_copy(
        dtype, N, cluster_n, vecsize
    )

    mW_expanded = expand(mW, dim=0, size=tiler_mn[0])

    rms_norm_kernel(
        mX, mW_expanded, mY, eps, tiler_mn, tiled_copy, threads_per_row, cluster_n
    ).launch(
        grid=[cute.ceil_div(M, tiler_mn[0]), cluster_n, 1],
        block=[num_threads, 1, 1],
        cluster=[1, cluster_n, 1] if const_expr(cluster_n > 1) else None,
    )


# =====================================================================================
# Testing
# =====================================================================================

if __name__ == "__main__":
    device = "cuda"
    eps = 1e-5
    USE_FP16 = True
    dtype_width = 16 if USE_FP16 else 32

    dtype = torch.float16 if USE_FP16 else torch.float32
    atol = 1e-2 if USE_FP16 else 1e-5
    rtol = 1e-2 if USE_FP16 else 1.3e-6
    print(f"Running with dtype={dtype}")

    test_configs = [
        # cluster_n=1 (single-CTA reduction, same behavior as A100)
        (256, 64),       # threads_per_row=8
        (256, 128),      # threads_per_row=16
        (1024, 1024),    # threads_per_row=32
        (1024, 2048),    # threads_per_row=32
        (4096, 4096),    # threads_per_row=64
        (4096, 8192),    # threads_per_row=128
        (65536, 1024),   # threads_per_row=32, large M
        # cluster_n > 1 (cross-CTA reduction via DSMEM + mbarrier)
        (1024, 32768),   # cluster_n=2 (fp16), threads_per_row=256
        (512, 65536),    # cluster_n=4 (fp16), threads_per_row=256
        (256, 131072),   # cluster_n=8 (fp16), threads_per_row=256
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

        cluster_n = _set_cluster_n(N, dtype_width)
        try:
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
            print(f"  PASSED: M={M:>6}, N={N:>6}  (threads_per_row={_threads_per_row(N)}, cluster_n={cluster_n})")
        except AssertionError as e:
            print(f"  FAILED: M={M:>6}, N={N:>6}  (threads_per_row={_threads_per_row(N)}, cluster_n={cluster_n})")
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
