# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Flattened, self-contained reimplementation of the SM80 Flash Attention Forward kernel.
# Original sources:
#   - https://github.com/Dao-AILab/flash-attention  (flash_attn/cute/)
#   - https://github.com/Dao-AILab/quack            (quack/)
#   - https://github.com/NVIDIA/cutlass              (CuTe DSL examples)
#
# Dependencies: nvidia-cutlass-dsl>=4.2.0, cuda-python, torch (with CUDA)
# No flash_attn or quack packages required.

import math
import enum
import operator
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial
from dataclasses import dataclass, fields

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Constexpr, Float32, Int32, const_expr, Boolean
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils_basic
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import NumericMeta
from cutlass.base_dsl.typing import JitArgument
from cutlass.cute import FastDivmodDivisor

# =============================================================================
# NamedBarrierFwd  (from flash_attn.cute.named_barrier)
# =============================================================================

class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()


# =============================================================================
# ParamsBase  (from quack.cute_dsl_utils)
# Needed as base class for Softmax so CuTe DSL can extract MLIR values.
# =============================================================================

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


# =============================================================================
# SeqlenInfoQK  (from flash_attn.cute.seqlen_info)
# =============================================================================

@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: cutlass.Int32
    offset_k: cutlass.Int32
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
    ):
        return SeqlenInfoQK(
            offset_q=0,
            offset_k=0,
            seqlen_q=seqlen_q_static,
            seqlen_k=seqlen_k_static,
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
        )


# =============================================================================
# BlockInfo  (from flash_attn.cute.block_info)
# =============================================================================

@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: cutlass.Int32 = 0,
        num_splits: cutlass.Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        if cutlass.const_expr(self.is_split_kv):
            num_n_blocks_per_split = (
                cutlass.Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)


# =============================================================================
# Layout utilities  (from quack.layout_utils)
# =============================================================================

def _transpose_view(a: cute.Tensor) -> cute.Tensor:
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def _convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def _reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def _convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # SM80: (4, MMA_M, MMA_N) -> ((4, 2), MMA_M, MMA_N / 2)
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def _reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_frgA(acc.layout))


# =============================================================================
# Utilities  (from flash_attn.cute.utils — SM80-relevant subset)
# =============================================================================

@dsl_user_op
def _fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    from cutlass import CUDA_VERSION
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc, ip=ip,
            )
        )
    else:
        return Float32(
            nvvm.fmax(
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc, ip=ip,
            )
        )


@cute.jit
def _fmax_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = _fmax(local_max[0], res[i + 0])
        local_max[1] = _fmax(local_max[1], res[i + 1])
        local_max[2] = _fmax(local_max[2], res[i + 2])
        local_max[3] = _fmax(local_max[3], res[i + 3])
    local_max[0] = _fmax(local_max[0], local_max[1])
    local_max[2] = _fmax(local_max[2], local_max[3])
    local_max[0] = _fmax(local_max[0], local_max[2])
    return local_max[0] if const_expr(init_val is None) else _fmax(local_max[0], init_val)


@cute.jit
def _fadd_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@cute.jit
def _warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = _warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def _predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def _get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


def _make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def _make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


@cute.jit
def _shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@cute.jit
def _scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    vec = cute.make_fragment(1, dtype)
    vec[0] = a
    return vec.load()


def _ssa_to_scalar(val):
    return val[0]


# =============================================================================
# Ampere GEMM helpers  (from flash_attn.cute.ampere_helpers)
# =============================================================================

def _get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        cutlass.const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size), order=(1, 0)
        ),
    )


@cute.jit
def _ampere_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(swap_AB):
        _ampere_gemm(
            tiled_mma, acc, tCrB, tCrA, tCsB, tCsA,
            smem_thr_copy_B, smem_thr_copy_A, hook_fn,
            A_in_regs=B_in_regs, B_in_regs=A_in_regs, swap_AB=False,
        )
    else:
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        if cutlass.const_expr(not A_in_regs):
            cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
        if cutlass.const_expr(not B_in_regs):
            cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                if cutlass.const_expr(not A_in_regs):
                    cute.copy(
                        smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1]
                    )
                if cutlass.const_expr(not B_in_regs):
                    cute.copy(
                        smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
                    )
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            if cutlass.const_expr(k == 0 and hook_fn is not None):
                hook_fn()


@cute.jit
def _ampere_gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()


# =============================================================================
# Softmax  (from flash_attn.cute.softmax — SM80 only)
# =============================================================================

@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = _reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = _fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_sum = _fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = _fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2
        row_sum.store(_warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                row_sum[r] += cute.math.exp2(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
                )
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = _reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


# =============================================================================
# AttentionMask  (from flash_attn.cute.mask — SM80 apply_mask only)
# Adapted to match the kernel's constructor: (tile_m, tile_n, seqlen_q, seqlen_k, ...)
# =============================================================================

@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_q: Int32
    seqlen_k: Int32
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        fastdiv_mods=None,
    ) -> None:
        assert not (mask_causal and mask_local)
        acc_S_mn = _reshape_acc_to_mn(acc_S)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape)
        tScS_mn = _reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0ScS_mn = _reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
        else:
            threads_per_row = thr_mma.tv_layout_C.shape[0][0]
            mma_m_idx = None
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert cute.arch.WARP_SIZE % threads_per_row == 0
                assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                tidx = thr_mma.thr_idx
                mma_m_idx = (
                    m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                ) // self.qhead_per_kvhead_packgqa
            causal_row_offset = (
                1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
            )
            if const_expr(mask_causal):
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    if const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                    else:
                        row_idx = _shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    col_limit_right = row_idx + causal_row_offset
                    if const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        acc_S_mn[r, c] = (
                            -Float32.inf
                            if t0ScS_mn[0, c][1] >= col_limit_right
                            else acc_S_mn[r, c]
                        )
            else:  # Local
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    if const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                    else:
                        row_idx = _shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    if const_expr(self.window_size_right is not None):
                        col_limit_right = row_idx + local_row_offset_right
                    else:
                        col_limit_right = self.tile_n
                    if const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                    col_limit_left = (
                        row_idx + local_row_offset_left
                        if const_expr(self.window_size_left is not None)
                        else 0
                    )
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col_idx = t0ScS_mn[0, c][1]
                        if col_idx >= col_limit_right or col_idx < col_limit_left:
                            acc_S_mn[r, c] = -Float32.inf


# =============================================================================
# Tensor alignment helper  (from flash_attn.cute.cute_dsl_utils)
# =============================================================================

def _assume_strides_aligned(t):
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])

def _assume_tensor_aligned(t):
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=_assume_strides_aligned(t)))


def _to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=False):
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


# =============================================================================
# PackGQA stub — only constructed, never invoked when pack_gqa=False
# =============================================================================

class PackGQA:
    def __init__(self, tile_m, tile_hdimv, check_hdim_v_oob, qhead_per_kvhead):
        self.tile_m = tile_m
        self.tile_hdimv = tile_hdimv
        self.check_hdim_v_oob = check_hdim_v_oob
        self.qhead_per_kvhead = qhead_per_kvhead

    def store_LSE(self, *args, **kwargs):
        raise NotImplementedError("PackGQA.store_LSE requires pack_gqa=True path")

    def store_O(self, *args, **kwargs):
        raise NotImplementedError("PackGQA.store_O requires pack_gqa=True path")


# =============================================================================
# FlashAttentionForwardBase
# =============================================================================

class FlashAttentionForwardBase:
    arch: int = 80

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = False,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int | None = None,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.q_subtile_factor = q_subtile_factor
        self.Q_in_regs = Q_in_regs
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = False  # SM80 only
        self.aux_tensors = None
        self.fastdiv_mods = None

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, tile_m, tile_n, num_stages, num_threads, is_causal, Q_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0 or num_threads % 32 != 0:
            return False
        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type, mK_type, mV_type, mO_type, mLSE_type,
        mCuSeqlensQ_type=None, mCuSeqlensK_type=None,
        mSeqUsedQ_type=None, mSeqUsedK_type=None,
    ):
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = (
            self._get_smem_layout_atom()
        )
        self.sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1))
        self.sK_layout = cute.tile_to_shape(sK_layout_atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2))
        self.sV_layout = cute.tile_to_shape(sV_layout_atom, (self.tile_n, self.tile_hdimv, self.num_stages), (0, 1, 2))
        self.sO_layout = cute.tile_to_shape(sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1))
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(sP_layout_atom, (self.tile_m, self.tile_n), (0, 1))
        else:
            self.sP_layout = None

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0
        assert self.num_producer_threads % tQK_shape_dim_1 == 0
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        assert self.tile_m % tO_layout.shape[0] == 0
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout
        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    def _get_smem_layout_atom(self):
        raise NotImplementedError()

    def _get_tiled_mma(self):
        raise NotImplementedError()

    def _get_shared_storage_cls(self):
        raise NotImplementedError()

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, mLSE, softmax_scale, stream):
        raise NotImplementedError()

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
        )
        smem_copy_atom_O = _get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )

        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = _reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = _reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = _reshape_acc_to_mn(tiled_mma.get_slice(0).partition_C(cO))
                if taccOcO[0][1] == 0:
                    for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])

        # SM80: use_tma_O is always False (arch < 90)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(not self.pack_gqa):
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = _predicate_k(tOcO, limit=mO.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(self, gmem_thr_copy, gQ, sQ, block, seqlen, headdim):
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = _predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                )

    @cute.jit
    def load_K(
        self, gmem_tiled_copy, tKgK, tKsK, tKcK, t0KcK, tKpK, block, smem_pipe_write, seqlen, need_predicates,
    ):
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.tile_n
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.tile_n
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.tile_n, self.tile_n)
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def load_V(
        self, gmem_tiled_copy, tVgV, tVsV, tVcV, t0VcV, tVpV, block, smem_pipe_write, seqlen, need_predicates,
    ):
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if (
                    is_even_n_smem_v
                    or n < cute.size(tVsV.shape[1]) - 1
                    or tVcV[0, n, 0][0] < self.tile_n
                ):
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (
                                    tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )


# =============================================================================
# FlashAttentionForwardSm80
# =============================================================================

class FlashAttentionForwardSm80(FlashAttentionForwardBase):

    def _get_smem_layout_atom(self):
        sQ_layout_atom = _get_smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = _get_smem_layout_atom(self.dtype, self.tile_hdimv)
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream,
        softmax_scale: Optional[Float32] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        aux_tensors=None,
    ):
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        self._check_type(
            *(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE))
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        mQ, mK, mV, mO = [_assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (mQ, mK, mV, mO)
        ]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        grid_dim = (
            cute.ceil_div(mQ.shape[0], self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = Float32(softmax_scale * LOG2_E)
            softmax_scale = Float32(0.0)
        else:
            softmax_scale_log2 = Float32(LOG2_E)
            softmax_scale = Float32(softmax_scale)

        self.kernel(
            mQ, mK, mV, mO, mLSE, softmax_scale_log2, softmax_scale,
            window_size_left, window_size_right,
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout,
            self.gmem_tiled_copy_Q, self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V, self.gmem_tiled_copy_O,
            tiled_mma_qk, tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mO: cute.Tensor, mLSE: cute.Tensor,
        softmax_scale_log2: Float32, softmax_scale: Float32,
        window_size_left, window_size_right,
        sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy, gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy, gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        fastdiv_mods = self.fastdiv_mods
        aux_tensors = self.aux_tensors

        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        block_info = BlockInfo(
            self.tile_m, self.tile_n, self.is_causal, self.is_local,
            False, window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfoQK.create(seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0])
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        n_block = n_block_max - 1

        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        gQ = cute.local_tile(mQ[None, None, num_head, batch_size], blkQ_shape, (m_block, 0))
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[None, None, num_head_kv, batch_size], blkK_shape, (None, 0))
        gV = cute.local_tile(mV[None, None, num_head_kv, batch_size], blkV_shape, (None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        sVt = _transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        acc_O.fill(0.0)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
        )
        smem_thr_copy_Q = _make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = _make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = _make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        tKpK = _predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = _predicate_k(tVcV, limit=mV.shape[1])

        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk, thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q, smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ, tSsK=tSsK, tOsVt=tOsVt,
        )
        load_K = partial(
            self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK, seqlen=seqlen.seqlen_k
        )
        load_V = partial(
            self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV, seqlen=seqlen.seqlen_k
        )

        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params, smem_copy_params=smem_copy_params,
            softmax=softmax, load_K=load_K, load_V=load_V,
            score_mod=self.score_mod,
            batch_idx=batch_size, head_idx=num_head, m_block=m_block,
            aux_tensors=aux_tensors, fastdiv_mods=fastdiv_mods,
        )

        # ---- Prologue ----
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ---- Mainloop ----
        mask = AttentionMask(
            self.tile_m, self.tile_n,
            seqlen.seqlen_q, seqlen.seqlen_k,
            window_size_left, window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask,
            m_block=m_block, thr_mma=thr_mma_qk,
            mask_causal=self.is_causal, mask_local=self.is_local,
            fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
        )

        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(
            n_block, smem_pipe_read, smem_pipe_write,
            is_first_n_block=True, check_inf=True,
            mask_fn=partial(mask_fn, mask_seqlen=True),
        )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(
                    n_block, smem_pipe_read, smem_pipe_write,
                    check_inf=True,
                    mask_fn=partial(mask_fn, mask_seqlen=False),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1, smem_pipe_read, smem_pipe_write, check_inf=True
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ---- Epilogue ----
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
            gmem_tiled_copy_O, None, tiled_mma_pv,
            tidx, m_block, num_head, batch_size,
        )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)
        sync()

        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1, smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        load_V_next()
        _ampere_gemm(
            mma_params.thr_mma_qk, acc_S,
            mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )

        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()

        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = _reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        _ampere_gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_O,
            tOrP, mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_V,
        )


# =============================================================================
# PyTorch invocation wrapper
# =============================================================================

import torch

_torch2cute_dtype = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

_compile_cache: dict = {}


def flash_attn_sm80_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    num_threads: int = 128,
    num_stages: int = 1,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Self-contained Flash Attention v2 forward pass for SM80 (Ampere) via CuTe DSL.

    Args:
        q, k, v: [B, S, H, D] contiguous fp16/bf16 CUDA tensors.
        softmax_scale: Defaults to 1/sqrt(D).
        causal: Apply causal mask.
        out: Optional pre-allocated output tensor.
    Returns:
        (out, lse) where lse is [B, H, S] float32 or None.
    """
    q, k, v = [t.contiguous() for t in (q, k, v)]
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    batch, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, num_heads_kv, head_dim_v = v.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    dtype = _torch2cute_dtype[q.dtype]
    device = q.device

    if out is None:
        out = torch.empty(batch, seqlen_q, num_heads, head_dim_v, dtype=q.dtype, device=device)
    # Always allocate LSE — the kernel is compiled with a concrete tensor so CuTe DSL
    # never sees None for mLSE (which would crash during tracing since @cute.jit
    # doesn't do Python-level None branch elimination).  The cost is negligible
    # (B*H*S float32).  We simply discard the result if the caller didn't ask for it.
    lse = torch.empty(batch, num_heads, seqlen_q, dtype=torch.float32, device=device)

    qhead_per_kvhead = num_heads // num_heads_kv

    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if local and window_size_left is None and window_size_right == 0:
        causal, local = True, False
        window_size_right = None

    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local,
        window_size_left is not None, window_size_right is not None,
        tile_m, tile_n, num_stages, num_threads,
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compile_key not in _compile_cache:
        fa_fwd = FlashAttentionForwardSm80(
            dtype, head_dim, head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal, is_local=local, pack_gqa=False,
            tile_m=tile_m, tile_n=tile_n,
            num_stages=num_stages, num_threads=num_threads,
            Q_in_regs=False,
        )

        q_tensor = _to_cute_tensor(q)
        k_tensor = _to_cute_tensor(k)
        v_tensor = _to_cute_tensor(v)
        o_tensor = _to_cute_tensor(out)
        lse_tensor = _to_cute_tensor(lse, assumed_align=4)

        softmax_scale_f32 = Float32(softmax_scale)
        _compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor, k_tensor, v_tensor, o_tensor,
            lse_tensor,
            current_stream,
            softmax_scale_f32,
            window_size_left,
            window_size_right,
            None,  # learnable_sink
            None,  # aux_tensors
        )

    _compile_cache[compile_key](
        q.detach(), k.detach(), v.detach(), out.detach(),
        lse,
        current_stream,
        Float32(softmax_scale),
        window_size_left,
        window_size_right,
        None,  # learnable_sink
        None,  # aux_tensors
    )

    return out, lse if return_lse else None
