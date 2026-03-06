# triton_kernels/flashattn/flash_attn_v2.py
import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_ptr,
    V_ptr,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            K_ptr
            + (start_n + offs_n[None, :]) * stride_kn
            + tl.arange(0, HEAD_DIM)[:, None] * stride_kd
        )
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(
            V_ptr
            + (start_n + offs_n[:, None]) * stride_vn
            + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_on,
    stride_od,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z, off_h = off_hz // H, off_hz % H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    Out += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(
        Q + offs_m[:, None] * stride_qn + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K,
            V,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K,
            V,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
        )

    acc = acc / l_i[:, None]
    tl.store(
        Out + offs_m[:, None] * stride_on + tl.arange(0, HEAD_DIM)[None, :] * stride_od,
        acc.to(tl.float16),
    )


def flash_attention_v2_wrapper(q, k, v, causal=False):
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
    STAGE = 3 if not causal else 1

    attn_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
    )
    return out
