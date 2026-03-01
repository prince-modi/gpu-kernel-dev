# Bhrugu Bharathi, A16641798, 02/28/2026

import torch
import sys
print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(sys.version)

# USE WHEEL FINDER HERE TO INSTALL THE RIGHT VERSION OF flash_attn https://flashattn.dev/#finder
# On Colab's A100 I used the following command:
# !uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl

import math

def compile_fa2_benchmark(benchmark_name: str, **kwargs):
    # For ease of integration, compile means bind the relevant kernel call.
    if "qkv" not in kwargs:
        raise Exception("Expected argument qkv in kwargs")
    qkv = kwargs["qkv"]

    causal = bool(kwargs.get("causal", False))
    dropout_p = float(kwargs.get("dropout_p", 0.0))
    softmax_scale = kwargs.get("softmax_scale", None)  # optional

    from flash_attn import flash_attn_qkvpacked_func

    # Ensure no hidden copies during timing
    qkv = qkv.contiguous()

    def compiled_code(**call_kwargs):
        q = call_kwargs.get("qkv", qkv).contiguous()
        return flash_attn_qkvpacked_func(
            q,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return compiled_code

def _sdpa_reference(qkv: torch.Tensor, causal: bool, softmax_scale=None):
    """
    Reference using PyTorch scaled_dot_product_attention on (q,k,v) split.
    qkv: [B, S, 3, H, D]
    returns: [B, S, H, D]
    """
    # PyTorch SDPA wants [B, H, S, D]
    q, k, v = qkv.unbind(dim=2)              # each [B, S, H, D]
    q = q.transpose(1, 2).contiguous()       # [B, H, S, D]
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # If scale is provided, SDPA lets you scale q by scale and use scale=None.
    # But SDPA already applies 1/sqrt(D) internally. FA2 uses softmax_scale if provided.
    # We'll match FA2 default behavior: if softmax_scale is None, rely on SDPA default.
    if softmax_scale is not None:
        q = q * softmax_scale * math.sqrt(q.size(-1))  # convert "softmax_scale" into equivalent q scaling

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )  # [B, H, S, D]
    return out.transpose(1, 2).contiguous()  # [B, S, H, D]

def bench_cuda(fn, warmup=50, iters=200):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms

def attn_flops_fwd(B, S, H, D, causal: bool):
    # Rough forward FLOPs: QK^T and PV; causal ~ half matrix.
    return 4.0 * B * H * (S * S) * D * (0.5 if causal else 1.0)

def tflops(flop, ms):
    return flop / (ms * 1e-3) / 1e12

# =====================================================================================
# Testing
# =====================================================================================

if __name__ == "__main__":
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    USE_FP16 = True  # Toggle
    dtype = torch.float16 if USE_FP16 else torch.bfloat16  # swap to bf16 if you want
    # For correctness vs SDPA, fp16 can differ a bit more; pick tolerances accordingly.
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    causal = True
    dropout_p = 0.0
    softmax_scale = None  # leave None to match FA2 default

    print("GPU:", torch.cuda.get_device_name(0))
    print("CC :", torch.cuda.get_device_capability(0))
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print(f"Running FA2 forward-only with dtype={dtype}, causal={causal}")

    # Configs: (B, S, H, D)
    test_configs = [
        (4, 512, 16, 64),
        (2, 1024, 16, 64),
        (2, 2048, 16, 64),
        (1, 4096, 16, 64),
        (2, 1024, 16, 128),
        (1, 2048, 16, 128),
        (1, 4096, 16, 128),
    ]

    print("\n=== Correctness Tests (vs PyTorch SDPA) ===")
    for (B, S, H, D) in test_configs:
        qkv = torch.randn(B, S, 3, H, D, device=device, dtype=dtype).contiguous()

        compiled = compile_fa2_benchmark(
            "fa2_qkvpacked",
            qkv=qkv,
            causal=causal,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        )

        # FA2 output
        with torch.no_grad():
            y = compiled(qkv=qkv)  # [B, S, H, D]

        # SDPA reference (float32 accumulation can help reduce ref noise)
        with torch.no_grad():
            y_ref = _sdpa_reference(qkv.to(torch.float32), causal=causal, softmax_scale=softmax_scale).to(dtype)

        try:
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
            print(f"  PASSED: B={B:>2}, S={S:>5}, H={H:>2}, D={D:>3}")
        except AssertionError as e:
            max_abs = (y - y_ref).abs().max().item()
            print(f"  FAILED: B={B:>2}, S={S:>5}, H={H:>2}, D={D:>3} | max_abs={max_abs:.4e}")
            print(f"          {e}")

    print("\n=== Benchmark (kernel-only) ===")
    bench_configs = [
        (4, 1024, 16, 128),
        (2, 2048, 16, 128),
        (1, 4096, 16, 128),
        (1, 8192, 16, 128),
    ]

    for (B, S, H, D) in bench_configs:
        qkv = torch.randn(B, S, 3, H, D, device=device, dtype=dtype).contiguous()
        compiled = compile_fa2_benchmark(
            "fa2_qkvpacked",
            qkv=qkv,
            causal=causal,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
        )

        with torch.no_grad():
            ms = bench_cuda(lambda: compiled(qkv=qkv), warmup=50, iters=200)
        tf = tflops(attn_flops_fwd(B, S, H, D, causal), ms)
        print(f"B={B:>2} S={S:>5} H={H:>2} D={D:>3} | {ms:7.3f} ms | {tf:7.1f} TF/s")