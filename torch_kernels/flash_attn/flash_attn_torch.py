import torch
from torch import Tensor
import math
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

flashatt_spec = torch.nn.functional.scaled_dot_product_attention

def sdpa_reference(q: Tensor, k: Tensor, v: Tensor, softmax_scale=None):
    """
    Reference using PyTorch scaled_dot_product_attention on (q,k,v) split.
    qkv: [B, S, 3, H, D]
    returns: [B, S, H, D]
    """
    # PyTorch SDPA wants [B, H, S, D]
    # If scale is provided, SDPA lets you scale q by scale and use scale=None.
    # But SDPA already applies 1/sqrt(D) internally. FA2 uses softmax_scale if provided.
    # We'll match FA2 default behavior: if softmax_scale is None, rely on SDPA default.
    if softmax_scale is not None:
        q = q * softmax_scale * math.sqrt(q.size(-1))  # convert "softmax_scale" into equivalent q scaling
    out = None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        try:
            out = F.scaled_dot_product_attention(q, k, v,
                                                    attn_mask=None,
                                                    dropout_p=0.0,
                                                    is_causal=False)
        except RuntimeError as e:
            print(f"Flash Attention V2 is not available. Reason: {e}")
            assert False

    return out.transpose(1, 2).contiguous()  # [B, S, H, D]