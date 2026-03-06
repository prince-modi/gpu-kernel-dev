import helion
import torch
import helion.language as hl
from generate_config import retrieve_configs

config_list = retrieve_configs('rms_norm_fwd')

#https://helionlang.com/examples/rms_norm.html
@helion.kernel(configs = config_list, static_shapes = False)
def rms_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)

        # Compute inverse RMS: 1/sqrt(mean(x^2) + eps)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)

        # Apply normalization and weight
        normalized = x_tile * inv_rms_tile[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)

    return out