import helion
import helion.language as hl
import torch
from torch import Tensor
import os


def retrieve_configs(benchmark_name: str):
  filtered_configs = []
  if os.path.exists('configs'):
    all_configs = os.listdir('configs')
    for conf in all_configs:
        if benchmark_name in conf and 'json' in conf:
            filtered_configs.append(helion.Config.load(os.path.join('configs',conf)))
  return filtered_configs


config_list = retrieve_configs('helion_rms_kernel')

@helion.kernel(
    configs = config_list,
    dot_precision="ieee",    # Using ieee precision to match spec
    static_shapes = False
)
def flashatt_fwd(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    seq_len, d_head = q.size()
    qk_scale = 1 / (d_head ** 0.5)

    out = torch.zeros_like(q, dtype=q.dtype, device=q.device)

    for tile_q in hl.tile(seq_len):
        q_tile = q[tile_q, :]

        l_i = hl.zeros([tile_q], dtype=q.dtype, device=q.device)
        m_i = hl.full([tile_q], -float("inf"), dtype=q.dtype, device=q.device)
        acc = hl.zeros([tile_q, d_head])

        for tile_kv in hl.tile(seq_len):
            k_tile = k[tile_kv, :]
            v_tile = v[tile_kv, :]

            qk = q_tile @ k_tile.T * qk_scale
            m_ij = torch.amax(qk, 1)
            p = torch.exp(qk - m_ij[:, None])
            l_ij = torch.sum(p, 1)
            m_i_new = torch.maximum(m_i, m_ij)
            alpha = torch.exp(m_i - m_i_new)
            beta = torch.exp(m_ij - m_i_new)

            l_i_new = alpha * l_i + beta * l_ij
            scale_factor = beta / l_i_new
            p = p * scale_factor[:, None]
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None] + p @ v_tile

            l_i = l_i_new
            m_i = m_i_new

        out[tile_q, :] = acc

    return out