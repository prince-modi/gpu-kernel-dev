@helion.kernel(
    autotune_effort="none", # Autotuning disabled for development
    dot_precision="ieee"    # Using ieee precision to match spec
)
def flashatt_fwd(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    seq_len, d_head = q.size()
    qk_scale = 1 / (d_head ** 0.5)

    out = torch.zeros_like(q, dtype=q.dtype, device=q.device)

    for tile_q in hl.tile(seq_len):
        q_tile = q[tile_q, :]

        l_j = hl.zeros([tile_q], dtype=q.dtype, device=q.device)
        m_i = hl.full([tile_q], -float("inf"), dtype=q.dtype, device=q.device)
        o_j = hl.zeros([tile_q, d_head])

        for tile_kv in hl.tile(seq_len):
            k_tile = k[tile_kv, :]
            v_tile = v[tile_kv, :]

            qk = q_tile @ k_tile.T * qk_scale

            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp(qk - m_ij[:, None])
            scale_factor = torch.exp(m_i - m_ij)
            l_j = scale_factor * l_j + torch.sum(p, -1)

            o_j = o_j * scale_factor[:, None] + p @ v_tile

            m_i = m_ij

        out[tile_q, :] = o_j / l_j[:, None]

    return out