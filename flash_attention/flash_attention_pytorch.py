import torch
from torch import Tensor, einsum
import math

from jaxtyping import Float, Int, Bool

def flash_attention(
        q: Float[Tensor, "batch seq_len d_k"],
        k: Float[Tensor, "batch seq_len d_k"],
        v: Float[Tensor, "batch seq_len d_v"],
        Bc: Int,
        Br: Int,
        causal: Bool = False,
    ) -> Float[Tensor, "batch seq_len d_v"]:

    dtype, device = q.dtype, q.device
    batch_size, seq_len, d_k = q.shape
    d_v = v.shape[2]
    scale = 1.0 / math.sqrt(d_k)

    out = torch.zeros_like(v, dtype=dtype, device=device)
    log_sum_exp = torch.full((batch_size, seq_len), float('-inf'), dtype=dtype, device=device)
    
    for i in range(0, seq_len, Br):
        row_start = i
        row_end = min(i + Br, seq_len)

        q_block = q[:, row_start:row_end, :]

        o_block = torch.zeros((batch_size, row_end - row_start, d_v), dtype=dtype, device=device)
        l_i = torch.zeros((batch_size, row_end - row_start), dtype=dtype, device=device)
        m_i = torch.full((batch_size, row_end - row_start), float('-inf'), dtype=dtype, device=device)

        for j in range(0, seq_len, Bc):
            col_start = j
            col_end = min(j + Bc, seq_len)

            if causal and col_start >= row_end:
                break

            k_block = k[:, col_start:col_end, :]
            v_block = v[:, col_start:col_end, :]
            attn_scores = einsum('b i d_k, b j d_k -> b i j', q_block, k_block) * scale

            if causal:
                mask = torch.full((row_end - row_start, col_end - col_start), float('-inf'), device=device, dtype=dtype)
                for row_idx in range(row_end - row_start):
                    for col_idx in range(col_end - col_start):
                        actual_row = row_start + row_idx
                        actual_col = col_start + col_idx
                        if actual_col <= actual_row:
                            mask[row_idx, col_idx] = 0.0
                attn_scores = attn_scores + mask.unsqueeze(0)

            m_ij = torch.max(m_i, attn_scores.max(dim=-1).values)
            P_ij = torch.exp(attn_scores - m_ij[:, :, None])
            l_ij = P_ij.sum(dim=-1) + l_i * torch.exp(m_i - m_ij)

            o_block_j = einsum('b i j, b j d_v -> b i d_v', P_ij, v_block) + o_block * torch.exp(m_i - m_ij)[:, :, None]

            m_i, l_i, o_block = m_ij, l_ij, o_block_j
        
        log_sum_exp[:, row_start:row_end] = m_i + torch.log(l_i)
        out[:, row_start:row_end, :] = o_block / l_i[:, :, None]

    return out
