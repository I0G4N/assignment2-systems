import torch
from torch import Tensor, einsum
import math
from typing import Tuple

from jaxtyping import Float, Int, Bool

def _flash_attention_forward(
    q: Float[Tensor, "batch seq_len d_k"],
    k: Float[Tensor, "batch seq_len d_k"],
    v: Float[Tensor, "batch seq_len d_v"],
    Bc: Int,
    Br: Int,
    causal: Bool = False,
) -> Tuple[Float[Tensor, "batch seq_len d_v"], Float[Tensor, "batch seq_len"]]:

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

    return out, log_sum_exp


def _flash_attention_backward(
    q: Float[Tensor, "batch seq_len d_k"],
    k: Float[Tensor, "batch seq_len d_k"],
    v: Float[Tensor, "batch seq_len d_v"],
    out: Float[Tensor, "batch seq_len d_v"],
    grad_out: Float[Tensor, "batch seq_len d_v"],
    log_sum_exp: Float[Tensor, "batch seq_len"],
    Bc: Int,
    Br: Int,
    causal: Bool = False,
) -> Tuple[Float[Tensor, "batch seq_len d_k"], Float[Tensor, "batch seq_len d_k"], Float[Tensor, "batch seq_len d_v"]]:
    
    dtype, device = q.dtype, q.device
    batch_size, seq_len, dim_k = q.shape
    dim_v = v.shape[2]
    scale = 1.0 / math.sqrt(dim_k)

    dQ = torch.zeros_like(q, dtype=dtype, device=device)
    dK = torch.zeros_like(k, dtype=dtype, device=device)
    dV = torch.zeros_like(v, dtype=dtype, device=device)
    D = torch.sum(grad_out * out, dim=-1)
    
    for i in range(0, seq_len, Bc):
        col_start = i
        col_end = min(i + Bc, seq_len)

        k_block = k[:, col_start:col_end, :]
        v_block = v[:, col_start:col_end, :]

        for j in range(0, seq_len, Br):
            row_start = j
            row_end = min(j + Br, seq_len)

            q_block = q[:, row_start:row_end, :]

            if causal and col_start >= row_end:
                continue

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

            P_ij = torch.exp(attn_scores - log_sum_exp[:, row_start:row_end][:, :, None])

            dV[:, col_start:col_end, :] += einsum('b i j, b i d_v -> b j d_v', P_ij, grad_out[:, row_start:row_end, :])
            dP_ij = einsum('b i d_v, b j d_v -> b i j', grad_out[:, row_start:row_end, :], v_block)
            dS_ij = P_ij * (dP_ij - D[:, row_start:row_end][:, :, None])
            dQ[:, row_start:row_end, :] += einsum('b i j, b j d_k -> b i d_k', dS_ij, k_block) * scale
            dK[:, col_start:col_end, :] += einsum('b i j, b i d_k -> b j d_k', dS_ij, q_block) * scale

    return dQ, dK, dV


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Float[Tensor, "batch seq_len d_k"],
        k: Float[Tensor, "batch seq_len d_k"],
        v: Float[Tensor, "batch seq_len d_v"],
        Bc: int,
        Br: int,
        causal: bool = False,
    ) -> Float[Tensor, "batch seq_len d_v"]:
        
        out, log_sum_exp = _flash_attention_forward(q=q, k=k, v=v, Bc=Bc, Br=Br, causal=causal)
        ctx.save_for_backward(q, k, v, out, log_sum_exp)
        ctx.Bc = Bc
        ctx.Br = Br
        ctx.causal = causal

        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        q, k, v, out, log_sum_exp = ctx.saved_tensors
        Bc, Br, causal = ctx.Bc, ctx.Br, ctx.causal

        dQ, dK, dV = _flash_attention_backward(q=q, k=k, v=v, out=out, grad_out=grad_output, log_sum_exp=log_sum_exp, Bc=Bc, Br=Br, causal=causal)

        return dQ, dK, dV, None, None, None




        
        


def flash_attention(
    q: Float[Tensor, "batch seq_len d_k"],
    k: Float[Tensor, "batch seq_len d_k"],
    v: Float[Tensor, "batch seq_len d_v"],
    Bc: Int,
    Br: Int,
    causal: Bool = False,
) -> Float[Tensor, "batch seq_len d_v"]:
    return FlashAttentionFunction.apply(q, k, v, Bc, Br, causal)
