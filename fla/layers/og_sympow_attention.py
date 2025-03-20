from __future__ import annotations

import math

import torch
import torch.nn as nn

from power_attention import power_full, power_full_triton
from power_attention._expansion.impl_triton import compute_expanded_dim
from fla.models.og_sympow.configuration_og_sympow import OGSympowConfig
from flash_attn import flash_attn_func

# It's already a CUDA kernel, we don't want Triton in our CUDA
power_full = torch.compiler.disable(power_full)
# It's already a Triton kernel, we don't want more Triton in our Triton
power_full_triton = torch.compiler.disable(power_full_triton)

def get_sinusoidal_embeddings(position, d, device):
    """Generate sinusoidal positional embeddings."""
    # position is [B, T, nh]
    T = position.shape[1]
    div_term = (2. * math.pi) / (float(T) ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d)).view(1, 1, 1, -1)
    sinusoid_inp = position.unsqueeze(-1) * div_term
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    return sin, cos # [B, T, nh, d]

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.view(*x.shape[:-2], x.shape[-2] * x.shape[-1])

def apply_rotary_position_embeddings(x, sincos):
    _, T, _, _ = x.shape
    sin, cos = sincos
    sin = sin.repeat_interleave(2, dim=3)[:, :T]
    cos = cos.repeat_interleave(2, dim=3)[:, :T]
    return ((x * cos) + (rotate_every_two(x) * sin)).to(dtype=x.dtype)


class OGSympowAttention(nn.Module):

    def __init__(self, config: OGSympowConfig):
        super().__init__()
        self.n_head = config.num_heads
        self.n_embd = config.hidden_size
        self.attention_kernel = config.attn_kernel
        self.chunk_size = config.chunk_size
        self.degree = config.degree
        self.head_size = config.head_size
        self.qhead_ratio = config.qhead_ratio
        self.log_space = config.log_space
        self.gating = config.gating
        self.device = config.device
        self.dtype = config.dtype
        if self.gating and self.attention_kernel == 'flash':
            print('WARNING: FlashAttention does not support gating, gating is ignored.')
        # key, query, value projections for all heads, but in a batch
        self.qkv_size = (config.qhead_ratio + 2) * self.n_head * self.head_size
        self.gating_size = config.num_heads if self.gating else 0
        self.c_attn = nn.Linear(config.hidden_size, self.qkv_size + self.gating_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.qhead_ratio * config.num_heads * config.head_size, config.hidden_size, bias=config.bias)

    def reset_state(self, batch_size):
        self.pos_idx = torch.zeros([1, 1, 1], dtype=torch.int32, device=self.device)
        if self.attention_kernel == 'power':
            self.state = torch.zeros((batch_size, self.n_head, compute_expanded_dim(self.head_size, self.degree), self.head_size), 
                                     dtype=torch.get_autocast_gpu_dtype(), device=self.device)
        else:
            self.k_cache = None
            self.v_cache = None

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        h = self.n_head
        hq = self.qhead_ratio * h
        d = self.head_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkvg = self.c_attn(x)

        qkv = qkvg[...,:self.qkv_size]
        q, k, v  = qkv.split([hq*d, h*d, h*d], dim=2)
        q = q.view(B, T, hq, d)
        k = k.view(B, T, h, d)
        v = v.view(B, T, h, d)

        if self.gating:
            # 6.906768 corresponds to initial gating of .999
            log_g = torch.nn.functional.logsigmoid(6.906768 + qkvg[...,self.qkv_size:].to(dtype=torch.float32)).contiguous()
        else:
            log_g = None

        # apply rotary position embeddings
        position = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(2) # [1, T, 1]
        sincos = get_sinusoidal_embeddings(position, d, q.device)
        q = apply_rotary_position_embeddings(q, sincos)
        k = apply_rotary_position_embeddings(k, sincos)

        if self.attention_kernel == 'flash':
            y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True, softmax_scale=1.0 / d**0.5)
        elif self.attention_kernel == 'power':
            out = power_full(q.contiguous(), k.contiguous(), v.contiguous(), log_g,
                deg=self.degree,
                scale=1.0 / d**0.5,
                chunk_size=self.chunk_size)
            y = out
        elif self.attention_kernel == 'power_triton':
            y = power_full_triton(q.contiguous(), k.contiguous(), v.contiguous(), log_g,
                deg=self.degree,
                scale=1.0 / d**0.5,
                chunk_size=self.chunk_size)
        else:
            msg = f'Unknown attention kernel: {self.attention_kernel}'
            raise NotImplementedError(msg)
        y = y.contiguous().view(B, T, hq * d) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
