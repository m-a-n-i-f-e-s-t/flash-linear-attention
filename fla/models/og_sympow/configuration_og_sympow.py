# -*- coding: utf-8 -*-

from typing import Optional

import torch
from transformers.configuration_utils import PretrainedConfig


class OGSympowConfig(PretrainedConfig):

    model_type = 'og_sympow'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_ratio: Optional[float] = 3.5, # as per llama 3.1
        num_hidden_layers: int = 12,
        num_heads: int = 12,
        max_position_embeddings: int = 1024,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 50257,

        dtype: torch.dtype = None,
        device: torch.device = None,

        bias: bool = False,

        # sympow specific
        attn_kernel: str = "power", # or "flash"
        gating: bool = True,
        head_size: int = 64,
        log_space: bool = False,
        chunk_size: int = 1024,
        degree: int = 2,
        qhead_ratio: int = 1,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

        self.bias = bias

        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.attn_kernel = attn_kernel
        self.gating = gating
        self.head_size = head_size
        self.log_space = log_space
        self.chunk_size = chunk_size
        self.degree = degree
        self.qhead_ratio = qhead_ratio

        self.device = device
        self.dtype = dtype

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
