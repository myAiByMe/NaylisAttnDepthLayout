# Core/TransformerBlock/transformer_block.py — Naylis v1
import torch
import torch.nn as nn
from typing import Optional, Tuple

from attention import NaylisAttention, RMSNorm, KVCache
from feedforward import FeedForward


class NaylisBlock(nn.Module):
    """
    Transformer Block Naylis.
    Pré-norm (RMSNorm) + NaylisAttention + SwiGLU FFN.
    Supporte sequence packing via cu_seqlens.
    """

    def __init__(
        self,
        embed_dim    : int,
        num_heads    : int,
        dropout      : float = 0.0,
        use_rope     : bool  = True,
        max_seq_len  : int   = 2048,
        use_yarn     : bool  = False,
        yarn_scale   : float = 1.0,
        yarn_original_max_len: int = 1024,
        use_swiglu   : bool  = True,
        n_kv_heads   : Optional[int] = None,
        use_qk_norm  : bool  = True,
        use_flash_attn: bool = True,
        soft_cap      : Optional[float] = None,
        rel_rank      : int   = 32,
        sym_heads     : int   = 0,
        vanilla_heads : int   = 0,
    ):
        super().__init__()

        self.ln1 = RMSNorm(embed_dim)
        self.attention = NaylisAttention(
            embed_dim             = embed_dim,
            num_heads             = num_heads,
            dropout               = dropout,
            use_rope              = use_rope,
            max_seq_len           = max_seq_len,
            use_yarn              = use_yarn,
            yarn_scale            = yarn_scale,
            yarn_original_max_len = yarn_original_max_len,
            n_kv_heads            = n_kv_heads,
            use_qk_norm           = use_qk_norm,
            use_flash_attn        = use_flash_attn,
            soft_cap              = soft_cap,
            rel_rank              = rel_rank,
            sym_heads             = sym_heads,
            vanilla_heads         = vanilla_heads,
        )
        self.ln2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu)

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        residual = x
        attn_out, new_kv = self.attention(
            self.ln1(x),
            mask         = mask,
            past_kv      = past_kv,
            use_kv_cache = use_kv_cache,
            cu_seqlens_q = cu_seqlens_q,
            cu_seqlens_k = cu_seqlens_k,
            max_seqlen_q = max_seqlen_q,
            max_seqlen_k = max_seqlen_k,
        )
        x = residual + attn_out

        residual = x
        x        = residual + self.ffn(self.ln2(x))

        return x, new_kv
