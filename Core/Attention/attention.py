# Core/Attention/attention.py — Naylis v1
"""
NaylisAttention — Hybrid Token-Graph Attention

ARCHITECTURE :
  Option 1.5 : toutes les dims participent à l'attention classique,
  un projecteur léger extrait les canaux relationnels DEPUIS toutes les dims.

  Deux projecteurs séparés rel_q_proj / rel_k_proj → biais ASYMÉTRIQUE.
  B[i,j] = <R_q(i), R_k(j)> ≠ B[j,i]  ← directionnalité réelle.

  graph_scale initialisé à 0 par head → le modèle commence comme un
  transformer classique stable et active les canaux quand c'est utile.

BACKENDS (priorité) :
  1. varlen  (flash_attn >= 2.0, cu_seqlens fournis) → sequence packing
  2. SDPA    (PyTorch >= 2.0)  → FA natif B200/H100, PRIORITAIRE
  3. FA std  (flash_attn)      → fallback si SDPA indisponible
  4. Manuel  (soft_cap ou mask custom)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================
# FLASH ATTENTION — Détection hiérarchique
# ============================================================
_FA_LEVEL       = 0
_FA_VARLEN_FUNC = None
_FA_FUNC        = None


def _detect_flash_attn():
    global _FA_LEVEL, _FA_VARLEN_FUNC, _FA_FUNC
    try:
        import flash_attn
        version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])

        if version >= (3, 0):
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap == (12, 0) or cap[0] > 12:
                    try:
                        from flash_attn.flash_attn_interface import (
                            flash_attn_func, flash_attn_varlen_func,
                        )
                        _FA_FUNC        = flash_attn_func
                        _FA_VARLEN_FUNC = flash_attn_varlen_func
                        _FA_LEVEL       = 4
                        print("  ⚡ FlashAttention-4 (Blackwell SM120) détecté")
                        return
                    except ImportError:
                        pass
                if cap[0] >= 9:
                    try:
                        from flash_attn.flash_attn_interface import (
                            flash_attn_func, flash_attn_varlen_func,
                        )
                        _FA_FUNC        = flash_attn_func
                        _FA_VARLEN_FUNC = flash_attn_varlen_func
                        _FA_LEVEL       = 3
                        print("  ⚡ FlashAttention-3 (Hopper SM90) détecté")
                        return
                    except ImportError:
                        pass

        if version >= (2, 0):
            try:
                from flash_attn.flash_attn_interface import (
                    flash_attn_func, flash_attn_varlen_func,
                )
                _FA_FUNC        = flash_attn_func
                _FA_VARLEN_FUNC = flash_attn_varlen_func
                _FA_LEVEL       = 2
                print(f"  ⚡ flash_attn {flash_attn.__version__} — varlen (packing)")
                return
            except ImportError:
                pass

    except ImportError:
        pass

    if hasattr(F, 'scaled_dot_product_attention'):
        _FA_LEVEL = 1
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 12:
                print("  ⚡ SDPA PyTorch — FA natif Blackwell SM120")
            elif cap[0] >= 10:
                print("  ⚡ SDPA PyTorch — FA natif Blackwell SM100")
            elif cap[0] >= 9:
                print("  ⚡ SDPA PyTorch — FA natif Hopper SM90")
            else:
                print("  ⚡ SDPA PyTorch natif")
        else:
            print("  ⚡ SDPA PyTorch natif (CPU)")
    else:
        _FA_LEVEL = 0
        print("  ⚠️  Aucune Flash Attention disponible (PyTorch < 2.0)")


_detect_flash_attn()


# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================
# RoPE + YaRN
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000,
                 use_yarn: bool = False, yarn_scale: float = 1.0,
                 yarn_original_max_len: int = 1024):
        super().__init__()
        self.dim                   = dim
        self.max_seq_len           = max_seq_len
        self.base                  = base
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        inv_freq = (self._compute_yarn_frequencies() if use_yarn
                    else 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.register_buffer('inv_freq', inv_freq)
        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self):
        freqs         = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        if self.yarn_scale == 1.0:
            return inv_freq_base
        alpha = self.yarn_scale
        beta  = max(self.dim // 2, int(self.dim * 0.25))
        dims  = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)
        )
        return inv_freq_base / scale

    def _update_cache(self, seq_len: int, device, dtype):
        if (seq_len != self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len   = q.shape[2]
        total_len = seq_len + position_offset
        cos, sin  = self._update_cache(total_len, q.device, q.dtype)
        cos = cos[position_offset:position_offset + seq_len][None, None]
        sin = sin[position_offset:position_offset + seq_len][None, None]
        return ((q * cos) + (self._rotate_half(q) * sin),
                (k * cos) + (self._rotate_half(k) * sin))


# ============================================================
# KV Cache
# ============================================================
KVCache = Tuple[torch.Tensor, torch.Tensor]


# ============================================================
# NAYLIS ATTENTION
# ============================================================
class NaylisAttention(nn.Module):
    """
    Attention hybride Token-Graph.

    Différences vs MultiHeadAttention classique :
      - rel_q_proj / rel_k_proj  : projecteurs séparés → biais ASYMÉTRIQUE
      - graph_scale               : paramètre learnable par head, init=0
                                    → modèle commence classique, active
                                    les canaux progressivement
      - graph_bias injecté AVANT SDPA via attn_mask (BF16, contiguous)
      - Pas de F.normalize → gradient sur magnitude préservé
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
        n_kv_heads   : Optional[int] = None,
        use_qk_norm  : bool  = True,
        use_flash_attn: bool = True,
        soft_cap      : Optional[float] = None,
        rel_rank      : int   = 32,
        sym_heads     : int   = 0,
        vanilla_heads : int   = 0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        graph_heads = num_heads - vanilla_heads
        assert 0 <= sym_heads <= graph_heads, \
            f"sym_heads ({sym_heads}) doit etre <= graph_heads ({graph_heads})"
        assert 0 <= vanilla_heads <= num_heads, \
            f"vanilla_heads ({vanilla_heads}) doit etre <= num_heads ({num_heads})"

        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.head_dim     = embed_dim // num_heads
        self.use_rope     = use_rope
        self.use_qk_norm  = use_qk_norm
        self.soft_cap     = soft_cap
        self.rel_rank     = rel_rank
        # Layout des tetes (dans l'ordre) :
        #   [0 .. sym_heads-1]            -> biais SYMETRIQUE  (B+Bt)/2
        #   [sym_heads .. graph_heads-1]  -> biais ASYMETRIQUE B[i,j]!=B[j,i]
        #   [graph_heads .. num_heads-1]  -> VANILLA  (aucun biais graph)
        self.sym_heads     = sym_heads
        self.vanilla_heads = vanilla_heads
        self.graph_heads   = graph_heads

        self.n_kv_heads         = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        self.kv_dim             = self.n_kv_heads * self.head_dim

        # ── Projections classiques ───────────────────────────────
        self.q_proj   = nn.Linear(embed_dim, embed_dim,   bias=False)
        self.k_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,   bias=False)
        self.dropout  = nn.Dropout(dropout)

        # ── QK Norm ──────────────────────────────────────────────
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        # ── RoPE ─────────────────────────────────────────────────
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
            )
        else:
            self.rope = None

        # ── Naylis : projecteurs relationnels (asym + sym seulement) ─
        # Les vanilla_heads n'ont pas de rel_proj — attention classique pure
        # rel_q_proj / rel_k_proj couvrent graph_heads tetes uniquement
        if graph_heads > 0:
            self.rel_q_proj = nn.Linear(embed_dim, graph_heads * rel_rank, bias=False)
            self.rel_k_proj = nn.Linear(embed_dim, graph_heads * rel_rank, bias=False)
            nn.init.normal_(self.rel_q_proj.weight, std=0.01)
            nn.init.normal_(self.rel_k_proj.weight, std=0.01)
            # graph_scale par tete graph, init=0 → transformer classique au step 0
            self.graph_scale = nn.Parameter(torch.zeros(graph_heads))
        else:
            self.rel_q_proj  = None
            self.rel_k_proj  = None
            self.graph_scale = None

        # ── Backend attention ────────────────────────────────────
        self._fa_level  = _FA_LEVEL if use_flash_attn else 0
        self._fa_varlen = _FA_VARLEN_FUNC
        self._fa_func   = _FA_FUNC
        self._sdpa_ok   = hasattr(F, 'scaled_dot_product_attention')

    def _attn_scale(self) -> float:
        if (self.use_rope and self.rope is not None
                and self.rope.use_yarn and self.rope.yarn_scale > 1.0):
            return math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
        return 1.0 / math.sqrt(self.head_dim)

    def _compute_graph_bias(
        self,
        x     : torch.Tensor,
        dtype : torch.dtype,
    ) -> torch.Tensor:
        """
        Calcule le biais de graphe pour les graph_heads tetes.

        Layout :
          [0 .. sym_heads-1]           -> SYMETRIQUE  (B+Bt)/2
          [sym_heads .. graph_heads-1] -> ASYMETRIQUE  B[i,j]!=B[j,i]
          [graph_heads .. num_heads-1] -> VANILLA : biais = 0 (pas de rel_proj)

        Shape retournee : [B, num_heads, S, S]
        graph_scale init=0 -> biais nul au step 0.
        """
        B, S, _ = x.shape
        H_g, R  = self.graph_heads, self.rel_rank

        # [B, S, H_g*R] -> [B, H_g, S, R]
        R_q = self.rel_q_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3)
        R_k = self.rel_k_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3)

        # [B, H_g, S, S]
        graph_bias = torch.matmul(R_q, R_k.transpose(-2, -1))

        # Symetrisation des sym_heads premieres tetes
        if self.sym_heads > 0:
            sym        = graph_bias[:, :self.sym_heads]
            sym        = (sym + sym.transpose(-2, -1)) * 0.5
            graph_bias = torch.cat([sym, graph_bias[:, self.sym_heads:]], dim=1)

        # Mise a l'echelle par tete graph
        scale      = self.graph_scale.view(1, H_g, 1, 1)
        graph_bias = (scale * graph_bias).to(dtype)

        # Pad avec zeros pour les vanilla_heads
        if self.vanilla_heads > 0:
            pad        = torch.zeros(B, self.vanilla_heads, S, S, dtype=dtype, device=x.device)
            graph_bias = torch.cat([graph_bias, pad], dim=1)

        return graph_bias.contiguous()

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

        B, S, _ = x.shape
        scale   = self._attn_scale()

        # ── Projections QKV ──────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.num_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        position_offset = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv_cache: Optional[KVCache] = (k, v) if use_kv_cache else None

        if self.n_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # ── Cast BF16 pour attention ─────────────────────────────
        orig_dtype = q.dtype
        if q.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        # ── Graph bias Naylis ────────────────────────────────────
        # Calculé depuis x (toutes les dims) — Option 1.5
        # Seulement si SDPA ou manuel (varlen ne supporte pas attn_mask custom)
        use_varlen = (
            cu_seqlens_q is not None
            and self._fa_varlen is not None
            and self.soft_cap is None
            and past_kv is None
        )
        # En decode (S=1 + KV cache) : pas de graph_bias
        # car x=[B,1,D] → on ne peut pas calculer R_k sur S_total
        # Le biais est appliqué uniquement en prefill (S>1)
        _use_graph = self.graph_heads > 0 and (not use_varlen) and (S > 1 or past_kv is None)
        graph_bias = (
            self._compute_graph_bias(x, q.dtype) if _use_graph else None
        )

        # ── Backends attention ───────────────────────────────────

        if use_varlen:
            # varlen : sequence packing — graph_bias non appliqué
            # (flash_attn_varlen ne supporte pas attn_mask dense)
            q_var = q.permute(0, 2, 1, 3).reshape(-1, self.num_heads,  self.head_dim)
            k_var = k.permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim)
            v_var = v.permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim)
            _msl_q = max_seqlen_q if max_seqlen_q is not None else S
            _msl_k = max_seqlen_k if max_seqlen_k is not None else S
            output = self._fa_varlen(
                q_var, k_var, v_var,
                cu_seqlens_q, cu_seqlens_k,
                _msl_q, _msl_k,
                dropout_p     = self.dropout.p if self.training else 0.0,
                softmax_scale = scale,
                causal        = True,
            )
            output = output.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        elif self._sdpa_ok and self.soft_cap is None:
            # SDPA prioritaire — graph_bias injecté via attn_mask
            # Si S=1 (decode) : graph_bias [B,H,1,S_kv], is_causal=False
            # Si S>1 (prefill) : graph_bias [B,H,S,S] + masque causal fusionné
            is_causal = (S > 1 and past_kv is None)

            if is_causal and graph_bias is not None:
                # Fusionne graph_bias + masque causal dans attn_mask
                # SDPA avec attn_mask explicite désactive is_causal interne
                S_q, S_kv = q.shape[2], k.shape[2]
                causal_mask = torch.triu(
                    torch.full((S_q, S_kv), float('-inf'),
                               device=q.device, dtype=q.dtype),
                    diagonal=1 + (past_kv[0].shape[2] if past_kv else 0),
                )
                attn_mask = (graph_bias + causal_mask.unsqueeze(0).unsqueeze(0)).contiguous()
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = attn_mask,
                    is_causal = False,
                    dropout_p = self.dropout.p if self.training else 0.0,
                    scale     = scale,
                )
            elif graph_bias is not None:
                # Decode (S=1) — .contiguous() obligatoire pour SDPA
                graph_bias = graph_bias.contiguous()
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = graph_bias,
                    is_causal = False,
                    dropout_p = self.dropout.p if self.training else 0.0,
                    scale     = scale,
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = None,
                    is_causal = is_causal,
                    dropout_p = self.dropout.p if self.training else 0.0,
                    scale     = scale,
                )

        elif self._fa_func is not None and self.soft_cap is None and mask is None:
            # FA std fallback (PyTorch < 2.0) — sans graph_bias
            is_causal = (S > 1 and past_kv is None)
            output = self._fa_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p     = self.dropout.p if self.training else 0.0,
                softmax_scale = scale,
                causal        = is_causal,
            )
            output = output.transpose(1, 2)

        else:
            # Manuel — soft_cap ou mask custom — graph_bias inclus
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
            if graph_bias is not None:
                scores = scores + graph_bias
            if S > 1 and past_kv is None:
                if mask is not None:
                    scores = scores.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                else:
                    causal_bool = torch.triu(
                        torch.ones(S, k.shape[2], device=q.device, dtype=torch.bool),
                        diagonal=1)
                    scores = scores.masked_fill(
                        causal_bool.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        # ── Reshape + projection ─────────────────────────────────
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        if output.dtype != self.out_proj.weight.dtype:
            output = output.to(self.out_proj.weight.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, new_kv_cache
