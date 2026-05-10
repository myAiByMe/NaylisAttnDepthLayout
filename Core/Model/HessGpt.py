# Core/Model/HessGpt.py — Naylis v1
"""
NaylisGPT — ~200M paramètres
  vocab_size  : 49152  (cosmo2-tokenizer)
  embed_dim   : 768
  num_heads   : 12
  num_layers  : 12
  n_kv_heads  : 4      (GQA 3:1)
  rel_rank    : 32     (canaux Naylis par head)
  max_seq_len : 1024

DEPTH LAYOUT — Configuration par zone de profondeur
  Plutôt qu'un ratio global (asym/vanilla) appliqué à tous les blocs,
  depth_layout assigne un type de tête différent selon la zone :

    bottom  (0 – 25% des layers)  : tête asym dominantes
                                     → patterns directionnels bruts, vecteurs relationnels tôt
    mid     (25 – 75%)            : mix goldilocks asym+vanilla
                                     → équilibre relationnel / recall, transition sémantique
    top     (75 – 100%)           : têtes vanilla dominantes
                                     → inférence finale libre, sans biais directionnel imposé

  Format depth_layout :
    depth_layout = {
        'bottom': {'asym': 8, 'sym': 0, 'vanilla': 0},
        'mid':    {'asym': 4, 'sym': 0, 'vanilla': 4},
        'top':    {'asym': 0, 'sym': 0, 'vanilla': 8},
    }
  Contrainte : asym + sym + vanilla == num_heads pour chaque zone.

  Si depth_layout=None, les paramètres globaux sym_heads / vanilla_heads
  sont appliqués uniformément à tous les blocs (comportement original).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union, Dict

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Attention'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TransformerBlock'))

from attention import RMSNorm, KVCache
from transformer_block import NaylisBlock


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire : calcul des zones depth layout
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_BOUNDARIES = {
    'bottom': (0.00, 0.25),
    'mid':    (0.25, 0.75),
    'top':    (0.75, 1.00),
}


def zone_for_layer(layer_idx: int, num_layers: int) -> str:
    """Retourne la zone ('bottom', 'mid', 'top') pour un layer donné."""
    frac = layer_idx / num_layers
    if frac < 0.25:
        return 'bottom'
    elif frac < 0.75:
        return 'mid'
    else:
        return 'top'


def build_layer_specs(
    num_layers   : int,
    num_heads    : int,
    depth_layout : Optional[Dict],
    global_sym   : int,
    global_van   : int,
) -> List[Tuple[int, int]]:
    """
    Construit la liste de (sym_heads, vanilla_heads) pour chaque layer.

    Si depth_layout est None → ratio global appliqué uniformément.
    Sinon → ratio par zone selon les frontières 0/25/75/100%.

    Retourne : [(sym_i, vanilla_i)] pour i in range(num_layers)
    """
    if depth_layout is None:
        return [(global_sym, global_van)] * num_layers

    required = {'bottom', 'mid', 'top'}
    missing  = required - set(depth_layout.keys())
    assert not missing, \
        f"depth_layout doit contenir les zones : {required}. Manquant : {missing}"

    specs = []
    for i in range(num_layers):
        zone = zone_for_layer(i, num_layers)
        cfg  = depth_layout[zone]

        asym_h = cfg.get('asym', 0)
        sym_h  = cfg.get('sym',  0)
        van_h  = cfg.get('vanilla', 0)

        total = asym_h + sym_h + van_h
        assert total == num_heads, (
            f"Layer {i} (zone '{zone}') : "
            f"asym({asym_h}) + sym({sym_h}) + vanilla({van_h}) = {total} "
            f"!= num_heads({num_heads})"
        )
        assert sym_h >= 0 and van_h >= 0 and asym_h >= 0, \
            f"Layer {i} (zone '{zone}') : les valeurs doivent être >= 0"

        specs.append((sym_h, van_h))

    return specs


def describe_depth_layout(
    num_layers   : int,
    num_heads    : int,
    depth_layout : Optional[Dict],
    global_sym   : int,
    global_van   : int,
) -> str:
    """Retourne une description lisible de la configuration depth layout."""
    if depth_layout is None:
        asym_g = num_heads - global_sym - global_van
        return (f"Global uniforme : {asym_g} asym + {global_sym} sym + {global_van} vanilla "
                f"(tous les {num_layers} layers)")

    lines = [f"Depth Layout ({num_layers} layers) :"]
    zone_counts = {'bottom': 0, 'mid': 0, 'top': 0}
    for i in range(num_layers):
        zone_counts[zone_for_layer(i, num_layers)] += 1

    for zone in ('bottom', 'mid', 'top'):
        cfg    = depth_layout[zone]
        asym_h = cfg.get('asym', 0)
        sym_h  = cfg.get('sym',  0)
        van_h  = cfg.get('vanilla', 0)
        n      = zone_counts[zone]
        lo, hi = _ZONE_BOUNDARIES[zone]
        lines.append(
            f"  {zone:6s}  ({int(lo*100):3d}–{int(hi*100):3d}%)  "
            f"{n:2d} layers  →  {asym_h} asym + {sym_h} sym + {van_h} vanilla"
        )
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NaylisGPT
# ─────────────────────────────────────────────────────────────────────────────

class NaylisGPT(nn.Module):
    def __init__(
        self,
        vocab_size            : int   = 49_152,
        embed_dim             : int   = 768,
        num_heads             : int   = 12,
        num_layers            : int   = 12,
        max_seq_len           : int   = 1024,
        dropout               : float = 0.0,
        use_rope              : bool  = True,
        use_yarn              : bool  = False,
        yarn_scale            : float = 1.0,
        yarn_original_max_len : int   = 1024,
        use_swiglu            : bool  = True,
        n_kv_heads            : Optional[int]   = 4,
        use_qk_norm           : bool  = True,
        soft_cap              : Optional[float] = None,
        use_flash_attn        : bool  = True,
        rel_rank              : int   = 32,
        # ── Mode global (depth_layout=None) ──────────────────────
        sym_heads             : int   = 0,
        vanilla_heads         : int   = 0,
        # ── Mode depth layout ────────────────────────────────────
        depth_layout          : Optional[Dict] = None,
    ):
        """
        depth_layout : dict avec les zones 'bottom', 'mid', 'top'.
          Chaque zone contient {'asym': int, 'sym': int, 'vanilla': int}.
          asym + sym + vanilla doit égaler num_heads pour chaque zone.

          Exemple (8 heads, config depth R8) :
            depth_layout = {
                'bottom': {'asym': 8, 'sym': 0, 'vanilla': 0},
                'mid':    {'asym': 4, 'sym': 0, 'vanilla': 4},
                'top':    {'asym': 0, 'sym': 0, 'vanilla': 8},
            }

          Si None → sym_heads / vanilla_heads appliqués uniformément (mode original).
        """
        super().__init__()

        assert vocab_size > 0
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) % num_heads ({num_heads}) != 0"
        if n_kv_heads is not None:
            assert num_heads % n_kv_heads == 0, \
                f"num_heads ({num_heads}) % n_kv_heads ({n_kv_heads}) != 0"

        self.vocab_size            = vocab_size
        self.embed_dim             = embed_dim
        self.num_heads             = num_heads
        self.num_layers            = num_layers
        self.max_seq_len           = max_seq_len
        self.n_kv_heads            = n_kv_heads
        self.rel_rank              = rel_rank
        self.sym_heads             = sym_heads
        self.vanilla_heads         = vanilla_heads
        self.depth_layout          = depth_layout

        # ── Calcul des specs par layer ────────────────────────────
        self._layer_specs = build_layer_specs(
            num_layers   = num_layers,
            num_heads    = num_heads,
            depth_layout = depth_layout,
            global_sym   = sym_heads,
            global_van   = vanilla_heads,
        )

        # ── Embeddings ───────────────────────────────────────────
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout          = nn.Dropout(dropout)

        # ── Blocs Naylis — chaque block reçoit son propre (sym, vanilla) ─
        self.blocks = nn.ModuleList([
            NaylisBlock(
                embed_dim             = embed_dim,
                num_heads             = num_heads,
                dropout               = dropout,
                use_rope              = use_rope,
                max_seq_len           = max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
                use_swiglu            = use_swiglu,
                n_kv_heads            = n_kv_heads,
                use_qk_norm           = use_qk_norm,
                use_flash_attn        = use_flash_attn,
                soft_cap              = soft_cap,
                rel_rank              = rel_rank,
                sym_heads             = self._layer_specs[i][0],
                vanilla_heads         = self._layer_specs[i][1],
            )
            for i in range(num_layers)
        ])

        # ── Norm finale + head ───────────────────────────────────
        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying token_embeddings ↔ output_head
        self.output_head.weight = self.token_embeddings.weight

        # Masque causal pré-alloué (compile-safe)
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer('_causal_mask', causal_mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids    : torch.Tensor,
        targets      : Optional[torch.Tensor]    = None,
        pad_token_id : Optional[int]             = None,
        past_kv      : Optional[List[KVCache]]   = None,
        use_kv_cache : bool                      = False,
        cu_seqlens_q : Optional[torch.Tensor]    = None,
        cu_seqlens_k : Optional[torch.Tensor]    = None,
        max_seqlen_q : Optional[int]             = None,
        max_seqlen_k : Optional[int]             = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:

        B, S = input_ids.shape
        x    = self.dropout(self.token_embeddings(input_ids))

        if x.device.type == 'cuda' and x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        new_past_kv: Optional[List[KVCache]] = [] if use_kv_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv  = block(
                x,
                past_kv      = layer_past,
                use_kv_cache = use_kv_cache,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
            )
            if use_kv_cache:
                new_past_kv.append(new_kv)

        x      = self.ln_final(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index = -100,
            )

        return logits, loss, new_past_kv

    # ─────────────────────────────────────────────────────────────
    # Génération — KV Cache + top_k + top_p
    # ─────────────────────────────────────────────────────────────
    def generate(
        self,
        input_ids     : torch.Tensor,
        max_new_tokens: int                          = 50,
        temperature   : float                        = 1.0,
        top_k         : Optional[int]                = None,
        top_p         : Optional[float]              = None,
        eos_token_id  : Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        if eos_token_id is None:
            eos_ids = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = set(eos_token_id)

        with torch.no_grad():
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            prefill_logits, _, past_kv = self.forward(input_ids, use_kv_cache=True)
            next_logits = prefill_logits[:, -1, :]

            for _ in range(max_new_tokens):
                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = next_logits / temperature
                    if top_k is not None:
                        k_         = min(top_k, logits.size(-1))
                        topk_v, _  = torch.topk(logits, k_)
                        logits     = logits.masked_fill(logits < topk_v[:, [-1]], float('-inf'))
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if eos_ids and next_token.item() in eos_ids:
                    break

                decode_logits, _, past_kv = self.forward(
                    next_token, past_kv=past_kv, use_kv_cache=True)
                next_logits = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return input_ids

    # ─────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────
    def count_parameters(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        naylis = sum(
            p.numel()
            for b in self.blocks
            for name, p in b.attention.named_parameters()
            if 'rel_q_proj' in name or 'rel_k_proj' in name or 'graph_scale' in name
        )
        return {
            'total_M'    : round(total  / 1e6, 2),
            'naylis_K'   : round(naylis / 1e3, 1),
            'naylis_pct' : f'{naylis / total * 100:.2f}%',
        }

    def get_config(self) -> dict:
        cfg = {
            'vocab_size'   : self.vocab_size,
            'embed_dim'    : self.embed_dim,
            'num_heads'    : self.num_heads,
            'num_layers'   : self.num_layers,
            'max_seq_len'  : self.max_seq_len,
            'n_kv_heads'   : self.n_kv_heads,
            'rel_rank'     : self.rel_rank,
            'depth_layout' : self.depth_layout,
        }
        if self.depth_layout is None:
            cfg['sym_heads']     = self.sym_heads
            cfg['vanilla_heads'] = self.vanilla_heads
        return cfg

    def get_layer_layout(self) -> List[Dict]:
        """
        Retourne la config complète par layer : index, zone, asym/sym/vanilla.
        Utile pour le logging et la vérification.
        """
        result = []
        for i, (sym_h, van_h) in enumerate(self._layer_specs):
            asym_h = self.num_heads - sym_h - van_h
            result.append({
                'layer'   : i,
                'zone'    : zone_for_layer(i, self.num_layers),
                'asym'    : asym_h,
                'sym'     : sym_h,
                'vanilla' : van_h,
            })
        return result

    def describe_layout(self) -> str:
        """Retourne la description lisible de la config depth layout."""
        return describe_depth_layout(
            self.num_layers, self.num_heads,
            self.depth_layout, self.sym_heads, self.vanilla_heads,
        )

    def get_graph_scales_per_zone(self) -> Dict[str, float]:
        """
        Retourne |graph_scale| moyen par zone (bottom/mid/top).
        Layers vanilla-only (graph_scale=None) sont exclus du calcul.
        """
        zone_vals: Dict[str, List[float]] = {'bottom': [], 'mid': [], 'top': []}
        for i, b in enumerate(self.blocks):
            if b.attention.graph_scale is not None:
                s    = b.attention.graph_scale.detach().abs().mean().item()
                zone = zone_for_layer(i, self.num_layers)
                zone_vals[zone].append(s)
        return {
            z: (sum(v) / len(v) if v else 0.0)
            for z, v in zone_vals.items()
        }

    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        old_emb = self.token_embeddings
        self.token_embeddings        = nn.Embedding(new_vocab_size, self.embed_dim)
        n = min(old_emb.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:n] = old_emb.weight.data[:n]
        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        self.vocab_size         = new_vocab_size
