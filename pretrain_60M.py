import os
os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import sys
import time
import math
import json
import gc
import traceback
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import Optional, List

import matplotlib
matplotlib.use('Agg')  # pas de display, sauvegarde en fichier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.set_float32_matmul_precision('high')

# -- Paths Core -----------------------------------------------------------------
_root = os.path.dirname(__file__)
sys.path.append(os.path.join(_root, 'Core', 'Model'))
sys.path.append(os.path.join(_root, 'Core', 'Attention'))
sys.path.append(os.path.join(_root, 'Core', 'FeedForward'))
sys.path.append(os.path.join(_root, 'Core', 'TransformerBlock'))

from HessGpt import NaylisGPT
from attention import KVCache

# -- Args -----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    return p.parse_args()

ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- Config ---------------------------------------------------------------------
CONFIG = {
    # Model
    'vocab_size'            : None,
    'embed_dim'             : 512,
    'num_heads'             : 8,
    'num_layers'            : 12,
    'max_seq_len'           : 512,
    'dropout'               : 0.0,
    'use_rope'              : True,
    'use_yarn'              : False,
    'yarn_scale'            : 1.0,
    'yarn_original_max_len' : 512,
    'use_swiglu'            : True,
    'n_kv_heads'            : 4,
    'use_qk_norm'           : True,
    'soft_cap'              : None,
    'use_flash_attn'        : True,
    'rel_rank'              : 16,
    # ── Depth Layout — config par zone de profondeur ──────────────────────────
    # Remplace sym_heads / vanilla_heads global par une config différente
    # selon la position du layer dans le réseau.
    #
    # Zones : bottom (0–25%)  mid (25–75%)  top (75–100%)
    # Contrainte : asym + sym + vanilla == num_heads pour chaque zone.
    #
    # Config R8 — Depth Layout Hypothesis (run de référence) :
    'depth_layout'          : {
        'bottom': {'asym': 4, 'sym': 0, 'vanilla': 4},   # layers 0–2  (3/12)
        'mid':    {'asym': 4, 'sym': 0, 'vanilla': 4},   # layers 3–8  (6/12)
        'top':    {'asym': 0, 'sym': 0, 'vanilla': 8},   # layers 9–11 (3/12)
    },
    # Pour revenir au mode global uniforme, mettre depth_layout=None
    # et définir sym_heads / vanilla_heads :
    # 'depth_layout'  : None,
    # 'sym_heads'     : 0,
    # 'vanilla_heads' : 4,
    # Training
    'batch_size'            : 220,
    'gradient_accumulation' : 1,
    'max_grad_norm'         : 1.0,
    'learning_rate'         : 3e-4,
    'weight_decay'          : 0.1,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.95,
    'adam_eps'              : 1e-8,
    # Data
    'data_file'             : './data/pretrain_data.bin',
    'val_tokens'            : 10_000_000,
    'warmup_ratio'          : 0.03,
    'decay_ratio'           : 0.15,
    'min_lr_ratio'          : 0.1,
    # Validation
    'validate_every_steps'  : 500,
    'val_batches'           : 50,
    'save_every_steps'      : 2000,
    # Checkpoint
    'checkpoint_file'       : './Model/naylis_pretrain_60M.pt',
    # Compile
    'use_compile'           : not ARGS.no_compile,
    'compile_mode'          : ARGS.compile_mode,
    # DataLoader
    'num_workers'           : 1,
    'use_packing'           : True,
    # Plot
    'plot_file'             : './Model/training_curves_60M.png',
}

print('=' * 70)
print('  Naylis v1 -- Pretrain')
print('=' * 70)
if DEVICE == 'cuda':
    print(f'  GPU  : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
    cap = torch.cuda.get_device_capability()
    print(f'  SM   : {cap[0]}{cap[1]}')
_dl = CONFIG.get('depth_layout')
if _dl is not None:
    print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
          f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}  '
          f'rel_rank={CONFIG["rel_rank"]}')
    print('  Depth Layout :')
    for _zone in ('bottom', 'mid', 'top'):
        _z = _dl[_zone]
        print(f'    {_zone:6s}  asym={_z.get("asym",0)}  sym={_z.get("sym",0)}  '
              f'vanilla={_z.get("vanilla",0)}')
else:
    _sym_h  = CONFIG.get('sym_heads', 0)
    _van_h  = CONFIG.get('vanilla_heads', 0)
    _asym_h = CONFIG['num_heads'] - _sym_h - _van_h
    print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
          f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}  '
          f'rel_rank={CONFIG["rel_rank"]}  '
          f'asym={_asym_h}  sym={_sym_h}  vanilla={_van_h}  [global uniforme]')


# -- Tokenizer ------------------------------------------------------------------
print('\nTokenizer...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f'  vocab={len(tokenizer)}  eos={tokenizer.eos_token_id}')


# -- Data -----------------------------------------------------------------------
_data_path = Path(CONFIG['data_file'])
if not _data_path.exists():
    print(f'\nERREUR : fichier introuvable -> {_data_path}')
    sys.exit(1)

_n_tokens   = _data_path.stat().st_size // 2
_val_size   = min(CONFIG['val_tokens'], int(_n_tokens * 0.05))
_train_size = _n_tokens - _val_size

def steps_for_tokens(n_tokens: int) -> int:
    samples = n_tokens // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    return max(math.ceil(batches / CONFIG['gradient_accumulation']), 1)

TOTAL_STEPS = steps_for_tokens(_train_size)
print(f'\n  Fichier  : {_data_path}  ({_n_tokens / 1e9:.3f}B tokens  '
      f'{_data_path.stat().st_size / 1e9:.2f} GB)')
print(f'  Train    : {_train_size / 1e9:.3f}B tokens')
print(f'  Val      : {_val_size / 1e6:.0f}M tokens')
print(f'  Steps    : {TOTAL_STEPS:,}')


# -- Live Plot ------------------------------------------------------------------

class LivePlot:
    """
    Sauvegarde un PNG a chaque update avec deux subplots :
      - haut  : train loss (gris) + val loss (bleu)
      - bas   : graph_scale par zone (bottom=rouge, mid=orange, top=vert)
                + moyenne globale (jaune pointillé)

    En mode global uniforme (depth_layout=None), une seule courbe globale.
    En mode depth layout, 3 courbes par zone pour diagnostiquer les contributions.
    """
    _ZONE_COLORS = {
        'bottom': '#f47067',  # rouge
        'mid':    '#e3b341',  # orange/jaune
        'top':    '#3fb950',  # vert
        'avg':    '#8b949e',  # gris (globale)
    }

    def __init__(self, path: str, total_steps: int):
        self.path         = path
        self.total_steps  = total_steps
        self.train_steps  : list = []
        self.train_losses : list = []
        self.val_steps    : list = []
        self.val_losses   : list = []
        # graph_scale global
        self.gs_steps     : list = []
        self.gs_values    : list = []
        # graph_scale par zone (depth layout)
        self.gs_bottom    : list = []
        self.gs_mid       : list = []
        self.gs_top       : list = []
        self._has_zones   : bool = False
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def add_train(self, step: int, loss: float):
        self.train_steps.append(step)
        self.train_losses.append(loss)

    def add_val(self, step: int, loss: float):
        self.val_steps.append(step)
        self.val_losses.append(loss)

    def add_graph_scale(self, step: int, value: float,
                        zone_avgs: Optional[dict] = None):
        """
        value     : moyenne globale de |graph_scale| sur tous les layers actifs
        zone_avgs : dict {'bottom': float, 'mid': float, 'top': float}
                    si fourni, active le mode per-zone sur le plot
        """
        self.gs_steps.append(step)
        self.gs_values.append(value)
        if zone_avgs is not None:
            self._has_zones = True
            self.gs_bottom.append(zone_avgs.get('bottom', 0.0))
            self.gs_mid.append(zone_avgs.get('mid',    0.0))
            self.gs_top.append(zone_avgs.get('top',    0.0))

    def save(self):
        fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
        gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

        ax_loss = fig.add_subplot(gs[0])
        ax_gs   = fig.add_subplot(gs[1])

        for ax in (ax_loss, ax_gs):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.title.set_color('#c9d1d9')
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')

        # --- Loss ---
        if self.train_steps:
            ax_loss.plot(self.train_steps, self.train_losses,
                         color='#484f58', linewidth=0.8, alpha=0.7, label='Train loss')
        if self.val_steps:
            ax_loss.plot(self.val_steps, self.val_losses,
                         color='#58a6ff', linewidth=1.8, marker='o',
                         markersize=4, label='Val loss')
        ax_loss.set_title('Loss', fontsize=13, fontweight='bold')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlim(0, self.total_steps)
        if self.train_losses:
            ymin = min(min(self.train_losses), min(self.val_losses, default=999)) * 0.95
            ymax = max(max(self.train_losses[:20] if len(self.train_losses) > 20
                           else self.train_losses), 0.1) * 1.05
            ax_loss.set_ylim(max(ymin, 0), ymax)
        ax_loss.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor='#c9d1d9', fontsize=9)
        ax_loss.grid(True, color='#21262d', linewidth=0.5)

        if self.val_steps:
            last_s, last_l = self.val_steps[-1], self.val_losses[-1]
            ax_loss.annotate(
                f'{last_l:.4f}',
                xy=(last_s, last_l),
                xytext=(8, 8), textcoords='offset points',
                color='#58a6ff', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=0.8),
            )

        # --- Graph Scale ---
        c = self._ZONE_COLORS
        if self._has_zones and self.gs_steps:
            # Mode depth layout : 3 courbes per-zone + globale
            ax_gs.plot(self.gs_steps, self.gs_values,
                       color=c['avg'], linewidth=1.0, linestyle='--',
                       alpha=0.6, label='avg global')
            for zone_key, vals, label in [
                ('bottom', self.gs_bottom, 'bottom (0–25%)'),
                ('mid',    self.gs_mid,    'mid (25–75%)'),
                ('top',    self.gs_top,    'top (75–100%)'),
            ]:
                if vals:
                    ax_gs.plot(self.gs_steps[:len(vals)], vals,
                               color=c[zone_key], linewidth=1.8,
                               marker='s', markersize=3, label=label)
                    ax_gs.fill_between(self.gs_steps[:len(vals)], vals,
                                       alpha=0.08, color=c[zone_key])
            ax_gs.set_title('Naylis Graph Scale — par zone de profondeur',
                            fontsize=13, fontweight='bold')
        elif self.gs_steps:
            # Mode global uniforme : une seule courbe
            ax_gs.plot(self.gs_steps, self.gs_values,
                       color=c['mid'], linewidth=1.5, marker='s',
                       markersize=4, label='|graph_scale| avg')
            ax_gs.fill_between(self.gs_steps, self.gs_values,
                               alpha=0.15, color=c['mid'])
            ax_gs.set_title('Naylis Graph Scale', fontsize=13, fontweight='bold')

        ax_gs.set_xlabel('Step')
        ax_gs.set_ylabel('|graph_scale| avg')
        ax_gs.set_xlim(0, self.total_steps)
        ax_gs.legend(facecolor='#161b22', edgecolor='#30363d',
                     labelcolor='#c9d1d9', fontsize=9)
        ax_gs.grid(True, color='#21262d', linewidth=0.5)

        if self.gs_values:
            last_s, last_v = self.gs_steps[-1], self.gs_values[-1]
            ax_gs.annotate(
                f'{last_v:.5f}',
                xy=(last_s, last_v),
                xytext=(8, 8), textcoords='offset points',
                color=c['avg'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=c['avg'], lw=0.8),
            )

        fig.suptitle(
            f'Naylis Pretrain  —  {self.total_steps:,} steps',
            color='#c9d1d9', fontsize=15, fontweight='bold', y=0.98,
        )

        plt.savefig(self.path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)


# -- WSD Scheduler --------------------------------------------------------------
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self) -> float:
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self): return [self.get_lr()]
    def state_dict(self):  return {'current_step': self.current_step}
    def load_state_dict(self, sd): self.current_step = sd['current_step']


# -- MmapData -------------------------------------------------------------------
class MmapData:
    def __init__(self):
        print(f'  mmap -> {_data_path}')
        t0  = time.time()
        arr = np.memmap(str(_data_path), dtype=np.uint16, mode='r', shape=(_n_tokens,))
        self._arr       = arr
        self._train_end = _train_size
        print(f'  mmap OK  train={_train_size/1e9:.3f}B  val={_val_size/1e6:.0f}M  '
              f'({time.time()-t0:.1f}s)')

    def train_dataset(self, seq_len, use_packing, eos_id):
        if use_packing:
            return MmapPackedDataset(self._arr[:self._train_end], seq_len, eos_id)
        return MmapDataset(self._arr[:self._train_end], seq_len)

    def val_dataset(self, seq_len):
        return MmapDataset(self._arr[self._train_end:], seq_len)

    def unload(self):
        del self._arr
        gc.collect()
        print('  mmap libere')


class MmapDataset(Dataset):
    def __init__(self, arr, seq_len):
        n            = len(arr) // (seq_len + 1)
        self._arr    = arr[:n * (seq_len + 1)]
        self.seq_len = seq_len
        self.n       = n

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * (self.seq_len + 1)
        c = torch.from_numpy(self._arr[s:s + self.seq_len + 1].astype(np.int32))
        return c[:-1].clone(), c[1:].clone()


class MmapPackedDataset(Dataset):
    def __init__(self, arr, seq_len, eos_token_id):
        n                 = len(arr) // (seq_len + 1)
        self._arr         = arr[:n * (seq_len + 1)]
        self.seq_len      = seq_len
        self.eos_token_id = eos_token_id
        self.n            = n

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * (self.seq_len + 1)
        b = torch.from_numpy(self._arr[s:s + self.seq_len + 1].astype(np.int32))
        return b[:-1].clone(), b[1:].clone()


def packed_collate_fn(batch, eos_token_id, seq_len):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    all_cu = [0]; max_sl = 1
    for i in range(x.size(0)):
        seq     = x[i]
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) == 0:
            all_cu.append(all_cu[-1] + seq_len); max_sl = max(max_sl, seq_len)
        else:
            prev = 0
            for pos in eos_pos.tolist():
                l = pos - prev + 1
                if l > 0: all_cu.append(all_cu[-1] + l); max_sl = max(max_sl, l)
                prev = pos + 1
            rem = seq_len - prev
            if rem > 0: all_cu.append(all_cu[-1] + rem); max_sl = max(max_sl, rem)
    return x, y, torch.tensor(all_cu, dtype=torch.int32), max_sl


# -- Checkpoint -----------------------------------------------------------------
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata):
        m           = model._orig_mod if hasattr(model, '_orig_mod') else model
        muon, adamw = optimizers
        cp = {
            'model_state_dict'    : m.state_dict(),
            'muon_state_dict'     : muon.state_dict(),
            'adamw_state_dict'    : adamw.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        info_path = self.path.replace('.pt', '_info.json')
        info      = {**metadata, 'last_save': datetime.now().isoformat(), 'config': CONFIG}
        tmp_json  = info_path + '.tmp'
        with open(tmp_json, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        tmp_pt = self.path + '.tmp'
        torch.save(cp, tmp_pt)
        os.replace(tmp_pt, self.path)
        os.replace(tmp_json, info_path)
        print(f'  SAVE  step={metadata["global_step"]:,}  [{self.path}]')

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f'\nCheckpoint trouve : {self.path}')
        cp        = torch.load(self.path, map_location='cpu', weights_only=False)
        info_path = self.path.replace('.pt', '_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            for k in ('global_step', 'current_epoch', 'chunk_within_epoch',
                      'total_training_time', 'chunk_start_step'):
                cp[k] = info.get(k, 0)
        else:
            cp.update({'global_step': 0, 'current_epoch': 1,
                       'chunk_within_epoch': 0, 'total_training_time': 0.0,
                       'chunk_start_step': 0})
        return cp


# -- Validation -----------------------------------------------------------------
@torch.no_grad()
def validate(model, val_loader, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y)
            total_loss += loss.item(); n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg


# -- Muon + MARS ----------------------------------------------------------------
def _zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                     ns_steps=ns_steps, weight_decay=weight_decay,
                                     use_mars=use_mars, mars_gamma=mars_gamma))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nest = group['lr'], group['momentum'], group['nesterov']
            ns, wd        = group['ns_steps'], group['weight_decay']
            use_mars, mg  = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g     = p.grad
                state = self.state[p]
                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev = state['prev_grad']
                    c_t  = torch.clamp(
                        (mg / (1. - mg)) * (g.norm() + 1e-8) / (prev.norm() + 1e-8),
                        max=1.0)
                    g = g + c_t * (g - prev)
                    state['prev_grad'].copy_(p.grad)
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mom).add_(g)
                g   = (g + mom * buf) if nest else buf
                g   = _zeropower_via_newtonschulz5(g, steps=ns)
                g   = g * max(g.size(0), g.size(1)) ** 0.5
                if wd: p.mul_(1. - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr, weight_decay, betas, eps):
    EXCLUDE = {'token_embeddings.weight', 'output_head.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
        elif p.dim() >= 2 and pn.startswith('blocks.'):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith('blocks.'):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr_muon, momentum=0.95, nesterov=True,
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True
    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
         {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False}],
        lr=lr, betas=betas, eps=eps, fused=(DEVICE == 'cuda'),
    )
    n_muon  = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_decay + adamw_nodecay)
    print(f'\n  Muon+MARS  : {n_muon / 1e6:.2f}M params  lr={lr_muon:.2e}')
    print(f'  AdamW      : {n_adamw / 1e6:.2f}M params  lr={lr:.2e}')
    return muon_opt, adamw_opt


# -- Train one pass -------------------------------------------------------------
def train_one_pass(
    model, data, optimizers, scheduler,
    ckpt_mgr, history, plot,
    global_step, total_time, start_step,
):
    muon_opt, adamw_opt = optimizers
    steps_done   = global_step - start_step
    batches_done = steps_done * CONFIG['gradient_accumulation']

    print(f'\n{"="*70}')
    print(f'  TRAINING  --  {TOTAL_STEPS:,} steps  |  {_train_size/1e9:.3f}B tokens')
    print(f'{"="*70}')

    train_ds = data.train_dataset(
        CONFIG['max_seq_len'], CONFIG['use_packing'], tokenizer.eos_token_id)
    val_ds   = data.val_dataset(CONFIG['max_seq_len'])

    total_seqs = len(train_ds)
    if batches_done >= math.ceil(total_seqs / CONFIG['batch_size']):
        print('  Pass deja terminee -- skip')
        data.unload()
        return global_step, total_time, start_step

    indices = list(range(batches_done * CONFIG['batch_size'], total_seqs))

    class IndexSampler(torch.utils.data.Sampler):
        def __init__(self, idx): self._idx = idx
        def __iter__(self): return iter(self._idx)
        def __len__(self):  return len(self._idx)

    collate = (partial(packed_collate_fn,
                       eos_token_id=tokenizer.eos_token_id,
                       seq_len=CONFIG['max_seq_len'])
               if CONFIG['use_packing'] else None)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        sampler=IndexSampler(indices),
        num_workers=CONFIG['num_workers'], pin_memory=True,
        persistent_workers=(CONFIG['num_workers'] > 0),
        prefetch_factor=2 if CONFIG['num_workers'] > 0 else None,
        drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=1, pin_memory=True,
    )

    total_batches = total_seqs // CONFIG['batch_size']
    print(f'  batches={total_batches:,}  restant={len(train_loader):,}  '
          f'packing={"ON" if CONFIG["use_packing"] else "OFF"}')

    model.train()
    ae            = (DEVICE == 'cuda')
    adt           = torch.bfloat16
    run_loss      = 0.0
    valid_batches = 0
    acc_steps     = 0
    t0            = time.time()

    pbar = tqdm(
        train_loader, desc='  train',
        initial=total_batches - len(train_loader),
        total=total_batches, leave=True, dynamic_ncols=True,
    )

    for batch in pbar:
        try:
            if CONFIG['use_packing'] and len(batch) == 4:
                x, y, cu_seqlens, max_sl = batch
                x          = x.to(DEVICE, non_blocking=True)
                y          = y.to(DEVICE, non_blocking=True)
                cu_seqlens = cu_seqlens.to(DEVICE, non_blocking=True)
            else:
                x, y       = batch[0].to(DEVICE), batch[1].to(DEVICE)
                cu_seqlens = max_sl = None

            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                _, loss, _ = model(
                    x, targets=y,
                    cu_seqlens_q = cu_seqlens,
                    cu_seqlens_k = cu_seqlens,
                    max_seqlen_q = int(max_sl) if max_sl is not None else None,
                    max_seqlen_k = int(max_sl) if max_sl is not None else None,
                )
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                acc_steps = 0
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            run_loss      += loss.item() * CONFIG['gradient_accumulation']
            valid_batches += 1
            acc_steps     += 1

            if acc_steps >= CONFIG['gradient_accumulation']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                muon_opt.step()
                adamw_opt.step()
                lr = scheduler.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                acc_steps   = 0
                global_step += 1

                avg_loss = run_loss / max(valid_batches, 1)
                plot.add_train(global_step, avg_loss)

                pbar.set_postfix(
                    loss=f'{avg_loss:.4f}',
                    lr  =f'{lr:.2e}',
                )

                # Validation
                if global_step % CONFIG['validate_every_steps'] == 0:
                    _, vloss = validate(model, val_loader, CONFIG['val_batches'])
                    pbar.write(f'  [val  step={global_step:,}] '
                               f'loss={vloss:.4f}')
                    history.setdefault('validations', []).append(
                        {'step': global_step, 'val_loss': vloss})
                    plot.add_val(global_step, vloss)
                    plot.save()

                # Checkpoint
                if global_step % CONFIG['save_every_steps'] == 0:
                    ckpt_mgr.save(model, optimizers, scheduler, {
                        'global_step': global_step,
                        'total_training_time': total_time,
                        'start_step': start_step,
                    })

                # Graph scale — global + per-zone (depth layout)
                if global_step % 1000 == 0 or global_step == 1:
                    raw        = model._orig_mod if hasattr(model, '_orig_mod') else model
                    zone_avgs  = raw.get_graph_scales_per_zone()
                    # Moyenne globale sur les layers avec graph_scale actif
                    all_vals   = [v for v in zone_avgs.values() if v > 0.0]
                    avg_s      = sum(all_vals) / len(all_vals) if all_vals else 0.0
                    has_dl     = raw.depth_layout is not None
                    if has_dl:
                        pbar.write(
                            f'  [naylis step={global_step:,}]  '
                            f'avg={avg_s:.5f}  '
                            f'bottom={zone_avgs["bottom"]:.5f}  '
                            f'mid={zone_avgs["mid"]:.5f}  '
                            f'top={zone_avgs["top"]:.5f}'
                        )
                        plot.add_graph_scale(global_step, avg_s, zone_avgs)
                    else:
                        pbar.write(f'  [naylis step={global_step:,}] '
                                   f'|graph_scale| avg={avg_s:.5f}')
                        plot.add_graph_scale(global_step, avg_s)

                # Plot save every 500 steps
                if global_step % 500 == 0 or global_step == 1:
                    plot.save()

        except torch.cuda.OutOfMemoryError:
            print(f'\n  OOM -- skip batch')
            torch.cuda.empty_cache()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)
            acc_steps = 0
            gc.collect()
            model.train()
            continue

    pbar.close()
    elapsed     = time.time() - t0
    total_time += elapsed
    avg_loss    = run_loss / max(valid_batches, 1)
    print(f'\n  Pass terminee | loss={avg_loss:.4f} | {elapsed / 60:.1f}min')

    history.setdefault('passes', []).append(
        {'loss': avg_loss, 'time_sec': elapsed, 'global_step': global_step})

    plot.save()

    data.unload()
    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, total_time, start_step


# -- Main -----------------------------------------------------------------------
def main():
    print('\n' + '=' * 70)
    print('  CREATION MODELE')
    print('=' * 70)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])
    plot     = LivePlot(CONFIG['plot_file'], TOTAL_STEPS)

    model = NaylisGPT(
        vocab_size            = CONFIG['vocab_size'],
        embed_dim             = CONFIG['embed_dim'],
        num_heads             = CONFIG['num_heads'],
        num_layers            = CONFIG['num_layers'],
        max_seq_len           = CONFIG['max_seq_len'],
        dropout               = CONFIG['dropout'],
        use_rope              = CONFIG['use_rope'],
        use_yarn              = CONFIG['use_yarn'],
        yarn_scale            = CONFIG['yarn_scale'],
        yarn_original_max_len = CONFIG['yarn_original_max_len'],
        use_swiglu            = CONFIG['use_swiglu'],
        n_kv_heads            = CONFIG['n_kv_heads'],
        use_qk_norm           = CONFIG['use_qk_norm'],
        soft_cap              = CONFIG['soft_cap'],
        use_flash_attn        = CONFIG['use_flash_attn'],
        rel_rank              = CONFIG['rel_rank'],
        # ── Depth layout (prioritaire) ou global uniforme ────────
        depth_layout          = CONFIG.get('depth_layout'),
        sym_heads             = CONFIG.get('sym_heads', 0),
        vanilla_heads         = CONFIG.get('vanilla_heads', 0),
    ).to(DEVICE)

    p = model.count_parameters()
    print(f'  Params total : {p["total_M"]}M')
    print(f'  Naylis       : {p["naylis_K"]}K = {p["naylis_pct"]}')
    print(model.describe_layout())

    if CONFIG['use_compile'] and DEVICE == 'cuda':
        print('\ntorch.compile...')
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors  = True
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL : {e}')
    else:
        print('\ntorch.compile : desactive')

    raw_model  = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model, CONFIG['learning_rate'], CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']), CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG['learning_rate'],
        total_steps=TOTAL_STEPS, warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'], min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    history     = {'config': CONFIG, 'passes': [], 'validations': []}
    global_step = 0
    total_time  = 0.0
    start_step  = 0

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(cp['model_state_dict'])
        muon_opt.load_state_dict(cp.get('muon_state_dict', {}))
        adamw_opt.load_state_dict(cp.get('adamw_state_dict', {}))
        scheduler.load_state_dict(cp.get('scheduler_state_dict', {}))
        global_step = cp.get('global_step', 0)
        total_time  = cp.get('total_training_time', 0.0)
        start_step  = cp.get('start_step', 0)
        if global_step >= TOTAL_STEPS:
            print('Training deja termine.')
            return

    print('\n' + '=' * 70)
    print(f'  TRAINING START -- {TOTAL_STEPS:,} steps')
    print(f'  Plot -> {CONFIG["plot_file"]}')
    print('=' * 70)

    data = MmapData()

    try:
        global_step, total_time, start_step = train_one_pass(
            model=model, data=data, optimizers=optimizers,
            scheduler=scheduler, ckpt_mgr=ckpt_mgr,
            history=history, plot=plot,
            global_step=global_step, total_time=total_time,
            start_step=start_step,
        )
    except KeyboardInterrupt:
        print('\n  CTRL+C')
        ckpt_mgr.save(model, optimizers, scheduler, {
            'global_step': global_step,
            'total_training_time': total_time,
            'start_step': start_step,
        })
        plot.save()
        return
    except Exception:
        print(f'\n  ERREUR :\n{traceback.format_exc()}')
        ckpt_mgr.save(model, optimizers, scheduler, {
            'global_step': global_step,
            'total_training_time': total_time,
            'start_step': start_step,
        })
        plot.save()
        raise

    print(f'\n{"="*70}\n  TRAINING TERMINE\n{"="*70}')
    print(f'  Steps : {global_step:,}  |  Temps : {total_time / 3600:.2f}h')

    ckpt_mgr.save(model, optimizers, scheduler, {
        'global_step': global_step,
        'total_training_time': total_time,
        'start_step': start_step,
    })
    hist_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'  History : {hist_path}')
    print(f'  Plot    : {CONFIG["plot_file"]}')
    print('  DONE')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')
