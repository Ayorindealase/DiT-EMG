"""
train.py — DiT-EMG: Diffusion Transformer for Synthetic sEMG Generation
=========================================================================
THIS FILE IS EDITED BY THE AUTORESEARCH AGENT.

The agent modifies hyperparameters, architecture choices, and the optimizer
to minimise val_fid (Fréchet distance between real and synthetic EMG).

Architecture: Denoising Diffusion Probabilistic Model (DDPM) with a
Transformer backbone (DiT), conditioned on gesture class labels via
adaptive layer normalisation (adaLN).

Metric: val_fid — lower is better (primary optimisation target)
Budget: TRAIN_TIME_SECONDS = 300 (5 minutes, fixed — DO NOT CHANGE)

Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.optim import AdamW

# Import everything from prepare.py (fixed — do not change these imports)
sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    get_dataloader, get_real_samples, evaluate,
    N_CHANNELS, WINDOW_SIZE, N_CLASSES,
    CACHE_TRAIN, CACHE_VAL,
)

# ─────────────────────────────────────────────────────────────────
# ❶  HYPERPARAMETERS  ← agent iterates on this entire block
# ─────────────────────────────────────────────────────────────────

# Training budget (FIXED — must not change)
TRAIN_TIME_SECONDS = 300          # 5-minute wall-clock budget

# Batch & optimiser
BATCH_SIZE         = 128          # per-step batch size
LEARNING_RATE      = 3e-4         # peak learning rate
WEIGHT_DECAY       = 1e-4         # AdamW weight decay
BETAS              = (0.9, 0.999) # AdamW beta parameters
GRAD_CLIP          = 1.0          # gradient clipping norm

# DiT architecture
PATCH_SIZE         = 10           # temporal patch size (WINDOW_SIZE must be divisible)
D_MODEL            = 256          # transformer hidden dimension
N_HEADS            = 8            # number of attention heads  (D_MODEL % N_HEADS == 0)
DEPTH              = 6            # number of DiT blocks
D_FF_MULT          = 4            # feedforward expansion factor (d_ff = D_MODEL * D_FF_MULT)
DROPOUT            = 0.1          # dropout rate in attention + FFN
CLASS_EMBED_DIM    = 128          # gesture class embedding dimension

# Diffusion schedule
T_STEPS            = 1000         # total diffusion timesteps
SCHEDULE           = "cosine"     # noise schedule: "cosine" | "linear"
BETA_START         = 1e-4         # linear schedule start (ignored if cosine)
BETA_END           = 0.02         # linear schedule end   (ignored if cosine)

# Sampling
SAMPLE_STEPS       = 250          # DDIM steps for fast inference (≤ T_STEPS)
SAMPLE_GUIDANCE    = 1.5          # classifier-free guidance scale (1.0 = no guidance)
CFG_DROPOUT        = 0.1          # probability of dropping class label during training

# Logging
LOG_INTERVAL       = 50           # print loss every N steps
EVAL_EVERY_STEPS   = 500          # run full evaluation every N steps
CHECKPOINT_DIR     = Path("checkpoints")
RESULTS_FILE       = Path("results.jsonl")

# ─────────────────────────────────────────────────────────────────
# ❷  NOISE SCHEDULE
# ─────────────────────────────────────────────────────────────────

def make_beta_schedule(schedule: str, T: int,
                        beta_start: float = 1e-4,
                        beta_end: float = 0.02) -> torch.Tensor:
    """
    Return beta schedule tensor of shape (T,).
    Cosine schedule from Nichol & Dhariwal 2021 (improved DDPM).
    Linear schedule from Ho et al. 2020 (original DDPM).
    """
    if schedule == "cosine":
        steps  = T + 1
        s      = 0.008  # small offset to prevent beta too small near t=0
        t_vals = torch.linspace(0, T, steps) / T
        f_t    = torch.cos((t_vals + s) / (1 + s) * math.pi / 2) ** 2
        f_0    = f_t[0]
        alphas_cumprod = f_t / f_0
        betas  = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas  = torch.clamp(betas, min=1e-5, max=0.999)
    elif schedule == "linear":
        betas  = torch.linspace(beta_start, beta_end, T)
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'cosine' or 'linear'.")
    return betas


class DiffusionSchedule(nn.Module):
    """
    Pre-computes and stores all diffusion schedule quantities.
    Registered as buffers so they move to GPU with .to(device).
    """

    def __init__(self, T: int, schedule: str,
                 beta_start: float, beta_end: float):
        super().__init__()
        betas              = make_beta_schedule(schedule, T, beta_start, beta_end)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",              betas)
        self.register_buffer("alphas",             alphas)
        self.register_buffer("alphas_cumprod",     alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",     torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                              torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance",
                              betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to x0 at timestep t.
        x0 : (B, C, W)
        t  : (B,) LongTensor of timestep indices
        Returns noisy x_t of same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        s_a = self.sqrt_alphas_cumprod[t]            # (B,)
        s_b = self.sqrt_one_minus_alphas_cumprod[t]  # (B,)

        # Reshape for broadcasting: (B,) → (B, 1, 1)
        s_a = s_a[:, None, None]
        s_b = s_b[:, None, None]

        return s_a * x0 + s_b * noise

    def predict_x0(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                   t: torch.Tensor) -> torch.Tensor:
        """Recover x0 estimate from predicted noise."""
        s_a = self.sqrt_alphas_cumprod[t][:, None, None]
        s_b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - s_b * noise_pred) / s_a


# ─────────────────────────────────────────────────────────────────
# ❸  DiT-EMG MODEL
# ─────────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep t.
    Encodes scalar t → dense vector of shape (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Linear projection after sinusoidal encoding
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) int64 → (B, dim)"""
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                                     # (half,)
        args  = t[:, None].float() * freqs[None, :]          # (B, half)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)                                 # (B, dim)


class AdaptiveLayerNorm(nn.Module):
    """
    adaLN — Adaptive Layer Normalisation conditioned on a context vector.
    Modulates scale (gamma) and shift (beta) via linear projections of
    the conditioning signal (timestep + class embeddings).

    From DiT: Peebles & Xie 2023.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Project conditioning signal → 2*dim (scale + shift)
        self.proj  = nn.Linear(cond_dim, 2 * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, dim)   — sequence of patch tokens
        cond : (B, cond_dim) — conditioning vector
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)   # each (B, dim)
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    """
    Single DiT transformer block.
    Structure: adaLN → Multi-Head Self-Attention → adaLN → FFN
    with residual connections and a zero-initialised output gate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 cond_dim: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Adaptive layer norms (pre-norm style)
        self.norm1 = AdaptiveLayerNorm(d_model, cond_dim)
        self.norm2 = AdaptiveLayerNorm(d_model, cond_dim)

        # Multi-head self-attention
        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # Position-wise feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Zero-initialised output scale gate (DiT design choice for stable init)
        self.gate_attn = nn.Parameter(torch.zeros(d_model))
        self.gate_ff   = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N_patches, d_model)
        cond : (B, cond_dim)
        """
        # Self-attention with adaLN
        x_norm, _ = self.attn(
            self.norm1(x, cond),
            self.norm1(x, cond),
            self.norm1(x, cond),
        )
        x = x + self.gate_attn.tanh() * x_norm

        # FFN with adaLN
        x = x + self.gate_ff.tanh() * self.ff(self.norm2(x, cond))

        return x


class DiTEMG(nn.Module):
    """
    Diffusion Transformer for multichannel sEMG signal generation.

    Architecture:
      1. Patch embedding  : (B, C, W) → (B, N_patches, d_model)
      2. Positional embed : learnable 1D positional embeddings
      3. Conditioning     : timestep embedding + class embedding → cond vector
      4. DiT blocks       : DEPTH × [adaLN → Attn → adaLN → FFN]
      5. Output head      : (B, N_patches, d_model) → (B, C, W) noise prediction
    """

    def __init__(
        self,
        n_channels:     int   = N_CHANNELS,
        window_size:    int   = WINDOW_SIZE,
        n_classes:      int   = N_CLASSES,
        patch_size:     int   = PATCH_SIZE,
        d_model:        int   = D_MODEL,
        n_heads:        int   = N_HEADS,
        depth:          int   = DEPTH,
        d_ff_mult:      int   = D_FF_MULT,
        dropout:        float = DROPOUT,
        class_embed_dim: int  = CLASS_EMBED_DIM,
        cfg_dropout:    float = CFG_DROPOUT,
    ):
        super().__init__()

        assert window_size % patch_size == 0, \
            f"WINDOW_SIZE ({window_size}) must be divisible by PATCH_SIZE ({patch_size})"

        self.n_channels    = n_channels
        self.window_size   = window_size
        self.patch_size    = patch_size
        self.n_patches     = window_size // patch_size
        self.d_model       = d_model
        self.cfg_dropout   = cfg_dropout
        self.n_classes     = n_classes

        # ── Patch embedding ──────────────────────────────────────────
        # Treat each (C × patch_size) segment as a "token"
        patch_dim = n_channels * patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # ── Conditioning ─────────────────────────────────────────────
        # Timestep: sinusoidal → MLP
        self.time_embed = SinusoidalTimestepEmbedding(d_model)

        # Class: learnable embedding table + null token for CFG
        # Index n_classes = "null class" (used during CFG training dropout)
        self.class_embed = nn.Embedding(n_classes + 1, class_embed_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # Fuse time + class into conditioning vector
        cond_dim = d_model + class_embed_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # ── DiT blocks ───────────────────────────────────────────────
        d_ff = d_model * d_ff_mult
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model  = d_model,
                n_heads  = n_heads,
                d_ff     = d_ff,
                cond_dim = cond_dim,
                dropout  = dropout,
            )
            for _ in range(depth)
        ])

        # Final layer norm (not adaptive — standard)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        # ── Output head ──────────────────────────────────────────────
        # Predict noise: d_model → patch_dim, then reshape back to (C, W)
        self.output_head = nn.Linear(d_model, patch_dim, bias=True)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, small init for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, W) → (B, N_patches, C * patch_size)
        Splits the temporal dimension into non-overlapping patches.
        """
        B, C, W = x.shape
        P = self.patch_size
        N = W // P
        x = x.reshape(B, C, N, P)           # (B, C, N, P)
        x = x.permute(0, 2, 1, 3)           # (B, N, C, P)
        x = x.reshape(B, N, C * P)          # (B, N, C*P)
        return x

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, N_patches, C * patch_size) → (B, C, W)
        Inverse of _patchify.
        """
        B, N, CP = x.shape
        C = self.n_channels
        P = self.patch_size
        x = x.reshape(B, N, C, P)           # (B, N, C, P)
        x = x.permute(0, 2, 1, 3)           # (B, C, N, P)
        x = x.reshape(B, C, N * P)          # (B, C, W)
        return x

    def forward(
        self,
        x_t:   torch.Tensor,   # (B, C, W)  noisy signal at timestep t
        t:     torch.Tensor,   # (B,)       diffusion timestep index
        y:     torch.Tensor,   # (B,)       class label (int)
        force_uncond: bool = False,  # force unconditional (for CFG inference)
    ) -> torch.Tensor:
        """
        Predict the noise added to x_t at timestep t, conditioned on class y.
        Returns predicted noise of shape (B, C, W).
        """
        B = x_t.shape[0]
        device = x_t.device

        # ── Classifier-Free Guidance dropout (training only) ─────────
        if self.training and self.cfg_dropout > 0:
            mask = torch.rand(B, device=device) < self.cfg_dropout
            null_token = torch.full_like(y, self.n_classes)  # null class index
            y = torch.where(mask, null_token, y)

        if force_uncond:
            y = torch.full_like(y, self.n_classes)

        # ── Patch + positional embedding ─────────────────────────────
        tokens = self._patchify(x_t)                     # (B, N, C*P)
        tokens = self.patch_embed(tokens)                # (B, N, d_model)
        tokens = tokens + self.pos_embed                 # (B, N, d_model)

        # ── Conditioning vector ──────────────────────────────────────
        t_emb   = self.time_embed(t)                     # (B, d_model)
        cls_emb = self.class_embed(y)                    # (B, class_embed_dim)
        cond    = torch.cat([t_emb, cls_emb], dim=-1)   # (B, cond_dim)
        cond    = self.cond_proj(cond)                   # (B, cond_dim)

        # ── DiT blocks ───────────────────────────────────────────────
        for block in self.blocks:
            tokens = block(tokens, cond)

        # ── Output ───────────────────────────────────────────────────
        tokens = self.final_norm(tokens)                 # (B, N, d_model)
        noise  = self.output_head(tokens)                # (B, N, C*P)
        noise  = self._unpatchify(noise)                 # (B, C, W)

        return noise


# ─────────────────────────────────────────────────────────────────
# ❹  CLASSIFIER-FREE GUIDANCE SAMPLING (DDIM)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    model:    DiTEMG,
    schedule: DiffusionSchedule,
    n_samples: int,
    labels:   torch.Tensor,         # (n_samples,) class indices
    device:   torch.device,
    steps:    int   = SAMPLE_STEPS,
    guidance: float = SAMPLE_GUIDANCE,
    eta:      float = 0.0,          # eta=0 → deterministic DDIM
) -> torch.Tensor:
    """
    DDIM sampler with Classifier-Free Guidance.
    Returns synthetic EMG signals: (n_samples, C, W).

    Guidance formula: eps = eps_uncond + guidance * (eps_cond - eps_uncond)
    """
    model.eval()

    # Start from pure Gaussian noise
    x = torch.randn(n_samples, model.n_channels, model.window_size, device=device)

    # Select evenly-spaced subset of timesteps for DDIM
    T = len(schedule.betas)
    t_seq  = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i, t_val in enumerate(t_seq):
        t_batch = t_val.expand(n_samples)

        # Conditional and unconditional predictions
        eps_cond   = model(x, t_batch, labels,         force_uncond=False)
        eps_uncond = model(x, t_batch, labels,         force_uncond=True)

        # CFG interpolation
        eps = eps_uncond + guidance * (eps_cond - eps_uncond)

        # DDIM update step
        a_t    = schedule.alphas_cumprod[t_val]
        a_prev = schedule.alphas_cumprod_prev[t_val]

        x0_pred  = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x0_pred  = x0_pred.clamp(-3.0, 3.0)    # clip x0 for stability

        sigma_t  = eta * ((1 - a_prev) / (1 - a_t) * schedule.betas[t_val]).sqrt()
        noise    = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

        x = a_prev.sqrt() * x0_pred + \
            (1 - a_prev - sigma_t ** 2).clamp(min=0).sqrt() * eps + \
            sigma_t * noise

    return x.cpu()


# ─────────────────────────────────────────────────────────────────
# ❺  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    AdamW with weight decay applied only to weight matrices (not biases / norms).
    Agent may replace this block with a different optimiser.
    """
    decay_params   = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay on 1D params (bias, layer norm, embeddings)
        if param.ndim == 1 or "embed" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = LEARNING_RATE,
        betas = BETAS,
    )


def cosine_lr_schedule(step: int, warmup_steps: int, total_steps: int,
                        lr_min: float = 1e-6) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (1.0 - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, metrics: Dict, path: Path):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":        step,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "metrics":     metrics,
        "hparams": {
            "BATCH_SIZE":      BATCH_SIZE,
            "LEARNING_RATE":   LEARNING_RATE,
            "D_MODEL":         D_MODEL,
            "N_HEADS":         N_HEADS,
            "DEPTH":           DEPTH,
            "PATCH_SIZE":      PATCH_SIZE,
            "T_STEPS":         T_STEPS,
            "SCHEDULE":        SCHEDULE,
            "SAMPLE_GUIDANCE": SAMPLE_GUIDANCE,
        },
    }, path)


def log_result(step: int, metrics: Dict, elapsed: float):
    """Append a result line to results.jsonl — used by analysis.ipynb."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "step":      step,
        "elapsed_s": round(elapsed, 1),
        **metrics,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────────
# ❻  SYNTHETIC SAMPLE GENERATION HELPER
# ─────────────────────────────────────────────────────────────────

def generate_synthetic_batch(
    model:    DiTEMG,
    schedule: DiffusionSchedule,
    n:        int,
    device:   torch.device,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n synthetic EMG windows with class labels.
    Used during evaluation to produce the synthetic set for FID/TSTR.

    Args:
        balanced : if True, generate equal samples per class
    Returns:
        X_syn : (n, C, W) numpy float32
        y_syn : (n,)      numpy int32
    """
    model.eval()
    all_x, all_y = [], []

    samples_per_class = max(1, n // N_CLASSES)
    per_batch = min(64, samples_per_class)  # cap batch size for VRAM

    for cls in range(N_CLASSES):
        remaining = samples_per_class
        while remaining > 0:
            bs  = min(per_batch, remaining)
            lbl = torch.full((bs,), cls, dtype=torch.long, device=device)
            x   = ddim_sample(model, schedule, bs, lbl, device)
            all_x.append(x.numpy())
            all_y.extend([cls] * bs)
            remaining -= bs

    X_syn = np.concatenate(all_x, axis=0).astype(np.float32)
    y_syn = np.array(all_y, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(y_syn))
    return X_syn[idx], y_syn[idx]


# ─────────────────────────────────────────────────────────────────
# ❼  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train():
    # ── Device setup ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("[WARN] No GPU detected — training will be very slow.")

    print(f"\n{'='*60}")
    print(f"  DiT-EMG Training")
    print(f"  Device    : {device}")
    print(f"{'='*60}")

    # ── Check data cache ──────────────────────────────────────────
    if not CACHE_TRAIN.exists():
        print("[ERROR] Data cache missing. Run: python prepare.py --synthetic")
        sys.exit(1)

    # ── Model & schedule ──────────────────────────────────────────
    model    = DiTEMG().to(device)
    schedule = DiffusionSchedule(T_STEPS, SCHEDULE, BETA_START, BETA_END).to(device)
    optim    = make_optimizer(model)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  D_MODEL={D_MODEL}  DEPTH={DEPTH}  N_HEADS={N_HEADS}  PATCH={PATCH_SIZE}")
    print(f"  T={T_STEPS}  Schedule={SCHEDULE}  Guidance={SAMPLE_GUIDANCE}")
    print(f"  Batch={BATCH_SIZE}  LR={LEARNING_RATE}  Time budget={TRAIN_TIME_SECONDS}s")
    print(f"{'='*60}\n")

    # ── Mixed precision scaler (CUDA only) ────────────────────────
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Dataloader ────────────────────────────────────────────────
    loader    = get_dataloader("train", BATCH_SIZE, shuffle=True)

    # Estimate warmup / total steps from time budget
    # Rough estimate: assume ~10 steps/sec on T4 GPU
    est_total_steps = TRAIN_TIME_SECONDS * 10
    warmup_steps    = min(500, est_total_steps // 20)

    # ── Training ──────────────────────────────────────────────────
    best_fid    = float("inf")
    best_ckpt   = CHECKPOINT_DIR / "best_model.pt"
    train_start = time.time()
    step        = 0
    losses      = []

    try:
        while True:
            elapsed = time.time() - train_start
            if elapsed >= TRAIN_TIME_SECONDS:
                print(f"\n✓ Time budget reached ({TRAIN_TIME_SECONDS}s). Stopping.")
                break

            # ── Load batch ────────────────────────────────────────
            X_batch, y_batch = next(loader)
            x0 = torch.from_numpy(X_batch).to(device)   # (B, C, W)
            y  = torch.from_numpy(y_batch).long().to(device)  # (B,)

            # ── Sample random timesteps ───────────────────────────
            t = torch.randint(0, T_STEPS, (x0.shape[0],), device=device)

            # ── Forward diffusion (add noise) ─────────────────────
            noise = torch.randn_like(x0)
            x_t   = schedule.q_sample(x0, t, noise)

            # ── Predict noise with DiT ────────────────────────────
            model.train()
            optim.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    noise_pred = model(x_t, t, y)
                    loss = F.mse_loss(noise_pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optim)
                scaler.update()
            else:
                noise_pred = model(x_t, t, y)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()

            # ── LR schedule ───────────────────────────────────────
            lr_scale = cosine_lr_schedule(step, warmup_steps, est_total_steps)
            for pg in optim.param_groups:
                pg["lr"] = LEARNING_RATE * lr_scale

            losses.append(loss.item())
            step += 1

            # ── Logging ───────────────────────────────────────────
            if step % LOG_INTERVAL == 0:
                avg_loss = np.mean(losses[-LOG_INTERVAL:])
                lr_now   = optim.param_groups[0]["lr"]
                print(f"  step {step:5d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | elapsed {elapsed:.0f}s")

            # ── Full evaluation ───────────────────────────────────
            if step % EVAL_EVERY_STEPS == 0:
                print(f"\n  ── Evaluating at step {step} ──")
                X_syn, y_syn = generate_synthetic_batch(
                    model, schedule, n=1000, device=device
                )
                metrics = evaluate(X_syn, y_syn, verbose=True)
                metrics["train_loss"] = float(np.mean(losses[-EVAL_EVERY_STEPS:]))
                metrics["step"]       = step
                metrics["elapsed_s"]  = elapsed

                log_result(step, metrics, elapsed)

                if metrics["fid"] < best_fid:
                    best_fid = metrics["fid"]
                    save_checkpoint(model, optim, step, metrics, best_ckpt)
                    print(f"  ✓ New best FID: {best_fid:.4f} — checkpoint saved")

                model.train()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # ── Final evaluation ──────────────────────────────────────────
    total_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time:.0f}s  ({step} steps)")
    print(f"{'='*60}")
    print("\n  Running final evaluation on best checkpoint...")

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])

    X_syn, y_syn = generate_synthetic_batch(
        model, schedule, n=2000, device=device
    )
    final_metrics = evaluate(X_syn, y_syn, verbose=True)
    final_metrics["step"]      = step
    final_metrics["elapsed_s"] = total_time
    final_metrics["best_fid"]  = best_fid
    log_result(step, final_metrics, total_time)

    # Print summary for autoresearch agent to parse
    print(f"\n── Final Result ────────────────────────────")
    print(f"  val_fid      : {final_metrics['fid']:.4f}   ← primary metric (lower=better)")
    print(f"  tstr_acc     : {final_metrics['tstr_acc']:.4f}")
    print(f"  trtr_acc     : {final_metrics['trtr_acc']:.4f}  (upper bound)")
    print(f"  tstr_f1      : {final_metrics['tstr_f1']:.4f}")
    print(f"  psd_error    : {final_metrics['psd_error']:.4f}")
    print(f"  dtw_mean     : {final_metrics['dtw_mean']:.4f}")
    print(f"  total steps  : {step}")
    print(f"────────────────────────────────────────────")

    return final_metrics


# ─────────────────────────────────────────────────────────────────
# ❽  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT-EMG Training")
    parser.add_argument("--sample-only", action="store_true",
                        help="Load best checkpoint and generate samples only")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to generate with --sample-only")
    args = parser.parse_args()

    if args.sample_only:
        ckpt_path = CHECKPOINT_DIR / "best_model.pt"
        if not ckpt_path.exists():
            print("[ERROR] No checkpoint found. Run training first.")
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DiTEMG().to(device)
        schedule = DiffusionSchedule(T_STEPS, SCHEDULE, BETA_START, BETA_END).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from step {ckpt['step']}, "
              f"FID={ckpt['metrics']['fid']:.4f}")

        X_syn, y_syn = generate_synthetic_batch(
            model, schedule, n=args.n_samples, device=device
        )
        out = Path("samples.npz")
        np.savez(out, X=X_syn, y=y_syn)
        print(f"Saved {len(y_syn)} samples → {out}")

    else:
        train()