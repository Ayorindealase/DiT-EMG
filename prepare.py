"""
prepare.py — NinaPro DB2 EMG Dataset Preparation & Runtime Utilities
=====================================================================
Adapted from karpathy/autoresearch for synthetic sEMG generation research.

This file is FIXED — the autoresearch agent does NOT modify it.
It provides:
  1. Constants and configuration
  2. One-time data download + preprocessing (NinaPro DB2 .mat files → numpy cache)
  3. Runtime utilities: DataLoader, evaluation (FID, TSTR, DTW, PSD)

NinaPro DB2 specs:
  - 40 subjects, 50 hand gestures + rest (52 classes total)
  - 12 EMG channels (Delsys Trigno Wireless), 2000 Hz sampling rate
  - 3 exercises: E1 (17 movements), E2 (17 movements), E3 (23 movements)
  - We use Exercise 1 (17 gestures + rest = 18 classes) as the default subset

Usage:
  python prepare.py          # one-time setup: download + preprocess + cache
  python prepare.py --check  # verify cache exists and print stats
"""

import os
import sys
import json
import argparse
import hashlib
import requests
import zipfile
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# ─────────────────────────────────────────────
# FIXED CONSTANTS  (agent must NOT change these)
# ─────────────────────────────────────────────

# Dataset
DATASET_DIR     = Path("data/ninapro_db2")
CACHE_DIR       = Path("data/cache")
CACHE_TRAIN     = CACHE_DIR / "emg_train.npz"
CACHE_VAL       = CACHE_DIR / "emg_val.npz"
CACHE_STATS     = CACHE_DIR / "dataset_stats.json"

# Signal parameters (DO NOT CHANGE — these affect model I/O dimensions)
N_CHANNELS      = 12        # EMG channels from Delsys Trigno
SAMPLING_RATE   = 2000      # Hz
WINDOW_SIZE     = 200       # samples = 100ms window @ 2kHz
WINDOW_STRIDE   = 100       # samples = 50ms hop (50% overlap)
N_CLASSES       = 18        # Exercise 1: 17 gestures + rest (class 0)

# Data splits
VAL_SUBJECTS    = [1, 2]    # Subject IDs held out for validation (1-indexed)
TRAIN_SUBJECTS  = list(range(3, 41))  # Subjects 3–40 for training

# Evaluation
EVAL_SAMPLES    = 1000      # Number of real samples for FID/metric computation
CLASSIFIER_EPOCHS = 30      # Epochs for TSTR classifier
TSTR_HIDDEN     = 128       # Hidden dim of TSTR classifier

# Normalisation (computed from training set, stored in CACHE_STATS)
# Will be populated by prepare.py on first run
EMG_MEAN: Optional[np.ndarray] = None
EMG_STD:  Optional[np.ndarray] = None

# ─────────────────────────────────────────────
# DOWNLOAD UTILITIES
# ─────────────────────────────────────────────

# NinaPro DB2 is publicly available after free registration at:
#   https://ninapro.hevs.ch/instructions/DB2.html
#
# For automated download we support two paths:
#   A) Official NinaPro ZIP (requires auth token in env var NINAPRO_TOKEN)
#   B) PhysioNet mirror / local .mat files placed in DATASET_DIR manually
#   ...
# Then run python prepare.py --local

NINAPRO_BASE_URL = "https://ninapro.hevs.ch/files/DB2_Preproc/"

def check_local_files() -> List[int]:
    """Return list of subject IDs whose E1 .mat files exist locally."""
    found = []
    for sid in range(1, 41):
        mat_path = DATASET_DIR / f"S{sid}_E1_A1.mat"
        if mat_path.exists():
            found.append(sid)
    return found


def download_subject(subject_id: int, token: str = "") -> bool:
    """
    Attempt to download a single subject's Exercise 1 file.
    Returns True on success. Requires NINAPRO_TOKEN env var or passed token.
    """
    fname = f"S{subject_id}_E1_A1.mat"
    url   = NINAPRO_BASE_URL + fname
    dest  = DATASET_DIR / fname

    if dest.exists():
        print(f"  [skip] {fname} already exists")
        return True

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        if resp.status_code == 200:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  [ok] Downloaded {fname}")
            return True
        else:
            print(f"  [warn] HTTP {resp.status_code} for {fname} — place files manually")
            return False
    except Exception as e:
        print(f"  [warn] Download failed for {fname}: {e}")
        return False


# ─────────────────────────────────────────────
# MAT FILE PARSING
# ─────────────────────────────────────────────

def load_subject_mat(subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Exercise 1 .mat file for one subject.

    Returns:
        emg    : (N_samples, N_CHANNELS) float32 array, raw EMG in mV
        labels : (N_samples,)            int32 array, class 0=rest, 1–17=gestures
    """
    mat_path = DATASET_DIR / f"S{subject_id}_E1_A1.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Missing: {mat_path}\n"
            f"Download from https://ninapro.hevs.ch or set NINAPRO_TOKEN env var."
        )

    mat = sio.loadmat(str(mat_path))

    # NinaPro DB2 .mat keys: 'emg', 'restimulus', 'rerepetition', 'stimulus'
    # We use 'restimulus' (re-labeled, cleaner than raw 'stimulus')
    emg    = mat["emg"].astype(np.float32)          # (T, 12)
    labels = mat["restimulus"].astype(np.int32).squeeze()  # (T,)

    assert emg.shape[1] == N_CHANNELS, \
        f"Expected {N_CHANNELS} channels, got {emg.shape[1]}"

    return emg, labels


def segment_signal(
    emg: np.ndarray,
    labels: np.ndarray,
    window: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    min_class_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation of continuous EMG signal.

    Args:
        emg             : (T, C) continuous signal
        labels          : (T,)   sample-wise labels
        window          : window length in samples
        stride          : hop size in samples
        min_class_ratio : minimum fraction of window that must share the same label

    Returns:
        windows : (N, C, window) float32 — channel-first for Conv/Attention
        win_labels : (N,) int32
    """
    n_samples = len(labels)
    win_list, lbl_list = [], []

    for start in range(0, n_samples - window + 1, stride):
        end    = start + window
        seg    = emg[start:end]          # (window, C)
        seg_lbl = labels[start:end]

        # Majority vote for window label
        vals, counts = np.unique(seg_lbl, return_counts=True)
        majority_lbl  = vals[np.argmax(counts)]
        majority_frac = counts.max() / window

        if majority_frac < min_class_ratio:
            continue  # skip ambiguous transition windows

        win_list.append(seg.T)           # (C, window) — channel first
        lbl_list.append(majority_lbl)

    windows    = np.stack(win_list,  axis=0).astype(np.float32)  # (N, C, W)
    win_labels = np.array(lbl_list,  dtype=np.int32)

    return windows, win_labels


# ─────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────

def compute_normalisation(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from training windows.
    windows: (N, C, W)  → stats shape: (C, 1) for broadcasting
    """
    # Reshape to (N*W, C) then compute stats
    N, C, W   = windows.shape
    flat      = windows.transpose(0, 2, 1).reshape(-1, C)  # (N*W, C)
    mean      = flat.mean(axis=0, keepdims=True).T          # (C, 1)
    std       = flat.std(axis=0, keepdims=True).T + 1e-8    # (C, 1)
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise to zero mean, unit variance per channel."""
    return (windows - mean[None, :, :]) / std[None, :, :]


def denormalise(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse of normalise."""
    return windows * std[None, :, :] + mean[None, :, :]


# ─────────────────────────────────────────────
# DATASET BUILDING  (one-time)
# ─────────────────────────────────────────────

def build_dataset():
    """
    Process all available subjects → segment → normalise → save npz cache.
    Splits subjects into train / val sets by subject ID.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    available = check_local_files()
    if not available:
        token = os.environ.get("NINAPRO_TOKEN", "")
        print("No local .mat files found. Attempting download...")
        for sid in range(1, 41):
            download_subject(sid, token)
        available = check_local_files()

    if not available:
        print(
            "\n[ERROR] No NinaPro DB2 files found.\n"
            "Options:\n"
            "  1) Register at https://ninapro.hevs.ch and set NINAPRO_TOKEN=<token>\n"
            "  2) Download manually and place S{N}_E1_A1.mat files in data/ninapro_db2/\n"
            "  3) Use the synthetic dataset path — see SYNTHETIC_FALLBACK in prepare.py\n"
        )
        sys.exit(1)

    train_ids = [s for s in available if s not in VAL_SUBJECTS]
    val_ids   = [s for s in available if s in VAL_SUBJECTS]

    print(f"\nBuilding dataset: {len(train_ids)} train subjects, {len(val_ids)} val subjects")

    def process_subjects(subject_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        all_windows, all_labels = [], []
        for sid in subject_ids:
            try:
                emg, labels = load_subject_mat(sid)
                windows, win_labels = segment_signal(emg, labels)
                all_windows.append(windows)
                all_labels.append(win_labels)
                print(f"  Subject {sid:2d}: {len(win_labels):6d} windows")
            except Exception as e:
                print(f"  Subject {sid:2d}: SKIP — {e}")
        return (
            np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels,  axis=0),
        )

    print("\nProcessing training subjects...")
    X_train, y_train = process_subjects(train_ids)

    print("\nProcessing validation subjects...")
    X_val,   y_val   = process_subjects(val_ids)

    # Compute normalisation from training set only
    print("\nComputing normalisation statistics from training set...")
    mean, std = compute_normalisation(X_train)

    X_train_n = normalise(X_train, mean, std)
    X_val_n   = normalise(X_val,   mean, std)

    # Save caches
    np.savez_compressed(CACHE_TRAIN, X=X_train_n, y=y_train)
    np.savez_compressed(CACHE_VAL,   X=X_val_n,   y=y_val)

    stats = {
        "n_train":       int(len(y_train)),
        "n_val":         int(len(y_val)),
        "n_channels":    N_CHANNELS,
        "window_size":   WINDOW_SIZE,
        "n_classes":     N_CLASSES,
        "class_counts_train": {str(c): int((y_train == c).sum()) for c in range(N_CLASSES)},
        "mean":          mean.tolist(),
        "std":           std.tolist(),
        "train_subjects": train_ids,
        "val_subjects":   val_ids,
    }
    with open(CACHE_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Cache written:")
    print(f"  Train: {len(y_train):,} windows  → {CACHE_TRAIN}")
    print(f"  Val:   {len(y_val):,} windows  → {CACHE_VAL}")
    print(f"  Stats: {CACHE_STATS}")


# ─────────────────────────────────────────────
# SYNTHETIC FALLBACK (no GPU / demo mode)
# ─────────────────────────────────────────────

def build_synthetic_dataset(n_train: int = 20000, n_val: int = 2000, seed: int = 42):
    """
    Build a synthetic EMG-like dataset for pipeline testing when
    real NinaPro data is not yet available.

    Signals are band-limited Gaussian noise shaped like real sEMG:
      - Dominant power between 20–500 Hz
      - Class-specific amplitude scaling (simulates gesture intensity)
    """
    print("[SYNTHETIC FALLBACK] Generating fake EMG data for pipeline testing...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def make_fake_emg(n: int) -> Tuple[np.ndarray, np.ndarray]:
        labels  = rng.integers(0, N_CLASSES, size=n).astype(np.int32)
        signals = np.zeros((n, N_CHANNELS, WINDOW_SIZE), dtype=np.float32)

        for i, cls in enumerate(labels):
            amp   = 0.5 + cls * 0.03          # class-specific amplitude
            noise = rng.standard_normal((N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
            # Simple bandpass shaping via convolution with Hanning window
            kernel = np.hanning(20).astype(np.float32)
            kernel /= kernel.sum()
            for ch in range(N_CHANNELS):
                filtered = np.convolve(noise[ch], kernel, mode="same")
                signals[i, ch] = filtered * amp

        return signals, labels

    X_train, y_train = make_fake_emg(n_train)
    X_val,   y_val   = make_fake_emg(n_val)

    np.savez_compressed(CACHE_TRAIN, X=X_train, y=y_train)
    np.savez_compressed(CACHE_VAL,   X=X_val,   y=y_val)

    stats = {
        "n_train": n_train, "n_val": n_val,
        "n_channels": N_CHANNELS, "window_size": WINDOW_SIZE,
        "n_classes": N_CLASSES, "synthetic": True,
        "mean": np.zeros((N_CHANNELS, 1)).tolist(),
        "std":  np.ones((N_CHANNELS, 1)).tolist(),
    }
    with open(CACHE_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Synthetic cache: {n_train} train, {n_val} val samples")


# ─────────────────────────────────────────────
# RUNTIME DATALOADER
# ─────────────────────────────────────────────

class EMGDataset:
    """
    Minimal numpy-based dataset. Used by train.py at runtime.
    No torch dependency — train.py handles device transfers.
    """

    def __init__(self, split: str = "train"):
        cache = CACHE_TRAIN if split == "train" else CACHE_VAL
        if not cache.exists():
            raise RuntimeError(
                f"Cache not found: {cache}\nRun: python prepare.py"
            )
        data       = np.load(cache)
        self.X     = data["X"]   # (N, C, W) float32, normalised
        self.y     = data["y"]   # (N,) int32
        self.n     = len(self.y)

    def __len__(self) -> int:
        return self.n

    def get_batch(self, indices: np.ndarray):
        """Return (X_batch, y_batch) for given indices."""
        return self.X[indices], self.y[indices]


def get_dataloader(split: str, batch_size: int, shuffle: bool = True):
    """
    Returns a generator that yields (X_batch, y_batch) indefinitely.
    X_batch: (B, C, W) float32
    y_batch: (B,)      int32
    """
    ds  = EMGDataset(split)
    idx = np.arange(ds.n)

    def _loader():
        nonlocal idx
        while True:
            if shuffle:
                np.random.shuffle(idx)
            for start in range(0, ds.n - batch_size + 1, batch_size):
                batch_idx = idx[start : start + batch_size]
                yield ds.get_batch(batch_idx)

    return _loader()


def get_real_samples(n: int = EVAL_SAMPLES, split: str = "val", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a fixed set of real EMG windows for evaluation.
    Used by compute_fid() and tstr_accuracy().
    """
    ds  = EMGDataset(split)
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.n, size=min(n, ds.n), replace=False)
    return ds.X[idx], ds.y[idx]


# ─────────────────────────────────────────────
# EVALUATION UTILITIES
# ─────────────────────────────────────────────

# ── FID (Fréchet Inception Distance for time-series) ──────────────────────────

def _compute_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of feature vectors.
    X: (N, D) — each row is a flattened or embedded window
    """
    mu  = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return mu, cov


def _sqrtm_real(A: np.ndarray) -> np.ndarray:
    """Real matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals  = np.maximum(eigvals, 0)          # clamp numerical negatives
    sqrt_A   = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_A


def compute_fid(real: np.ndarray, fake: np.ndarray) -> float:
    """
    Fréchet Distance between real and synthetic EMG distributions.

    Uses flattened (C×W) feature vectors — no inception network needed
    for EMG signals. For publication-quality results, replace with a
    pre-trained EMG feature extractor (e.g. fine-tuned ResNet-1D).

    Args:
        real : (N, C, W) real EMG windows
        fake : (M, C, W) synthetic EMG windows
    Returns:
        fid  : float (lower is better)
    """
    def flatten(X):
        N = X.shape[0]
        return X.reshape(N, -1).astype(np.float64)

    r = flatten(real)
    f = flatten(fake)

    mu_r, cov_r = _compute_stats(r)
    mu_f, cov_f = _compute_stats(f)

    diff    = mu_r - mu_f
    cov_mix = _sqrtm_real(cov_r @ cov_f)

    # Numerical cleanup
    if np.iscomplexobj(cov_mix):
        cov_mix = cov_mix.real

    fid = float(diff @ diff + np.trace(cov_r + cov_f - 2 * cov_mix))
    return max(fid, 0.0)


# ── DTW Distance (distribution) ───────────────────────────────────────────────

def compute_mean_dtw(real: np.ndarray, fake: np.ndarray,
                     n_pairs: int = 200, seed: int = 0) -> float:
    """
    Average Dynamic Time Warping distance between random real/fake pairs.
    Uses a fast DTW approximation (Sakoe-Chiba band = 10% of signal length).

    Args:
        real, fake : (N, C, W) windows
        n_pairs    : number of random pairs to compare
    Returns:
        mean DTW distance (lower = more similar temporal dynamics)
    """
    rng  = np.random.default_rng(seed)
    ri   = rng.choice(len(real), size=n_pairs, replace=True)
    fi   = rng.choice(len(fake), size=n_pairs, replace=True)

    total = 0.0
    band  = max(1, int(0.1 * WINDOW_SIZE))  # Sakoe-Chiba band

    for r_idx, f_idx in zip(ri, fi):
        # Compare channel-averaged signal (mean across channels)
        r_sig = real[r_idx].mean(axis=0)   # (W,)
        f_sig = fake[f_idx].mean(axis=0)   # (W,)
        total += _fast_dtw(r_sig, f_sig, band)

    return total / n_pairs


def _fast_dtw(s: np.ndarray, t: np.ndarray, band: int) -> float:
    """DTW with Sakoe-Chiba band constraint."""
    n, m  = len(s), len(t)
    D     = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost   = (s[i - 1] - t[j - 1]) ** 2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return float(np.sqrt(D[n, m]))


# ── Power Spectral Density comparison ─────────────────────────────────────────

def compute_psd_error(real: np.ndarray, fake: np.ndarray) -> float:
    """
    Compare power spectral density profiles between real and synthetic.
    Returns mean absolute relative error across frequency bins and channels.

    Args:
        real, fake : (N, C, W) windows
    Returns:
        psd_error  : float (lower is better)
    """
    def mean_psd(X: np.ndarray) -> np.ndarray:
        # X: (N, C, W) → average PSD across samples → (C, W//2+1)
        psds = np.abs(np.fft.rfft(X, axis=-1)) ** 2   # (N, C, F)
        return psds.mean(axis=0)                        # (C, F)

    psd_r = mean_psd(real)
    psd_f = mean_psd(fake)

    rel_error = np.abs(psd_r - psd_f) / (psd_r + 1e-8)
    return float(rel_error.mean())


# ── TSTR (Train on Synthetic, Test on Real) ───────────────────────────────────

class _SimpleMLP:
    """
    Minimal numpy-only MLP for TSTR evaluation.
    For real experiments, replace with a PyTorch 1D-CNN classifier
    (see tstr_accuracy_torch below if torch is available).
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int, lr: float = 1e-3):
        scale      = np.sqrt(2.0 / in_dim)
        self.W1    = np.random.randn(in_dim, hidden).astype(np.float32) * scale
        self.b1    = np.zeros(hidden, dtype=np.float32)
        self.W2    = np.random.randn(hidden, n_classes).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2    = np.zeros(n_classes, dtype=np.float32)
        self.lr    = lr

    def forward(self, X: np.ndarray):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)     # ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2
        exp_z   = np.exp(self.Z2 - self.Z2.max(axis=1, keepdims=True))
        self.P  = exp_z / exp_z.sum(axis=1, keepdims=True)
        return self.P

    def loss(self, y: np.ndarray) -> float:
        n   = len(y)
        eps = 1e-9
        return -float(np.log(self.P[np.arange(n), y] + eps).mean())

    def backward(self, X: np.ndarray, y: np.ndarray):
        n       = len(y)
        dZ2     = self.P.copy()
        dZ2[np.arange(n), y] -= 1
        dZ2    /= n
        dW2     = self.A1.T @ dZ2
        db2     = dZ2.sum(axis=0)
        dA1     = dZ2 @ self.W2.T
        dZ1     = dA1 * (self.Z1 > 0)
        dW1     = X.T @ dZ1
        db1     = dZ1.sum(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)


def tstr_accuracy(
    synthetic_X: np.ndarray,
    synthetic_y: np.ndarray,
    real_X:      np.ndarray,
    real_y:      np.ndarray,
    epochs:      int  = CLASSIFIER_EPOCHS,
    batch_size:  int  = 256,
    seed:        int  = 42,
) -> Dict[str, float]:
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.

    Trains a simple MLP on synthetic data, evaluates on real held-out data.
    Also computes TRTR (Train on Real, Test on Real) as upper bound.

    Args:
        synthetic_X : (N_syn, C, W) synthetic windows
        synthetic_y : (N_syn,)      synthetic labels
        real_X      : (N_real, C, W) real windows (test set)
        real_y      : (N_real,)      real labels
        epochs      : training epochs

    Returns:
        dict with keys:
          'tstr_acc'   : TSTR accuracy (synthetic train → real test)
          'trtr_acc'   : TRTR accuracy (real train → real test) upper bound
          'tstr_f1'    : macro-F1 on TSTR
    """
    np.random.seed(seed)

    def flatten(X):
        return X.reshape(len(X), -1).astype(np.float32)

    syn_X_flat  = flatten(synthetic_X)
    real_X_flat = flatten(real_X)
    in_dim      = syn_X_flat.shape[1]

    def train_mlp(X_tr, y_tr, X_te, y_te):
        mlp = _SimpleMLP(in_dim, TSTR_HIDDEN, N_CLASSES)
        idx = np.arange(len(y_tr))

        for ep in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx) - batch_size + 1, batch_size):
                bi  = idx[start : start + batch_size]
                Xb, yb = X_tr[bi], y_tr[bi]
                mlp.forward(Xb)
                mlp.backward(Xb, yb)

        preds = mlp.predict(X_te)
        acc   = float((preds == y_te).mean())

        # Macro F1
        f1s = []
        for c in range(N_CLASSES):
            tp = int(((preds == c) & (y_te == c)).sum())
            fp = int(((preds == c) & (y_te != c)).sum())
            fn = int(((preds != c) & (y_te == c)).sum())
            prec  = tp / (tp + fp + 1e-8)
            rec   = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        f1 = float(np.mean(f1s))
        return acc, f1

    tstr_acc, tstr_f1 = train_mlp(syn_X_flat,  synthetic_y, real_X_flat, real_y)

    # TRTR — subsample real training data to match synthetic set size for fairness
    n_real_train = min(len(synthetic_y), len(real_X_flat))
    idx_r = np.random.choice(len(real_X_flat), n_real_train, replace=False)
    trtr_acc, _ = train_mlp(
        real_X_flat[idx_r], real_y[idx_r], real_X_flat, real_y
    )

    return {
        "tstr_acc":  round(tstr_acc,  4),
        "trtr_acc":  round(trtr_acc,  4),
        "tstr_f1":   round(tstr_f1,   4),
    }


# ── Full evaluation suite (called by train.py agent loop) ─────────────────────

def evaluate(
    synthetic_X: np.ndarray,
    synthetic_y: np.ndarray,
    n_real: int = EVAL_SAMPLES,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run all evaluation metrics and return a summary dict.
    This is the single function that train.py calls to assess model quality.

    Metrics returned:
      fid         — Fréchet distance (primary, LOWER is better)
      tstr_acc    — TSTR classification accuracy
      trtr_acc    — TRTR upper bound
      tstr_f1     — TSTR macro-F1
      psd_error   — PSD relative error
      dtw_mean    — mean DTW distance

    The autoresearch agent uses 'fid' as the primary optimisation target (val_fid).
    """
    real_X, real_y = get_real_samples(n=n_real, split="val")

    # Subsample synthetic to match real count for fair comparison
    n   = min(len(synthetic_y), n_real)
    idx = np.random.choice(len(synthetic_y), n, replace=False)
    syn_X = synthetic_X[idx]
    syn_y = synthetic_y[idx]

    results = {}
    results["fid"]       = compute_fid(real_X, syn_X)
    results["psd_error"] = compute_psd_error(real_X, syn_X)
    results["dtw_mean"]  = compute_mean_dtw(real_X, syn_X, n_pairs=100)

    tstr = tstr_accuracy(syn_X, syn_y, real_X, real_y)
    results.update(tstr)

    if verbose:
        print("\n── Evaluation Results ──────────────────")
        for k, v in results.items():
            print(f"  {k:<15} {v:.4f}")
        print("────────────────────────────────────────")

    return results


# ─────────────────────────────────────────────
# DATASET STATS LOADER  (for train.py)
# ─────────────────────────────────────────────

def load_stats() -> Dict:
    """Load cached dataset statistics. Raises if prepare.py hasn't been run."""
    if not CACHE_STATS.exists():
        raise RuntimeError("Run: python prepare.py  (stats cache missing)")
    with open(CACHE_STATS) as f:
        return json.load(f)


def load_normalisation() -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) arrays from cached stats, shape (C, 1) each."""
    stats = load_stats()
    mean  = np.array(stats["mean"], dtype=np.float32)
    std   = np.array(stats["std"],  dtype=np.float32)
    return mean, std


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NinaPro DB2 data preparation")
    parser.add_argument("--local",     action="store_true", help="Use local .mat files (no download)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic fallback data")
    parser.add_argument("--check",     action="store_true", help="Verify cache and print stats")
    parser.add_argument("--n-train",   type=int, default=20000, help="Synthetic fallback: train samples")
    parser.add_argument("--n-val",     type=int, default=2000,  help="Synthetic fallback: val samples")
    args = parser.parse_args()

    if args.check:
        if CACHE_TRAIN.exists() and CACHE_VAL.exists():
            stats = load_stats()
            print("\n── Dataset Cache ───────────────────────")
            print(f"  Train samples : {stats['n_train']:,}")
            print(f"  Val samples   : {stats['n_val']:,}")
            print(f"  Channels      : {stats['n_channels']}")
            print(f"  Window size   : {stats['window_size']} samples ({stats['window_size'] / SAMPLING_RATE * 1000:.0f} ms)")
            print(f"  Classes       : {stats['n_classes']}")
            if stats.get("synthetic"):
                print("  ⚠  Synthetic fallback data — NOT real NinaPro signals")
            print("────────────────────────────────────────")
        else:
            print("Cache not found. Run: python prepare.py")
        return

    if args.synthetic:
        build_synthetic_dataset(n_train=args.n_train, n_val=args.n_val)
        return

    build_dataset()


if __name__ == "__main__":
    main()