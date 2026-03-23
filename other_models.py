#!/usr/bin/env python3
"""BiLSTM, CNN-BiLSTM, and Transformer baselines for EMG to joint angle prediction.

Usage: uv run other_models.py [--model bilstm|cnnlstm|transformer] [--epochs N]
"""

import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, Subset


# Constants

SENSOR_NAMES = [
    "CMC1_F", "CMC1_A", "MCP1_F", "IP1_F",
    "MCP2_F", "MCP2-3_A", "PIP2_F",
    "MCP3_F", "PIP3_F",
    "MCP4_F", "MCP3-4_A", "PIP4_F",
    "CMC5_F", "MCP5_F", "MCP4-5_A", "PIP5_F",
    "DIP2_F", "DIP3_F", "DIP4_F", "DIP5_F",
    "PALM_ARCH", "WRIST_F",
]
ROM_DEG = {
    "CMC1_F":   (0, 50),   "CMC1_A":   (-30, 30),
    "MCP1_F":   (0, 60),   "IP1_F":    (0, 80),
    "MCP2_F":   (0, 90),   "MCP2-3_A": (-20, 20),
    "PIP2_F":   (0, 100),  "DIP2_F":   (0, 80),
    "MCP3_F":   (0, 90),   "MCP3-4_A": (-20, 20),
    "PIP3_F":   (0, 100),  "DIP3_F":   (0, 80),
    "MCP4_F":   (0, 90),   "MCP4-5_A": (-20, 20),
    "PIP4_F":   (0, 100),  "DIP4_F":   (0, 80),
    "CMC5_F":   (0, 30),
    "MCP5_F":   (0, 90),
    "PIP5_F":   (0, 100),  "DIP5_F":   (0, 80),
    "PALM_ARCH": (0, 30),
    "WRIST_F":  (-60, 60),
}
FINGER_BASES = {
    "thumb":  np.array([-0.9, 0.6, 0.0]),
    "index":  np.array([-0.4, 1.0, 0.0]),
    "middle": np.array([0.0,  1.05, 0.0]),
    "ring":   np.array([0.4,  1.0, 0.0]),
    "little": np.array([0.8,  0.9, 0.0]),
}
FINGER_LENGTHS = {
    "thumb":  (0.55, 0.40, 0.35),
    "index":  (0.65, 0.45, 0.35),
    "middle": (0.70, 0.50, 0.38),
    "ring":   (0.68, 0.48, 0.36),
    "little": (0.55, 0.38, 0.30),
}
JOINT_NAMES = [
    "Wrist",
    "Thumb MCP", "Thumb PIP", "Thumb DIP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP","Middle PIP","Middle DIP","Middle Tip",
    "Ring MCP",  "Ring PIP",  "Ring DIP",  "Ring Tip",
    "Little MCP","Little PIP","Little DIP","Little Tip",
]
FINGER_JOINT_RANGES = {
    "Wrist":  [0],
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Little": [17, 18, 19, 20],
}


# Data loading helpers

def load_ninapro_mat(path):
    m = loadmat(path)
    return (np.asarray(m["emg"],          dtype=np.float32),
            np.asarray(m["glove"],        dtype=np.float32),
            np.asarray(m["restimulus"],   dtype=np.int32).ravel(),
            np.asarray(m["rerepetition"], dtype=np.int32).ravel())


def extract_segment(glove, restim, rerep, m_id, r_id):
    mask = (restim == m_id) & (rerep == r_id)
    idx  = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No samples.")
    splits = np.where(np.diff(idx) > 1)[0]
    blocks, start = [], 0
    for s in splits:
        blocks.append(idx[start:s+1]); start = s+1
    blocks.append(idx[start:])
    return glove[max(blocks, key=len)], max(blocks, key=len)


def robust_minmax(x, qlo=0.02, qhi=0.98):
    x  = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    lo = np.quantile(x, qlo, axis=0)
    hi = np.quantile(x, qhi, axis=0)
    return lo, hi, np.where((hi - lo) < 1e-6, 1., hi - lo)


def glove_to_angles(glove_raw):
    lo, _, denom = robust_minmax(glove_raw)
    u   = np.clip((glove_raw - lo) / denom, 0., 1.)
    deg = np.zeros_like(u)
    for j, name in enumerate(SENSOR_NAMES):
        a, b = ROM_DEG.get(name, (0, 90))
        deg[:, j] = a + u[:, j] * (b - a)
    return np.deg2rad(deg)


def _R_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

def _R_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

def fk_finger(base, mcp_flex, pip_flex, dip_flex, mcp_abd, lengths):
    L1, L2, L3 = lengths
    R  = _R_z(mcp_abd) @ _R_x(mcp_flex)
    p1 = base
    p2 = p1 + R  @ np.array([0, L1, 0.])
    R2 = R  @ _R_x(pip_flex)
    p3 = p2 + R2 @ np.array([0, L2, 0.])
    R3 = R2 @ _R_x(dip_flex)
    p4 = p3 + R3 @ np.array([0, L3, 0.])
    return p1, p2, p3, p4

def hand_keypoints(angles_rad):
    T   = angles_rad.shape[0]
    out = np.zeros((T, 21, 3), dtype=np.float32)
    col = {n: i for i, n in enumerate(SENSOR_NAMES)}
    for t in range(T):
        a = angles_rad[t]; out[t, 0] = 0.
        thumb  = fk_finger(FINGER_BASES["thumb"],
                    a[col["MCP1_F"]] + 0.5*a[col["CMC1_F"]],
                    0.6*a[col["IP1_F"]], 0.4*a[col["IP1_F"]],
                    a[col["CMC1_A"]], FINGER_LENGTHS["thumb"])
        idx_f  = fk_finger(FINGER_BASES["index"],
                    a[col["MCP2_F"]], a[col["PIP2_F"]], a[col["DIP2_F"]],
                    a[col["MCP2-3_A"]], FINGER_LENGTHS["index"])
        mid    = fk_finger(FINGER_BASES["middle"],
                    a[col["MCP3_F"]], a[col["PIP3_F"]], a[col["DIP3_F"]],
                    0.5*(a[col["MCP2-3_A"]] + a[col["MCP3-4_A"]]), FINGER_LENGTHS["middle"])
        ring   = fk_finger(FINGER_BASES["ring"],
                    a[col["MCP4_F"]], a[col["PIP4_F"]], a[col["DIP4_F"]],
                    a[col["MCP3-4_A"]], FINGER_LENGTHS["ring"])
        little = fk_finger(FINGER_BASES["little"],
                    a[col["MCP5_F"]], a[col["PIP5_F"]], a[col["DIP5_F"]],
                    a[col["MCP4-5_A"]], FINGER_LENGTHS["little"])
        k = 1
        for finger in [thumb, idx_f, mid, ring, little]:
            for j in finger: out[t, k] = j; k += 1
    return out


def safe_zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    mu   = flat.mean(0); sd = flat.std(0)
    return mu.astype(np.float32), np.where(sd < 1e-6, 1., sd).astype(np.float32)

def safe_zscore_apply(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

def windowize_seq(emg, angles, win=64, stride=16):
    T = emg.shape[0]; X, Y = [], []
    for s in range(0, T - win + 1, stride):
        X.append(emg[s:s+win]); Y.append(angles[s:s+win])
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)) if X else (None, None)


def build_windows_from_mat(path, ds=3, win=64, stride=16, min_len=80):
    emg, glove, restim, rerep = load_ninapro_mat(path)
    X_all, Y_all, kept = [], [], 0
    for m_id in [int(x) for x in np.unique(restim) if int(x) > 0]:
        for r_id in [int(x) for x in np.unique(rerep)  if int(x) > 0]:
            try:   gs, idx = extract_segment(glove, restim, rerep, m_id, r_id)
            except ValueError: continue
            es  = emg[idx][::ds]; gs = gs[::ds]
            if gs.shape[0] < min_len or es.shape[0] < win: continue
            ang = glove_to_angles(gs).astype(np.float32)
            es  = np.nan_to_num(es,  nan=0., posinf=0., neginf=0.).astype(np.float32)
            ang = np.nan_to_num(ang, nan=0., posinf=0., neginf=0.).astype(np.float32)
            X, Y = windowize_seq(es, ang, win, stride)
            if X is None: continue
            X_all.append(X); Y_all.append(Y); kept += 1
    if not X_all: return None, None, 0
    return np.concatenate(X_all), np.concatenate(Y_all), kept


def build_split(base_dir, subjects, ds=3, win=64, stride=16, min_len=80):
    Xs, Ys = [], []; fcount = scount = 0
    for s in subjects:
        for mf in sorted(glob.glob(os.path.join(base_dir, s, f"{s.upper()}_A1_E*.mat"))):
            X, Y, segs = build_windows_from_mat(mf, ds, win, stride, min_len)
            fcount += 1; scount += segs
            if X is None: continue
            Xs.append(X); Ys.append(Y)
    if not Xs: raise ValueError(f"No windows for subjects {subjects}")
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    print(f"  Split | subjects={len(subjects)} files={fcount} segs={scount} windows={X.shape[0]}")
    return X, Y


def build_within_subject_split(base_dir, subjects, ds=3, win=64, stride=16,
                               min_len=80, val_rep=6):
    """Within-subject split by repetition index.

    All subjects contribute to BOTH partitions.
    Train: every repetition whose index != val_rep.
    Val:   repetition val_rep only.
    """
    Xtr, Ytr, Xva, Yva = [], [], [], []
    fcount = tr_segs = va_segs = 0
    for s in subjects:
        for mf in sorted(glob.glob(os.path.join(base_dir, s, f"{s.upper()}_A1_E*.mat"))):
            emg, glove, restim, rerep = load_ninapro_mat(mf)
            fcount += 1
            mov_ids = [int(x) for x in np.unique(restim) if int(x) > 0]
            rep_ids = [int(x) for x in np.unique(rerep)  if int(x) > 0]
            for m_id in mov_ids:
                for r_id in rep_ids:
                    try:
                        gs, idx = extract_segment(glove, restim, rerep, m_id, r_id)
                    except ValueError:
                        continue
                    es  = emg[idx][::ds]; gs = gs[::ds]
                    if gs.shape[0] < min_len or es.shape[0] < win:
                        continue
                    ang = glove_to_angles(gs).astype(np.float32)
                    es  = np.nan_to_num(es,  nan=0., posinf=0., neginf=0.).astype(np.float32)
                    ang = np.nan_to_num(ang, nan=0., posinf=0., neginf=0.).astype(np.float32)
                    X, Y = windowize_seq(es, ang, win, stride)
                    if X is None:
                        continue
                    if r_id == val_rep:
                        Xva.append(X); Yva.append(Y); va_segs += 1
                    else:
                        Xtr.append(X); Ytr.append(Y); tr_segs += 1
    if not Xtr:
        raise ValueError("No training windows — check data directory and --val-rep")
    if not Xva:
        raise ValueError(f"No validation windows for rep {val_rep} — try a different --val-rep")
    Xtr = np.concatenate(Xtr).astype(np.float32)
    Ytr = np.concatenate(Ytr).astype(np.float32)
    Xva = np.concatenate(Xva).astype(np.float32)
    Yva = np.concatenate(Yva).astype(np.float32)
    print(f"  Within-subject | subjects={len(subjects)} files={fcount} "
          f"train_segs={tr_segs} val_segs={va_segs}")
    print(f"  Train: {Xtr.shape[0]} windows  Val: {Xva.shape[0]} windows")
    return Xtr, Ytr, Xva, Yva


# Dataset

class EMGAngleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()   # (N, win, C)
        self.Y = torch.from_numpy(Y).float()   # (N, win, 22) – angles in radians
    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i].transpose(0, 1), self.Y[i]   # (C, win), (win, 22)


# ===========================================================================
# MODEL ARCHITECTURES
# ===========================================================================

# Shared building blocks

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)   # (B, n_patches, d_model)


# 1. BiLSTM

class EMG_BiLSTM(nn.Module):
    """
    Bidirectional LSTM for EMG → joint angle prediction.

    Input:  (B, C, win)
    Output: (B, 22, win)
    """
    def __init__(self, c_in: int, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.2, out_dim: int = 22):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, win) → transpose for LSTM → (B, win, C)
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, win, 2*hidden)
        out     = self.drop(self.norm(out))
        out     = self.proj(out)                  # (B, win, out_dim)
        return out.permute(0, 2, 1)               # (B, out_dim, win)


# 2. CNN-BiLSTM

class EMG_CNN_BiLSTM(nn.Module):
    """
    Conv1d feature extractor followed by a Bidirectional LSTM.

    The CNN captures local temporal patterns; the BiLSTM models long-range
    dependencies. A common hybrid architecture for biosignal regression.

    Input:  (B, C, win)
    Output: (B, 22, win)
    """
    def __init__(self, c_in: int, cnn_channels: int = 64, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.2, out_dim: int = 22):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(c_in,          cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels,  cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, win)
        feat = self.cnn(x)                        # (B, cnn_channels, win)
        out, _ = self.lstm(feat.permute(0, 2, 1)) # (B, win, 2*hidden)
        out     = self.drop(self.norm(out))
        out     = self.proj(out)                  # (B, win, out_dim)
        return out.permute(0, 2, 1)               # (B, out_dim, win)


# 3. Standard Transformer

class EMG_Transformer(nn.Module):
    """
    Standard multi-head self-attention Transformer (pre-norm).

    Uses the same patch-embedding + sinusoidal PE + linear interpolation
    upsampling strategy as PET, but replaces external attention with
    PyTorch's built-in TransformerEncoderLayer (O(n²) self-attention).

    Comparison with PET reveals the benefit of external attention.

    Input:  (B, C, win)
    Output: (B, 22, win)
    """
    def __init__(self, c_in: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, patch_size: int = 8, ffn_mult: int = 2,
                 dropout: float = 0.2, out_dim: int = 22, win: int = 64):
        super().__init__()
        self.win        = win
        self.patch_embed = PatchEmbedding(c_in, d_model, patch_size)
        self.pos_enc     = SinusoidalPE(d_model, max_len=win // patch_size + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for better gradient flow
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop     = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, out_dim)
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, win)
        z = self.patch_embed(x)                              # (B, P, d_model)
        z = self.pos_enc(z)
        z = self.drop(z)
        z = self.encoder(z)                                  # (B, P, d_model)
        # upsample P → win
        z = F.interpolate(z.permute(0, 2, 1), size=self.win,
                          mode="linear", align_corners=False) # (B, d_model, win)
        z   = z.permute(0, 2, 1)                             # (B, win, d_model)
        out = self.out_head(z)                               # (B, win, out_dim)
        return out.permute(0, 2, 1)                          # (B, out_dim, win)


# Model registry

MODEL_REGISTRY = {
    "bilstm":         EMG_BiLSTM,
    "bilstm_small":   EMG_BiLSTM,
    "cnnlstm":        EMG_CNN_BiLSTM,
    "cnnlstm_small":  EMG_CNN_BiLSTM,
    "transformer":    EMG_Transformer,
}

def build_model(name: str, c_in: int, args, device) -> nn.Module:
    if name == "bilstm":
        m = EMG_BiLSTM(c_in=c_in, hidden=args.hidden, n_layers=args.n_layers,
                       dropout=args.dropout, out_dim=22)
    elif name == "bilstm_small":
        m = EMG_BiLSTM(c_in=c_in, hidden=64, n_layers=1,
                       dropout=args.dropout, out_dim=22)
    elif name == "cnnlstm":
        m = EMG_CNN_BiLSTM(c_in=c_in, cnn_channels=args.cnn_channels,
                           hidden=args.hidden, n_layers=args.n_layers,
                           dropout=args.dropout, out_dim=22)
    elif name == "cnnlstm_small":
        m = EMG_CNN_BiLSTM(c_in=c_in, cnn_channels=32, hidden=64, n_layers=1,
                           dropout=args.dropout, out_dim=22)
    elif name == "transformer":
        assert args.win % args.patch_size == 0, \
            f"--win ({args.win}) must be divisible by --patch-size ({args.patch_size})"
        m = EMG_Transformer(c_in=c_in, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.transformer_layers, patch_size=args.patch_size,
                            ffn_mult=args.ffn_mult, dropout=args.dropout,
                            out_dim=22, win=args.win)
    else:
        raise ValueError(f"Unknown model: {name!r}")
    return m.to(device)


# ===========================================================================
# TRAINING UTILITIES
# ===========================================================================

def pred_to_angles(pred):
    """(B, 22, win) → (B, win, 22)"""
    return pred.permute(0, 2, 1).contiguous()

def angles_batch_to_kpts(angles_np):
    """(B, win, 22) numpy → (B, win, 21, 3) numpy via forward kinematics."""
    B, W, _ = angles_np.shape
    flat = angles_np.reshape(B * W, 22)
    return hand_keypoints(flat).reshape(B, W, 21, 3)

def angle_mae(pred_a, tgt_a):
    return (pred_a - tgt_a).abs().mean()


@torch.no_grad()
def eval_epoch(model, dl, loss_fn, device):
    model.eval()
    tot_l = tot_e = n = 0
    for x, y in dl:
        x, y   = x.to(device), y.to(device)
        pred_a = pred_to_angles(model(x))
        tot_l += loss_fn(pred_a, y).item() * x.size(0)
        tot_e += angle_mae(pred_a, y).item() * x.size(0)
        n     += x.size(0)
    return tot_l / n, tot_e / n


def train_one_epoch(model, dl, loss_fn, opt, device):
    model.train()
    gnorms = []
    for x, y in dl:
        x, y   = x.to(device), y.to(device)
        pred_a = pred_to_angles(model(x))
        loss   = loss_fn(pred_a, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()
    return gnorms


@torch.no_grad()
def per_joint_mpjpe(model, dl, device):
    model.eval()
    acc = np.zeros(21); n = 0
    for x, y in dl:
        pred_a  = pred_to_angles(model(x.to(device))).cpu().numpy()
        pred_kp = angles_batch_to_kpts(pred_a)
        gt_kp   = angles_batch_to_kpts(y.numpy())
        acc    += np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=(0, 1)); n += 1
    return acc / n


@torch.no_grad()
def per_window_mpjpe(model, dl, device):
    model.eval()
    errs = []
    for x, y in dl:
        pred_a  = pred_to_angles(model(x.to(device))).cpu().numpy()
        pred_kp = angles_batch_to_kpts(pred_a)
        gt_kp   = angles_batch_to_kpts(y.numpy())
        errs.append(np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=(1, 2)))
    return np.concatenate(errs)


@torch.no_grad()
def sample_predictions(model, ds, device, n=3):
    """Return (gt_kpts, pred_kpts) via FK for overlay plots."""
    model.eval()
    idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)
    gts, preds = [], []
    for i in idxs:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(hand_keypoints(y.numpy()))
        preds.append(hand_keypoints(pred_a))
    return gts, preds


@torch.no_grad()
def sample_angle_predictions(model, ds, device, n=3):
    """Return (gt_angles, pred_angles) as (win, 22) arrays."""
    model.eval()
    idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)
    gts, preds = [], []
    for i in idxs:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(y.numpy()); preds.append(pred_a)
    return gts, preds


# ===========================================================================
# PLOTTING
# ===========================================================================

STYLE = dict(dpi=150, bbox_inches="tight")

def _finger_colors():
    cmap = plt.cm.tab10(np.linspace(0, 1, 6))
    return [cmap[fi] for fi, (_, idxs) in enumerate(FINGER_JOINT_RANGES.items()) for _ in idxs]


def plot_training_curves(tl, vl, te, ve, model_name, save_dir):
    epochs = range(1, len(tl) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, tl, "o-", ms=4, label="Train")
    axes[0].plot(epochs, vl, "s-", ms=4, label="Val")
    axes[0].set(xlabel="Epoch", ylabel="SmoothL1 Loss",
                title=f"{model_name} – Angle Regression Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, te, "o-", ms=4, label="Train")
    axes[1].plot(epochs, ve, "s-", ms=4, label="Val")
    axes[1].set(xlabel="Epoch", ylabel="Mean Angle Error (rad)",
                title=f"{model_name} – Mean Angle Error")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_per_joint_mpjpe(joint_errs, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(JOINT_NAMES, joint_errs, color=_finger_colors(), edgecolor="black", linewidth=0.4)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="MPJPE", title=f"{model_name} – Per-Joint MPJPE (Validation)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    cmap = plt.cm.tab10(np.linspace(0, 1, 6))
    ax.legend(handles=[Patch(color=cmap[i], label=n) for i, n in enumerate(FINGER_JOINT_RANGES)],
              fontsize=8)
    plt.tight_layout()
    out = os.path.join(save_dir, "per_joint_mpjpe.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_per_finger_mpjpe(joint_errs, model_name, save_dir):
    names = list(FINGER_JOINT_RANGES.keys())
    vals  = [np.mean(joint_errs[idxs]) for idxs in FINGER_JOINT_RANGES.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=plt.cm.Set2(np.linspace(0, 1, len(names))), edgecolor="black")
    ax.set(ylabel="MPJPE", title=f"{model_name} – MPJPE by Finger")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "per_finger_mpjpe.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_error_distribution(pw_errs, model_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(pw_errs, bins=60, color="steelblue", edgecolor="white", lw=0.3)
    ax.axvline(np.median(pw_errs), color="red",    ls="--", label=f"Median={np.median(pw_errs):.4f}")
    ax.axvline(np.mean(pw_errs),   color="orange", ls="--", label=f"Mean={np.mean(pw_errs):.4f}")
    ax.set(xlabel="MPJPE", ylabel="Count",
           title=f"{model_name} – Per-Window MPJPE Distribution")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    se = np.sort(pw_errs); cdf = np.arange(1, len(se) + 1) / len(se)
    ax.plot(se, cdf, lw=1.5, color="steelblue")
    for pct, col in [(0.5, "red"), (0.9, "orange"), (0.95, "purple")]:
        ax.axhline(pct, color=col, ls="--", alpha=0.6, label=f"{int(pct*100)}th %ile")
    ax.set(xlabel="MPJPE", ylabel="CDF", title=f"{model_name} – CDF of Per-Window MPJPE")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "error_distribution.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_joint_error_heatmap(joint_errs, model_name, save_dir):
    mat = np.full((6, 4), np.nan); mat[0, 0] = joint_errs[0]; k = 1
    for fi in range(1, 6):
        for jj in range(4): mat[fi, jj] = joint_errs[k]; k += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Base/MCP", "PIP", "DIP", "Tip"])
    ax.set_yticks(range(6)); ax.set_yticklabels(["Wrist","Thumb","Index","Middle","Ring","Little"])
    ax.set_title(f"{model_name} – Per-Joint MPJPE Heatmap")
    plt.colorbar(im, ax=ax, label="MPJPE")
    vmax = np.nanmax(mat)
    for fi in range(6):
        for jj in range(4):
            v = mat[fi, jj]
            if not np.isnan(v):
                ax.text(jj, fi, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * vmax else "black")
    plt.tight_layout()
    out = os.path.join(save_dir, "joint_error_heatmap.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_prediction_overlays(gts, preds, model_name, save_dir, joint_idx=5):
    n = len(gts)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n))
    if n == 1: axes = axes[np.newaxis, :]
    for row, (gt, pred) in enumerate(zip(gts, preds)):
        t = np.arange(gt.shape[0])
        for col, cl in enumerate(["X", "Y", "Z"]):
            ax = axes[row, col]
            ax.plot(t, gt[:,   joint_idx, col], lw=1.5, label="GT",   color="royalblue")
            ax.plot(t, pred[:, joint_idx, col], lw=1.5, label="Pred", color="tomato", ls="--")
            ax.set(xlabel="Frame", ylabel=cl,
                   title=f"Sample {row+1} – {JOINT_NAMES[joint_idx]} {cl}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle(f"{model_name} – Prediction Overlay: {JOINT_NAMES[joint_idx]}", fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, f"prediction_overlay_joint{joint_idx}.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_angle_overlays(gts, preds, model_name, save_dir,
                        sensor_indices=None):
    if sensor_indices is None:
        sensor_indices = [0, 2, 4, 7, 11, 21]
    n = len(gts); m = len(sensor_indices)
    fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n))
    if n == 1: axes = axes[np.newaxis, :]
    if m == 1: axes = axes[:, np.newaxis]
    for row, (gt, pred) in enumerate(zip(gts, preds)):
        t = np.arange(gt.shape[0])
        for col, si in enumerate(sensor_indices):
            ax = axes[row, col]
            ax.plot(t, np.rad2deg(gt[:,   si]), lw=1.5, label="GT",   color="royalblue")
            ax.plot(t, np.rad2deg(pred[:, si]), lw=1.5, label="Pred", color="tomato", ls="--")
            ax.set(xlabel="Frame", ylabel="Angle (°)",
                   title=f"S{row+1} – {SENSOR_NAMES[si]}")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
    plt.suptitle(f"{model_name} – Predicted vs GT Joint Angles", fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, "angle_overlays.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_per_sensor_angle_error(model, dl, device, model_name, save_dir):
    """Bar chart of mean absolute angle error (°) per CyberGlove sensor."""
    model.eval()
    acc = np.zeros(22); n = 0
    with torch.no_grad():
        for x, y in dl:
            pred_a = pred_to_angles(model(x.to(device))).cpu().numpy()
            gt_a   = y.numpy()
            acc   += np.abs(pred_a - gt_a).mean(axis=(0, 1)); n += 1
    sensor_mae_deg = np.rad2deg(acc / n)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 22))
    ax.bar(SENSOR_NAMES, sensor_mae_deg, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticklabels(SENSOR_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="MAE (°)", title=f"{model_name} – Per-Sensor Angle MAE (Validation)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "per_sensor_angle_mae.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_grad_norm(gnorms, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(gnorms, lw=0.6, color="darkorange", alpha=0.7)
    if len(gnorms) > 20:
        w = max(1, len(gnorms) // 50)
        ax.plot(np.arange(w - 1, len(gnorms)),
                np.convolve(gnorms, np.ones(w) / w, "valid"),
                lw=1.5, color="red", label=f"Avg (w={w})")
        ax.legend()
    ax.set(xlabel="Batch", ylabel="Gradient norm",
           title=f"{model_name} – Gradient Norm During Training")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "grad_norm.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_data_ablation(fracs, vals, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([f * 100 for f in fracs], vals, "o-", ms=6, lw=1.5, color="steelblue")
    for f, v in zip(fracs, vals):
        ax.annotate(f"{v:.4f}", xy=(f * 100, v), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="Training Data (%)", ylabel="Val MPJPE",
           title=f"{model_name} – Data Ablation: Val MPJPE vs Dataset Size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "data_ablation.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_model_comparison(results: dict, save_dir: str):
    """
    Bar chart comparing best val MPJPE across all trained models.
    results = {model_name: best_val_mpjpe, ...}
    """
    names = list(results.keys())
    vals  = [results[n] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Val MPJPE (lower is better)",
           title="Model Comparison – Validation MPJPE")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


# ===========================================================================
# TRAINING LOOP (shared by all models)
# ===========================================================================

def train_model(model, model_name, train_dl, val_dl, train_ds, val_ds,
                loss_fn, args, device):
    """Full training + evaluation + plot generation for one model."""
    save_dir = os.path.join(args.plot_root, f"{model_name}_plots")
    os.makedirs(save_dir, exist_ok=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=1e-5)

    tl_hist, vl_hist, te_hist, ve_hist = [], [], [], []
    gnorms_all = []
    best_mpjpe = float("inf")
    ckpt_path  = os.path.join(args.plot_root, f"best_{model_name}.pt")

    print(f"\n=== Training {model_name.upper()} ({args.epochs} epochs) ===")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_l = ep_e = n = 0
        for x, y in train_dl:
            x, y   = x.to(device), y.to(device)
            pred_a = pred_to_angles(model(x))
            loss   = loss_fn(pred_a, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorms_all.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            opt.step()
            bs = x.size(0)
            ep_l += loss.item() * bs
            ep_e += angle_mae(pred_a, y).item() * bs
            n    += bs
        tl_hist.append(ep_l / n); te_hist.append(ep_e / n)

        vl, ve = eval_epoch(model, val_dl, loss_fn, device)
        vl_hist.append(vl); ve_hist.append(ve)
        val_mpjpe = float(np.mean(per_window_mpjpe(model, val_dl, device)))
        sched.step()

        improved = ""
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mpjpe": val_mpjpe}, ckpt_path)
            improved = " ✓"
        print(f"  epoch {epoch:02d} | tr loss {tl_hist[-1]:.4f} | "
              f"vl loss {vl:.4f} | val MAE {ve:.4f} | val MPJPE {val_mpjpe:.4f}{improved}")

    print(f"\n  Best val MPJPE: {best_mpjpe:.4f}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- plots ---
    print(f"  Generating plots → {save_dir}/")
    plot_training_curves(tl_hist, vl_hist, te_hist, ve_hist, model_name, save_dir)

    joint_errs = per_joint_mpjpe(model, val_dl, device)
    plot_per_joint_mpjpe(joint_errs, model_name, save_dir)
    plot_per_finger_mpjpe(joint_errs, model_name, save_dir)

    pw_errs = per_window_mpjpe(model, val_dl, device)
    plot_error_distribution(pw_errs, model_name, save_dir)
    plot_joint_error_heatmap(joint_errs, model_name, save_dir)

    gts, preds = sample_predictions(model, val_ds, device)
    for ji in args.overlay_joints:
        plot_prediction_overlays(gts, preds, model_name, save_dir, joint_idx=ji)

    ang_gts, ang_preds = sample_angle_predictions(model, val_ds, device)
    plot_angle_overlays(ang_gts, ang_preds, model_name, save_dir)

    plot_per_sensor_angle_error(model, val_dl, device, model_name, save_dir)
    plot_grad_norm(gnorms_all, model_name, save_dir)

    # data ablation
    print(f"  Running data ablation ({len(args.ablation_fractions)} fractions) …")
    abl_errs = []
    pin = torch.cuda.is_available()
    for frac in args.ablation_fractions:
        n_use  = max(args.batch_size, int(frac * len(train_ds)))
        idx    = np.random.choice(len(train_ds), n_use, replace=False)
        sub_dl = DataLoader(Subset(train_ds, idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=0, pin_memory=pin)
        m_abl  = build_model(model_name, train_ds[0][0].shape[0], args, device)
        o_abl  = torch.optim.AdamW(m_abl.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        best_abl = float("inf")
        for _ in range(args.ablation_epochs):
            train_one_epoch(m_abl, sub_dl, loss_fn, o_abl, device)
            _, ve = eval_epoch(m_abl, val_dl, loss_fn, device)
            best_abl = min(best_abl, ve)
        abl_errs.append(best_abl)
        print(f"    frac={frac*100:.0f}%  n={n_use}  best_mae={best_abl:.4f}")
    plot_data_ablation(args.ablation_fractions, abl_errs, model_name, save_dir)

    best_val_mpjpe = float(np.mean(per_window_mpjpe(model, val_dl, device)))
    return best_val_mpjpe


# ===========================================================================
# ARGPARSE + MAIN
# ===========================================================================

def get_device():
    if torch.cuda.is_available():   return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="BiLSTM / CNN-BiLSTM / Transformer: EMG → Joint Angles")

    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ninapro_DB1")
    desktop_data_dir = os.path.expanduser("~/Desktop/Ninapro_DB1")
    resolved_data_dir = default_data_dir if os.path.isdir(default_data_dir) else desktop_data_dir

    # which model(s) to train
    p.add_argument("--model", type=str, default="all",
                   choices=["bilstm", "bilstm_small", "cnnlstm", "cnnlstm_small", "transformer", "all"],
                   help="Which model to train (default: all). bilstm_small / cnnlstm_small fit nRF arena.")
    # data
    p.add_argument("--data-dir",     type=str, default=resolved_data_dir)
    p.add_argument("--val-subjects", type=str, nargs="+", default=["s1"])
    p.add_argument("--cv-mode",      type=str, default="cross_subject",
                   choices=["cross_subject", "within_subject"],
                   help="cross_subject: hold out val-subjects; "
                        "within_subject: hold out val-rep from all subjects")
    p.add_argument("--val-rep",      type=int, default=6,
                   help="Repetition index held out as validation (within_subject mode only)")
    p.add_argument("--win",          type=int, default=64)
    p.add_argument("--stride",       type=int, default=16)
    p.add_argument("--ds",           type=int, default=3)
    p.add_argument("--min-len",      type=int, default=80)
    # training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    # BiLSTM / CNN-BiLSTM shared
    p.add_argument("--hidden",       type=int, default=128, help="LSTM hidden size")
    p.add_argument("--n-layers",     type=int, default=2,   help="LSTM layers")
    p.add_argument("--cnn-channels", type=int, default=64,  help="CNN-BiLSTM conv channels")
    # Transformer-specific
    p.add_argument("--d-model",           type=int, default=128, help="Transformer embedding dim")
    p.add_argument("--n-heads",           type=int, default=4,   help="Attention heads")
    p.add_argument("--transformer-layers",type=int, default=4,   help="Transformer encoder layers")
    p.add_argument("--patch-size",        type=int, default=8,   help="Patch size for Transformer")
    p.add_argument("--ffn-mult",          type=int, default=2,   help="FFN expansion factor")
    # plots / ablation
    p.add_argument("--plot-root",         type=str, default=".",
                   help="Root directory for per-model plot subdirs")
    p.add_argument("--overlay-joints",    type=int, nargs="+", default=[0, 5, 9])
    p.add_argument("--ablation-fractions",type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--ablation-epochs",   type=int, default=5)

    args, _ = p.parse_known_args()
    return args


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.plot_root, exist_ok=True)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir!r}\n"
            "Pass --data-dir /path/to/Ninapro_DB1"
        )

    # --- subjects ---
    all_subs = sorted([d for d in os.listdir(args.data_dir)
                       if d.startswith("s") and os.path.isdir(os.path.join(args.data_dir, d))])

    # --- data (built once, shared across all models) ---
    if args.cv_mode == "within_subject":
        print(f"\nCV mode: within_subject (val rep = {args.val_rep})")
        print(f"All subjects: {all_subs}")
        print("\nBuilding within-subject split …")
        Xtr, Ytr, Xva, Yva = build_within_subject_split(
            args.data_dir, all_subs,
            args.ds, args.win, args.stride, args.min_len, args.val_rep,
        )
    else:
        val_subs   = args.val_subjects
        train_subs = [s for s in all_subs if s not in val_subs]
        print(f"\nCV mode: cross_subject")
        print(f"Train: {train_subs}\nVal:   {val_subs}")
        print("\nBuilding training windows …")
        Xtr, Ytr = build_split(args.data_dir, train_subs, args.ds, args.win, args.stride, args.min_len)
        print("Building validation windows …")
        Xva, Yva = build_split(args.data_dir, val_subs,   args.ds, args.win, args.stride, args.min_len)

    mu, sd = safe_zscore_fit(Xtr)
    Xtr    = safe_zscore_apply(Xtr, mu, sd)
    Xva    = safe_zscore_apply(Xva, mu, sd)
    print(f"Xtr {Xtr.shape}  Ytr {Ytr.shape}")
    print(f"Xva {Xva.shape}  Yva {Yva.shape}")

    train_ds = EMGAngleDataset(Xtr, Ytr)
    val_ds   = EMGAngleDataset(Xva, Yva)
    pin      = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    c_in    = Xtr.shape[-1]
    loss_fn = nn.SmoothL1Loss()

    models_to_train = (list(MODEL_REGISTRY.keys())
                       if args.model == "all" else [args.model])

    comparison = {}
    for name in models_to_train:
        model = build_model(name, c_in, args, device)
        best_mpjpe = train_model(model, name, train_dl, val_dl,
                                 train_ds, val_ds, loss_fn, args, device)
        comparison[name] = best_mpjpe

    # --- cross-model comparison plot ---
    if len(comparison) > 1:
        print(f"\nGenerating model comparison plot → {args.plot_root}/")
        plot_model_comparison(comparison, args.plot_root)

    print("\n=== Summary ===")
    for name, mpjpe in comparison.items():
        print(f"  {name:<14} val MPJPE = {mpjpe:.4f}")


if __name__ == "__main__":
    main()
