#!/usr/bin/env python3
"""PET (Parallel Efficient Transformer) for EMG to joint angle prediction.

Re-implementation of Lin et al. 2025 (doi:10.1038/s41598-025-16268-y).
Uses external attention (O(n·s) vs O(n²)) and parallel encoder branches.
Same CLI as main.py; extra flags: --d-model, --n-heads, --n-branches, --n-layers, --mem-size, --patch-size.
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



# Constants (identical to TCN script)

SENSOR_NAMES = [
    "CMC1_F", "CMC1_A", "MCP1_F", "IP1_F",
    "MCP2_F", "MCP2-3_A", "PIP2_F", "MCP3_F", "PIP3_F",
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
    "PALM_ARCH":(0, 30),
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
    "Thumb MCP","Thumb PIP","Thumb DIP","Thumb Tip",
    "Index MCP","Index PIP","Index DIP","Index Tip",
    "Middle MCP","Middle PIP","Middle DIP","Middle Tip",
    "Ring MCP","Ring PIP","Ring DIP","Ring Tip",
    "Little MCP","Little PIP","Little DIP","Little Tip",
]
FINGER_JOINT_RANGES = {
    "Wrist":  [0],
    "Thumb":  [1,2,3,4],
    "Index":  [5,6,7,8],
    "Middle": [9,10,11,12],
    "Ring":   [13,14,15,16],
    "Little": [17,18,19,20],
}


# Data loading helpers (unchanged from TCN script)

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
    block = max(blocks, key=len)
    return glove[block], block


def robust_minmax(x, qlo=0.02, qhi=0.98):
    x   = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    lo  = np.quantile(x, qlo, axis=0)
    hi  = np.quantile(x, qhi, axis=0)
    return lo, hi, np.where((hi-lo)<1e-6, 1., hi-lo)


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
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def _R_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])

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
        a = angles_rad[t]; out[t,0] = 0.
        thumb = fk_finger(FINGER_BASES["thumb"],
            a[col["MCP1_F"]]+0.5*a[col["CMC1_F"]],
            0.6*a[col["IP1_F"]], 0.4*a[col["IP1_F"]],
            a[col["CMC1_A"]], FINGER_LENGTHS["thumb"])
        idx_f = fk_finger(FINGER_BASES["index"],
            a[col["MCP2_F"]], a[col["PIP2_F"]], a[col["DIP2_F"]],
            a[col["MCP2-3_A"]], FINGER_LENGTHS["index"])
        mid   = fk_finger(FINGER_BASES["middle"],
            a[col["MCP3_F"]], a[col["PIP3_F"]], a[col["DIP3_F"]],
            0.5*(a[col["MCP2-3_A"]]+a[col["MCP3-4_A"]]), FINGER_LENGTHS["middle"])
        ring  = fk_finger(FINGER_BASES["ring"],
            a[col["MCP4_F"]], a[col["PIP4_F"]], a[col["DIP4_F"]],
            a[col["MCP3-4_A"]], FINGER_LENGTHS["ring"])
        little= fk_finger(FINGER_BASES["little"],
            a[col["MCP5_F"]], a[col["PIP5_F"]], a[col["DIP5_F"]],
            a[col["MCP4-5_A"]], FINGER_LENGTHS["little"])
        k = 1
        for finger in [thumb, idx_f, mid, ring, little]:
            for j in finger: out[t,k] = j; k+=1
    return out


def safe_zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    mu   = flat.mean(0); sd = flat.std(0)
    return mu.astype(np.float32), np.where(sd<1e-6, 1., sd).astype(np.float32)

def safe_zscore_apply(X, mu, sd):
    return ((X-mu)/sd).astype(np.float32)

def windowize_seq(emg, kpts, win=64, stride=16):
    T = emg.shape[0]; X, Y = [], []
    for s in range(0, T-win+1, stride):
        X.append(emg[s:s+win]); Y.append(kpts[s:s+win])
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)) if X else (None, None)


def build_windows_from_mat(path, ds=3, win=64, stride=16, min_len=80):
    emg, glove, restim, rerep = load_ninapro_mat(path)
    X_all, Y_all, kept = [], [], 0
    for m_id in [int(x) for x in np.unique(restim) if int(x)>0]:
        for r_id in [int(x) for x in np.unique(rerep)  if int(x)>0]:
            try:   gs, idx = extract_segment(glove, restim, rerep, m_id, r_id)
            except ValueError: continue
            es  = emg[idx][::ds]; gs = gs[::ds]
            if gs.shape[0]<min_len or es.shape[0]<win: continue
            ang = glove_to_angles(gs).astype(np.float32)          # (T, 22) angles in radians
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

class Emg2KptsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()   # (N, win, C)
        self.Y = torch.from_numpy(Y).float()   # (N, win, 22) – joint angles in radians
    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i].transpose(0,1), self.Y[i]   # (C,win), (win,22)


# ===========================================================================
# PET MODEL
# ===========================================================================

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding for sequence position."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class ExternalAttention(nn.Module):
    """
    External Attention (Guo et al. 2021, "Beyond Self-attention").

    Complexity: O(n·s) instead of O(n²).

    Two shared external memory matrices M_k ∈ ℝ^{s×d_head} and
    M_v ∈ ℝ^{s×d_head} replace the input-derived K and V.

    Multi-head variant: each head has its own M_k, M_v pair.

    Forward
    -------
    x : (B, T, d_model)
    returns (B, T, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, mem_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.mem_size = mem_size

        # Project input to queries
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # External memory parameters: (n_heads, mem_size, d_head)
        # M_k: used to compute attention weights
        # M_v: used to aggregate values
        self.M_k = nn.Parameter(torch.empty(n_heads, mem_size, self.d_head))
        self.M_v = nn.Parameter(torch.empty(n_heads, mem_size, self.d_head))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Queries / output projection: xavier
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # M_k: kaiming (like a linear weight matrix)
        nn.init.kaiming_uniform_(self.M_k, a=math.sqrt(5))
        # M_v: near-zero so early attention output is stable
        nn.init.uniform_(self.M_v, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, S, D = self.n_heads, self.mem_size, self.d_head

        # --- compute queries ---
        Q = self.q_proj(x)                            # (B, T, d_model)
        Q = Q.view(B, T, H, D).permute(0, 2, 1, 3)   # (B, H, T, D)

        # --- attention with external memory ---
        # attn_w: (B, H, T, S)
        #   Each query token attends to s memory slots instead of all n tokens.
        scale    = D ** -0.5
        attn_w   = torch.matmul(Q, self.M_k.transpose(-1, -2)) * scale  # (B,H,T,S)
        attn_w   = F.softmax(attn_w, dim=-1)

        # Normalise along the memory axis as well (double-normalisation trick
        # from the original paper — helps avoid attention collapse)
        attn_w   = attn_w / (attn_w.sum(dim=-2, keepdim=True) + 1e-6)
        attn_w   = self.drop(attn_w)

        # --- aggregate values from memory ---
        out = torch.matmul(attn_w, self.M_v)          # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).contiguous()    # (B, T, H, D)
        out = out.view(B, T, H * D)                   # (B, T, d_model)
        return self.out_proj(out)


class PETEncoderLayer(nn.Module):
    """
    One layer of a PET encoder branch:
      ExternalAttention → Add & Norm → FFN → Add & Norm
    """
    def __init__(self, d_model: int, n_heads: int, mem_size: int,
                 ffn_mult: int = 2, dropout: float = 0.1, ffn_act: str = "gelu"):
        super().__init__()
        self.attn   = ExternalAttention(d_model, n_heads, mem_size, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        act = nn.ReLU() if ffn_act == "relu" else nn.GELU()
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm2  = nn.LayerNorm(d_model)
        self._init_ffn()

    def _init_ffn(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PETBranch(nn.Module):
    """
    A single parallel encoder branch: stack of PETEncoderLayers.
    """
    def __init__(self, d_model, n_heads, mem_size, n_layers, ffn_mult, dropout, ffn_act="gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            PETEncoderLayer(d_model, n_heads, mem_size, ffn_mult, dropout, ffn_act)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Splits the EMG window into non-overlapping patches along the time axis
    and projects each patch to d_model.

    Input:  (B, C, win)     – C EMG channels, win time steps
    Output: (B, n_patches, d_model)
    """
    def __init__(self, c_in: int, d_model: int, patch_size: int):
        super().__init__()
        # Conv1d with kernel=stride=patch_size is equivalent to non-overlapping patches
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, C, win) → (B, d_model, n_patches) → (B, n_patches, d_model)
        return self.proj(x).transpose(1, 2)


class EMG_PET(nn.Module):
    """
    PET – Parallel Efficient Transformer for EMG → Hand Keypoints.

    Architecture
    ------------
    1. Patch embedding  : (B, C, win) → (B, P, d_model)  where P = win // patch_size
    2. Positional enc   : add sinusoidal PE
    3. Parallel branches: N independent stacks of PETEncoderLayers
    4. Merge            : concat along feature dim → project back to d_model
    5. Temporal upsample: linear interp from P back to win
    6. Output head      : (B, win, d_model) → (B, 63, win)

    Produces the same output shape as EMG_TCN_SEQ so training loop is identical.
    """
    def __init__(
        self,
        c_in       : int,
        d_model    : int = 128,
        n_heads    : int = 4,
        n_branches : int = 3,
        n_layers   : int = 2,
        mem_size   : int = 64,
        patch_size : int = 8,
        ffn_mult   : int = 2,
        dropout    : float = 0.1,
        out_dim    : int = 63,
        win        : int = 64,
        ffn_act    : str = "gelu",
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.win         = win
        self.n_patches   = win // patch_size

        # --- patch embedding ---
        self.patch_embed = PatchEmbedding(c_in, d_model, patch_size)

        # --- positional encoding ---
        self.pos_enc = SinusoidalPE(d_model, max_len=self.n_patches + 1)

        # --- parallel branches ---
        self.branches = nn.ModuleList([
            PETBranch(d_model, n_heads, mem_size, n_layers, ffn_mult, dropout, ffn_act)
            for _ in range(n_branches)
        ])

        # --- merge projection: concat(n_branches × d_model) → d_model ---
        merge_act = nn.ReLU() if ffn_act == "relu" else nn.GELU()
        self.merge = nn.Sequential(
            nn.Linear(d_model * n_branches, d_model),
            merge_act,
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.merge[0].weight)

        # --- output projection: d_model → out_dim (63) ---
        self.out_head = nn.Linear(d_model, out_dim)
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, win)
        returns (B, out_dim, win)  – same shape as EMG_TCN_SEQ output
        """
        B = x.size(0)

        # 1. Patch embed + PE
        z = self.patch_embed(x)           # (B, P, d_model)
        z = self.pos_enc(z)
        z = self.drop(z)

        # 2. Run all branches on the same input, collect outputs
        branch_outs = [branch(z) for branch in self.branches]  # each (B, P, d_model)

        # 3. Merge: concatenate → project
        merged = torch.cat(branch_outs, dim=-1)   # (B, P, d_model * n_branches)
        merged = self.merge(merged)               # (B, P, d_model)

        # 4. Temporal upsample: P → win via linear interpolation
        #    F.interpolate expects (B, C, L); we have (B, L, C)
        merged = merged.permute(0, 2, 1)                     # (B, d_model, P)
        merged = F.interpolate(merged, size=self.win,
                               mode="linear", align_corners=False)  # (B, d_model, win)
        merged = merged.permute(0, 2, 1)                     # (B, win, d_model)

        # 5. Output head
        out = self.out_head(merged)               # (B, win, out_dim=22)
        return out.permute(0, 2, 1)               # (B, 22, win)  ← same shape convention as TCN


# Training utilities (identical API to TCN script)

def pred_to_angles(pred):
    """(B, 22, win) → (B, win, 22) – model output to angle sequence."""
    return pred.permute(0, 2, 1).contiguous()

def angles_batch_to_kpts(angles_np):
    """(B, win, 22) numpy → (B, win, 21, 3) numpy via forward kinematics."""
    B, W, _ = angles_np.shape
    flat = angles_np.reshape(B * W, 22)
    kpts = hand_keypoints(flat)        # (B*W, 21, 3)
    return kpts.reshape(B, W, 21, 3)

def angle_mae(pred_a, tgt_a):
    """Mean absolute angle error in radians."""
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
    return tot_l/n, tot_e/n

def train_epoch(model, dl, loss_fn, opt, device, grad_log=None):
    model.train()
    for x, y in dl:
        x, y   = x.to(device), y.to(device)
        pred_a = pred_to_angles(model(x))
        loss   = loss_fn(pred_a, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if grad_log is not None: grad_log.append(gn)
        opt.step()

# Evaluation helpers

@torch.no_grad()
def per_joint_mpjpe(model, dl, device):
    model.eval()
    acc = np.zeros(21); n = 0
    for x, y in dl:
        pred_a  = pred_to_angles(model(x.to(device))).cpu().numpy()
        pred_kp = angles_batch_to_kpts(pred_a)
        gt_kp   = angles_batch_to_kpts(y.numpy())
        err     = np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=(0, 1))
        acc    += err; n += 1
    return acc / n

@torch.no_grad()
def per_window_mpjpe(model, dl, device):
    model.eval()
    errs = []
    for x, y in dl:
        pred_a  = pred_to_angles(model(x.to(device))).cpu().numpy()
        pred_kp = angles_batch_to_kpts(pred_a)
        gt_kp   = angles_batch_to_kpts(y.numpy())
        err     = np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=(1, 2))  # (B,)
        errs.append(err)
    return np.concatenate(errs)

@torch.no_grad()
def sample_predictions(model, ds, device, n=3):
    """Return (gt_kpts, pred_kpts) pairs of shape (win, 21, 3) via FK."""
    model.eval()
    idxs = np.random.choice(len(ds), size=min(n,len(ds)), replace=False)
    gts, preds = [], []
    for i in idxs:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(hand_keypoints(y.numpy()))
        preds.append(hand_keypoints(pred_a))
    return gts, preds

@torch.no_grad()
def sample_angle_predictions(model, ds, device, n=3):
    """Return (gt_angles, pred_angles) pairs of shape (win, 22) in radians."""
    model.eval()
    idxs = np.random.choice(len(ds), size=min(n,len(ds)), replace=False)
    gts, preds = [], []
    for i in idxs:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(y.numpy()); preds.append(pred_a)
    return gts, preds


# Plotting (same as TCN script)

STYLE = dict(dpi=150, bbox_inches="tight")

def _finger_colors():
    cmap = plt.cm.tab10(np.linspace(0,1,6))
    return [cmap[fi] for fi,(_, idxs) in enumerate(FINGER_JOINT_RANGES.items()) for _ in idxs]

def plot_training_curves(tl, vl, te, ve, save_dir):
    epochs = range(1, len(tl)+1)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(epochs, tl, "o-", ms=4, label="Train"); axes[0].plot(epochs, vl, "s-", ms=4, label="Val")
    axes[0].set(xlabel="Epoch", ylabel="SmoothL1 Loss", title="Angle Regression Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, te, "o-", ms=4, label="Train"); axes[1].plot(epochs, ve, "s-", ms=4, label="Val")
    axes[1].set(xlabel="Epoch", ylabel="Mean Angle Error (rad)", title="Mean Angle Error"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir,"training_curves.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_per_joint_mpjpe(joint_errs, save_dir):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.bar(JOINT_NAMES, joint_errs, color=_finger_colors(), edgecolor="black", linewidth=0.4)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="MPJPE", title="Per-Joint MPJPE (Validation)"); ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    cmap = plt.cm.tab10(np.linspace(0,1,6))
    ax.legend(handles=[Patch(color=cmap[i],label=n) for i,n in enumerate(FINGER_JOINT_RANGES)], fontsize=8)
    plt.tight_layout(); out=os.path.join(save_dir,"per_joint_mpjpe.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_per_finger_mpjpe(joint_errs, save_dir):
    names = list(FINGER_JOINT_RANGES.keys())
    vals  = [np.mean(joint_errs[idxs]) for idxs in FINGER_JOINT_RANGES.values()]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(names, vals, color=plt.cm.Set2(np.linspace(0,1,len(names))), edgecolor="black")
    ax.set(ylabel="MPJPE", title="MPJPE by Finger"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); out=os.path.join(save_dir,"per_finger_mpjpe.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_error_distribution(pw_errs, save_dir):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    ax=axes[0]; ax.hist(pw_errs,bins=60,color="steelblue",edgecolor="white",lw=0.3)
    ax.axvline(np.median(pw_errs),color="red",ls="--",label=f"Median={np.median(pw_errs):.4f}")
    ax.axvline(np.mean(pw_errs),  color="orange",ls="--",label=f"Mean={np.mean(pw_errs):.4f}")
    ax.set(xlabel="MPJPE",ylabel="Count",title="Per-Window MPJPE Distribution"); ax.legend(); ax.grid(alpha=0.3)
    ax=axes[1]; se=np.sort(pw_errs); cdf=np.arange(1,len(se)+1)/len(se)
    ax.plot(se,cdf,lw=1.5,color="steelblue")
    for pct,col in [(0.5,"red"),(0.9,"orange"),(0.95,"purple")]:
        ax.axhline(pct,color=col,ls="--",alpha=0.6,label=f"{int(pct*100)}th %ile")
    ax.set(xlabel="MPJPE",ylabel="CDF",title="CDF of Per-Window MPJPE"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); out=os.path.join(save_dir,"error_distribution.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_joint_error_heatmap(joint_errs, save_dir):
    mat = np.full((6,4), np.nan); mat[0,0] = joint_errs[0]; k=1
    for fi in range(1,6):
        for jj in range(4): mat[fi,jj]=joint_errs[k]; k+=1
    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Base/MCP","PIP","DIP","Tip"])
    ax.set_yticks(range(6)); ax.set_yticklabels(["Wrist","Thumb","Index","Middle","Ring","Little"])
    ax.set_title("Per-Joint MPJPE Heatmap"); plt.colorbar(im,ax=ax,label="MPJPE")
    vmax=np.nanmax(mat)
    for fi in range(6):
        for jj in range(4):
            v=mat[fi,jj]
            if not np.isnan(v): ax.text(jj,fi,f"{v:.3f}",ha="center",va="center",fontsize=7,color="white" if v>0.6*vmax else "black")
    plt.tight_layout(); out=os.path.join(save_dir,"joint_error_heatmap.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_prediction_overlays(gts, preds, save_dir, joint_idx=5):
    n=len(gts); fig, axes=plt.subplots(n,3,figsize=(14,3*n))
    if n==1: axes=axes[np.newaxis,:]
    for row,(gt,pred) in enumerate(zip(gts,preds)):
        t=np.arange(gt.shape[0])
        for col,cl in enumerate(["X","Y","Z"]):
            ax=axes[row,col]
            ax.plot(t, gt[:,joint_idx,col],   lw=1.5, label="GT",   color="royalblue")
            ax.plot(t, pred[:,joint_idx,col], lw=1.5, label="Pred", color="tomato", ls="--")
            ax.set(xlabel="Frame",ylabel=cl, title=f"Sample {row+1} – {JOINT_NAMES[joint_idx]} {cl}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle(f"Prediction Overlay – {JOINT_NAMES[joint_idx]}", fontsize=11, y=1.01)
    plt.tight_layout(); out=os.path.join(save_dir,f"prediction_overlay_joint{joint_idx}.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_angle_overlays(gts, preds, save_dir, sensor_indices=None):
    """Plot predicted vs GT angle trajectories for selected sensors (in degrees)."""
    if sensor_indices is None:
        sensor_indices = [0, 2, 4, 7, 11, 21]   # CMC1_F, MCP1_F, MCP2_F, PIP2_F, PIP3_F, WRIST_F
    n = len(gts); m = len(sensor_indices)
    fig, axes = plt.subplots(n, m, figsize=(3*m, 3*n))
    if n == 1: axes = axes[np.newaxis, :]
    if m == 1: axes = axes[:, np.newaxis]
    for row, (gt, pred) in enumerate(zip(gts, preds)):
        t = np.arange(gt.shape[0])
        for col, si in enumerate(sensor_indices):
            ax = axes[row, col]
            ax.plot(t, np.rad2deg(gt[:,   si]), lw=1.5, label="GT",   color="royalblue")
            ax.plot(t, np.rad2deg(pred[:, si]), lw=1.5, label="Pred", color="tomato", ls="--")
            ax.set(xlabel="Frame", ylabel="Angle (°)", title=f"S{row+1} – {SENSOR_NAMES[si]}")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
    plt.suptitle("Predicted vs GT Joint Angles", fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, "angle_overlays.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_grad_norm(gnorms, save_dir):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(gnorms, lw=0.6, color="darkorange", alpha=0.7)
    if len(gnorms)>20:
        w=max(1,len(gnorms)//50)
        ax.plot(np.arange(w-1,len(gnorms)), np.convolve(gnorms,np.ones(w)/w,"valid"),
                lw=1.5, color="red", label=f"Avg (w={w})")
        ax.legend()
    ax.set(xlabel="Batch",ylabel="Gradient norm",title="Gradient Norm During Training"); ax.grid(alpha=0.3)
    plt.tight_layout(); out=os.path.join(save_dir,"grad_norm.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_data_ablation(fracs, vals, save_dir):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot([f*100 for f in fracs], vals, "o-", ms=6, lw=1.5, color="steelblue")
    for f,v in zip(fracs,vals): ax.annotate(f"{v:.4f}",xy=(f*100,v),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Training Data (%)",ylabel="Val MPJPE",title="Data Ablation – Val MPJPE vs Dataset Size"); ax.grid(alpha=0.3)
    plt.tight_layout(); out=os.path.join(save_dir,"data_ablation.png"); fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")

def plot_attention_memory_weights(model, save_dir):
    """
    PET-specific: visualise the learned external memory M_k and M_v
    from every branch × layer × head.

    Shows which memory slots have the largest magnitude (i.e. which
    "muscle activation archetypes" the model found most useful).
    """
    rows = []
    for bi, branch in enumerate(model.branches):
        for li, layer in enumerate(branch.layers):
            attn = layer.attn
            # M_k, M_v: (n_heads, mem_size, d_head)
            mk_norm = attn.M_k.detach().cpu().norm(dim=-1)   # (n_heads, mem_size)
            mv_norm = attn.M_v.detach().cpu().norm(dim=-1)
            rows.append((f"B{bi+1}L{li+1}", mk_norm, mv_norm))

    n_rows  = len(rows)
    n_heads = rows[0][1].shape[0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3*n_rows))
    if n_rows == 1: axes = axes[np.newaxis,:]

    for row_i, (label, mk, mv) in enumerate(rows):
        for col_i, (mat, mname) in enumerate([(mk,"M_k norm"),(mv,"M_v norm")]):
            ax  = axes[row_i, col_i]
            im  = ax.imshow(mat.numpy(), cmap="viridis", aspect="auto")
            ax.set_xlabel("Memory slot (s)"); ax.set_ylabel("Head")
            ax.set_title(f"{label} – {mname}")
            ax.set_yticks(range(n_heads)); ax.set_yticklabels([f"H{i}" for i in range(n_heads)])
            plt.colorbar(im, ax=ax, fraction=0.03)

    plt.suptitle("External Memory Norms (‖M_k‖, ‖M_v‖) per Branch/Layer/Head\n"
                 "High-norm slots = frequently activated 'muscle archetypes'",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir,"attention_memory_weights.png")
    fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")


def plot_attention_patterns(model, val_ds, device, save_dir, n_samples=4):
    """
    PET-specific: for a few validation windows, show the attention weight
    matrix  A ∈ ℝ^{T × s}  (patches × memory slots) for every branch and head.

    Unlike standard self-attention this is NOT square — it reveals which
    memory slots each time-step activates, giving interpretable "gesture templates".
    """
    model.eval()
    idxs = np.random.default_rng(0).choice(len(val_ds), size=min(n_samples,len(val_ds)), replace=False)

    # Hook to capture attention weights from the first layer of branch 0
    captured = {}
    def _hook(module, inp, out):
        # Recompute attn weights from stored Q and M_k
        # (we store them during the forward of ExternalAttention)
        captured["attn_w"] = module._last_attn_w

    # Monkey-patch ExternalAttention.forward to cache attn weights
    original_forward = ExternalAttention.forward

    def patched_forward(self, x):
        B, T, _ = x.shape
        H, S, D = self.n_heads, self.mem_size, self.d_head
        Q       = self.q_proj(x).view(B,T,H,D).permute(0,2,1,3)
        scale   = D**-0.5
        attn_w  = F.softmax(torch.matmul(Q, self.M_k.transpose(-1,-2))*scale, dim=-1)
        attn_w  = attn_w / (attn_w.sum(dim=-2,keepdim=True)+1e-6)
        self._last_attn_w = attn_w.detach().cpu()   # cache
        out     = torch.matmul(attn_w, self.M_v)
        out     = out.permute(0,2,1,3).contiguous().view(B,T,H*D)
        return self.out_proj(out)

    ExternalAttention.forward = patched_forward

    first_attn = model.branches[0].layers[0].attn
    n_heads    = first_attn.n_heads
    mem_size   = first_attn.mem_size

    for si, idx in enumerate(idxs):
        x, _ = val_ds[idx]
        with torch.no_grad():
            model(x.unsqueeze(0).to(device))
        attn_w = first_attn._last_attn_w[0]   # (n_heads, P, S)

        fig, axes = plt.subplots(1, n_heads, figsize=(4*n_heads, 4))
        if n_heads == 1: axes = [axes]
        for hi, ax in enumerate(axes):
            im = ax.imshow(attn_w[hi].numpy(), cmap="Blues", aspect="auto")
            ax.set_xlabel("Memory slot (s)")
            ax.set_ylabel("Patch (time)")
            ax.set_title(f"Head {hi+1}")
            plt.colorbar(im, ax=ax, fraction=0.03)
        plt.suptitle(f"External Attention Weights – Sample {si+1}\n"
                     "Branch 1, Layer 1  |  rows=patches, cols=memory slots",
                     fontsize=9, y=1.02)
        plt.tight_layout()
        out = os.path.join(save_dir, f"attention_pattern_sample{si+1}.png")
        fig.savefig(out, **STYLE); plt.close(fig); print(f"  Saved: {out}")

    # Restore original forward
    ExternalAttention.forward = original_forward


def plot_branch_diversity(model, val_ds, device, save_dir, n_samples=64):
    """
    PET-specific: measure how differently each parallel branch represents
    the same input by computing pairwise cosine similarity between branch
    output vectors (averaged over the sampled batch).

    Low similarity = branches have specialised → good for ensemble diversity.
    High similarity = branches are redundant → may want fewer branches.
    """
    model.eval()
    n_branches = len(model.branches)
    idxs = np.random.default_rng(1).choice(len(val_ds), size=min(n_samples,len(val_ds)), replace=False)
    xs   = torch.stack([val_ds[i][0] for i in idxs]).to(device)

    with torch.no_grad():
        z     = model.pos_enc(model.patch_embed(xs))
        bouts = [b(z).mean(dim=1) for b in model.branches]  # each (B, d_model)

    cos_sim = np.zeros((n_branches, n_branches))
    for i in range(n_branches):
        for j in range(n_branches):
            a = F.normalize(bouts[i], dim=-1)
            b = F.normalize(bouts[j], dim=-1)
            cos_sim[i,j] = (a * b).sum(dim=-1).mean().item()

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cos_sim, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(n_branches)); ax.set_xticklabels([f"B{i+1}" for i in range(n_branches)])
    ax.set_yticks(range(n_branches)); ax.set_yticklabels([f"B{i+1}" for i in range(n_branches)])
    ax.set_title("Branch Output Cosine Similarity\n(lower = more diverse = better)")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    for i in range(n_branches):
        for j in range(n_branches):
            ax.text(j, i, f"{cos_sim[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    out = os.path.join(save_dir,"branch_diversity.png")
    fig.savefig(out,**STYLE); plt.close(fig); print(f"  Saved: {out}")


# Main

def get_device():
    if torch.cuda.is_available():     return "cuda"
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="PET: EMG → Hand Keypoints")
    # data
    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ninapro_DB1")
    desktop_data_dir = os.path.expanduser("~/Desktop/Ninapro_DB1")
    resolved_data_dir = default_data_dir if os.path.isdir(default_data_dir) else desktop_data_dir
    p.add_argument("--data-dir",      type=str,   default=resolved_data_dir)
    p.add_argument("--val-subjects",  type=str,   nargs="+", default=["s1"])
    p.add_argument("--cv-mode",       type=str,   default="cross_subject",
                   choices=["cross_subject", "within_subject"],
                   help="cross_subject: hold out val-subjects; "
                        "within_subject: hold out val-rep from all subjects")
    p.add_argument("--val-rep",       type=int,   default=6,
                   help="Repetition index held out as validation (within_subject mode only)")
    p.add_argument("--win",           type=int,   default=64)
    p.add_argument("--stride",        type=int,   default=16)
    p.add_argument("--ds",            type=int,   default=3)
    p.add_argument("--min-len",       type=int,   default=80)
    # training
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--save-path",     type=str,   default="checkpoints/best_pet.pt")
    # model
    p.add_argument("--d-model",       type=int,   default=128,  help="Embedding dimension")
    p.add_argument("--n-heads",       type=int,   default=4,    help="Attention heads per branch")
    p.add_argument("--n-branches",    type=int,   default=3,    help="Parallel encoder branches")
    p.add_argument("--n-layers",      type=int,   default=2,    help="Encoder layers per branch")
    p.add_argument("--mem-size",      type=int,   default=64,   help="External memory size s")
    p.add_argument("--patch-size",    type=int,   default=8,    help="Frames per patch (win %% patch_size == 0)")
    p.add_argument("--ffn-mult",      type=int,   default=2,    help="FFN hidden = d_model * ffn_mult")
    p.add_argument("--ffn-act",       type=str,   default="gelu", choices=["gelu","relu"],
                   help="FFN activation (use relu for TFLM deployment, gelu for full model)")
    p.add_argument("--dropout",       type=float, default=0.2)
    # plots / ablation
    p.add_argument("--plot-dir",      type=str,   default="./plots/pet")
    p.add_argument("--overlay-joints",type=int,   nargs="+", default=[0,5,9])
    p.add_argument("--ablation-fractions", type=float, nargs="+", default=[0.1,0.25,0.5,0.75,1.0])
    p.add_argument("--ablation-epochs",    type=int,   default=5)
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()

    assert args.win % args.patch_size == 0, \
        f"--win ({args.win}) must be divisible by --patch-size ({args.patch_size})"

    os.makedirs(args.plot_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Local data directory not found: {args.data_dir!r}\n"
            "Expected either ./Ninapro_DB1 next to pet.py or ~/Desktop/Ninapro_DB1, or pass --data-dir /full/path/to/Ninapro_DB1"
        )

    # --- subjects ---
    all_subjects = sorted([d for d in os.listdir(args.data_dir)
                           if d.startswith("s") and os.path.isdir(os.path.join(args.data_dir,d))])

    # --- data ---
    if args.cv_mode == "within_subject":
        print(f"\nCV mode: within_subject (val rep = {args.val_rep})")
        print(f"All subjects: {all_subjects}")
        print("\nBuilding within-subject split...")
        Xtr, Ytr, Xva, Yva = build_within_subject_split(
            args.data_dir, all_subjects,
            args.ds, args.win, args.stride, args.min_len, args.val_rep,
        )
    else:
        val_subjects   = args.val_subjects
        train_subjects = [s for s in all_subjects if s not in val_subjects]
        print(f"\nCV mode: cross_subject")
        print(f"Train: {train_subjects}\nVal:   {val_subjects}")
        print("\nBuilding training windows...")
        Xtr, Ytr = build_split(args.data_dir, train_subjects, args.ds, args.win, args.stride, args.min_len)
        print("Building validation windows...")
        Xva, Yva = build_split(args.data_dir, val_subjects,   args.ds, args.win, args.stride, args.min_len)

    mu, sd = safe_zscore_fit(Xtr)
    Xtr    = safe_zscore_apply(Xtr, mu, sd)
    Xva    = safe_zscore_apply(Xva, mu, sd)
    print(f"Xtr {Xtr.shape}  Ytr {Ytr.shape}")
    print(f"Xva {Xva.shape}  Yva {Yva.shape}")

    train_ds = Emg2KptsDataset(Xtr, Ytr)
    val_ds   = Emg2KptsDataset(Xva, Yva)
    pin      = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # --- model ---
    c_in  = Xtr.shape[-1]
    model = EMG_PET(
        c_in=c_in, d_model=args.d_model, n_heads=args.n_heads,
        n_branches=args.n_branches, n_layers=args.n_layers,
        mem_size=args.mem_size, patch_size=args.patch_size,
        ffn_mult=args.ffn_mult, dropout=args.dropout,
        out_dim=22, win=args.win, ffn_act=args.ffn_act,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPET parameters: {params:,}")
    print(model)

    loss_fn = nn.SmoothL1Loss()
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # --- training loop ---
    print(f"\n=== Training PET ({args.epochs} epochs) ===")
    tl_hist, vl_hist, te_hist, ve_hist, gnorms = [], [], [], [], []
    best_mpjpe = float("inf")

    for epoch in range(1, args.epochs+1):
        model.train()
        ep_l = ep_e = n = 0
        for x, y in train_dl:
            x, y   = x.to(device), y.to(device)
            pred_a = pred_to_angles(model(x))
            loss   = loss_fn(pred_a, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            gnorms.append(gn); opt.step()
            bs = x.size(0)
            ep_l += loss.item()*bs; ep_e += angle_mae(pred_a, y).item()*bs; n+=bs
        tl_hist.append(ep_l/n); te_hist.append(ep_e/n)

        vl, ve = eval_epoch(model, val_dl, loss_fn, device)
        vl_hist.append(vl); ve_hist.append(ve)
        sched.step()

        improved = ""
        if ve < best_mpjpe:
            best_mpjpe = ve
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "val_mpjpe":ve,"args":vars(args)}, args.save_path)
            improved = " ✓"
        print(f"  epoch {epoch:02d} | tr loss {tl_hist[-1]:.4f} | vl loss {vl:.4f} | "
              f"val MPJPE {ve:.4f}{improved}")

    print(f"\nBest val MPJPE: {best_mpjpe:.4f}")

    # reload best
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- plots ---
    print(f"\nGenerating plots → {args.plot_dir}/")
    plot_training_curves(tl_hist, vl_hist, te_hist, ve_hist, args.plot_dir)

    joint_errs = per_joint_mpjpe(model, val_dl, device)
    plot_per_joint_mpjpe(joint_errs, args.plot_dir)
    plot_per_finger_mpjpe(joint_errs, args.plot_dir)

    pw_errs = per_window_mpjpe(model, val_dl, device)
    plot_error_distribution(pw_errs, args.plot_dir)
    plot_joint_error_heatmap(joint_errs, args.plot_dir)

    gts, preds = sample_predictions(model, val_ds, device)
    for ji in args.overlay_joints:
        plot_prediction_overlays(gts, preds, args.plot_dir, joint_idx=ji)

    ang_gts, ang_preds = sample_angle_predictions(model, val_ds, device)
    plot_angle_overlays(ang_gts, ang_preds, args.plot_dir)

    plot_grad_norm(gnorms, args.plot_dir)

    # PET-native visualisations
    plot_attention_memory_weights(model, args.plot_dir)
    plot_attention_patterns(model, val_ds, device, args.plot_dir)
    plot_branch_diversity(model, val_ds, device, args.plot_dir)

    # data ablation
    print("\n=== Data ablation ===")
    abl_errs = []
    for frac in args.ablation_fractions:
        n_use = max(args.batch_size, int(frac*len(train_ds)))
        idx   = np.random.choice(len(train_ds), n_use, replace=False)
        sub_dl= DataLoader(Subset(train_ds,idx), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        m_abl = EMG_PET(c_in=c_in, d_model=args.d_model, n_heads=args.n_heads,
                        n_branches=args.n_branches, n_layers=args.n_layers,
                        mem_size=args.mem_size, patch_size=args.patch_size,
                        ffn_mult=args.ffn_mult, dropout=args.dropout,
                        out_dim=22, win=args.win, ffn_act=args.ffn_act).to(device)
        o_abl = torch.optim.AdamW(m_abl.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_abl = float("inf")
        print(f"  frac={frac*100:.0f}%  n={n_use}")
        for ep in range(args.ablation_epochs):
            train_epoch(m_abl, sub_dl, loss_fn, o_abl, device)
            _, ve = eval_epoch(m_abl, val_dl, loss_fn, device)
            best_abl = min(best_abl, ve)
        abl_errs.append(best_abl)
    plot_data_ablation(args.ablation_fractions, abl_errs, args.plot_dir)

    print(f"\nAll plots saved to: {args.plot_dir}/")
    for f in sorted(os.listdir(args.plot_dir)):
        if f.endswith(".png"): print(f"  {f}")


if __name__ == "__main__":
    main()