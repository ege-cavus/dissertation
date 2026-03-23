#!/usr/bin/env python3
"""Train a TCN to predict 22 hand joint angles from sEMG (Ninapro DB1).

Data dir defaults to ~/Desktop/Ninapro_DB1; override with --data-dir.
Plots saved to --plot-dir (default: ./plots/tcn/).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Local data directory helper

def _check_local_data_dir(data_dir):
    expanded = os.path.expanduser(data_dir)
    if os.path.isdir(expanded):
        return expanded
    raise FileNotFoundError(
        f"Data directory not found: {expanded!r}\n"
        "Make sure Ninapro_DB1 exists at that path, for example:\n"
        "  ~/Desktop/Ninapro_DB1"
    )


import torch
import torch.nn as nn
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
    "CMC1_F":   (0, 50),  "CMC1_A":   (-30, 30),
    "MCP1_F":   (0, 60),  "IP1_F":    (0, 80),
    "MCP2_F":   (0, 90),  "MCP2-3_A": (-20, 20),
    "PIP2_F":   (0, 100), "DIP2_F":   (0, 80),
    "MCP3_F":   (0, 90),  "MCP3-4_A": (-20, 20),
    "PIP3_F":   (0, 100), "DIP3_F":   (0, 80),
    "MCP4_F":   (0, 90),  "MCP4-5_A": (-20, 20),
    "PIP4_F":   (0, 100), "DIP4_F":   (0, 80),
    "CMC5_F":   (0, 30),
    "MCP5_F":   (0, 90),
    "PIP5_F":   (0, 100), "DIP5_F":   (0, 80),
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

# Data loading & preprocessing helpers

def load_ninapro_mat(path):
    m = loadmat(path)
    emg   = np.asarray(m["emg"],         dtype=np.float32)
    glove = np.asarray(m["glove"],        dtype=np.float32)
    restim= np.asarray(m["restimulus"],   dtype=np.int32).ravel()
    rerep = np.asarray(m["rerepetition"], dtype=np.int32).ravel()
    return emg, glove, restim, rerep


def extract_segment(glove, restim, rerep, movement_id, repetition_id):
    mask = (restim == movement_id) & (rerep == repetition_id)
    idx  = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No samples found for that (movement_id, repetition_id).")
    splits = np.where(np.diff(idx) > 1)[0]
    blocks, start = [], 0
    for s in splits:
        blocks.append(idx[start:s + 1])
        start = s + 1
    blocks.append(idx[start:])
    block = max(blocks, key=len)
    return glove[block], block


def robust_minmax(x, qlo=0.02, qhi=0.98):
    x   = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo  = np.quantile(x, qlo, axis=0)
    hi  = np.quantile(x, qhi, axis=0)
    den = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
    return lo, hi, den


def glove_to_angles(glove_raw):
    lo, hi, denom = robust_minmax(glove_raw)
    u = np.clip((glove_raw - lo) / denom, 0.0, 1.0)
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
    R   = _R_z(mcp_abd) @ _R_x(mcp_flex)
    mcp = base
    pip = mcp + R  @ np.array([0, L1, 0.0])
    R2  = R @ _R_x(pip_flex)
    dip = pip + R2 @ np.array([0, L2, 0.0])
    R3  = R2 @ _R_x(dip_flex)
    tip = dip + R3 @ np.array([0, L3, 0.0])
    return mcp, pip, dip, tip


def hand_keypoints(angles_rad):
    T   = angles_rad.shape[0]
    out = np.zeros((T, 21, 3), dtype=np.float32)
    col = {name: i for i, name in enumerate(SENSOR_NAMES)}
    for t in range(T):
        a = angles_rad[t]
        out[t, 0] = [0.0, 0.0, 0.0]   # wrist

        thumb_mcp_f = a[col["MCP1_F"]] + 0.5 * a[col["CMC1_F"]]
        thumb = fk_finger(
            FINGER_BASES["thumb"], thumb_mcp_f,
            0.6 * a[col["IP1_F"]], 0.4 * a[col["IP1_F"]],
            a[col["CMC1_A"]], FINGER_LENGTHS["thumb"],
        )
        idx_f = fk_finger(
            FINGER_BASES["index"],
            a[col["MCP2_F"]], a[col["PIP2_F"]], a[col["DIP2_F"]],
            a[col["MCP2-3_A"]], FINGER_LENGTHS["index"],
        )
        mid_abd = 0.5 * (a[col["MCP2-3_A"]] + a[col["MCP3-4_A"]])
        mid = fk_finger(
            FINGER_BASES["middle"],
            a[col["MCP3_F"]], a[col["PIP3_F"]], a[col["DIP3_F"]],
            mid_abd, FINGER_LENGTHS["middle"],
        )
        ring = fk_finger(
            FINGER_BASES["ring"],
            a[col["MCP4_F"]], a[col["PIP4_F"]], a[col["DIP4_F"]],
            a[col["MCP3-4_A"]], FINGER_LENGTHS["ring"],
        )
        little = fk_finger(
            FINGER_BASES["little"],
            a[col["MCP5_F"]], a[col["PIP5_F"]], a[col["DIP5_F"]],
            a[col["MCP4-5_A"]], FINGER_LENGTHS["little"],
        )
        k = 1
        for finger in [thumb, idx_f, mid, ring, little]:
            for j in finger:
                out[t, k] = j
                k += 1
    return out


# Windowed dataset construction

def safe_zscore_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def safe_zscore_apply(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def windowize_seq(emg_seg, kpts, win=64, stride=16):
    T = emg_seg.shape[0]
    X, Y = [], []
    for start in range(0, T - win + 1, stride):
        end = start + win
        X.append(emg_seg[start:end])
        Y.append(kpts[start:end])
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


def build_windows_from_mat_seq(mat_path, ds=3, win=64, stride=16, min_len=80):
    emg, glove, restim, rerep = load_ninapro_mat(mat_path)
    mov_ids = [int(x) for x in np.unique(restim) if int(x) > 0]
    rep_ids = [int(x) for x in np.unique(rerep)  if int(x) > 0]

    X_all, Y_all, kept = [], [], 0
    for m_id in mov_ids:
        for r_id in rep_ids:
            try:
                glove_seg, idx = extract_segment(glove, restim, rerep, m_id, r_id)
            except ValueError:
                continue
            emg_seg   = emg[idx][::ds]
            glove_seg = glove_seg[::ds]
            if glove_seg.shape[0] < min_len or emg_seg.shape[0] < win:
                continue
            ang     = glove_to_angles(glove_seg).astype(np.float32)   # (T, 22) angles in radians
            emg_seg = np.nan_to_num(emg_seg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            ang     = np.nan_to_num(ang,     nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            X, Y = windowize_seq(emg_seg, ang, win=win, stride=stride)
            if X is None:
                continue
            X_all.append(X); Y_all.append(Y); kept += 1

    if not X_all:
        return None, None, 0
    return np.concatenate(X_all), np.concatenate(Y_all), kept


def build_split_seq(base_dir, subject_list, ds=3, win=64, stride=16, min_len=80):
    X_all, Y_all = [], []
    files_seen = segs_seen = 0
    for s in subject_list:
        subj_dir  = os.path.join(base_dir, s)
        mat_files = sorted(glob.glob(os.path.join(subj_dir, f"{s.upper()}_A1_E*.mat")))
        for mf in mat_files:
            X, Y, segs = build_windows_from_mat_seq(mf, ds=ds, win=win, stride=stride, min_len=min_len)
            files_seen += 1
            segs_seen  += segs
            if X is None:
                continue
            X_all.append(X); Y_all.append(Y)
    if not X_all:
        raise ValueError(f"No windows created for subjects {subject_list}.")
    X = np.concatenate(X_all).astype(np.float32)
    Y = np.concatenate(Y_all).astype(np.float32)
    print(f"  Split | subjects={len(subject_list)} files={files_seen} "
          f"segments={segs_seen} windows={X.shape[0]}")
    return X, Y


def build_within_subject_split_seq(base_dir, subjects, ds=3, win=64, stride=16,
                                   min_len=80, val_rep=6):
    """Within-subject split by repetition index.

    All subjects contribute to BOTH partitions.
    Train: every repetition whose index != val_rep.
    Val:   repetition val_rep only.
    """
    Xtr, Ytr, Xva, Yva = [], [], [], []
    fcount = tr_segs = va_segs = 0
    for s in subjects:
        subj_dir  = os.path.join(base_dir, s)
        mat_files = sorted(glob.glob(os.path.join(subj_dir, f"{s.upper()}_A1_E*.mat")))
        for mf in mat_files:
            emg, glove, restim, rerep = load_ninapro_mat(mf)
            fcount += 1
            mov_ids = [int(x) for x in np.unique(restim) if int(x) > 0]
            rep_ids = [int(x) for x in np.unique(rerep)  if int(x) > 0]
            for m_id in mov_ids:
                for r_id in rep_ids:
                    try:
                        glove_seg, idx = extract_segment(glove, restim, rerep, m_id, r_id)
                    except ValueError:
                        continue
                    emg_seg   = emg[idx][::ds]
                    glove_seg = glove_seg[::ds]
                    if glove_seg.shape[0] < min_len or emg_seg.shape[0] < win:
                        continue
                    ang     = glove_to_angles(glove_seg).astype(np.float32)
                    emg_seg = np.nan_to_num(emg_seg, nan=0., posinf=0., neginf=0.).astype(np.float32)
                    ang     = np.nan_to_num(ang,     nan=0., posinf=0., neginf=0.).astype(np.float32)
                    X, Y = windowize_seq(emg_seg, ang, win=win, stride=stride)
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


# PyTorch Dataset

class Emg2KptsSeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()   # (N, win, 22) – joint angles in radians

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i].transpose(0, 1), self.Y[i]   # (C,win), (win,22)


# TCN Model

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in,  c_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.down  = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        T = x.shape[-1]
        y = self.drop(self.act(self.conv1(x)))[..., :T]
        y = self.drop(self.act(self.conv2(y)))[..., :T]
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class EMG_TCN_SEQ(nn.Module):
    def __init__(self, c_in, hidden=128, levels=5, dropout=0.1, out_dim=22):
        super().__init__()
        blocks = []
        c = c_in
        for i in range(levels):
            blocks.append(TCNBlock(c, hidden, k=3, dilation=2**i, dropout=dropout))
            c = hidden
        self.tcn  = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden, out_dim, 1)

    def forward(self, x):
        return self.proj(self.tcn(x))


# Training utilities

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
    tot_loss = tot_mae = n = 0
    for x, y in dl:
        x, y   = x.to(device), y.to(device)
        pred_a = pred_to_angles(model(x))
        l      = loss_fn(pred_a, y)
        e      = angle_mae(pred_a, y)
        bs     = x.size(0)
        tot_loss += l.item() * bs
        tot_mae  += e.item() * bs
        n        += bs
    return tot_loss / n, tot_mae / n


def train_epoch(model, dl, loss_fn, opt, device, grad_norm_log=None):
    model.train()
    for x, y in dl:
        x, y   = x.to(device), y.to(device)
        pred_a = pred_to_angles(model(x))
        loss   = loss_fn(pred_a, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if grad_norm_log is not None:
            grad_norm_log.append(gn)
        opt.step()


# Evaluation helpers that compute per-joint errors

@torch.no_grad()
def per_joint_mpjpe(model, dl, device):
    """Returns array of shape (21,) – mean MPJPE per joint via FK."""
    model.eval()
    joint_err = np.zeros(21, dtype=np.float64)
    n = 0
    for x, y in dl:
        pred_a  = pred_to_angles(model(x.to(device))).cpu().numpy()
        pred_kp = angles_batch_to_kpts(pred_a)
        gt_kp   = angles_batch_to_kpts(y.numpy())
        err     = np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=(0, 1))
        joint_err += err; n += 1
    return joint_err / n


@torch.no_grad()
def per_window_mpjpe(model, dl, device):
    """Returns 1-D array – MPJPE (via FK) for every window in the loader."""
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
def sample_predictions(model, ds, device, n_samples=3):
    """Return (gt_kpts, pred_kpts) pairs of shape (win, 21, 3) via FK."""
    model.eval()
    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    gts, preds = [], []
    for i in indices:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(hand_keypoints(y.numpy()))
        preds.append(hand_keypoints(pred_a))
    return gts, preds


@torch.no_grad()
def sample_angle_predictions(model, ds, device, n_samples=3):
    """Return (gt_angles, pred_angles) pairs of shape (win, 22) in radians."""
    model.eval()
    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    gts, preds = [], []
    for i in indices:
        x, y   = ds[i]
        pred_a = pred_to_angles(model(x.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        gts.append(y.numpy()); preds.append(pred_a)
    return gts, preds


# Plotting functions

STYLE = dict(dpi=150, bbox_inches="tight")

def plot_training_curves(train_losses, val_losses, train_mpjpes, val_mpjpes, save_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, train_losses,  label="Train loss",  marker="o", ms=4)
    ax.plot(epochs, val_losses,    label="Val loss",    marker="s", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("SmoothL1 Loss")
    ax.set_title("Angle Regression Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_mpjpes, label="Train", marker="o", ms=4)
    ax.plot(epochs, val_mpjpes,   label="Val",   marker="s", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Angle Error (rad)")
    ax.set_title("Mean Angle Error")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_joint_mpjpe(joint_errors, save_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    finger_colors = []
    finger_map = []
    for fi, (fname, idxs) in enumerate(FINGER_JOINT_RANGES.items()):
        for _ in idxs:
            finger_colors.append(colors[fi])
            finger_map.append(fname)

    bars = ax.bar(JOINT_NAMES, joint_errors, color=finger_colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Joint"); ax.set_ylabel("Mean MPJPE")
    ax.set_title("Per-Joint MPJPE on Validation Set")
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Legend for finger groups
    from matplotlib.patches import Patch
    legend_els = [Patch(color=colors[i], label=name)
                  for i, name in enumerate(FINGER_JOINT_RANGES.keys())]
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, "per_joint_mpjpe.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_finger_mpjpe(joint_errors, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    names, vals = [], []
    for fname, idxs in FINGER_JOINT_RANGES.items():
        names.append(fname)
        vals.append(np.mean(joint_errors[idxs]))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax.bar(names, vals, color=cmap, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean MPJPE"); ax.set_title("MPJPE Aggregated by Finger")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "per_finger_mpjpe.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_error_distribution(per_window_errors, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(per_window_errors, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(per_window_errors), color="red",   linestyle="--", label=f"Median={np.median(per_window_errors):.4f}")
    ax.axvline(np.mean(per_window_errors),   color="orange",linestyle="--", label=f"Mean={np.mean(per_window_errors):.4f}")
    ax.set_xlabel("MPJPE"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Window MPJPE (Validation)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    sorted_e = np.sort(per_window_errors)
    cdf      = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax.plot(sorted_e, cdf, color="steelblue", lw=1.5)
    ax.axhline(0.5,  color="red",    linestyle="--", alpha=0.6, label="50th pctile")
    ax.axhline(0.9,  color="orange", linestyle="--", alpha=0.6, label="90th pctile")
    ax.axhline(0.95, color="purple", linestyle="--", alpha=0.6, label="95th pctile")
    ax.set_xlabel("MPJPE"); ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of Per-Window MPJPE (Validation)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "error_distribution.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_prediction_overlays(gts, preds, save_dir, joint_idx=5):
    """Plot predicted vs GT coordinate trajectories for a chosen joint across sample windows."""
    n = len(gts)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n), sharey=False)
    if n == 1:
        axes = axes[np.newaxis, :]
    coord_labels = ["X", "Y", "Z"]
    for row, (gt, pred) in enumerate(zip(gts, preds)):
        win = gt.shape[0]
        t   = np.arange(win)
        for col, cl in enumerate(coord_labels):
            ax = axes[row, col]
            ax.plot(t, gt[:, joint_idx, col],   lw=1.5, label="GT",   color="royalblue")
            ax.plot(t, pred[:, joint_idx, col], lw=1.5, label="Pred", color="tomato", linestyle="--")
            ax.set_xlabel("Frame"); ax.set_ylabel(cl)
            ax.set_title(f"Sample {row+1} – {JOINT_NAMES[joint_idx]} {cl}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle(f"Prediction Overlay – {JOINT_NAMES[joint_idx]}", y=1.01, fontsize=11)
    plt.tight_layout()
    out = os.path.join(save_dir, f"prediction_overlay_joint{joint_idx}.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


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


def plot_grad_norm(grad_norms_per_epoch, save_dir):
    """Plot gradient norm over training iterations."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grad_norms_per_epoch, lw=0.6, color="darkorange", alpha=0.7)
    # smoothed
    if len(grad_norms_per_epoch) > 20:
        w = max(1, len(grad_norms_per_epoch) // 50)
        smooth = np.convolve(grad_norms_per_epoch, np.ones(w)/w, mode="valid")
        ax.plot(np.arange(w-1, len(grad_norms_per_epoch)), smooth, lw=1.5,
                color="red", label=f"Running avg (w={w})")
        ax.legend()
    ax.set_xlabel("Training step (batch)"); ax.set_ylabel("Gradient norm (post-clip)")
    ax.set_title("Gradient Norm During Training")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "grad_norm.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_joint_error_heatmap(joint_errors, save_dir):
    """Reshape 21-joint errors into a finger × joint-position matrix."""
    fingers = ["Wrist","Thumb","Index","Middle","Ring","Little"]
    # Build a (6,4) matrix – wrist is special (only 1 joint)
    mat = np.full((6, 4), np.nan)
    mat[0, 0] = joint_errors[0]   # wrist
    k = 1
    for fi in range(1, 6):
        for jj in range(4):
            mat[fi, jj] = joint_errors[k]
            k += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Base/MCP","PIP","DIP","Tip"])
    ax.set_yticks(range(6))
    ax.set_yticklabels(fingers)
    ax.set_title("Per-Joint MPJPE Heatmap (Validation)")
    plt.colorbar(im, ax=ax, label="MPJPE")
    for fi in range(6):
        for jj in range(4):
            v = mat[fi, jj]
            if not np.isnan(v):
                ax.text(jj, fi, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if v < mat[~np.isnan(mat)].max() * 0.6 else "white")
    plt.tight_layout()
    out = os.path.join(save_dir, "joint_error_heatmap.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_data_ablation(fractions, val_mpjpes, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([f * 100 for f in fractions], val_mpjpes,
            marker="o", ms=6, lw=1.5, color="steelblue")
    for f, v in zip(fractions, val_mpjpes):
        ax.annotate(f"{v:.4f}", xy=(f * 100, v),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Val MPJPE")
    ax.set_title("Data Ablation – Val MPJPE vs Training Set Size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "data_ablation.png")
    fig.savefig(out, **STYLE)
    plt.close(fig)
    print(f"  Saved: {out}")


# Main

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train EMG -> Hand Keypoint TCN (with plots)")
    p.add_argument("--data-dir",     type=str,   default="~/Desktop/Ninapro_DB1")
    p.add_argument("--val-subjects", type=str,   nargs="+", default=["s1"])
    p.add_argument("--cv-mode",      type=str,   default="cross_subject",
                   choices=["cross_subject", "within_subject"],
                   help="cross_subject: hold out val-subjects; "
                        "within_subject: hold out val-rep from all subjects")
    p.add_argument("--val-rep",      type=int,   default=6,
                   help="Repetition index held out as validation (within_subject mode only)")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--levels",       type=int,   default=5)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--win",          type=int,   default=64)
    p.add_argument("--stride",       type=int,   default=16)
    p.add_argument("--ds",           type=int,   default=3)
    p.add_argument("--min-len",      type=int,   default=80)
    p.add_argument("--save-path",    type=str,   default="checkpoints/best_tcn.pt")
    p.add_argument("--plot-dir",     type=str,   default="./plots/tcn",
                   help="Directory to save all evaluation plots")
    p.add_argument("--ablation-fractions", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="Training-set fractions to use in data ablation study")
    p.add_argument("--ablation-epochs", type=int, default=5,
                   help="Epochs to train each ablation model (keep low for speed)")
    p.add_argument("--overlay-joints", type=int, nargs="+", default=[0, 5, 9],
                   help="Joint indices for prediction overlay plots")
    args, _ = p.parse_known_args()
    return args


def make_model(c_in, args, device):
    m = EMG_TCN_SEQ(
        c_in=c_in, hidden=args.hidden, levels=args.levels,
        dropout=args.dropout, out_dim=22,
    ).to(device)
    return m


def run_training(model, train_dl, val_dl, loss_fn, args, device):
    """Full training loop; returns per-epoch metrics and gradient norms."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mpjpe = float("inf")
    train_losses, val_losses, train_mpjpes, val_mpjpes, grad_norms = [], [], [], [], []

    for epoch in range(1, args.epochs + 1):
        # --- training ---
        model.train()
        ep_grad = []
        ep_loss = ep_mpjpe = n = 0
        for x, y in train_dl:
            x, y   = x.to(device), y.to(device)
            pred_a = pred_to_angles(model(x))
            loss   = loss_fn(pred_a, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            ep_grad.append(gn)
            opt.step()
            bs = x.size(0)
            ep_loss  += loss.item() * bs
            ep_mpjpe += angle_mae(pred_a, y).item() * bs
            n        += bs
        train_losses.append(ep_loss / n)
        train_mpjpes.append(ep_mpjpe / n)
        grad_norms.extend(ep_grad)

        # --- validation ---
        vl, ve = eval_epoch(model, val_dl, loss_fn, device)
        val_losses.append(vl)
        val_mpjpes.append(ve)

        improved = ""
        if ve < best_mpjpe:
            best_mpjpe = ve
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": vl, "val_mpjpe": ve,
            }, args.save_path)
            improved = " *"
        print(f"  epoch {epoch:02d} | train loss {train_losses[-1]:.4f} "
              f"| val loss {vl:.4f} | val MPJPE {ve:.4f}{improved}")

    return train_losses, val_losses, train_mpjpes, val_mpjpes, grad_norms, best_mpjpe


def main():
    args = parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    args.data_dir = _check_local_data_dir(args.data_dir)
    print(f"Using local data directory: {args.data_dir}")

    # --- Discover subjects ---------------------------------------------------
    all_subjects = sorted([
        d for d in os.listdir(args.data_dir)
        if d.startswith("s") and os.path.isdir(os.path.join(args.data_dir, d))
    ])

    # --- Build windows -------------------------------------------------------
    if args.cv_mode == "within_subject":
        print(f"\nCV mode: within_subject (val rep = {args.val_rep})")
        print(f"All subjects: {all_subjects}")
        print("\nBuilding within-subject split...")
        Xtr, Ytr, Xva, Yva = build_within_subject_split_seq(
            args.data_dir, all_subjects,
            ds=args.ds, win=args.win, stride=args.stride, min_len=args.min_len,
            val_rep=args.val_rep,
        )
    else:
        val_subjects   = args.val_subjects
        train_subjects = [s for s in all_subjects if s not in val_subjects]
        print(f"\nCV mode: cross_subject")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val   subjects ({len(val_subjects)}):   {val_subjects}")
        print("\nBuilding training windows...")
        Xtr, Ytr = build_split_seq(
            args.data_dir, train_subjects,
            ds=args.ds, win=args.win, stride=args.stride, min_len=args.min_len,
        )
        print("Building validation windows...")
        Xva, Yva = build_split_seq(
            args.data_dir, val_subjects,
            ds=args.ds, win=args.win, stride=args.stride, min_len=args.min_len,
        )

    # --- Normalize EMG -------------------------------------------------------
    mu, sd = safe_zscore_fit(Xtr)
    Xtr    = safe_zscore_apply(Xtr, mu, sd)
    Xva    = safe_zscore_apply(Xva, mu, sd)

    print(f"\nXtr: {Xtr.shape}  Ytr: {Ytr.shape}")
    print(f"Xva: {Xva.shape}  Yva: {Yva.shape}")

    # --- Datasets & loaders --------------------------------------------------
    train_ds = Emg2KptsSeqDataset(Xtr, Ytr)
    val_ds   = Emg2KptsSeqDataset(Xva, Yva)
    pin      = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    # --- Full-data training --------------------------------------------------
    print(f"\n=== Full training ({args.epochs} epochs) ===")
    c_in  = Xtr.shape[-1]
    model = make_model(c_in, args, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss_fn = nn.SmoothL1Loss()

    (train_losses, val_losses,
     train_mpjpes, val_mpjpes,
     grad_norms, best_mpjpe) = run_training(model, train_dl, val_dl, loss_fn, args, device)

    print(f"\nBest val MPJPE: {best_mpjpe:.4f}")
    print(f"Checkpoint saved: {args.save_path}")

    # --- Load best checkpoint for evaluation ---------------------------------
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- Generate plots ------------------------------------------------------
    print(f"\nGenerating plots -> {args.plot_dir}/")

    # 1. Training curves
    plot_training_curves(train_losses, val_losses, train_mpjpes, val_mpjpes, args.plot_dir)

    # 2. Per-joint MPJPE bar chart
    joint_errs = per_joint_mpjpe(model, val_dl, device)
    plot_per_joint_mpjpe(joint_errs, args.plot_dir)

    # 3. Per-finger MPJPE
    plot_per_finger_mpjpe(joint_errs, args.plot_dir)

    # 4. Error distribution (histogram + CDF)
    pw_errs = per_window_mpjpe(model, val_dl, device)
    plot_error_distribution(pw_errs, args.plot_dir)

    # 5. Prediction overlays for selected joints (FK keypoints)
    gts, preds = sample_predictions(model, val_ds, device, n_samples=3)
    for ji in args.overlay_joints:
        plot_prediction_overlays(gts, preds, args.plot_dir, joint_idx=ji)

    # 5b. Angle trajectory overlays (direct regression targets)
    ang_gts, ang_preds = sample_angle_predictions(model, val_ds, device, n_samples=3)
    plot_angle_overlays(ang_gts, ang_preds, args.plot_dir)

    # 6. Gradient norm
    plot_grad_norm(grad_norms, args.plot_dir)

    # 7. Joint error heatmap
    plot_joint_error_heatmap(joint_errs, args.plot_dir)

    # 8. Data ablation study
    print(f"\n=== Data ablation study (fractions={args.ablation_fractions}) ===")
    abl_mpjpes = []
    N_tr = len(train_ds)
    for frac in args.ablation_fractions:
        n_use = max(args.batch_size, int(frac * N_tr))
        idx   = np.random.choice(N_tr, size=n_use, replace=False)
        sub_ds = Subset(train_ds, idx)
        sub_dl = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=pin)

        abl_model  = make_model(c_in, args, device)
        abl_args   = argparse.Namespace(**vars(args))
        abl_args.epochs    = args.ablation_epochs
        abl_args.save_path = os.path.join(args.plot_dir, f"_abl_{frac}.pt")

        print(f"  Fraction {frac*100:.0f}%  (n={n_use})")
        _, _, _, abl_val_mpjpes, _, best = run_training(
            abl_model, sub_dl, val_dl, loss_fn, abl_args, device
        )
        abl_mpjpes.append(best)
        # clean up temp checkpoint
        if os.path.exists(abl_args.save_path):
            os.remove(abl_args.save_path)

    plot_data_ablation(args.ablation_fractions, abl_mpjpes, args.plot_dir)

    print(f"\nAll plots saved to: {args.plot_dir}/")
    print("\nPlot summary:")
    for f in sorted(os.listdir(args.plot_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()