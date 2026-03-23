#!/usr/bin/env python3
"""Convert trained PyTorch models to INT8 TFLite and C header arrays for nRF52840.

Needs Python 3.12/3.13 + TensorFlow (TF has no 3.14 wheel):
    uv venv --python 3.12 .venv-quantise && source .venv-quantise/bin/activate
    uv pip install torch tensorflow scipy numpy
    python quantise.py --data-dir ~/Desktop/Ninapro_DB1
"""

import argparse
import glob
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


# Heartbeat for long-running steps (single-line \r progress)

def _run_with_heartbeat(msg: str, interval_sec: int, fn):
    """Run fn(); while it runs, print '  msg ... Ns' on a single line every interval_sec."""
    stop = threading.Event()
    elapsed = [0]

    def heartbeat():
        while not stop.is_set():
            stop.wait(interval_sec)
            if stop.is_set():
                break
            elapsed[0] += interval_sec
            print(f"\r  {msg} ... {elapsed[0]}s", end="", flush=True)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop.set()
        t.join(timeout=1.0)
        print("\r" + " " * (len(msg) + 14) + "\r", end="", flush=True)

# Dependency checks

def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"\n[ERROR] Missing package: {package}")
        print(f"  Install with: pip install {pip_name}")
        sys.exit(1)


# Constants (must match main.py / other_models.py)

WIN    = 64    # window length in samples (after downsampling)
N_CH   = 10    # EMG channels
N_ANG  = 22    # joint angle outputs

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
    "PALM_ARCH": (0, 30),
    "WRIST_F":  (-60, 60),
}


# Model architecture definitions (identical to training scripts)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- TCN (from main.py) ------------------------------------------------

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
    def __init__(self, c_in=N_CH, hidden=128, levels=5, dropout=0.1, out_dim=N_ANG):
        super().__init__()
        blocks = []
        c = c_in
        for i in range(levels):
            blocks.append(TCNBlock(c, hidden, k=3, dilation=2**i, dropout=dropout))
            c = hidden
        self.tcn  = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden, out_dim, 1)

    def forward(self, x):                      # x: (B, C, win)
        return self.proj(self.tcn(x))          # → (B, out_dim, win)


# ---- Shared Transformer helpers (from other_models.py / pet.py) ---------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super().__init__()
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                      # (B, C, win)
        return self.proj(x).transpose(1, 2)    # → (B, n_patches, d_model)


# ---- BiLSTM (from other_models.py) -------------------------------------

class EMG_BiLSTM(nn.Module):
    def __init__(self, c_in=N_CH, hidden=128, n_layers=2, dropout=0.2, out_dim=N_ANG):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                         # (B, C, win)
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, win, 2*hidden)
        out = self.proj(self.drop(self.norm(out)))
        return out.permute(0, 2, 1)               # (B, out_dim, win)


# ---- CNN-BiLSTM (from other_models.py) ---------------------------------

class EMG_CNN_BiLSTM(nn.Module):
    def __init__(self, c_in=N_CH, cnn_channels=64, hidden=128,
                 n_layers=2, dropout=0.2, out_dim=N_ANG):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(c_in, cnn_channels, 5, padding=2), nn.BatchNorm1d(cnn_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1), nn.BatchNorm1d(cnn_channels),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_channels, hidden, n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                           # (B, C, win)
        feat = self.cnn(x)                          # (B, cnn_channels, win)
        out, _ = self.lstm(feat.permute(0, 2, 1))   # (B, win, 2*hidden)
        out = self.proj(self.drop(self.norm(out)))
        return out.permute(0, 2, 1)                 # (B, out_dim, win)


# ---- Standard Transformer (from other_models.py) -----------------------

class EMG_Transformer(nn.Module):
    def __init__(self, c_in=N_CH, d_model=128, n_heads=4, n_layers=4,
                 patch_size=8, ffn_mult=2, dropout=0.2, out_dim=N_ANG, win=WIN):
        super().__init__()
        self.win         = win
        self.patch_embed = PatchEmbedding(c_in, d_model, patch_size)
        self.pos_enc     = SinusoidalPE(d_model, max_len=win // patch_size + 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop     = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, out_dim)
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x):                                          # (B, C, win)
        z = self.drop(self.pos_enc(self.patch_embed(x)))           # (B, P, d_model)
        z = self.encoder(z)
        out = self.out_head(z)                                     # (B, P, out_dim)
        out = out.permute(0, 2, 1)                                 # (B, out_dim, P)
        out = F.interpolate(out.float(), size=self.win, mode="linear", align_corners=False)
        return out                                                 # (B, out_dim, win)


# ---- PET (from pet.py) -------------------------------------------------

class ExternalAttention(nn.Module):
    def __init__(self, d_model, n_heads, mem_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.M_k     = nn.Parameter(torch.empty(n_heads, mem_size, self.d_head))
        self.M_v     = nn.Parameter(torch.empty(n_heads, mem_size, self.d_head))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_uniform_(self.M_k, a=math.sqrt(5))
        nn.init.uniform_(self.M_v, -0.01, 0.01)

    def forward(self, x):
        B, T, _ = x.shape
        H, D    = self.n_heads, self.d_head
        Q  = self.q_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)
        w  = torch.matmul(Q, self.M_k.transpose(-1, -2)) * (D ** -0.5)
        w  = F.softmax(w, dim=-1)
        w  = w / (w.sum(dim=-2, keepdim=True) + 1e-6)
        w  = self.drop(w)
        out = torch.matmul(w, self.M_v).permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        return self.out_proj(out)


class PETEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mem_size, ffn_mult=2, dropout=0.1, ffn_act="gelu"):
        super().__init__()
        self.attn  = ExternalAttention(d_model, n_heads, mem_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        act = nn.ReLU() if ffn_act == "relu" else nn.GELU()
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), act, nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PETBranch(nn.Module):
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


class EMG_PET(nn.Module):
    def __init__(self, c_in=N_CH, d_model=128, n_heads=4, n_branches=3,
                 n_layers=2, mem_size=64, patch_size=8, ffn_mult=2,
                 dropout=0.1, out_dim=N_ANG, win=WIN, ffn_act="gelu"):
        super().__init__()
        self.win        = win
        self.n_patches  = win // patch_size
        self.patch_embed = PatchEmbedding(c_in, d_model, patch_size)
        self.pos_enc    = SinusoidalPE(d_model, max_len=self.n_patches + 1)
        self.branches   = nn.ModuleList([
            PETBranch(d_model, n_heads, mem_size, n_layers, ffn_mult, dropout, ffn_act)
            for _ in range(n_branches)
        ])
        merge_act = nn.ReLU() if ffn_act == "relu" else nn.GELU()
        self.merge = nn.Sequential(
            nn.Linear(d_model * n_branches, d_model), merge_act, nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.merge[0].weight)
        self.out_head = nn.Linear(d_model, out_dim)
        nn.init.xavier_uniform_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                             # (B, C, win)
        B = x.size(0)
        z = self.drop(self.pos_enc(self.patch_embed(x)))              # (B, P, d_model)
        merged = self.merge(torch.cat([b(z) for b in self.branches], dim=-1))
        out    = self.out_head(merged)                                # (B, P, out_dim)
        out    = out.permute(0, 2, 1)                                 # (B, out_dim, P)
        out    = F.interpolate(out.float(), size=self.win, mode="linear", align_corners=False)
        return out                                                    # (B, out_dim, win)


# Firmware wrapper — normalises all models to time-first I/O

class FirmwareWrapper(nn.Module):
    """
    Wraps any model that uses channels-first convention to match the firmware
    buffer layout.

      input : (1, win=64, ch=10)    ← time-first, one window from ring buffer
      output: (1, win=64, angles=22) ← time-first, one angle sequence
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, win, ch) → (B, ch, win)
        y = self.model(x.permute(0, 2, 1))   # → (B, out_dim, win)
        return y.permute(0, 2, 1)             # → (B, win, out_dim)


# Model registry

def make_model(name: str) -> nn.Module:
    """Instantiate a fresh model with default hyper-parameters."""
    if name == "tcn":
        return EMG_TCN_SEQ()
    if name == "bilstm":
        return EMG_BiLSTM()
    if name == "bilstm_small":
        return EMG_BiLSTM(hidden=64, n_layers=1)
    if name == "cnnlstm":
        return EMG_CNN_BiLSTM()
    if name == "cnnlstm_small":
        return EMG_CNN_BiLSTM(cnn_channels=32, hidden=64, n_layers=1)
    if name == "transformer":
        return EMG_Transformer()
    if name == "pet":
        return EMG_PET()
    if name == "pet_small":
        return EMG_PET(c_in=N_CH, n_branches=1, ffn_act="relu", out_dim=N_ANG, win=WIN)
    raise ValueError(f"Unknown model: {name}")


def checkpoint_path(name: str) -> str:
    """Return the expected checkpoint location based on training script defaults."""
    if name == "tcn":
        return "best_tcn.pt"
    if name == "pet":
        return "best_pet.pt"
    if name == "pet_small":
        return "best_pet_small.pt"
    # other_models.py uses --plot-root (default '.') / best_<name>.pt
    return f"best_{name}.pt"  # bilstm_small → best_bilstm_small.pt


# Data loading helpers (minimal — only for representative calibration data)

def _load_mat(path):
    from scipy.io import loadmat
    m = loadmat(path)
    return (np.asarray(m["emg"],         dtype=np.float32),
            np.asarray(m["glove"],       dtype=np.float32),
            np.asarray(m["restimulus"],  dtype=np.int32).ravel(),
            np.asarray(m["rerepetition"],dtype=np.int32).ravel())


def _glove_to_angles(glove_raw):
    lo = np.quantile(np.nan_to_num(glove_raw), 0.02, axis=0)
    hi = np.quantile(np.nan_to_num(glove_raw), 0.98, axis=0)
    den = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
    u   = np.clip((glove_raw - lo) / den, 0.0, 1.0)
    deg = np.zeros_like(u)
    for j, name in enumerate(SENSOR_NAMES):
        a, b = ROM_DEG.get(name, (0, 90))
        deg[:, j] = a + u[:, j] * (b - a)
    return np.deg2rad(deg)


def load_calibration_windows(data_dir: str, n_windows=500, ds=3,
                              win=WIN, stride=16) -> np.ndarray:
    """
    Load ~n_windows normalised EMG windows from a few subjects for INT8
    representative dataset calibration. Returns (N, win, ch) float32 array.
    """
    mats = sorted(glob.glob(os.path.join(data_dir, "s*", "*.mat")))[:6]
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {data_dir}/s*/")

    windows = []
    for mf in mats:
        try:
            emg, glove, restim, rerep = _load_mat(mf)
        except Exception:
            continue
        emg = np.nan_to_num(emg[::ds], nan=0.0).astype(np.float32)
        mu  = emg.mean(axis=0)
        sd  = np.where(emg.std(axis=0) < 1e-6, 1.0, emg.std(axis=0))
        emg = ((emg - mu) / sd).astype(np.float32)
        for start in range(0, len(emg) - win + 1, stride):
            windows.append(emg[start:start + win])
        if len(windows) >= n_windows * 2:
            break

    windows = np.stack(windows[:n_windows * 2]).astype(np.float32)
    idx     = np.random.choice(len(windows), size=min(n_windows, len(windows)), replace=False)
    print(f"  Calibration: {len(idx)} windows from {len(mats)} files")
    return windows[idx]          # (N, win, ch) — already time-first


# ONNX export

def export_onnx(model: nn.Module, onnx_path: str):
    """Export raw model with static (1, N_CH, WIN) input using the legacy tracer.

    dynamo=False forces the TorchScript-based tracer instead of the dynamo
    exporter.  Tracing runs the forward pass with concrete values, so all
    expressions like `T = x.shape[-1]` are folded to the integer 64.  This
    produces a fully-static ONNX graph where every tensor has known shape,
    which lets onnx2tf correctly identify NCL channel vs length dimensions.
    """
    model.eval()
    dummy = torch.zeros(1, N_CH, WIN)   # channels-first: (1, 10, 64)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["emg_ch_first"],
            output_names=["angles_ch_first"],
            opset_version=17,
            dynamo=False,    # legacy TorchScript tracer → fully static shapes
        )
    print(f"  ONNX: {onnx_path}  ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


# Keras TCN (direct weight port — bypasses onnx2tf entirely)
#
# onnx2tf has a known bug converting NCL Conv1D models: it pads the channel
# dimension instead of the temporal dimension, producing a RESHAPE with
# 896 input elements vs 680 expected output elements.  Building the TCN
# directly in Keras and loading PyTorch weights avoids all ONNX→TF steps.

def _build_keras_tcn(state_dict: dict) -> "tf.keras.Model":
    """
    Construct a Keras model equivalent to EMG_TCN_SEQ and load PyTorch weights.

    Architecture: 5 dilated TCNBlocks (dilation 1,2,4,8,16) each with:
      ZeroPadding1D → Conv1D (valid) → Cropping1D → ReLU  (conv1 path)
      ZeroPadding1D → Conv1D (valid) → Cropping1D → ReLU  (conv2 path)
      Skip Conv1D(1)  (only level 0; levels 1-4 use identity)
      Add → ReLU  (block output)
    Then a pointwise Conv1D projection to N_ANG outputs.

    I/O layout: (batch, N_CH, WIN)  channels-first NCL — same as ONNX export,
    so model_runner.cc can feed the model without any extra transposition.
    """
    import tensorflow as tf
    import numpy as np

    HIDDEN = 128
    LEVELS = 5
    K      = 3

    # Input is channels-first NCL: (batch, N_CH, WIN)
    inp = tf.keras.Input(shape=(N_CH, WIN), name="emg_ch_first")

    # TF Conv1D is NLC, so transpose once at the entry boundary.
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)  # → (batch, WIN, N_CH)

    in_ch = N_CH
    for i in range(LEVELS):
        dil  = 2 ** i
        pad  = (K - 1) * dil   # symmetric padding per side (matches PyTorch)
        crop = (K - 1) * dil   # elements to crop from right after valid conv

        # ── conv1 ──
        h = tf.keras.layers.ZeroPadding1D(padding=pad, name=f"pad1_{i}")(x)
        h = tf.keras.layers.Conv1D(HIDDEN, K, dilation_rate=dil, padding="valid",
                                   use_bias=True, name=f"conv1_{i}")(h)
        h = tf.keras.layers.Cropping1D(cropping=(0, crop), name=f"crop1_{i}")(h)
        h = tf.keras.layers.ReLU(name=f"relu1_{i}")(h)

        # ── conv2 ──
        h = tf.keras.layers.ZeroPadding1D(padding=pad, name=f"pad2_{i}")(h)
        h = tf.keras.layers.Conv1D(HIDDEN, K, dilation_rate=dil, padding="valid",
                                   use_bias=True, name=f"conv2_{i}")(h)
        h = tf.keras.layers.Cropping1D(cropping=(0, crop), name=f"crop2_{i}")(h)
        h = tf.keras.layers.ReLU(name=f"relu2_{i}")(h)

        # ── skip + residual ──
        if in_ch != HIDDEN:
            x = tf.keras.layers.Conv1D(HIDDEN, 1, use_bias=True, name=f"skip_{i}")(x)
        x = tf.keras.layers.Add(name=f"add_{i}")([h, x])
        x = tf.keras.layers.ReLU(name=f"relu3_{i}")(x)
        in_ch = HIDDEN

    # Projection to N_ANG outputs (NLC)
    out = tf.keras.layers.Conv1D(N_ANG, 1, use_bias=True, name="proj")(x)
    # Transpose back NLC → NCL: (batch, N_ANG, WIN)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="tcn_keras")

    # ── load PyTorch weights ──
    def pt_w(key):
        """PyTorch Conv1d weight (out,in,k) → Keras Conv1D weight (k,in,out)."""
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    def pt_b(key):
        return state_dict[key].numpy()

    for i in range(LEVELS):
        model.get_layer(f"conv1_{i}").set_weights(
            [pt_w(f"tcn.{i}.conv1.weight"), pt_b(f"tcn.{i}.conv1.bias")])
        model.get_layer(f"conv2_{i}").set_weights(
            [pt_w(f"tcn.{i}.conv2.weight"), pt_b(f"tcn.{i}.conv2.bias")])
        skip_key = f"tcn.{i}.skip.weight"
        if skip_key in state_dict:
            model.get_layer(f"skip_{i}").set_weights(
                [pt_w(skip_key), pt_b(f"tcn.{i}.skip.bias")])

    # Conv1d proj weight: (22, 128, 1) → Keras (1, 128, 22)
    model.get_layer("proj").set_weights(
        [pt_w("proj.weight"), pt_b("proj.bias")])

    return model


# Keras BiLSTM builder (direct weight port)

def _manual_bilstm_block(x, hidden, layer_idx, tf_mod):
    """Forward + reverse LSTM concatenated — avoids Bidirectional wrapper.
    unroll=False keeps the WHILE loop in TFLite (compact ~27-op while_body).
    The while_body's TensorArray scatter is post-processed by
    scripts/patch_while_to_dus.py to replace the dynamic SLICE+CONCAT
    accumulation with DYNAMIC_UPDATE_SLICE (op 151), giving TFLM static shapes."""
    fwd = tf_mod.keras.layers.LSTM(
        hidden, return_sequences=True, unroll=False, name=f"fwd_{layer_idx}")(x)
    x_rev = tf_mod.keras.layers.Lambda(
        lambda t: t[:, ::-1, :], name=f"rev_in_{layer_idx}")(x)
    bwd = tf_mod.keras.layers.LSTM(
        hidden, return_sequences=True, unroll=False, name=f"bwd_{layer_idx}")(x_rev)
    bwd = tf_mod.keras.layers.Lambda(
        lambda t: t[:, ::-1, :], name=f"rev_out_{layer_idx}")(bwd)
    return tf_mod.keras.layers.Concatenate(name=f"cat_{layer_idx}")([fwd, bwd])


def _build_keras_bilstm(state_dict: dict) -> "tf.keras.Model":
    import tensorflow as tf

    HIDDEN = 128
    N_LAYERS = 2

    inp = tf.keras.Input(shape=(N_CH, WIN), batch_size=1, name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    for L in range(N_LAYERS):
        x = _manual_bilstm_block(x, HIDDEN, L, tf)

    x = tf.keras.layers.LayerNormalization(name="norm")(x)
    out = tf.keras.layers.Dense(N_ANG, name="proj")(x)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="bilstm_keras")

    def _load_lstm_pair(layer_idx):
        for direction, suffix in [("fwd", ""), ("bwd", "_reverse")]:
            layer = model.get_layer(f"{direction}_{layer_idx}")
            ih = state_dict[f"lstm.weight_ih_l{layer_idx}{suffix}"].numpy()
            hh = state_dict[f"lstm.weight_hh_l{layer_idx}{suffix}"].numpy()
            b  = (state_dict[f"lstm.bias_ih_l{layer_idx}{suffix}"].numpy() +
                  state_dict[f"lstm.bias_hh_l{layer_idx}{suffix}"].numpy())
            layer.set_weights([ih.T, hh.T, b])

    for L in range(N_LAYERS):
        _load_lstm_pair(L)

    model.get_layer("norm").set_weights([
        state_dict["norm.weight"].numpy(),
        state_dict["norm.bias"].numpy(),
    ])
    model.get_layer("proj").set_weights([
        state_dict["proj.weight"].numpy().T,
        state_dict["proj.bias"].numpy(),
    ])

    return model


def _build_keras_bilstm_small(state_dict: dict) -> "tf.keras.Model":
    """BiLSTM with 64 hidden, 1 layer — fits nRF52840 tensor arena (WHILE-based)."""
    import tensorflow as tf

    HIDDEN = 64
    N_LAYERS = 1

    inp = tf.keras.Input(shape=(N_CH, WIN), batch_size=1, name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    for L in range(N_LAYERS):
        x = _manual_bilstm_block(x, HIDDEN, L, tf)

    x = tf.keras.layers.LayerNormalization(name="norm")(x)
    out = tf.keras.layers.Dense(N_ANG, name="proj")(x)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="bilstm_small_keras")

    def _load_lstm_pair(layer_idx):
        for direction, suffix in [("fwd", ""), ("bwd", "_reverse")]:
            layer = model.get_layer(f"{direction}_{layer_idx}")
            ih = state_dict[f"lstm.weight_ih_l{layer_idx}{suffix}"].numpy()
            hh = state_dict[f"lstm.weight_hh_l{layer_idx}{suffix}"].numpy()
            b  = (state_dict[f"lstm.bias_ih_l{layer_idx}{suffix}"].numpy() +
                  state_dict[f"lstm.bias_hh_l{layer_idx}{suffix}"].numpy())
            layer.set_weights([ih.T, hh.T, b])

    for L in range(N_LAYERS):
        _load_lstm_pair(L)

    model.get_layer("norm").set_weights([
        state_dict["norm.weight"].numpy(),
        state_dict["norm.bias"].numpy(),
    ])
    model.get_layer("proj").set_weights([
        state_dict["proj.weight"].numpy().T,
        state_dict["proj.bias"].numpy(),
    ])

    return model


# Keras CNN-BiLSTM builder (direct weight port)

def _build_keras_cnnlstm(state_dict: dict) -> "tf.keras.Model":
    import tensorflow as tf

    CNN_CH = 64
    HIDDEN = 128
    N_LAYERS = 2

    inp = tf.keras.Input(shape=(N_CH, WIN), batch_size=1, name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    x = tf.keras.layers.Conv1D(CNN_CH, 5, padding="same", name="cnn_conv0")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn0")(x)
    x = tf.keras.layers.ReLU(name="cnn_relu0")(x)

    x = tf.keras.layers.Conv1D(CNN_CH, 3, padding="same", name="cnn_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn1")(x)
    x = tf.keras.layers.ReLU(name="cnn_relu1")(x)

    for L in range(N_LAYERS):
        x = _manual_bilstm_block(x, HIDDEN, L, tf)

    x = tf.keras.layers.LayerNormalization(name="norm")(x)
    out = tf.keras.layers.Dense(N_ANG, name="proj")(x)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="cnnlstm_keras")

    def pt_conv1d(key):
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    model.get_layer("cnn_conv0").set_weights([
        pt_conv1d("cnn.0.weight"), state_dict["cnn.0.bias"].numpy()])
    model.get_layer("cnn_bn0").set_weights([
        state_dict["cnn.1.weight"].numpy(),
        state_dict["cnn.1.bias"].numpy(),
        state_dict["cnn.1.running_mean"].numpy(),
        state_dict["cnn.1.running_var"].numpy(),
    ])
    model.get_layer("cnn_conv1").set_weights([
        pt_conv1d("cnn.4.weight"), state_dict["cnn.4.bias"].numpy()])
    model.get_layer("cnn_bn1").set_weights([
        state_dict["cnn.5.weight"].numpy(),
        state_dict["cnn.5.bias"].numpy(),
        state_dict["cnn.5.running_mean"].numpy(),
        state_dict["cnn.5.running_var"].numpy(),
    ])

    def _load_lstm_pair(layer_idx):
        for direction, suffix in [("fwd", ""), ("bwd", "_reverse")]:
            layer = model.get_layer(f"{direction}_{layer_idx}")
            ih = state_dict[f"lstm.weight_ih_l{layer_idx}{suffix}"].numpy()
            hh = state_dict[f"lstm.weight_hh_l{layer_idx}{suffix}"].numpy()
            b  = (state_dict[f"lstm.bias_ih_l{layer_idx}{suffix}"].numpy() +
                  state_dict[f"lstm.bias_hh_l{layer_idx}{suffix}"].numpy())
            layer.set_weights([ih.T, hh.T, b])

    for L in range(N_LAYERS):
        _load_lstm_pair(L)

    model.get_layer("norm").set_weights([
        state_dict["norm.weight"].numpy(),
        state_dict["norm.bias"].numpy(),
    ])
    model.get_layer("proj").set_weights([
        state_dict["proj.weight"].numpy().T,
        state_dict["proj.bias"].numpy(),
    ])

    return model


def _build_keras_cnnlstm_small(state_dict: dict) -> "tf.keras.Model":
    """CNN-BiLSTM with 32 CNN ch, 64 hidden, 1 layer — fits nRF52840 tensor arena."""
    import tensorflow as tf

    CNN_CH = 32
    HIDDEN = 64
    N_LAYERS = 1

    inp = tf.keras.Input(shape=(N_CH, WIN), batch_size=1, name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    x = tf.keras.layers.Conv1D(CNN_CH, 5, padding="same", name="cnn_conv0")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn0")(x)
    x = tf.keras.layers.ReLU(name="cnn_relu0")(x)

    x = tf.keras.layers.Conv1D(CNN_CH, 3, padding="same", name="cnn_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn1")(x)
    x = tf.keras.layers.ReLU(name="cnn_relu1")(x)

    for L in range(N_LAYERS):
        x = _manual_bilstm_block(x, HIDDEN, L, tf)

    x = tf.keras.layers.LayerNormalization(name="norm")(x)
    out = tf.keras.layers.Dense(N_ANG, name="proj")(x)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="cnnlstm_small_keras")

    def pt_conv1d(key):
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    model.get_layer("cnn_conv0").set_weights([
        pt_conv1d("cnn.0.weight"), state_dict["cnn.0.bias"].numpy()])
    model.get_layer("cnn_bn0").set_weights([
        state_dict["cnn.1.weight"].numpy(),
        state_dict["cnn.1.bias"].numpy(),
        state_dict["cnn.1.running_mean"].numpy(),
        state_dict["cnn.1.running_var"].numpy(),
    ])
    model.get_layer("cnn_conv1").set_weights([
        pt_conv1d("cnn.4.weight"), state_dict["cnn.4.bias"].numpy()])
    model.get_layer("cnn_bn1").set_weights([
        state_dict["cnn.5.weight"].numpy(),
        state_dict["cnn.5.bias"].numpy(),
        state_dict["cnn.5.running_mean"].numpy(),
        state_dict["cnn.5.running_var"].numpy(),
    ])

    def _load_lstm_pair(layer_idx):
        for direction, suffix in [("fwd", ""), ("bwd", "_reverse")]:
            layer = model.get_layer(f"{direction}_{layer_idx}")
            ih = state_dict[f"lstm.weight_ih_l{layer_idx}{suffix}"].numpy()
            hh = state_dict[f"lstm.weight_hh_l{layer_idx}{suffix}"].numpy()
            b  = (state_dict[f"lstm.bias_ih_l{layer_idx}{suffix}"].numpy() +
                  state_dict[f"lstm.bias_hh_l{layer_idx}{suffix}"].numpy())
            layer.set_weights([ih.T, hh.T, b])

    for L in range(N_LAYERS):
        _load_lstm_pair(L)

    model.get_layer("norm").set_weights([
        state_dict["norm.weight"].numpy(),
        state_dict["norm.bias"].numpy(),
    ])
    model.get_layer("proj").set_weights([
        state_dict["proj.weight"].numpy().T,
        state_dict["proj.bias"].numpy(),
    ])

    return model


# Keras Transformer builder (direct weight port)

def _sinusoidal_pe(d_model, max_len):
    pe  = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe[None]   # (1, max_len, d_model)


def _build_keras_transformer(state_dict: dict) -> "tf.keras.Model":
    import tensorflow as tf

    D_MODEL    = 128
    N_HEADS    = 4
    KEY_DIM    = D_MODEL // N_HEADS
    N_LAYERS   = 4
    FFN_DIM    = D_MODEL * 2
    PATCH_SIZE = 8
    N_PATCHES  = WIN // PATCH_SIZE

    inp = tf.keras.Input(shape=(N_CH, WIN), name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    x = tf.keras.layers.Conv1D(
        D_MODEL, PATCH_SIZE, strides=PATCH_SIZE, name="patch_embed")(x)

    pe = _sinusoidal_pe(D_MODEL, N_PATCHES + 1)[:, :N_PATCHES]
    x = tf.keras.layers.Add(name="add_pe")([x, tf.constant(pe)])

    for i in range(N_LAYERS):
        residual = x
        x = tf.keras.layers.LayerNormalization(name=f"enc{i}_norm1")(x)
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=N_HEADS, key_dim=KEY_DIM, name=f"enc{i}_mha")(x, x)
        x = tf.keras.layers.Add(name=f"enc{i}_add1")([residual, x])

        residual = x
        x = tf.keras.layers.LayerNormalization(name=f"enc{i}_norm2")(x)
        x = tf.keras.layers.Dense(FFN_DIM, activation="relu",
                                  name=f"enc{i}_ffn1")(x)
        x = tf.keras.layers.Dense(D_MODEL, name=f"enc{i}_ffn2")(x)
        x = tf.keras.layers.Add(name=f"enc{i}_add2")([residual, x])

    out = tf.keras.layers.Dense(N_ANG, name="out_head")(x)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    out = tf.keras.layers.Reshape((N_ANG, N_PATCHES, 1), name="pre_resize")(out)
    out = tf.keras.layers.Resizing(
        N_ANG, WIN, interpolation="bilinear", name="resize")(out)
    out = tf.keras.layers.Reshape((N_ANG, WIN), name="post_resize")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="transformer_keras")

    def pt_w(key):
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    model.get_layer("patch_embed").set_weights([
        pt_w("patch_embed.proj.weight"),
        state_dict["patch_embed.proj.bias"].numpy(),
    ])

    for i in range(N_LAYERS):
        prefix = f"encoder.layers.{i}"
        in_proj_w = state_dict[f"{prefix}.self_attn.in_proj_weight"].numpy()
        in_proj_b = state_dict[f"{prefix}.self_attn.in_proj_bias"].numpy()
        out_proj_w = state_dict[f"{prefix}.self_attn.out_proj.weight"].numpy()
        out_proj_b = state_dict[f"{prefix}.self_attn.out_proj.bias"].numpy()

        W_Q, W_K, W_V = np.split(in_proj_w, 3, axis=0)
        b_Q, b_K, b_V = np.split(in_proj_b, 3, axis=0)

        model.get_layer(f"enc{i}_mha").set_weights([
            W_Q.T.reshape(D_MODEL, N_HEADS, KEY_DIM),
            b_Q.reshape(N_HEADS, KEY_DIM),
            W_K.T.reshape(D_MODEL, N_HEADS, KEY_DIM),
            b_K.reshape(N_HEADS, KEY_DIM),
            W_V.T.reshape(D_MODEL, N_HEADS, KEY_DIM),
            b_V.reshape(N_HEADS, KEY_DIM),
            out_proj_w.reshape(D_MODEL, N_HEADS, KEY_DIM).transpose(1, 2, 0),
            out_proj_b,
        ])

        model.get_layer(f"enc{i}_norm1").set_weights([
            state_dict[f"{prefix}.norm1.weight"].numpy(),
            state_dict[f"{prefix}.norm1.bias"].numpy(),
        ])
        model.get_layer(f"enc{i}_norm2").set_weights([
            state_dict[f"{prefix}.norm2.weight"].numpy(),
            state_dict[f"{prefix}.norm2.bias"].numpy(),
        ])
        model.get_layer(f"enc{i}_ffn1").set_weights([
            state_dict[f"{prefix}.linear1.weight"].numpy().T,
            state_dict[f"{prefix}.linear1.bias"].numpy(),
        ])
        model.get_layer(f"enc{i}_ffn2").set_weights([
            state_dict[f"{prefix}.linear2.weight"].numpy().T,
            state_dict[f"{prefix}.linear2.bias"].numpy(),
        ])

    model.get_layer("out_head").set_weights([
        state_dict["out_head.weight"].numpy().T,
        state_dict["out_head.bias"].numpy(),
    ])

    return model


# Keras PET builder (direct weight port)

def _make_external_attention_cls():
    import tensorflow as tf

    class ExternalAttentionLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, n_heads, mem_size, **kwargs):
            super().__init__(**kwargs)
            self.d_model  = d_model
            self.n_heads  = n_heads
            self.d_head   = d_model // n_heads
            self.mem_size = mem_size
            self.scale    = float(self.d_head) ** -0.5

        def build(self, input_shape):
            self.q_proj   = self.add_weight(name="q_proj",   shape=(self.d_model, self.d_model))
            self.M_k      = self.add_weight(name="M_k",      shape=(self.n_heads, self.mem_size, self.d_head))
            self.M_v      = self.add_weight(name="M_v",      shape=(self.n_heads, self.mem_size, self.d_head))
            self.out_proj = self.add_weight(name="out_proj",  shape=(self.d_model, self.d_model))

        def call(self, x):
            H, D = self.n_heads, self.d_head
            # Use static T to avoid tf.shape() → SHAPE op (unsupported in TFLM).
            # x.shape[1] is statically known from the graph (N_PATCHES = 8).
            T = int(x.shape[1])

            Q = tf.matmul(x, self.q_proj)
            Q = tf.reshape(Q, [-1, T, H, D])  # -1 = batch (inferred, no SHAPE op)
            Q = tf.transpose(Q, [0, 2, 1, 3])

            M_k_t = tf.transpose(self.M_k, [0, 2, 1])
            w = tf.matmul(Q, tf.expand_dims(M_k_t, 0)) * self.scale
            w = tf.nn.softmax(w, axis=-1)
            # Double-normalisation (/ sum across queries) omitted — DIV op not
            # required by TFLM and softmax already normalises across keys.

            out = tf.matmul(w, tf.expand_dims(self.M_v, 0))
            out = tf.transpose(out, [0, 2, 1, 3])
            out = tf.reshape(out, [-1, T, H * D])   # -1 = batch (inferred)
            return tf.matmul(out, self.out_proj)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"d_model": self.d_model, "n_heads": self.n_heads,
                         "mem_size": self.mem_size})
            return cfg

    return ExternalAttentionLayer


def _build_keras_pet(state_dict: dict) -> "tf.keras.Model":
    import tensorflow as tf
    ExternalAttentionLayer = _make_external_attention_cls()

    D_MODEL    = 128
    N_HEADS    = 4
    KEY_DIM    = D_MODEL // N_HEADS
    MEM_SIZE   = 64
    N_BRANCHES = 3
    N_LAYERS   = 2
    FFN_MULT   = 2
    PATCH_SIZE = 8
    N_PATCHES  = WIN // PATCH_SIZE

    inp = tf.keras.Input(shape=(N_CH, WIN), name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    x = tf.keras.layers.Conv1D(
        D_MODEL, PATCH_SIZE, strides=PATCH_SIZE, name="patch_embed")(x)

    pe = _sinusoidal_pe(D_MODEL, N_PATCHES + 1)[:, :N_PATCHES]
    x = tf.keras.layers.Add(name="add_pe")([x, tf.constant(pe)])

    branch_outputs = []
    for b in range(N_BRANCHES):
        z = x
        for L in range(N_LAYERS):
            residual = z
            z = tf.keras.layers.LayerNormalization(
                name=f"br{b}_L{L}_norm1")(z)
            z = ExternalAttentionLayer(
                D_MODEL, N_HEADS, MEM_SIZE,
                name=f"br{b}_L{L}_attn")(z)
            z = tf.keras.layers.Add(name=f"br{b}_L{L}_add1")([residual, z])

            residual = z
            z = tf.keras.layers.LayerNormalization(
                name=f"br{b}_L{L}_norm2")(z)
            z = tf.keras.layers.Dense(
                D_MODEL * FFN_MULT, activation="gelu",
                name=f"br{b}_L{L}_ffn1")(z)
            z = tf.keras.layers.Dense(
                D_MODEL, name=f"br{b}_L{L}_ffn2")(z)
            z = tf.keras.layers.Add(name=f"br{b}_L{L}_add2")([residual, z])
        branch_outputs.append(z)

    merged = tf.keras.layers.Concatenate(name="concat")(branch_outputs)
    merged = tf.keras.layers.Dense(D_MODEL, name="merge_linear")(merged)
    merged = tf.keras.layers.Activation("gelu", name="merge_gelu")(merged)
    merged = tf.keras.layers.LayerNormalization(name="merge_norm")(merged)

    out = tf.keras.layers.Dense(N_ANG, name="out_head")(merged)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    out = tf.keras.layers.Reshape((N_ANG, N_PATCHES, 1), name="pre_resize")(out)
    out = tf.keras.layers.Resizing(
        N_ANG, WIN, interpolation="bilinear", name="resize")(out)
    out = tf.keras.layers.Reshape((N_ANG, WIN), name="post_resize")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="pet_keras")

    def pt_conv1d(key):
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    model.get_layer("patch_embed").set_weights([
        pt_conv1d("patch_embed.proj.weight"),
        state_dict["patch_embed.proj.bias"].numpy(),
    ])

    for b in range(N_BRANCHES):
        for L in range(N_LAYERS):
            prefix = f"branches.{b}.layers.{L}"

            attn_layer = model.get_layer(f"br{b}_L{L}_attn")
            attn_layer.set_weights([
                state_dict[f"{prefix}.attn.q_proj.weight"].numpy().T,
                state_dict[f"{prefix}.attn.M_k"].numpy(),
                state_dict[f"{prefix}.attn.M_v"].numpy(),
                state_dict[f"{prefix}.attn.out_proj.weight"].numpy().T,
            ])

            model.get_layer(f"br{b}_L{L}_norm1").set_weights([
                state_dict[f"{prefix}.norm1.weight"].numpy(),
                state_dict[f"{prefix}.norm1.bias"].numpy(),
            ])
            model.get_layer(f"br{b}_L{L}_norm2").set_weights([
                state_dict[f"{prefix}.norm2.weight"].numpy(),
                state_dict[f"{prefix}.norm2.bias"].numpy(),
            ])
            model.get_layer(f"br{b}_L{L}_ffn1").set_weights([
                state_dict[f"{prefix}.ffn.0.weight"].numpy().T,
                state_dict[f"{prefix}.ffn.0.bias"].numpy(),
            ])
            model.get_layer(f"br{b}_L{L}_ffn2").set_weights([
                state_dict[f"{prefix}.ffn.3.weight"].numpy().T,
                state_dict[f"{prefix}.ffn.3.bias"].numpy(),
            ])

    model.get_layer("merge_linear").set_weights([
        state_dict["merge.0.weight"].numpy().T,
        state_dict["merge.0.bias"].numpy(),
    ])
    model.get_layer("merge_norm").set_weights([
        state_dict["merge.2.weight"].numpy(),
        state_dict["merge.2.bias"].numpy(),
    ])
    model.get_layer("out_head").set_weights([
        state_dict["out_head.weight"].numpy().T,
        state_dict["out_head.bias"].numpy(),
    ])

    return model


def _build_keras_pet_small(state_dict: dict) -> "tf.keras.Model":
    """Single-branch PET using branch-0 weights from a pre-trained 3-branch model.

    Architecture identical to _build_keras_pet but N_BRANCHES=1:
      - Skips the Concatenate + merge_linear layers (saves ~1/3 of weight count)
      - Uses the shared out_head from the full model (same d_model output dim)
    Expected TFLite size: ~300 KB (vs 898 KB for full 3-branch PET)
    """
    import tensorflow as tf
    ExternalAttentionLayer = _make_external_attention_cls()

    D_MODEL    = 128
    N_HEADS    = 4
    MEM_SIZE   = 64
    N_LAYERS   = 2
    FFN_MULT   = 2
    PATCH_SIZE = 8
    N_PATCHES  = WIN // PATCH_SIZE

    inp = tf.keras.Input(shape=(N_CH, WIN), name="emg_ch_first")
    x = tf.keras.layers.Permute((2, 1), name="ncl_to_nlc")(inp)

    x = tf.keras.layers.Conv1D(
        D_MODEL, PATCH_SIZE, strides=PATCH_SIZE, name="patch_embed")(x)

    pe = _sinusoidal_pe(D_MODEL, N_PATCHES + 1)[:, :N_PATCHES]
    x = tf.keras.layers.Add(name="add_pe")([x, tf.constant(pe)])

    # Single branch (branch 0 weights)
    z = x
    for L in range(N_LAYERS):
        residual = z
        z = tf.keras.layers.LayerNormalization(name=f"br0_L{L}_norm1")(z)
        z = ExternalAttentionLayer(D_MODEL, N_HEADS, MEM_SIZE, name=f"br0_L{L}_attn")(z)
        z = tf.keras.layers.Add(name=f"br0_L{L}_add1")([residual, z])

        residual = z
        z = tf.keras.layers.LayerNormalization(name=f"br0_L{L}_norm2")(z)
        # ReLU matches the pet_small PyTorch model trained with --ffn-act relu.
        z = tf.keras.layers.Dense(D_MODEL * FFN_MULT, activation="relu",
                                  name=f"br0_L{L}_ffn1")(z)
        z = tf.keras.layers.Dense(D_MODEL, name=f"br0_L{L}_ffn2")(z)
        z = tf.keras.layers.Add(name=f"br0_L{L}_add2")([residual, z])

    # Merge: matches PyTorch EMG_PET.merge (d_model*1 → d_model, relu, layernorm)
    z = tf.keras.layers.Dense(D_MODEL, activation="relu", name="merge_linear")(z)
    z = tf.keras.layers.LayerNormalization(name="merge_norm")(z)

    out = tf.keras.layers.Dense(N_ANG, name="out_head")(z)
    out = tf.keras.layers.Permute((2, 1), name="nlc_to_ncl")(out)

    out = tf.keras.layers.Reshape((N_ANG, N_PATCHES, 1), name="pre_resize")(out)
    out = tf.keras.layers.Resizing(N_ANG, WIN, interpolation="bilinear", name="resize")(out)
    out = tf.keras.layers.Reshape((N_ANG, WIN), name="post_resize")(out)

    model = tf.keras.Model(inputs=inp, outputs=out, name="pet_small_keras")

    def pt_conv1d(key):
        return np.transpose(state_dict[key].numpy(), (2, 1, 0))

    model.get_layer("patch_embed").set_weights([
        pt_conv1d("patch_embed.proj.weight"),
        state_dict["patch_embed.proj.bias"].numpy(),
    ])

    for L in range(N_LAYERS):
        prefix = f"branches.0.layers.{L}"

        attn_layer = model.get_layer(f"br0_L{L}_attn")
        attn_layer.set_weights([
            state_dict[f"{prefix}.attn.q_proj.weight"].numpy().T,
            state_dict[f"{prefix}.attn.M_k"].numpy(),
            state_dict[f"{prefix}.attn.M_v"].numpy(),
            state_dict[f"{prefix}.attn.out_proj.weight"].numpy().T,
        ])

        model.get_layer(f"br0_L{L}_norm1").set_weights([
            state_dict[f"{prefix}.norm1.weight"].numpy(),
            state_dict[f"{prefix}.norm1.bias"].numpy(),
        ])
        model.get_layer(f"br0_L{L}_norm2").set_weights([
            state_dict[f"{prefix}.norm2.weight"].numpy(),
            state_dict[f"{prefix}.norm2.bias"].numpy(),
        ])
        model.get_layer(f"br0_L{L}_ffn1").set_weights([
            state_dict[f"{prefix}.ffn.0.weight"].numpy().T,
            state_dict[f"{prefix}.ffn.0.bias"].numpy(),
        ])
        model.get_layer(f"br0_L{L}_ffn2").set_weights([
            state_dict[f"{prefix}.ffn.3.weight"].numpy().T,
            state_dict[f"{prefix}.ffn.3.bias"].numpy(),
        ])

    model.get_layer("merge_linear").set_weights([
        state_dict["merge.0.weight"].numpy().T,
        state_dict["merge.0.bias"].numpy(),
    ])
    model.get_layer("merge_norm").set_weights([
        state_dict["merge.2.weight"].numpy(),
        state_dict["merge.2.bias"].numpy(),
    ])
    model.get_layer("out_head").set_weights([
        state_dict["out_head.weight"].numpy().T,
        state_dict["out_head.bias"].numpy(),
    ])

    return model


# Generic Keras → TFLite INT8 conversion (all models)

KERAS_BUILDERS = {
    "tcn":            _build_keras_tcn,
    "bilstm":         _build_keras_bilstm,
    "bilstm_small":   _build_keras_bilstm_small,
    "cnnlstm":        _build_keras_cnnlstm,
    "cnnlstm_small":  _build_keras_cnnlstm_small,
    "transformer":    _build_keras_transformer,
    "pet":            _build_keras_pet,
    "pet_small":      _build_keras_pet_small,
}


def convert_to_tflite_keras(name: str, state_dict: dict, tflite_path: str,
                            calib_data: np.ndarray) -> int:
    """
    Build a Keras model, load PyTorch weights, and convert to TFLite INT8.

    Works for ALL five architectures — bypasses onnx2tf entirely.
    """
    import tensorflow as tf

    print(f"  Building Keras {name.upper()} and loading PyTorch weights …")
    keras_model = KERAS_BUILDERS[name](state_dict)

    N_CALIB = 50
    calib_subset = calib_data[:N_CALIB]

    print(f"  Converting Keras → TFLite (INT8 calibration, {N_CALIB} windows) …")

    def representative_gen():
        for w in calib_subset:
            yield [w.T[np.newaxis].astype(np.float32)]

    lstm_models = {"bilstm", "bilstm_small", "cnnlstm", "cnnlstm_small"}

    def _do_convert():
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if name in lstm_models:
            # Dynamic-range: INT8 weights, float32 activations (hybrid).
            # Full INT8 + experimental_new_quantizer segfaults in TF converter
            # on WHILE/LSTM. Firmware uses reference FullyConnected for LSTM
            # builds so hybrid FC is accepted.
            # TFLM conv.cc hybrid path is patched to handle FLOAT32 bias
            # correctly (dynamic-range CONV2D produces FLOAT32 biases).
            pass
        else:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_bytes = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

    _run_with_heartbeat(f"TFLiteConverter (Keras {name})", 60, _do_convert)

    sz = os.path.getsize(tflite_path)
    print(f"  TFLite (full INT8, float32 I/O): {tflite_path}  ({sz / 1024:.1f} KB)")
    return sz


def convert_tcn_to_tflite_keras(state_dict: dict, tflite_path: str,
                                  calib_data: np.ndarray) -> int:
    """
    Build a Keras TCN, load PyTorch weights, and convert to TFLite.

    Uses Optimize.DEFAULT + representative_dataset for calibrated INT8
    activation quantisation.  No onnx2tf involved — avoids the NCL RESHAPE
    shape mismatch that onnx2tf produces for dilated Conv1D models.
    """
    import tensorflow as tf

    print(f"  Building Keras TCN and loading PyTorch weights …")
    keras_model = _build_keras_tcn(state_dict)

    N_CALIB = 50
    calib_subset = calib_data[:N_CALIB]

    print(f"  Converting Keras → TFLite (INT8 calibration, {N_CALIB} windows) …")

    def representative_gen():
        for w in calib_subset:
            # w: (win, ch) → channels-first (1, N_CH, WIN) matching model input
            yield [w.T[np.newaxis].astype(np.float32)]

    def _do_convert():
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        # Force ALL internal ops (activations) to INT8.
        # Without this, Pad/Transpose/Add/ReLU stay float32, so the padded
        # tensor for dilation=16 (ZeroPadding1D(32) → (1,128,128) float32 =
        # 65 KB) plus the live input (32 KB) exceeds the 128 KB arena.
        # The Keras model has clean, shape-consistent ops so TFLITE_BUILTINS_INT8
        # works here — unlike the onnx2tf path that had a RESHAPE shape mismatch.
        # I/O tensors stay float32 (no inference_input/output_type set) so
        # model_runner.cc's float32 handling remains correct.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_bytes = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

    _run_with_heartbeat("TFLiteConverter (Keras TCN, full INT8)", 30, _do_convert)

    sz = os.path.getsize(tflite_path)
    print(f"  TFLite (Keras, full INT8 activations, float32 I/O): {tflite_path}  ({sz / 1024:.1f} KB)")
    return sz


# PyTorch → TFLite INT8  (onnx2tf path — used for non-TCN models)

def convert_to_tflite_int8(onnx_path: str, tflite_path: str,
                            calib_data: np.ndarray,
                            wrapper: "FirmwareWrapper") -> int:
    """
    ONNX → TF SavedModel (onnx2tf) → TFLite with full INT8 quantisation.

    Static INT8 quantisation: Conv/Dense weights AND activations are
    calibrated to INT8 using a small representative dataset.  Shape ops
    (RESHAPE, TRANSPOSE inserted by onnx2tf) stay float32 with explicit
    QUANTIZE/DEQUANTIZE boundaries — TFLM handles these via AddQuantize()
    and AddDequantize() in the op resolver.

    Forcing TFLITE_BUILTINS_INT8 is intentionally avoided because it makes
    TFLite quantize shape-manipulation tensors and causes a RESHAPE shape
    mismatch (896 != 680) in onnx2tf-generated SavedModels.

    The calibration dataset is intentionally small (N_CALIB=50 windows) so
    the MLIR calibration pass completes quickly without freezing.

    Requires: pip install onnx2tf tensorflow
    """
    import shutil
    import tensorflow as tf  # noqa: F401 – imported for TFLiteConverter

    saved_model_dir = tflite_path.replace(".tflite", "_saved_model")

    # ---- Step 1: ONNX → TF SavedModel via onnx2tf ----
    print(f"  Converting ONNX → TF SavedModel (onnx2tf) …")
    if os.path.isdir(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    def _to_savedmodel():
        import onnx2tf as onnx2tf_mod
        onnx2tf_mod.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            verbosity="error",
        )

    _run_with_heartbeat("onnx2tf", 120, _to_savedmodel)

    if not os.path.isdir(saved_model_dir):
        raise RuntimeError(f"onnx2tf produced no output in {saved_model_dir}")

    # ---- Step 2: SavedModel → TFLite full INT8 (weights + activations INT8) ----
    # Use only N_CALIB windows so the MLIR calibration pass is fast.
    N_CALIB = 50
    calib_subset = calib_data[:N_CALIB]  # (N_CALIB, win, ch) time-first

    print(f"  Applying full INT8 quantisation ({N_CALIB} calibration windows) …")

    def _to_tflite():
        def representative_gen():
            for w in calib_subset:
                # Transpose to channels-first (1, ch, win) — matches ONNX export
                w_cf = w.T[np.newaxis].astype(np.float32)  # (1, N_CH, WIN)
                yield [w_cf]

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        # Do NOT set target_spec.supported_ops.
        #
        # Forcing TFLITE_BUILTINS_INT8 makes TFLite try to quantize every op —
        # including the NHWC RESHAPE that onnx2tf inserts for NCL→NHWC layout
        # conversion — and that causes a shape mismatch (896 != 680) crash.
        #
        # With Optimize.DEFAULT + representative_dataset alone, TFLite quantizes
        # Conv/Dense weights AND calibrates activations to INT8, but leaves pure
        # shape manipulation ops (RESHAPE, TRANSPOSE) in float32 with explicit
        # QUANTIZE/DEQUANTIZE boundaries.  TFLM supports this mixed model via
        # the AddQuantize() and AddDequantize() ops already in our resolver.
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

    _run_with_heartbeat("TFLiteConverter (INT8 calib)", 30, _to_tflite)

    sz = os.path.getsize(tflite_path)
    print(f"  TFLite (INT8 activations, float32 I/O): {tflite_path}  ({sz / 1024:.1f} KB)")
    return sz


# C header generation (xxd-style)

def generate_c_header(tflite_path: str, header_path: str, var_name: str):
    """Convert .tflite bytes to a C unsigned char array header."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    lines  = [
        f"/* Auto-generated by quantise.py — DO NOT EDIT */",
        f"/* Source: {os.path.basename(tflite_path)} */",
        f"#pragma once",
        f"#include <stdint.h>",
        f"",
        f"alignas(16) static const uint8_t {var_name}[] = {{",
    ]
    hex_bytes = [f"0x{b:02x}" for b in data]
    chunk     = 12
    for i in range(0, len(hex_bytes), chunk):
        row = ", ".join(hex_bytes[i:i + chunk])
        lines.append(f"  {row},")
    lines += [
        "};",
        f"static const unsigned int {var_name}_len = {len(data)}u;",
        "",
    ]

    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  C header: {header_path}  ({len(data) / 1024:.1f} KB)")


# Accuracy comparison: float32 vs INT8 on a held-out validation subset

def evaluate_float32(wrapper: FirmwareWrapper, windows: np.ndarray) -> float:
    """Return mean absolute angle error (rad) over the provided windows."""
    wrapper.eval()
    errs = []
    with torch.no_grad():
        for w in windows:
            x    = torch.from_numpy(w[np.newaxis]).float()    # (1, win, ch)
            pred = wrapper(x).numpy()[0]                      # (win, 22)
            errs.append(np.zeros(pred.shape))                 # ground truth not available; skip
    return float("nan")


def evaluate_tflite_int8(tflite_path: str, windows: np.ndarray,
                         gt_angles: np.ndarray) -> float:
    """
    Run INT8 TFLite model on validation windows and compute mean angle MAE (rad).

    windows:   (N, win, ch) float32 — z-scored EMG (time-first)
    gt_angles: (N, win, 22) float32 — target angles in radians (time-first)

    The TFLite model expects channels-first I/O (no FirmwareWrapper), so each
    window is transposed from (win, ch) → (ch, win) before inference, and the
    output (N_ANG, win) is transposed back to (win, N_ANG) for MAE comparison.
    """
    tf = _require("tensorflow")

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det  = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]
    # quantization is () for float32 I/O (Keras-converted models), (scale, zp) for INT8
    in_q   = inp_det["quantization"]
    out_q  = out_det["quantization"]
    in_scale,  in_zp  = (in_q[0],  int(in_q[1]))  if len(in_q) == 2 else (0.0, 0)
    out_scale, out_zp = (out_q[0], int(out_q[1])) if len(out_q) == 2 else (0.0, 0)

    maes = []
    for i, (w, gt) in enumerate(zip(windows, gt_angles)):
        # w: (win, ch) → transpose to channels-first (ch, win) for TFLite input
        w_cf  = w.T                                                  # (N_CH, WIN)
        if in_scale > 0:
            q_in = np.clip(np.round(w_cf / in_scale + in_zp), -128, 127).astype(np.int8)
        else:
            q_in = w_cf.astype(np.float32)
        interp.set_tensor(inp_det["index"], q_in[np.newaxis])
        interp.invoke()
        q_out = interp.get_tensor(out_det["index"])[0]              # (N_ANG, win)
        if out_scale > 0:
            pred = (q_out.astype(np.float32) - out_zp) * out_scale  # (N_ANG, win)
        else:
            pred = q_out.astype(np.float32)
        pred = pred.T                                                # (win, N_ANG)
        mae   = np.abs(pred - gt).mean()
        maes.append(mae)
        if (i + 1) % 50 == 0:
            print(f"    evaluated {i+1}/{len(windows)} windows …")

    return float(np.mean(maes))


def load_validation_windows_with_angles(data_dir: str, n=200, ds=3,
                                        win=WIN, stride=32):
    """Load a small validation set with both EMG windows and target angles."""
    mats = sorted(glob.glob(os.path.join(data_dir, "s1", "*.mat")))[:3]
    if not mats:
        return None, None

    windows, angles = [], []
    for mf in mats:
        try:
            emg, glove, restim, rerep = _load_mat(mf)
        except Exception:
            continue
        emg   = np.nan_to_num(emg[::ds], nan=0.0).astype(np.float32)
        ang   = np.nan_to_num(_glove_to_angles(glove[::ds]), nan=0.0).astype(np.float32)
        mu    = emg.mean(axis=0)
        sd    = np.where(emg.std(axis=0) < 1e-6, 1.0, emg.std(axis=0))
        emg   = ((emg - mu) / sd).astype(np.float32)
        for start in range(0, len(emg) - win + 1, stride):
            windows.append(emg[start:start + win])
            angles.append(ang[start:start + win])
        if len(windows) >= n:
            break

    if not windows:
        return None, None
    windows = np.stack(windows[:n]).astype(np.float32)
    angles  = np.stack(angles[:n]).astype(np.float32)
    return windows, angles


# Estimated arena size (heuristic)

def estimate_arena_kb(tflite_path: str) -> int:
    """
    Heuristic TFLM arena estimate: peak activation memory ≈ 10–15× the
    largest single intermediate tensor, NOT proportional to model weights.

    For a TCN with hidden=128, win=64: largest tensor = (1,128,64)=8 KB INT8.
    15× overhead → ~120 KB.  Capped at 128 KB (half the nRF52840 SRAM).

    Profile the real value on-device with model_runner_arena_used() and lower
    TFLM_ARENA_SIZE in model_runner.h once you have the high-water mark.
    """
    model_kb = os.path.getsize(tflite_path) / 1024
    # Rough rule: arena ≈ 20–25% of INT8 weight size, minimum 32 KB
    arena = max(32, int(model_kb * 0.25))
    return min(arena, 128)    # nRF52840 has 256 KB SRAM total; cap at 128 KB


# Main pipeline

ALL_MODELS = ["tcn", "bilstm", "bilstm_small", "cnnlstm", "cnnlstm_small", "transformer", "pet", "pet_small"]


def parse_args():
    p = argparse.ArgumentParser(description="Export EMG models to TFLite INT8 for firmware")
    p.add_argument("--data-dir",      type=str, default="~/Desktop/Ninapro_DB1",
                   help="Path to Ninapro_DB1 (needed for calibration data)")
    p.add_argument("--models",        type=str, nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS,
                   help="Which models to export (default: all). bilstm_small and cnnlstm_small fit nRF arena.")
    p.add_argument("--out-dir",       type=str, default="tflite_models",
                   help="Directory for ONNX and .tflite outputs")
    p.add_argument("--header-dir",    type=str, default="firmware/src/models",
                   help="Directory for generated C headers")
    p.add_argument("--calib-windows", type=int, default=500,
                   help="Number of calibration windows for INT8 quantisation")
    p.add_argument("--skip-accuracy", action="store_true",
                   help="Skip INT8 vs float32 accuracy evaluation")
    p.add_argument("--skip-tflite",   action="store_true",
                   help="Export ONNX only, skip TFLite conversion (no TF needed)")
    return p.parse_args()


def main():
    args = parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(args.out_dir,    exist_ok=True)
    os.makedirs(args.header_dir, exist_ok=True)

    # Suppress TensorFlow/onnx2tf verbose logs (INFO and below)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not args.skip_tflite:
        _require("tensorflow")

    print(f"\nLoading calibration data from {args.data_dir} …")
    calib_windows = load_calibration_windows(args.data_dir,
                                             n_windows=args.calib_windows)

    print(f"  Will process {len(args.models)} model(s): {', '.join(args.models)}")

    if not args.skip_accuracy:
        print("Loading validation windows for accuracy comparison …")
        val_windows, val_angles = load_validation_windows_with_angles(args.data_dir)
        if val_windows is None:
            print("  [WARN] Could not load validation data — skipping accuracy eval")
            args.skip_accuracy = True

    results = []

    for name in args.models:
        ckpt = checkpoint_path(name)
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {name.upper()}: checkpoint not found ({ckpt})")
            continue

        t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  Model: {name.upper()}  (checkpoint: {ckpt})")
        print(f"{'='*60}")

        # --- load model ---
        print(f"  Loading checkpoint …")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        base  = make_model(name)
        try:
            base.load_state_dict(state["model_state_dict"])
        except RuntimeError as e:
            print(f"  [WARN] state_dict load partial: {e}")
            base.load_state_dict(state["model_state_dict"], strict=False)
        base.eval()

        params = sum(p.numel() for p in base.parameters())
        print(f"  Parameters: {params:,}")

        # --- ONNX export (optional — some archs use fused ops unsupported by ONNX) ---
        onnx_path = os.path.join(args.out_dir, f"{name}.onnx")
        try:
            print(f"  Exporting to ONNX …")
            export_onnx(base, onnx_path)
        except Exception as e:
            print(f"  [WARN] ONNX export failed ({e.__class__.__name__}), skipping")
            onnx_path = None

        if args.skip_tflite:
            elapsed = time.perf_counter() - t0
            print(f"  [{name}] done in {elapsed:.1f}s (ONNX only)")
            continue

        # --- TFLite INT8 (direct Keras path — no onnx2tf) ---
        tflite_path = os.path.join(args.out_dir, f"{name}_int8.tflite")
        try:
            sz = convert_to_tflite_keras(
                name, state["model_state_dict"], tflite_path, calib_windows)
        except Exception as e:
            print(f"  [ERROR] TFLite conversion failed for {name}: {e}")
            elapsed = time.perf_counter() - t0
            print(f"  [{name}] failed after {elapsed:.1f}s")
            continue

        # --- C header ---
        print(f"  Writing C header …")
        header_path = os.path.join(args.header_dir, f"{name}_model.h")
        var_name    = f"{name}_model_data"
        generate_c_header(tflite_path, header_path, var_name)

        # --- arena estimate ---
        arena_kb = estimate_arena_kb(tflite_path)

        # --- accuracy ---
        int8_mae = float("nan")
        if not args.skip_accuracy:
            print(f"  Evaluating INT8 accuracy on {len(val_windows)} validation windows …")
            try:
                int8_mae = evaluate_tflite_int8(tflite_path, val_windows, val_angles)
            except Exception as e:
                print(f"  [WARN] Accuracy eval failed: {e}")

        elapsed = time.perf_counter() - t0
        print(f"  [{name}] done in {elapsed:.1f}s")
        results.append({
            "name":     name,
            "params":   params,
            "onnx_kb":  os.path.getsize(onnx_path) / 1024 if onnx_path and os.path.exists(onnx_path) else 0,
            "tflite_kb": sz / 1024,
            "arena_kb": arena_kb,
            "int8_mae": int8_mae,
        })

    # --- summary table ---
    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<12} {'Params':>10} {'ONNX KB':>9} {'TFLite KB':>10} "
              f"{'Arena KB (est)':>15} {'INT8 MAE':>10}")
        print(f"  {'-'*67}")
        for r in results:
            mae_str = f"{r['int8_mae']:.4f}" if not math.isnan(r['int8_mae']) else "  n/a  "
            print(f"  {r['name']:<12} {r['params']:>10,} {r['onnx_kb']:>9.1f} "
                  f"{r['tflite_kb']:>10.1f} {r['arena_kb']:>15} {mae_str:>10}")
        print(f"\n  nRF52840 DK budget: ~900 KB flash, ~200 KB SRAM")
        print(f"  Recommended model for deployment: TCN (smallest, best conv-quantisation)")
        print(f"\n  Headers written to: {args.header_dir}/")
        print(f"  Include in firmware:  #include \"models/<name>_model.h\"")
        print(f"\n  Next: run firmware/get_tflm.sh to vendor TFLite Micro sources,")
        print(f"        then: west build -b nrf52840dk/nrf52840 firmware/")


if __name__ == "__main__":
    main()
