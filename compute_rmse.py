#!/usr/bin/env python3
"""Compute angle RMSE and NRMSE for all saved checkpoints on the validation split."""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from main import (
    build_within_subject_split_seq,
    Emg2KptsSeqDataset,
    safe_zscore_fit,
    safe_zscore_apply,
    EMG_TCN_SEQ,
)
from main import pred_to_angles as tcn_pred_to_angles
from other_models import (
    EMG_BiLSTM,
    EMG_CNN_BiLSTM,
    EMG_Transformer,
    build_within_subject_split,
    safe_zscore_fit as om_zscore_fit,
    safe_zscore_apply as om_zscore_apply,
    EMGAngleDataset,
)
from other_models import pred_to_angles as om_pred_to_angles
from pet import EMG_PET, Emg2KptsDataset
from pet import pred_to_angles as pet_pred_to_angles
from pet import (
    build_within_subject_split as pet_build_split,
    safe_zscore_fit as pet_zscore_fit,
    safe_zscore_apply as pet_zscore_apply,
)

# Config
DATA_DIR = os.path.expanduser("~/Desktop/Ninapro_DB1")
DEVICE   = torch.device("cpu")
WIN, STRIDE, DS, VAL_REP = 64, 16, 3, 6
BATCH    = 256

# ROM in degrees for each of the 22 sensors (same order as SENSOR_NAMES)
# Used to compute NRMSE denominator = ROM in radians
ROM_DEG = [
    50, 60, 60, 80,      # CMC1_F, CMC1_A, MCP1_F, IP1_F  (thumb)
    90, 40, 100,          # MCP2_F, MCP2-3_A, PIP2_F
    80,                   # DIP2_F
    90, 40, 100, 80,      # MCP3_F, MCP3-4_A, PIP3_F, DIP3_F
    90, 40, 100, 80,      # MCP4_F, MCP4-5_A, PIP4_F, DIP4_F
    30, 90, 100, 80,      # CMC5_F, MCP5_F, PIP5_F, DIP5_F
    30, 120,              # PALM_ARCH, WRIST_F
]
ROM_RAD = np.array([math.radians(d) for d in ROM_DEG], dtype=np.float32)

# Helpers

def load_subjects(data_dir):
    return sorted([
        d for d in os.listdir(data_dir)
        if d.startswith("s") and os.path.isdir(os.path.join(data_dir, d))
    ])


@torch.no_grad()
def compute_metrics(model, val_dl, pred_fn):
    """Returns angle RMSE in rad, deg, NRMSE, and per-joint RMSE in rad."""
    model.eval()
    preds, gts = [], []
    for x, y in val_dl:
        pa = pred_fn(model(x.to(DEVICE)))   # (B, win, 22)
        preds.append(pa.cpu().numpy())
        gts.append(y.numpy())

    pred = np.concatenate(preds).reshape(-1, 22)  # (N*win, 22)
    gt   = np.concatenate(gts).reshape(-1, 22)

    # Per-joint RMSE
    pj_rmse = np.sqrt(np.mean((pred - gt)**2, axis=0))   # (22,)

    # Global RMSE
    rmse_rad = float(np.sqrt(np.mean((pred - gt)**2)))
    rmse_deg = math.degrees(rmse_rad)

    # NRMSE: per-joint RMSE / per-joint ROM, then mean over joints
    nrmse = float(np.mean(pj_rmse / ROM_RAD))

    return rmse_rad, rmse_deg, nrmse, pj_rmse


def load_ckpt(model, path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model


# Build validation set  (within-subject, val_rep=6, all subjects)
# We build it once using main.py's pipeline; all models share the same
# val windows and normalisation statistics.
print("Loading subjects …")
subjects = load_subjects(DATA_DIR)

print("Building within-subject split (this takes ~1 min) …")
Xtr, Ytr, Xva, Yva = build_within_subject_split_seq(
    DATA_DIR, subjects,
    ds=DS, win=WIN, stride=STRIDE, val_rep=VAL_REP,
)

mu, sd = safe_zscore_fit(Xtr)
Xva_n  = safe_zscore_apply(Xva, mu, sd)

val_ds  = Emg2KptsSeqDataset(Xva_n, Yva)
val_dl  = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

C_IN = Xtr.shape[-1]   # 10 EMG channels
print(f"Val windows: {len(val_ds)}   C_IN={C_IN}\n")

# Models to evaluate
MODELS = [
    # (label, model_instance, ckpt_path, pred_fn)
    ("CNN-BiLSTM",
        EMG_CNN_BiLSTM(C_IN, cnn_channels=64, hidden=128, n_layers=2, dropout=0.2),
        "best_cnnlstm.pt", om_pred_to_angles),

    ("BiLSTM",
        EMG_BiLSTM(C_IN, hidden=128, n_layers=2, dropout=0.2),
        "best_bilstm.pt", om_pred_to_angles),

    ("Transformer",
        EMG_Transformer(C_IN, d_model=128, n_heads=4, n_layers=4,
                        patch_size=8, dropout=0.2, out_dim=22, win=64),
        "best_transformer.pt", om_pred_to_angles),

    ("TCN",
        EMG_TCN_SEQ(C_IN, hidden=128, levels=5, dropout=0.1, out_dim=22),
        "best_tcn.pt", tcn_pred_to_angles),

    ("PET",
        EMG_PET(C_IN, d_model=128, n_heads=4, n_branches=3, n_layers=2,
                mem_size=64, patch_size=8, ffn_mult=2, dropout=0.2,
                out_dim=22, win=64, ffn_act="gelu"),
        "best_pet.pt", pet_pred_to_angles),

    ("BiLSTM-small",
        EMG_BiLSTM(C_IN, hidden=64, n_layers=1, dropout=0.2),
        "best_bilstm_small.pt", om_pred_to_angles),

    ("PET-small",
        EMG_PET(C_IN, d_model=128, n_heads=4, n_branches=1, n_layers=2,
                mem_size=64, patch_size=8, ffn_mult=2, dropout=0.2,
                out_dim=22, win=64, ffn_act="relu"),
        "best_pet_small.pt", pet_pred_to_angles),
]

# Evaluate
results = {}
for label, model, ckpt, pred_fn in MODELS:
    model = load_ckpt(model, ckpt).to(DEVICE)
    rmse_rad, rmse_deg, nrmse, pj = compute_metrics(model, val_dl, pred_fn)
    results[label] = dict(rmse_rad=rmse_rad, rmse_deg=rmse_deg,
                          nrmse=nrmse, pj_rmse=pj)
    print(f"{label:<16}  RMSE={rmse_rad:.4f} rad  "
          f"({rmse_deg:.2f}°)   NRMSE={nrmse:.4f}")

# Summary table  (for pasting into the thesis)
print("\n" + "="*70)
print(f"{'Model':<16}  {'RMSE (rad)':>10}  {'RMSE (deg)':>10}  {'NRMSE':>8}")
print("-"*70)
for label, r in results.items():
    print(f"{label:<16}  {r['rmse_rad']:>10.4f}  {r['rmse_deg']:>9.2f}°  {r['nrmse']:>8.4f}")

print("\n--- Lin et al. 2025 reference (NinaPro DB2, 10 joints) ---")
print(f"{'sTransformer-EMFN':<16}  {'---':>10}  {10.77:>9.2f}°  {0.1158:>8.4f}")
print(f"{'TCN (Lin)':<16}  {'---':>10}  {12.24:>9.2f}°  {0.1390:>8.4f}")
print(f"{'LSTM (Lin)':<16}  {'---':>10}  {12.39:>9.2f}°  {0.1326:>8.4f}")
print("(DB2 uses 10 joints, RMS features, 2 kHz; not directly comparable)")

# Per-joint RMSE breakdown for best model
from main import SENSOR_NAMES
best = "CNN-BiLSTM"
print(f"\n--- Per-joint RMSE for {best} (radians) ---")
pj = results[best]["pj_rmse"]
for name, err in zip(SENSOR_NAMES, pj):
    print(f"  {name:<14}  {err:.4f} rad  ({math.degrees(err):.2f}°)")
