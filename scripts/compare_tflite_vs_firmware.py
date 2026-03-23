#!/usr/bin/env python3
"""Compare TFLite and firmware predictions on the same input window.

Saves tflite_pred_0.npy; use --tflite-only to skip the firmware step.
"""
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO)

# Reuse benchmark's data loading
from benchmark import load_ninapro_windows

WIN = 64
N_CH = 10
N_ANG = 22


def run_tflite_first_pred(tflite_path, data_dir, n_windows=1):
    """Run TFLite on first window, return prediction and ground truth."""
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow required. Install: uv sync --extra quantise")
        sys.exit(1)

    windows, angles = load_ninapro_windows(data_dir, n_windows)
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    w = windows[0]
    w_cf = w.T[np.newaxis].astype(np.float32)  # (1, N_CH, WIN)
    interp.set_tensor(inp["index"], w_cf)
    interp.invoke()
    pred = interp.get_tensor(out["index"])[0]
    if pred.shape == (N_ANG, WIN):
        pred = pred.T  # → (WIN, N_ANG)
    return pred, angles[0], w


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tflite", default=os.path.join(REPO, "tflite_models", "cnnlstm_small_int8.tflite"))
    p.add_argument("--data-dir", default=os.path.expanduser("~/Desktop/Ninapro_DB1"))
    p.add_argument("--tflite-only", action="store_true", help="Only run TFLite, save pred to tflite_pred_0.npy")
    p.add_argument("--firmware-pred", type=str, help="Path to firmware pred .npy (from benchmark --save-pred)")
    args = p.parse_args()

    tflite_path = os.path.expanduser(args.tflite)
    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isfile(tflite_path):
        print(f"ERROR: {tflite_path} not found")
        sys.exit(1)
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    print("Running TFLite on first window …")
    pred, gt, _ = run_tflite_first_pred(tflite_path, data_dir, 1)
    np.save("tflite_pred_0.npy", pred)
    print(f"  Saved tflite_pred_0.npy  shape={pred.shape}")
    print(f"  First row (8 vals): {pred[0, :8]}")
    print(f"  min={pred.min():.4f} max={pred.max():.4f} mean={pred.mean():.4f}")
    print(f"  Ground truth row 0 (8 vals): {gt[0, :8]}")

    if args.tflite_only:
        print("\nTo compare with firmware:")
        print("  1. Run: uv run python benchmark.py --port /dev/cu.usbmodemXXX --data-dir ... --n-windows 1 --debug")
        print("  2. The benchmark prints firmware first pred. Manually compare with tflite_pred_0.npy")
        print("  3. Or add --save-pred to benchmark to save firmware_pred_0.npy (not yet implemented)")
        return

    if args.firmware_pred and os.path.isfile(args.firmware_pred):
        fw = np.load(args.firmware_pred)
        print(f"\nFirmware pred shape={fw.shape}")
        print(f"  First row (8 vals): {fw[0, :8]}")
        print(f"  min={fw.min():.4f} max={fw.max():.4f} mean={fw.mean():.4f}")
        diff = np.abs(pred - fw)
        print(f"\n  Max |TFLite - Firmware| = {diff.max():.6f}")
        if diff.max() < 1e-5:
            print("  → Outputs match (within float precision)")
        else:
            print("  → Outputs differ — check firmware output copy or model")


if __name__ == "__main__":
    main()
