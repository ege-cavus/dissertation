#!/usr/bin/env python3
"""
benchmark.py — Ninapro dataset benchmark over UART for nRF52840 firmware.

Streams EMG windows from the Ninapro dataset to the nRF52840 over serial,
collects per-inference timing and predicted angles, then computes:
  - Inference latency (us): mean, median, min, max, std
  - Throughput (inferences/sec)
  - MAE (degrees) vs ground-truth angles
  - Estimated energy per inference (uJ)
  - SRAM arena usage (reported by firmware)

Protocol (binary, little-endian):
  Host  → FW:  0xAA                             (SYNC)
  FW    → Host: 0x55 + arena_used(4B)           (ACK)
  Host  → FW:  0x01 + float32[64*10]            (window)
  FW    → Host: inference_us(4B) + float32[64*22] (result)
  ...repeat for each window...
  Host  → FW:  0xFF                             (END)
  FW    → Host: count(4B) + avg_us(4B) + min_us(4B) + max_us(4B) (summary)

Usage:
  python benchmark.py --port /dev/tty.usbmodem14101 --data-dir ~/Desktop/Ninapro_DB1
"""

import argparse
import glob
import os
import struct
import sys
import time

import numpy as np
import serial
from serial.tools import list_ports

WIN = 64
N_CH = 10
N_ANG = 22

INPUT_FLOATS = WIN * N_CH
OUTPUT_FLOATS = WIN * N_ANG
INPUT_BYTES = INPUT_FLOATS * 4
OUTPUT_BYTES = OUTPUT_FLOATS * 4

SYNC = 0xAA
ACK = 0x55
CMD_WINDOW = 0x01
CMD_END = 0xFF

NRF52840_VOLTAGE = 3.3        # V
NRF52840_ACTIVE_MA = 5.3      # mA at 64 MHz, CPU active with FPU


def load_ninapro_windows(data_dir, n_windows=100, ds=3, win=WIN, stride=32):
    """Load normalised EMG windows and ground-truth angles from Ninapro .mat files."""
    from scipy.io import loadmat

    mats = sorted(glob.glob(os.path.join(data_dir, "s1", "*.mat")))[:3]
    if not mats:
        print(f"ERROR: no .mat files found in {data_dir}/s1/")
        sys.exit(1)

    ROM_DEG = {
        "ThumbRotate": (0, 90), "ThumbMCP": (0, 90), "ThumbIP": (0, 90),
        "ThumbAb": (-20, 20),
        "IndexMCP": (0, 90), "IndexPIP": (0, 100), "IndexDIP": (0, 70),
        "MiddleMCP": (0, 90), "MiddlePIP": (0, 100), "MiddleDIP": (0, 70),
        "RingMCP": (0, 90), "RingPIP": (0, 100), "RingDIP": (0, 70),
        "PinkyMCP": (0, 90), "PinkyPIP": (0, 100), "PinkyDIP": (0, 70),
        "PinkyAb": (-15, 15),
        "WristFlex": (-70, 70), "WristAb": (-20, 30),
        "ForearmPron": (-80, 80),
        "IndexMiddleAb": (-15, 15), "Palmarch": (0, 45),
    }
    SENSOR_NAMES = [
        "ThumbRotate", "ThumbMCP", "ThumbIP", "ThumbAb",
        "IndexMCP", "IndexPIP", "IndexDIP",
        "MiddleMCP", "MiddlePIP", "MiddleDIP",
        "RingMCP", "RingPIP", "RingDIP",
        "PinkyMCP", "PinkyPIP", "PinkyDIP", "PinkyAb",
        "WristFlex", "WristAb", "ForearmPron",
        "IndexMiddleAb", "Palmarch",
    ]

    def glove_to_angles(glove_raw):
        x = np.nan_to_num(glove_raw, nan=0.0, posinf=0.0, neginf=0.0)
        lo = np.quantile(x, 0.02, axis=0)
        hi = np.quantile(x, 0.98, axis=0)
        denom = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
        u = np.clip((x - lo) / denom, 0.0, 1.0)
        deg = np.zeros_like(u)
        for j, name in enumerate(SENSOR_NAMES):
            a, b = ROM_DEG.get(name, (0, 90))
            deg[:, j] = a + u[:, j] * (b - a)
        return np.deg2rad(deg).astype(np.float32)

    windows, angles = [], []
    for mf in mats:
        try:
            m = loadmat(mf)
            emg = np.asarray(m["emg"], dtype=np.float32)
            glove = np.asarray(m["glove"], dtype=np.float32)
        except Exception as e:
            print(f"  WARN: skipping {mf}: {e}")
            continue

        emg = np.nan_to_num(emg[::ds], nan=0.0).astype(np.float32)
        ang = glove_to_angles(glove[::ds])
        mu = emg.mean(axis=0)
        sd = np.where(emg.std(axis=0) < 1e-6, 1.0, emg.std(axis=0))
        emg = ((emg - mu) / sd).astype(np.float32)

        for start in range(0, len(emg) - win + 1, stride):
            windows.append(emg[start:start + win])
            angles.append(ang[start:start + win])
        if len(windows) >= n_windows:
            break

    windows = np.stack(windows[:n_windows]).astype(np.float32)
    angles = np.stack(angles[:n_windows]).astype(np.float32)
    print(f"  Loaded {len(windows)} windows (shape {windows.shape})")
    return windows, angles


def run_tflite_eval(tflite_path, data_dir, n_windows=100):
    """Run Python TFLite on same Ninapro data — no board needed. Use to verify model vs firmware."""
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("ERROR: --tflite-eval requires tensorflow.")
        print("  Install: uv sync --extra quantise  (needs Python 3.10–3.13; no 3.14 wheels)")
        sys.exit(1)

    tflite_path = os.path.expanduser(tflite_path)
    if not os.path.isfile(tflite_path):
        print(f"ERROR: TFLite file not found: {tflite_path}")
        print("  Generate with: uv run python quantise.py --models cnnlstm_small --data-dir <path>")
        sys.exit(1)

    data_dir = os.path.expanduser(data_dir or "~/Desktop/Ninapro_DB1")
    print(f"Loading Ninapro data from {data_dir} …")
    windows, angles = load_ninapro_windows(data_dir, n_windows)
    print(f"Running TFLite eval: {tflite_path}  ({len(windows)} windows) …")

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    predicted = []
    for w in windows:
        w_cf = w.T[np.newaxis].astype(np.float32)  # channels-first (1, N_CH, WIN)
        interp.set_tensor(inp["index"], w_cf)
        interp.invoke()
        pred = interp.get_tensor(out["index"])[0]  # (N_ANG, WIN) or (WIN, N_ANG)
        if pred.shape == (N_ANG, WIN):
            pred = pred.T  # → (WIN, N_ANG)
        predicted.append(pred)
    predicted = np.stack(predicted)

    mae_rad = np.mean(np.abs(predicted - angles))
    mae_deg = np.rad2deg(mae_rad)
    print(f"\n  Python TFLite MAE: {mae_deg:.2f} deg  ({mae_rad:.4f} rad)")
    if predicted.size > 0:
        p0 = predicted[0]
        print(f"  First pred: min={p0.min():.4f}, max={p0.max():.4f}, mean={p0.mean():.4f}")
    if mae_deg > 100:
        print("  → Model likely broken (MAE > 100°). Check conversion or weights.")
    else:
        print("  → Model OK. If firmware MAE differs, fix firmware output handling.")


def read_u32(ser):
    data = ser.read(4)
    if len(data) < 4:
        raise TimeoutError("UART read timed out waiting for u32")
    return struct.unpack("<I", data)[0]


def run_benchmark(ser, windows, timeout_per_window=30.0, debug_uart=False):
    """Stream windows to firmware and collect results."""
    n = len(windows)

    # Board may reset when serial port is opened; retry SYNC until ACK or give up.
    # If the same UART is used for logs, drain log bytes until we see 0x55.
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    got_ack = False
    total_discarded = 0
    for attempt in range(1, 6):
        print(f"  Sending SYNC (attempt {attempt}/5) …")
        ser.write(bytes([SYNC]))
        ser.flush()
        deadline = time.monotonic() + 5.0
        discarded_this_attempt = 0
        while time.monotonic() < deadline:
            try:
                b = ser.read(1)
            except serial.SerialException as e:
                err = str(e)
                if "Device not configured" in err or "Errno 6" in err:
                    print("\n  ERROR: Serial device disconnected (board may have reset during flash).")
                    print("  Close this script, wait for the board to boot, then run the benchmark again.")
                    sys.exit(1)
                raise
            if len(b) == 0:
                continue
            if b[0] == ACK:
                got_ack = True
                break
            discarded_this_attempt += 1
        total_discarded += discarded_this_attempt
        if got_ack:
            break
        if attempt < 5:
            time.sleep(1.5)
            ser.reset_input_buffer()

    if not got_ack:
        print("  ERROR: no ACK (0x55) received within 5 s of SYNC.")
        if total_discarded > 0:
            print(f"  (Received {total_discarded} non-ACK byte(s) — wrong port or firmware not in benchmark mode.)")
        else:
            print("  (Received 0 bytes — board may still be booting or firmware not in benchmark mode.)")
        print("  1) Rebuild: cd firmware && ./build_benchmark.sh cnnlstm_small")
        print("  2) Or: west build -b nrf52840dk/nrf52840 --no-sysbuild -p . -- -DCONF_FILE=\"prj.conf;benchmark.conf\" -DMODEL=cnnlstm_small")
        print("  3) Flash: west flash --runner jlink")
        print("  4) Try --boot-wait 40 (cnnlstm_small init can be slow)")
        print("  4) Find UART0: python benchmark.py --port /dev/cu.usbmodemXXXX --probe (try each port).")
        sys.exit(1)

    arena_bytes = read_u32(ser)
    print(f"  ACK received — arena used: {arena_bytes} B ({arena_bytes / 1024:.1f} KB)")

    latencies_us = []
    predicted_angles = []

    for i in range(n):
        ser.write(bytes([CMD_WINDOW]))
        ser.write(windows[i].tobytes())

        old_timeout = ser.timeout
        ser.timeout = timeout_per_window
        try:
            inf_us = read_u32(ser)
            angle_bytes = ser.read(OUTPUT_BYTES)
            if len(angle_bytes) < OUTPUT_BYTES:
                print(f"  ERROR: short read on window {i}: {len(angle_bytes)}/{OUTPUT_BYTES}")
                break
        finally:
            ser.timeout = old_timeout

        angles_flat = np.frombuffer(angle_bytes, dtype=np.float32)
        predicted_angles.append(angles_flat.reshape(WIN, N_ANG))
        latencies_us.append(inf_us)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] last inference: {inf_us} us")

        # After first window, optionally request firmware output debug (shape + raw[8])
        if i == 0 and debug_uart:
            ser.write(bytes([0x02]))  # BENCH_DEBUG
            old_t = ser.timeout
            ser.timeout = 2.0
            try:
                rank_b = ser.read(4)
                dim1_b = ser.read(4)
                dim2_b = ser.read(4)
                raw_b = ser.read(8 * 4)
            finally:
                ser.timeout = old_t
            if len(rank_b) == 4 and len(raw_b) == 32:
                rank = struct.unpack("<I", rank_b)[0]
                dim1 = struct.unpack("<I", dim1_b)[0]
                dim2 = struct.unpack("<I", dim2_b)[0]
                raw_floats = np.frombuffer(raw_b, dtype=np.float32)
                print(f"  [DEBUG] Firmware output tensor: rank={rank} dim1={dim1} dim2={dim2}")
                print(f"  [DEBUG] Firmware raw out[0..7]= {raw_floats[:8]}")
            else:
                print(f"  [DEBUG] Firmware debug not available (rebuild firmware for BENCH_DEBUG)")

    ser.write(bytes([CMD_END]))

    try:
        fw_count = read_u32(ser)
        fw_avg = read_u32(ser)
        fw_min = read_u32(ser)
        fw_max = read_u32(ser)
        print(f"  FW summary: {fw_count} inferences, avg={fw_avg} us, "
              f"min={fw_min} us, max={fw_max} us")
    except TimeoutError:
        print("  WARN: no firmware summary received (timed out)")

    return arena_bytes, np.array(latencies_us), np.stack(predicted_angles)


def compute_metrics(latencies_us, predicted, ground_truth, arena_bytes):
    """Compute and print all benchmark metrics."""
    n = len(latencies_us)
    lat = latencies_us.astype(np.float64)

    mae_rad = np.mean(np.abs(predicted - ground_truth[:n]))
    mae_deg = np.rad2deg(mae_rad)

    mean_us = np.mean(lat)
    median_us = np.median(lat)
    std_us = np.std(lat)
    min_us = np.min(lat)
    max_us = np.max(lat)

    throughput = 1e6 / mean_us if mean_us > 0 else 0

    energy_uj = mean_us * NRF52840_VOLTAGE * NRF52840_ACTIVE_MA / 1000.0

    print()
    print("=" * 64)
    print("  BENCHMARK RESULTS")
    print("=" * 64)
    print(f"  Windows evaluated:   {n}")
    print(f"  Arena used:          {arena_bytes} B ({arena_bytes / 1024:.1f} KB)")
    print()
    print(f"  Inference latency:")
    print(f"    Mean:     {mean_us:10.0f} us  ({mean_us / 1000:.1f} ms)")
    print(f"    Median:   {median_us:10.0f} us")
    print(f"    Std:      {std_us:10.0f} us")
    print(f"    Min:      {min_us:10.0f} us")
    print(f"    Max:      {max_us:10.0f} us")
    print()
    print(f"  Throughput:          {throughput:.1f} inferences/sec")
    print(f"  Energy/inference:    {energy_uj:.1f} uJ  "
          f"(at {NRF52840_VOLTAGE}V, {NRF52840_ACTIVE_MA}mA active)")
    print()
    print(f"  Accuracy (MAE):     {mae_deg:.2f} deg  ({mae_rad:.4f} rad)")
    print("=" * 64)

    return {
        "n_windows": n,
        "arena_kb": arena_bytes / 1024,
        "mean_us": mean_us,
        "median_us": median_us,
        "std_us": std_us,
        "min_us": min_us,
        "max_us": max_us,
        "throughput": throughput,
        "energy_uj": energy_uj,
        "mae_deg": mae_deg,
        "mae_rad": mae_rad,
    }


def get_serial_port(port_arg):
    """Resolve serial port: use port_arg if it exists, else try to auto-detect USB serial."""
    if port_arg and os.path.exists(port_arg):
        return port_arg
    # Only consider USB serial (ignore Bluetooth-Incoming-Port, modems, etc.)
    candidates = []
    for p in list_ports.comports():
        if not p.device:
            continue
        dev = p.device
        desc = (p.description or "").lower()
        if "bluetooth" in desc or "bluetooth" in dev:
            continue
        if "usb" in dev.lower() or "usb" in desc or "nrf" in desc or "j-link" in desc or "jlink" in desc or "cdc" in desc:
            candidates.append(dev)
    for pattern in ("/dev/cu.usbmodem*", "/dev/tty.usbmodem*"):
        for path in glob.glob(pattern):
            if path not in candidates:
                candidates.append(path)
    candidates = sorted(set(candidates))
    if candidates:
        chosen = candidates[0]
        if not port_arg:
            print(f"  No --port given; using first available: {chosen}")
        else:
            print(f"  Port {port_arg} not found. Available: {candidates}")
            print(f"  Using: {chosen}")
        return chosen
    # No port found
    if port_arg:
        print(f"ERROR: Port {port_arg} not found.")
    else:
        print("ERROR: No USB serial port detected (Bluetooth and other non-USB ports are ignored).")
    print("  Plug in the nRF52840 DK USB cable, then run: ls /dev/cu.usbmodem*")
    print("  Use that path with: --port /dev/cu.usbmodemXXXXX")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ninapro dataset benchmark over UART")
    parser.add_argument("--port", default=None,
                        help="Serial port (e.g. /dev/tty.usbmodem14101). Auto-detect if omitted.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--data-dir", default=None,
                        help="Path to Ninapro DB1 directory (required unless --listen)")
    parser.add_argument("--n-windows", type=int, default=100,
                        help="Number of windows to benchmark")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-window UART timeout in seconds (BiLSTM/WHILE can be 30–90 s per inference)")
    parser.add_argument("--csv", default=None,
                        help="Optional: append results to CSV file")
    parser.add_argument("--model-name", default="unknown",
                        help="Model name for CSV output")
    parser.add_argument("--listen", action="store_true",
                        help="Listen on port for 15s and print received bytes (hex). Requires --port.")
    parser.add_argument("--probe", action="store_true",
                        help="Send SYNC (0xAA) and wait for ACK (0x55). Use to find which port is UART0. Requires --port.")
    parser.add_argument("--probe-all", action="store_true",
                        help="Try all USB serial ports, report which one returns ACK (for finding UART0).")
    parser.add_argument("--boot-wait", type=float, default=25.0,
                        help="Seconds to wait for board to boot before SYNC (default 25). Use 30+ for BiLSTM/Transformer/PET.")
    parser.add_argument("--debug", action="store_true",
                        help="Print first prediction stats (min/max/mean) to diagnose garbage output.")
    parser.add_argument("--tflite-eval", type=str, default=None, metavar="PATH",
                        help="Run Python TFLite on same data (no board). Path: tflite_models/cnnlstm_small_int8.tflite")
    args = parser.parse_args()

    if args.tflite_eval:
        if not args.data_dir:
            print("ERROR: --tflite-eval requires --data-dir")
            sys.exit(1)
        run_tflite_eval(args.tflite_eval, args.data_dir, args.n_windows)
        return

    if args.probe_all:
        print("Probing all USB serial ports for UART0 (SYNC→ACK) …")
        ports = []
        for p in list_ports.comports():
            if not p.device:
                continue
            dev = p.device
            desc = (p.description or "").lower()
            if "bluetooth" in desc or "bluetooth" in dev:
                continue
            if "usb" in dev.lower() or "usb" in desc or "nrf" in desc or "j-link" in desc or "jlink" in desc or "cdc" in desc:
                ports.append(dev)
        for path in glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/tty.usbmodem*"):
            if path not in ports:
                ports.append(path)
        ports = sorted(set(ports))
        if not ports:
            print("  No USB serial ports found. Plug in the nRF52840 DK.")
            sys.exit(1)
        found = None
        for port in ports:
            try:
                ser = serial.Serial(port, args.baud, timeout=2.0)
                ser.write(bytes([0xAA]))
                b = ser.read(1)
                ser.close()
                if b and b[0] == 0x55:
                    print(f"  {port}: ACK (0x55) — this is UART0. Use: --port {port}")
                    found = port
                else:
                    print(f"  {port}: no ACK (got {f'0x{b[0]:02X}' if b else 'nothing'})")
            except serial.SerialException as e:
                print(f"  {port}: {e}")
        if not found:
            print("\n  No port returned ACK. Ensure benchmark firmware is flashed and board is in 'waiting for SYNC'.")
        return

    if args.probe:
        if not args.port:
            print("ERROR: --probe requires --port")
            sys.exit(1)
        port = get_serial_port(args.port)
        print(f"Probing {port}: sending SYNC (0xAA), waiting 3 s for ACK …")
        try:
            ser = serial.Serial(port, args.baud, timeout=3.0)
        except serial.SerialException as e:
            print(f"ERROR: Could not open {port}: {e}")
            sys.exit(1)
        ser.write(bytes([0xAA]))
        b = ser.read(1)
        ser.close()
        if b and b[0] == 0x55:
            print("  Received 0x55 (ACK) — this port is UART0. Use it for benchmark: --port", port)
        else:
            print("  No ACK (got", f"0x{b[0]:02X}" if b else "nothing", ") — try the other /dev/cu.usbmodem* port.")
        return

    if args.listen:
        if not args.port:
            print("ERROR: --listen requires --port")
            sys.exit(1)
        port = get_serial_port(args.port)
        print(f"Listening on {port} at {args.baud} baud for 15 s. Reset the board or send SYNC from another terminal …")
        try:
            ser = serial.Serial(port, args.baud, timeout=0.5)
        except serial.SerialException as e:
            print(f"ERROR: Could not open {port}: {e}")
            sys.exit(1)
        time.sleep(5.0)
        deadline = time.monotonic() + 15.0
        n = 0
        while time.monotonic() < deadline:
            b = ser.read(1)
            if b:
                n += 1
                print(f"  [{n}] 0x{b[0]:02X} {repr(b)}")
        print(f"  Done. Received {n} byte(s).")
        ser.close()
        return

    if not args.data_dir:
        print("ERROR: --data-dir is required (or use --listen to test the port)")
        sys.exit(1)

    port = get_serial_port(args.port)

    print(f"Loading Ninapro data from {args.data_dir} …")
    windows, angles = load_ninapro_windows(args.data_dir, args.n_windows)

    print(f"Connecting to {port} at {args.baud} baud …")
    try:
        ser = serial.Serial(port, args.baud, timeout=5.0)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {port}: {e}")
        print("  Is the board connected? Try: ls /dev/cu.usbmodem*")
        sys.exit(1)
    # Opening the port can trigger a board reset (DTR). Wait for boot then send SYNC.
    # Do not close/reopen — a second open often resets the board again and then no ACK.
    total_wait = max(5.0, args.boot_wait)
    print(f"  Waiting for board to boot ({total_wait:.0f} s) …", end="", flush=True)
    step = 5.0
    waited = 0.0
    while waited < total_wait:
        time.sleep(min(step, total_wait - waited))
        waited += step
        if waited <= total_wait:
            print(f" {waited:.0f}s", end="", flush=True)
    print()

    print(f"Starting benchmark ({len(windows)} windows) …")
    arena_bytes, latencies, predicted = run_benchmark(
        ser, windows, args.timeout, debug_uart=args.debug
    )
    ser.close()

    metrics = compute_metrics(latencies, predicted, angles, arena_bytes)

    if args.debug and len(predicted) > 0:
        p0 = predicted[0]
        print("\n  [DEBUG] First prediction stats:")
        print(f"    shape: {p0.shape}, dtype: {p0.dtype}")
        print(f"    min={p0.min():.6f}, max={p0.max():.6f}, mean={p0.mean():.6f}, std={p0.std():.6f}")
        print(f"    first 8 values (row 0): {p0[0, :8]}")
        print(f"    ground truth row 0 (first 8): {angles[0, 0, :8]}")
        nan_count = np.isnan(p0).sum()
        inf_count = np.isinf(p0).sum()
        if nan_count or inf_count:
            print(f"    WARN: {nan_count} NaN, {inf_count} Inf")

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if write_header:
                f.write("model,n_windows,arena_kb,mean_us,median_us,std_us,"
                        "min_us,max_us,throughput,energy_uj,mae_deg\n")
            f.write(f"{args.model_name},{metrics['n_windows']},"
                    f"{metrics['arena_kb']:.1f},{metrics['mean_us']:.0f},"
                    f"{metrics['median_us']:.0f},{metrics['std_us']:.0f},"
                    f"{metrics['min_us']:.0f},{metrics['max_us']:.0f},"
                    f"{metrics['throughput']:.1f},{metrics['energy_uj']:.1f},"
                    f"{metrics['mae_deg']:.2f}\n")
        print(f"\nResults appended to {args.csv}")


if __name__ == "__main__":
    main()
