#!/usr/bin/env bash
# Build and flash firmware in BENCHMARK mode (waits for SYNC on UART for benchmark.py).
# Usage: ./build_benchmark.sh [tcn|bilstm_small|cnnlstm_small|...]   (default: tcn)
#
# Requires nRF Connect SDK in PATH (west, Zephyr). Source it first, e.g.:
#   source ~/ncs/zephyr/zephyr-env.sh
# or open an "nRF Connect for VS Code" terminal.
set -e
cd "$(dirname "$0")"
# Ensure nRF Connect SDK environment (west + Zephyr)
if ! command -v west &>/dev/null || [ -z "${ZEPHYR_BASE:-}" ]; then
  NCS_ROOT="${NCS_ROOT:-/opt/nordic/ncs}"
  ZEPHYR_ENV="$NCS_ROOT/v3.2.4/zephyr/zephyr-env.sh"
  TOOLCHAIN_BIN="$NCS_ROOT/toolchains/185bb0e3b6/bin"
  if [ -f "$ZEPHYR_ENV" ]; then
    # shellcheck source=/dev/null
    . "$ZEPHYR_ENV"
  fi
  if [ -d "$TOOLCHAIN_BIN" ]; then
    export PATH="$TOOLCHAIN_BIN:$PATH"
  fi
  ZEPHYR_SDK="$NCS_ROOT/toolchains/185bb0e3b6/opt/zephyr-sdk"
  if [ -d "$ZEPHYR_SDK" ]; then
    export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK"
  fi
fi
if ! command -v west &>/dev/null; then
  echo "ERROR: west not found. Source nRF Connect SDK first:"
  echo "  source /opt/nordic/ncs/v3.2.4/zephyr/zephyr-env.sh"
  echo "  export PATH=/opt/nordic/ncs/toolchains/185bb0e3b6/bin:\$PATH"
  echo "Or use the 'nRF Connect for VS Code' terminal."
  exit 1
fi
MODEL="${1:-tcn}"
FIRMWARE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$FIRMWARE_DIR/build"
echo "=== Building benchmark firmware (MODEL=${MODEL}, BENCHMARK=ON) ==="
echo ""
# West must run from NCS workspace; build app in firmware/build
NCS_WORKSPACE="${NCS_ROOT:-/opt/nordic/ncs}/v3.2.4"
if [ ! -d "$NCS_WORKSPACE/.west" ]; then
  echo "ERROR: NCS workspace not found at $NCS_WORKSPACE"
  echo "Set NCS_ROOT if your nRF Connect SDK is elsewhere."
  exit 1
fi
export MODEL
(cd "$NCS_WORKSPACE" && west build -b nrf52840dk/nrf52840 --no-sysbuild -p always -d "$BUILD_DIR" "$FIRMWARE_DIR" -- -DCONF_FILE="prj.conf;benchmark.conf" -DMODEL=${MODEL})
echo ""
echo "Verify above: 'Selected model: ${MODEL}' and CONFIG_ES327_BENCHMARK_MODE in build"
echo ""
echo "Flashing (J-Link) ..."
(cd "$NCS_WORKSPACE" && west flash -d "$BUILD_DIR" --runner jlink)
echo ""
echo "Done. Press RESET on the board if it doesn't reboot automatically."
echo "RTT should show: 'Benchmark mode — UART ready, waiting for SYNC'"
echo "Then run: cd .. && uv run python benchmark.py --probe-all  # find UART0 port"
echo "         uv run python benchmark.py --port /dev/cu.usbmodemXXXX --data-dir <Ninapro_DB1> --model-name $MODEL --n-windows 100 --boot-wait 30 --timeout 120"
