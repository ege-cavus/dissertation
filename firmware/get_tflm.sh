#!/usr/bin/env bash
# get_tflm.sh — Clone TFLite Micro into firmware/tflm/
#
# The old approach used create_tflm_tree.py which calls 'make third_party_downloads'.
# That Makefile has been removed from the TFLM repo.  This script instead clones
# the repo directly with --recurse-submodules so that the flatbuffers headers
# (a submodule at third_party/flatbuffers/) are populated.
# CMakeLists.txt globs sources straight from this tree — no generation step needed.
#
# Run once from the firmware/ directory:
#   bash get_tflm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_DIR="${SCRIPT_DIR}/tflm"
TFLM_REPO="https://github.com/tensorflow/tflite-micro.git"

echo ""
echo "=== Fetching TFLite Micro into ${TFLM_DIR} ==="

if [ -d "${TFLM_DIR}/.git" ]; then
    echo "  tflm/ already exists."
    echo "  Updating submodules (flatbuffers etc.) …"
    git -C "${TFLM_DIR}" submodule update --init --recursive --depth=1
else
    echo "  Cloning …"
    git clone --depth=1 --recurse-submodules "${TFLM_REPO}" "${TFLM_DIR}"
fi

# flatbuffers is no longer a git submodule in the new TFLM repo — it was
# downloaded by 'make third_party_downloads' (now removed).  Clone it directly.
FB_DIR="${TFLM_DIR}/third_party/flatbuffers"
FB_HEADER="${FB_DIR}/include/flatbuffers/flatbuffers.h"

if [ ! -f "${FB_HEADER}" ]; then
    echo ""
    echo "  Cloning flatbuffers into ${FB_DIR} …"
    if [ -d "${FB_DIR}/.git" ]; then
        echo "  (already cloned — skipping)"
    else
        mkdir -p "${FB_DIR}"
        git clone --depth=1 https://github.com/google/flatbuffers.git "${FB_DIR}"
    fi
fi

if [ ! -f "${FB_HEADER}" ]; then
    echo "[ERROR] flatbuffers header still missing — check network and retry."
    exit 1
fi
echo "  flatbuffers OK"

# gemmlowp (needed by some TFLM reference kernel math)
GM_DIR="${TFLM_DIR}/third_party/gemmlowp"
if [ ! -d "${GM_DIR}/fixedpoint" ]; then
    echo "  Cloning gemmlowp into ${GM_DIR} …"
    git clone --depth=1 https://github.com/google/gemmlowp.git "${GM_DIR}"
fi
echo "  gemmlowp OK"

echo ""
echo "=== Done ==="
echo "  TFLM source tree: ${TFLM_DIR}/"
echo ""
echo "  Build the firmware:"
echo "    cd ${SCRIPT_DIR}"
echo "    west build -b nrf52840dk/nrf52840 ."
