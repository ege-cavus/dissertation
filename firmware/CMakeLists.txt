# ES327 firmware — nRF52840 DK
cmake_minimum_required(VERSION 3.20.0)

# Benchmark uses UART0 on board pins (J-Link exposes it as one of the two /dev/cu.usbmodem* ports).
# No USB overlay — so no need for a second cable.

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(es327_firmware)

# ---- Model selection (build-time) ----------------------------------------
# nRF sysbuild does NOT forward -D flags to child images. Use environment
# variables: MODEL=bilstm_small west build ... (the env is always visible).
if(DEFINED ENV{MODEL} AND NOT "$ENV{MODEL}" STREQUAL "")
  set(MODEL "$ENV{MODEL}" CACHE STRING "Which model to deploy" FORCE)
elseif(NOT DEFINED MODEL OR "${MODEL}" STREQUAL "")
  set(MODEL "tcn" CACHE STRING "Which model to deploy" FORCE)
endif()
set_property(CACHE MODEL PROPERTY STRINGS tcn bilstm bilstm_small cnnlstm cnnlstm_small transformer pet pet_small)
message(STATUS "Selected model: ${MODEL}")

# ---- Application sources -------------------------------------------------
target_sources(app PRIVATE
    src/main.c
    src/model_runner.cc
)

target_include_directories(app PRIVATE
    src
    src/models
)

string(TOUPPER "${MODEL}" MODEL_UPPER)
target_compile_definitions(app PRIVATE MODEL_${MODEL_UPPER})

# Benchmark mode is controlled by Kconfig CONFIG_ES327_BENCHMARK_MODE.
# To enable: west build ... -- -DCONF_FILE="prj.conf;benchmark.conf"

# ---- TFLite Micro --------------------------------------------------------
# Run  firmware/get_tflm.sh  once to populate tflm/ and its third-party deps.
#
# Compilation strategy:
#  - Use GLOB (not GLOB_RECURSE) for the root micro/ directory so we only get
#    platform-neutral .cc files.
#  - Explicitly add the Zephyr platform subdir (zephyr/) — NOT cortex_m_generic,
#    ceva, arc_custom or cortex_m_corstone_300 which need hardware-specific headers.
#  - Add the ARM toolchain's full C++ stdlib headers as SYSTEM includes because
#    Zephyr builds with -nostdinc++ and its minimal C++ shim is missing <algorithm>,
#    <utility>, <initializer_list> etc. that TFLM needs.

set(TFLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tflm)

if(EXISTS "${TFLM_DIR}/tensorflow/lite/micro/micro_interpreter.h")
    message(STATUS "TFLite Micro found at ${TFLM_DIR}")

    # --- Platform-neutral root sources ---
    file(GLOB TFLM_MICRO_ROOT
        "${TFLM_DIR}/tensorflow/lite/micro/*.cc"
    )
    # Exclude test/fake/benchmark files and the generic debug_log.cc.
    # We supply firmware/src/debug_log.cc which routes TFLM messages to Zephyr
    # printk so they appear in RTT alongside application logs.
    list(FILTER TFLM_MICRO_ROOT EXCLUDE REGEX ".*(test|fake|benchmark).*\\.cc$")
    list(FILTER TFLM_MICRO_ROOT EXCLUDE REGEX ".*debug_log\\.cc$")

    # --- Zephyr platform setup (system_setup + micro_time) ---
    # The zephyr/ subdir may not exist in all TFLM checkouts; guard against empty glob.
    file(GLOB TFLM_ZEPHYR_SRCS
        "${TFLM_DIR}/tensorflow/lite/micro/zephyr/*.cc"
    )

    # --- Arena allocator and memory planner ---
    file(GLOB TFLM_ALLOC_SRCS
        "${TFLM_DIR}/tensorflow/lite/micro/arena_allocator/*.cc"
        "${TFLM_DIR}/tensorflow/lite/micro/memory_planner/*.cc"
    )
    list(FILTER TFLM_ALLOC_SRCS EXCLUDE REGEX ".*(test|benchmark).*\\.cc$")

    # --- TFLite bridge (flatbuffer → TFLM conversion helpers) ---
    file(GLOB TFLM_BRIDGE_SRCS
        "${TFLM_DIR}/tensorflow/lite/micro/tflite_bridge/*.cc"
    )
    list(FILTER TFLM_BRIDGE_SRCS EXCLUDE REGEX ".*(test|benchmark).*\\.cc$")

    # --- Reference kernels (platform-neutral baseline) ---
    file(GLOB TFLM_KERNEL_SRCS
        "${TFLM_DIR}/tensorflow/lite/micro/kernels/*.cc"
    )
    list(FILTER TFLM_KERNEL_SRCS EXCLUDE REGEX ".*(test|benchmark).*\\.cc$")

    # --- CMSIS-NN optimised kernels (optional) ---
    # Set USE_CMSIS_NN=OFF to use reference kernels only if Invoke() aborts with
    # CMSIS-NN (e.g. unsupported conv params: dilation, padding, or alignment).
    set(USE_CMSIS_NN ON CACHE BOOL "Use CMSIS-NN optimized kernels")
    if(USE_CMSIS_NN)
        # nRF SDK v3.2.4 ships CMSIS-NN v18.0.0.  For every op that has a
        # cmsis_nn/ replacement we add it and drop the reference kernel.
        file(GLOB TFLM_CMSIS_KERNEL_SRCS
            "${TFLM_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/*.cc"
        )
        list(FILTER TFLM_CMSIS_KERNEL_SRCS EXCLUDE REGEX ".*(test|benchmark).*\\.cc$")
        list(FILTER TFLM_CMSIS_KERNEL_SRCS EXCLUDE REGEX ".*/unidirectional_sequence_lstm\\.cc$")

        # LSTM/CNNLSTM use dynamic-range quant (INT8 weights, float32 activations).
        # CMSIS-NN FullyConnected and Conv reject hybrid; use reference for those builds.
        if(MODEL STREQUAL "bilstm" OR MODEL STREQUAL "bilstm_small" OR MODEL STREQUAL "cnnlstm" OR MODEL STREQUAL "cnnlstm_small")
            list(FILTER TFLM_CMSIS_KERNEL_SRCS EXCLUDE REGEX ".*/fully_connected\\.cc$")
        endif()
        if(MODEL STREQUAL "cnnlstm" OR MODEL STREQUAL "cnnlstm_small")
            list(FILTER TFLM_CMSIS_KERNEL_SRCS EXCLUDE REGEX ".*/conv\\.cc$")
        endif()

        foreach(_KN add batch_matmul conv depthwise_conv fully_connected
                    maximum_minimum mul pad pooling softmax svdf transpose
                    transpose_conv)
            list(FILTER TFLM_KERNEL_SRCS EXCLUDE REGEX ".*/${_KN}\\.cc$")
        endforeach()

        # Re-add reference FullyConnected for LSTM so hybrid (float in, int8 weights) runs.
        if(MODEL STREQUAL "bilstm" OR MODEL STREQUAL "bilstm_small" OR MODEL STREQUAL "cnnlstm" OR MODEL STREQUAL "cnnlstm_small")
            list(APPEND TFLM_KERNEL_SRCS
                "${TFLM_DIR}/tensorflow/lite/micro/kernels/fully_connected.cc")
        endif()
        # Re-add reference Conv for CNNLSTM/cnnlstm_small so hybrid Conv (float in, int8 weights) runs.
        if(MODEL STREQUAL "cnnlstm" OR MODEL STREQUAL "cnnlstm_small")
            list(APPEND TFLM_KERNEL_SRCS
                "${TFLM_DIR}/tensorflow/lite/micro/kernels/conv.cc")
        endif()
    else()
        set(TFLM_CMSIS_KERNEL_SRCS "")
        message(STATUS "CMSIS-NN disabled — using reference kernels")
    endif()

    # --- Lite core helpers ---
    # core/c/common.cc                  → TfLiteTypeGetName, TfLiteIntArrayEqual, TfLiteIntArrayGetSizeInBytes
    # core/api/*.cc                     → flatbuffer_conversions, op_resolver helpers
    # compiler/mlir/.../error_reporter.cc → ErrorReporter::Report / ReportError
    #   (tensorflow/lite/core/api/error_reporter.h is a forwarding header to the mlir path)
    # kernels/internal/tensor_ctypes.cc → GetTensorShape(TfLiteTensor*)
    # kernels/internal/common.cc        → MultiplyByQuantizedMultiplier
    # kernels/internal/portable_tensor_utils.cc → UnpackDenseInt4IntoInt8
    # kernels/internal/quantization_util.cc    → quantization helpers
    # kernels/kernel_util.cc                   → HaveSameShapes
    # compiler/mlir/lite/schema/schema_utils.cc → GetBuiltinCode (same mlir relocation pattern)
    file(GLOB TFLM_LITE_SRCS
        "${TFLM_DIR}/tensorflow/lite/core/c/common.cc"
        "${TFLM_DIR}/tensorflow/lite/core/api/*.cc"
        "${TFLM_DIR}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc"
        "${TFLM_DIR}/tensorflow/lite/kernels/internal/tensor_ctypes.cc"
        "${TFLM_DIR}/tensorflow/lite/kernels/internal/common.cc"
        "${TFLM_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc"
        # portable_tensor_utils.cc is at the kernels/internal/ root (NOT .../reference/)
        "${TFLM_DIR}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc"
        "${TFLM_DIR}/tensorflow/lite/kernels/kernel_util.cc"
        # reference/comparisons.cc — BroadcastComparison4DSlowPreprocess used by micro/kernels/comparisons.cc (LESS/LESS_EQUAL)
        "${TFLM_DIR}/tensorflow/lite/kernels/internal/reference/comparisons.cc"
        # schema_utils.cc (GetBuiltinCode) lives under the mlir path, same pattern as error_reporter
        "${TFLM_DIR}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc"
    )
    list(FILTER TFLM_LITE_SRCS EXCLUDE REGEX ".*(test|benchmark).*\\.cc$")

    target_sources(app PRIVATE
        ${TFLM_MICRO_ROOT}
        ${TFLM_ZEPHYR_SRCS}
        ${TFLM_ALLOC_SRCS}
        ${TFLM_BRIDGE_SRCS}
        ${TFLM_KERNEL_SRCS}
        ${TFLM_CMSIS_KERNEL_SRCS}
        ${TFLM_LITE_SRCS}
    )

    # --- C++ standard library headers from the Zephyr SDK toolchain ---
    # Zephyr's -nostdinc++ flag hides the toolchain's C++ headers and only
    # exposes its minimal shim.  TFLM needs <algorithm>, <utility>, etc.
    # Adding the ARM toolchain C++ headers as SYSTEM includes restores them
    # without breaking any Zephyr-internal header ordering.
    if(DEFINED ZEPHYR_SDK_INSTALL_DIR)
        file(GLOB _ARM_CXX_DIRS
            "${ZEPHYR_SDK_INSTALL_DIR}/arm-zephyr-eabi/arm-zephyr-eabi/include/c++/[0-9]*"
        )
        if(_ARM_CXX_DIRS)
            list(GET _ARM_CXX_DIRS 0 _ARM_CXX_DIR)
            message(STATUS "ARM C++ headers: ${_ARM_CXX_DIR}")
            target_include_directories(app SYSTEM PRIVATE
                ${_ARM_CXX_DIR}
                ${_ARM_CXX_DIR}/arm-zephyr-eabi
            )
        else()
            message(WARNING "Could not locate ARM C++ headers under ${ZEPHYR_SDK_INSTALL_DIR}")
        endif()
    endif()

    # TFLM + third-party include dirs
    target_include_directories(app PRIVATE
        ${TFLM_DIR}
        ${TFLM_DIR}/third_party/flatbuffers/include
        ${TFLM_DIR}/third_party/gemmlowp
        ${TFLM_DIR}/third_party/ruy
    )

    if(USE_CMSIS_NN)
        if(DEFINED ZEPHYR_CMSIS_NN_MODULE_DIR)
            target_include_directories(app PRIVATE ${ZEPHYR_CMSIS_NN_MODULE_DIR})
            message(STATUS "CMSIS-NN headers: ${ZEPHYR_CMSIS_NN_MODULE_DIR}/Include")
        else()
            message(WARNING "ZEPHYR_CMSIS_NN_MODULE_DIR not set — CMSIS-NN kernels will not compile")
        endif()
        target_compile_definitions(app PRIVATE CMSIS_NN)
    endif()

    target_compile_definitions(app PRIVATE
        TF_LITE_STATIC_MEMORY=1
        TF_LITE_MICRO=1
        FLATBUFFERS_LOCALE_INDEPENDENT=0
    )

    target_compile_options(app PRIVATE
        -Wno-unused-function
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-type-limits
        -Wno-shadow
    )

else()
    message(WARNING
        "TFLite Micro not found at ${TFLM_DIR}.\n"
        "Run:  bash firmware/get_tflm.sh\n"
        "Building without inference support (stub only)."
    )
    target_compile_definitions(app PRIVATE TFLM_STUB=1)
    target_sources(app PRIVATE src/model_runner_stub.c)
    get_target_property(APP_SRCS app SOURCES)
    list(REMOVE_ITEM APP_SRCS src/model_runner.cc)
    set_property(TARGET app PROPERTY SOURCES ${APP_SRCS})
endif()
