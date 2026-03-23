/*
 * model_runner.h — C-compatible API for TFLite Micro EMG inference
 *
 * This header exposes a plain-C interface so that main.c (C) can call into
 * the C++ TFLite Micro runtime without knowing anything about it.
 *
 * Supported models (selected at build time via -DMODEL=xxx):
 *   tcn, bilstm, bilstm_small, cnnlstm, cnnlstm_small, transformer, pet
 *
 * Calling sequence
 * ----------------
 *   1. model_runner_init()          — once at startup
 *   2. model_runner_infer(in, out)  — every WIN_SAMPLES once per stride
 *   3. model_runner_arena_used()    — optional: log peak RAM
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>   /* size_t */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Dimensions — shared across all models (match quantise.py constants)
 * ------------------------------------------------------------------------- */
#define EMG_CHANNELS   10     /* sEMG input channels                         */
#define WIN_SAMPLES    64     /* window length in (downsampled) samples       */
#define N_ANGLES       22     /* output: CyberGlove joint angles              */

/* -------------------------------------------------------------------------
 * Tensor arena size
 *
 * nRF52840 has 256 KB RAM total.  After Zephyr OS overhead (~30 KB),
 * ring buffers, and stack, ~200 KB remains for the tensor arena.
 *
 * Arena usage varies by model — call model_runner_arena_used() after one
 * inference to see the actual high-water mark.
 *
 * BiLSTM with WHILE (non-unrolled) can request ~860 KB from TFLM's planner,
 * which exceeds nRF52840 RAM. To fit, use a smaller model: shorter sequence
 * (e.g. WIN=32), fewer hidden units, or fewer LSTM layers; or deploy on a
 * board with more RAM.
 *
 * Measured / estimated arena usage per model (INT8):
 *   TCN:           ~71 KB
 *   bilstm_small:  ~164 KB (fits)
 *   cnnlstm_small: ~120–170 KB (fits; 32 CNN ch, 64 hidden, 1 LSTM layer)
 *   BiLSTM/cnnlstm: WHILE-based graph often requests ~860 KB (does not fit)
 *   Transformer:   ~90 KB  (estimate)
 *   PET:           ~95 KB  (estimate)
 * ------------------------------------------------------------------------- */
#define TFLM_ARENA_SIZE (172 * 1024)  /* 172 KB — bilstm_small needs ~164 KB at runtime; heap reduced in prj.conf */

/* -------------------------------------------------------------------------
 * Error codes
 * ------------------------------------------------------------------------- */
#define MODEL_OK            0
#define MODEL_ERR_ALLOC    -1
#define MODEL_ERR_INPUT    -2
#define MODEL_ERR_INVOKE   -3

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

/**
 * model_runner_init
 *
 * Initialises TFLite Micro interpreter, allocates the tensor arena, and
 * loads the embedded model (selected at build time via -DMODEL=xxx,
 * from firmware/src/models/<name>_model.h).
 *
 * Must be called once before any call to model_runner_infer().
 *
 * @return MODEL_OK on success, negative error code on failure.
 */
int model_runner_init(void);

/**
 * model_runner_infer
 *
 * Run one forward pass of the INT8 model.
 *
 * @param emg_window   Pointer to a [WIN_SAMPLES][EMG_CHANNELS] float32 array.
 *                     Values must be z-score normalised (mean=0, std=1) using
 *                     the same statistics computed during training.
 * @param angles_out   Pointer to a [WIN_SAMPLES][N_ANGLES] float32 array.
 *                     Filled with predicted joint angles in radians on return.
 *
 * @return MODEL_OK on success, negative error code on failure.
 */
int model_runner_infer(const float emg_window[WIN_SAMPLES][EMG_CHANNELS],
                       float       angles_out[WIN_SAMPLES][N_ANGLES]);

/**
 * model_runner_arena_used
 *
 * @return Number of bytes currently used in the tensor arena.
 *         Call after at least one successful model_runner_infer() for a
 *         meaningful value.
 */
size_t model_runner_arena_used(void);

/**
 * model_runner_get_output_debug
 *
 * Returns output tensor shape and first 8 raw floats from the last inference.
 * Use for UART debug when RTT is unavailable.
 *
 * @param rank  Output: tensor rank (typically 3)
 * @param dim1  Output: shape[1] (22 or 64)
 * @param dim2  Output: shape[2] (64 or 22)
 * @param raw   Output: first 8 floats from interpreter output buffer
 */
void model_runner_get_output_debug(int *rank, int *dim1, int *dim2, float raw[8]);

#ifdef __cplusplus
}  /* extern "C" */
#endif
