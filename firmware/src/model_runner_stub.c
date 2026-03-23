/*
 * model_runner_stub.c — no-op model runner compiled when TFLM is absent
 *
 * This file is compiled instead of model_runner.cc when the tflm/ directory
 * is missing (i.e. get_tflm.sh has not been run yet). It allows the
 * firmware to compile and run as a blinky so the board bring-up workflow
 * is not gated on the TFLM dependency.
 *
 * Once get_tflm.sh has been run and model_runner.cc is active, this file
 * is excluded by the CMakeLists.txt guard.
 */

#ifdef TFLM_STUB

#include "model_runner.h"
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_DECLARE(emg_inference, LOG_LEVEL_INF);

int model_runner_init(void)
{
    LOG_WRN("TFLM stub: no model loaded (run firmware/get_tflm.sh)");
    return MODEL_OK;
}

int model_runner_infer(const float emg_window[WIN_SAMPLES][EMG_CHANNELS],
                       float       angles_out[WIN_SAMPLES][N_ANGLES])
{
    /* Return zeros — stub behaviour */
    memset(angles_out, 0, sizeof(float) * WIN_SAMPLES * N_ANGLES);
    return MODEL_OK;
}

size_t model_runner_arena_used(void)
{
    return 0u;
}

void model_runner_get_output_debug(int *rank, int *dim1, int *dim2, float raw[8])
{
    *rank = 3;
    *dim1 = 22;
    *dim2 = 64;
    for (int i = 0; i < 8; ++i) raw[i] = 0.0f;
}

#endif /* TFLM_STUB */
