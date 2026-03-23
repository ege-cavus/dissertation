/*
 * model_runner.cc — TFLite Micro inference engine for EMG → joint angles
 *
 * Implements the C interface declared in model_runner.h using the
 * TensorFlow Lite for Microcontrollers (TFLM) C++ API.
 *
 * Supports five model architectures (selected at build time via -DMODEL=xxx):
 *   TCN, BiLSTM, CNN-BiLSTM, Transformer, PET
 *
 * All models share the same I/O contract:
 *   Input:  channels-first [1, EMG_CHANNELS, WIN_SAMPLES] = [1, 10, 64]
 *   Output: channels-first [1, N_ANGLES,     WIN_SAMPLES] = [1, 22, 64]
 *   I/O type: float32 (internal ops are INT8 with QUANTIZE/DEQUANTIZE at
 *             the model boundary)
 *
 * The public API accepts time-first arrays [WIN_SAMPLES][channels] so that
 * main.c can pass the ring-buffer window directly.  The transposition is
 * done here, keeping main.c layout-agnostic.
 */

#include "model_runner.h"

/* ---- TFLite Micro headers ---- */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ---- Generated model flatbuffer (selected at build time via -DMODEL=xxx) ---- */
#if defined(MODEL_TCN)
#include "models/tcn_model.h"
static const uint8_t *s_model_data     = tcn_model_data;
static const char   *s_model_name      = "tcn";
#elif defined(MODEL_BILSTM)
#include "models/bilstm_model.h"
static const uint8_t *s_model_data     = bilstm_model_data;
static const char   *s_model_name      = "bilstm";
#elif defined(MODEL_BILSTM_SMALL)
#include "models/bilstm_small_model.h"
static const uint8_t *s_model_data     = bilstm_small_model_data;
static const char   *s_model_name      = "bilstm_small";
#elif defined(MODEL_CNNLSTM)
#include "models/cnnlstm_model.h"
static const uint8_t *s_model_data     = cnnlstm_model_data;
static const char   *s_model_name      = "cnnlstm";
#elif defined(MODEL_CNNLSTM_SMALL)
#include "models/cnnlstm_small_model.h"
static const uint8_t *s_model_data     = cnnlstm_small_model_data;
static const char   *s_model_name      = "cnnlstm_small";
#elif defined(MODEL_TRANSFORMER)
#include "models/transformer_model.h"
static const uint8_t *s_model_data     = transformer_model_data;
static const char   *s_model_name      = "transformer";
#elif defined(MODEL_PET)
#include "models/pet_model.h"
static const uint8_t *s_model_data     = pet_model_data;
static const char   *s_model_name      = "pet";
#elif defined(MODEL_PET_SMALL)
#include "models/pet_small_model.h"
static const uint8_t *s_model_data     = pet_small_model_data;
static const char   *s_model_name      = "pet_small";
#else
#error "No model selected. Pass -DMODEL=tcn|bilstm|bilstm_small|cnnlstm|cnnlstm_small|transformer|pet|pet_small to cmake."
#endif

#include <cstring>
#include <cstdio>
#include <zephyr/sys/printk.h>
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(model_runner, LOG_LEVEL_DBG);

/* -------------------------------------------------------------------------
 * Zephyr-native DebugLog — routes TFLM error messages to printk/RTT.
 * Must match the signature expected by micro_log.cc: (const char* format, va_list args)
 * so that MicroPrintf("opcode %d (%s)", code, name) is formatted correctly.
 * ------------------------------------------------------------------------- */
#include <cstdarg>
#include <cstdio>

extern "C" void DebugLog(const char *format, va_list args)
{
    char buf[200];
    int n = std::vsnprintf(buf, sizeof(buf), format, args);
    if (n > 0 && n < (int)sizeof(buf)) {
        printk("%s", buf);
    } else if (format && format[0]) {
        printk("%s", format);  /* fallback if vsnprintf fails */
    }
}

/* -------------------------------------------------------------------------
 * Operator resolver — superset of all ops used by the five model architectures
 *
 * TCN:         Conv2D, Add, ReLU, Pad, StridedSlice, Reshape, Transpose,
 *              SpaceToBatchNd, BatchToSpaceNd, ExpandDims, Squeeze, Slice
 * BiLSTM:      UnidirectionalSequenceLstm, FullyConnected, Tanh, Logistic
 * CNN-BiLSTM:  (BiLSTM ops) + Conv2D, DepthwiseConv2D
 * Transformer: BatchMatMul, Softmax, Gather, ResizeBilinear, Sub, Rsqrt
 * PET:         (Transformer ops) + Concatenation
 *
 * Common:      Quantize, Dequantize, Mul, Mean, Reshape, Transpose, Add
 *
 * If AllocateTensors returns -1, a required op is not registered.
 * Increase the template parameter and add the missing AddXxx() call.
 * ------------------------------------------------------------------------- */
using OpResolver = tflite::MicroMutableOpResolver<52>;

static OpResolver make_resolver()
{
    OpResolver resolver;

    /* --- Convolution / pooling --- */
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();

    /* --- Elementwise / activation --- */
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddRelu();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddSoftmax();
    resolver.AddRsqrt();

    /* --- Linear algebra --- */
    resolver.AddFullyConnected();
    resolver.AddBatchMatMul();

    /* --- Shape manipulation --- */
    resolver.AddReshape();
    resolver.AddTranspose();
    resolver.AddExpandDims();
    resolver.AddSqueeze();
    resolver.AddConcatenation();
    resolver.AddGather();
    resolver.AddPack();
    resolver.AddUnpack();
    resolver.AddSplit();        /* BiLSTM: split input/state tensors */

    /* --- TCN dilated-conv support --- */
    resolver.AddPad();
    resolver.AddStridedSlice();
    resolver.AddSlice();
    resolver.AddSpaceToBatchNd();
    resolver.AddBatchToSpaceNd();

    /* --- Quantisation boundary --- */
    resolver.AddQuantize();
    resolver.AddDequantize();

    /* --- Type/shape ops (converter often inserts these for LSTM/Transformer) --- */
    resolver.AddCast();
    resolver.AddFloor();
    resolver.AddExp();
    resolver.AddNeg();
    resolver.AddShape();        /* PET LayerNorm decomposition */

    /* --- Reduction --- */
    resolver.AddMean();

    /* --- Sequence models (BiLSTM, CNN-BiLSTM) --- */
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddLess();       /* WHILE condition: i < N */
    resolver.AddLessEqual();  /* WHILE condition: i <= N */
    resolver.AddLogicalAnd(); /* WHILE condition: (i < N) && ... */
    resolver.AddWhile();              /* LSTM loop when not unrolled */
    resolver.AddDynamicUpdateSlice(); /* LSTM h-state scatter (patched while_body) */

    /* --- Transformer / PET interpolation + activation --- */
    resolver.AddResizeBilinear();
    resolver.AddResizeNearestNeighbor();

    /* --- Sum / reduce ops used by LayerNorm, attention normalization --- */
    resolver.AddReduceMax();
    resolver.AddSum();
    resolver.AddSquaredDifference();

    return resolver;
}

/* -------------------------------------------------------------------------
 * Static state
 * ------------------------------------------------------------------------- */
alignas(16) static uint8_t s_arena[TFLM_ARENA_SIZE];
static OpResolver                     s_resolver;
static const tflite::Model           *s_model   = nullptr;
static tflite::MicroInterpreter      *s_interp  = nullptr;
static TfLiteTensor                  *s_input   = nullptr;
static TfLiteTensor                  *s_output  = nullptr;

/* Last inference output debug (for UART debug command) */
static int   s_last_rank = 0, s_last_dim1 = 0, s_last_dim2 = 0;
static float s_last_raw[8] = {0};

/* All models have float32 I/O.  The QUANTIZE / DEQUANTIZE ops inside the
 * flatbuffer handle float32 ↔ INT8 conversion at the model boundary. */

/* -------------------------------------------------------------------------
 * model_runner_init
 * ------------------------------------------------------------------------- */
int model_runner_init(void)
{
    tflite::InitializeTarget();

    s_model = tflite::GetModel(s_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        return -10;   /* schema version mismatch: model vs TFLM header */
    }

    LOG_INF("Loading model '%s', TFLM_ARENA_SIZE=%u B", s_model_name,
            (unsigned)TFLM_ARENA_SIZE);

    s_resolver = make_resolver();

    /* Placement-new into a small static buffer so we don't need dynamic alloc */
    static uint8_t s_interp_buf[sizeof(tflite::MicroInterpreter)];
    s_interp = new (s_interp_buf)
        tflite::MicroInterpreter(s_model, s_resolver, s_arena, TFLM_ARENA_SIZE);

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        LOG_ERR(
            "model_runner_init: tensor arena too small. Model '%s', "
            "TFLM_ARENA_SIZE=%u B. See TFLM message above for requested size. "
            "Increase TFLM_ARENA_SIZE in model_runner.h or use a smaller model.",
            s_model_name, (unsigned)TFLM_ARENA_SIZE);
        return -20;   /* AllocateTensors failed: unregistered op or arena too small */
    }

    s_input  = s_interp->input(0);
    s_output = s_interp->output(0);

    if (s_input->type != kTfLiteFloat32 || s_output->type != kTfLiteFloat32) {
        LOG_ERR("I/O type mismatch: input=%d output=%d (expected float32=1)",
                (int)s_input->type, (int)s_output->type);
        return -30;
    }

    return MODEL_OK;
}

/* -------------------------------------------------------------------------
 * model_runner_infer
 * ------------------------------------------------------------------------- */
int model_runner_infer(const float emg_window[WIN_SAMPLES][EMG_CHANNELS],
                       float       angles_out[WIN_SAMPLES][N_ANGLES])
{
    if (!s_interp) {
        return MODEL_ERR_INPUT;
    }

    /* Copy EMG window into the float32 input tensor (channels-first layout).
     *
     *   Public API layout:   emg_window[t][c]  — time-first   [WIN][CH]
     *   TFLite input layout: in_data[c * WIN + t] — channels-first [CH][WIN]
     *
     * The model's internal QUANTIZE op converts float32 → INT8 automatically.
     */
    float *in_data = s_input->data.f;
    for (int c = 0; c < EMG_CHANNELS; ++c) {
        for (int t = 0; t < WIN_SAMPLES; ++t) {
            in_data[c * WIN_SAMPLES + t] = emg_window[t][c];
        }
    }

    /* Run inference */
    LOG_INF("Invoke() starting...");
    TfLiteStatus invoke_status = s_interp->Invoke();
    LOG_INF("Invoke() returned %d", (int)invoke_status);
    if (invoke_status != kTfLiteOk) {
        LOG_ERR("Invoke() failed with status %d", (int)invoke_status);
        return MODEL_ERR_INVOKE;
    }

    /* Debug: log raw interpreter output (first 8 floats) to verify TFLM vs copy */
    {
        const float *raw = s_output->data.f;
        LOG_INF("raw out[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                (double)raw[0], (double)raw[1], (double)raw[2], (double)raw[3],
                (double)raw[4], (double)raw[5], (double)raw[6], (double)raw[7]);
    }

    /* Copy float32 output tensor to time-first angles_out[t][a].
     * Model may output (1, ANG, WIN) or (1, WIN, ANG) — detect from shape. */
    const float *out_data = s_output->data.f;
    const auto out_shape = tflite::GetTensorShape(s_output);
    const int rank = out_shape.DimensionsCount();
    /* Typical: (1, 22, 64) [ANG][WIN] or (1, 64, 22) [WIN][ANG] */
    const int dim1 = (rank >= 2) ? out_shape.Dims(1) : 1;
    const int dim2 = (rank >= 3) ? out_shape.Dims(2) : 1;
    if (dim1 == N_ANGLES && dim2 == WIN_SAMPLES) {
        /* [batch][ANG][WIN] — index = a * WIN + t */
        for (int t = 0; t < WIN_SAMPLES; ++t) {
            for (int a = 0; a < N_ANGLES; ++a) {
                angles_out[t][a] = out_data[a * WIN_SAMPLES + t];
            }
        }
    } else if (dim1 == WIN_SAMPLES && dim2 == N_ANGLES) {
        /* [batch][WIN][ANG] — index = t * ANG + a */
        for (int t = 0; t < WIN_SAMPLES; ++t) {
            for (int a = 0; a < N_ANGLES; ++a) {
                angles_out[t][a] = out_data[t * N_ANGLES + a];
            }
        }
    } else {
        /* Fallback: assume [ANG][WIN] */
        for (int t = 0; t < WIN_SAMPLES; ++t) {
            for (int a = 0; a < N_ANGLES; ++a) {
                angles_out[t][a] = out_data[a * WIN_SAMPLES + t];
            }
        }
    }

    /* Store debug for UART debug command */
    s_last_rank = rank;
    s_last_dim1 = dim1;
    s_last_dim2 = dim2;
    for (int i = 0; i < 8; ++i) {
        s_last_raw[i] = out_data[i];
    }

    return MODEL_OK;
}

/* -------------------------------------------------------------------------
 * model_runner_get_output_debug
 * ------------------------------------------------------------------------- */
void model_runner_get_output_debug(int *rank, int *dim1, int *dim2, float raw[8])
{
    *rank = s_last_rank;
    *dim1 = s_last_dim1;
    *dim2 = s_last_dim2;
    for (int i = 0; i < 8; ++i) {
        raw[i] = s_last_raw[i];
    }
}

/* -------------------------------------------------------------------------
 * model_runner_arena_used
 * ------------------------------------------------------------------------- */
size_t model_runner_arena_used(void)
{
    if (!s_interp) return 0u;
    return s_interp->arena_used_bytes();
}
