/*
 * ES327 firmware — EMG inference on nRF52840 DK
 *
 * Architecture
 * ------------
 *  - A circular ring buffer holds the most recent WIN_SAMPLES × EMG_CHANNELS
 *    z-score-normalised EMG samples.
 *  - Every STRIDE_SAMPLES new samples, a window is copied out of the ring
 *    buffer and passed to the TFLite Micro model runner.
 *  - The model outputs WIN_SAMPLES × N_ANGLES joint angles in radians, of
 *    which the most-recent frame (index WIN_SAMPLES-1) is used as the
 *    current angle estimate and logged over RTT.
 *
 * Signal path (placeholder — replace with real ADC driver)
 * ---------------------------------------------------------
 *  simulate_emg_sample() generates a synthetic sample every SAMPLE_PERIOD_MS.
 *  In a real deployment this would be replaced by a DMA interrupt callback
 *  from the sEMG analogue front-end (e.g. ADS1299 or similar).
 *
 * Memory budget (TCN INT8)
 * -------------------------
 *   Ring buffer:      WIN_SAMPLES × EMG_CHANNELS × 4 B = 64×10×4 = 2.5 KB
 *   Inference input:  WIN_SAMPLES × EMG_CHANNELS × 4 B = 2.5 KB
 *   Inference output: WIN_SAMPLES × N_ANGLES     × 4 B = 64×22×4 = 5.6 KB
 *   TFLM arena:       TFLM_ARENA_SIZE                  = 64 KB
 *   ------------------------------------------------------------------
 *   Total extra RAM:                                    ≈ 74.6 KB
 *   (fits well within the 256 KB SRAM after Zephyr OS ~30 KB overhead)
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>

#ifdef CONFIG_ES327_BENCHMARK_MODE
#include <zephyr/drivers/uart.h>
#endif

#include "model_runner.h"

LOG_MODULE_REGISTER(emg_inference, LOG_LEVEL_INF);

/* -------------------------------------------------------------------------
 * LED
 * ------------------------------------------------------------------------- */
#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

/* -------------------------------------------------------------------------
 * Timing
 * ------------------------------------------------------------------------- */
#define SAMPLE_PERIOD_MS   2    /* 500 Hz after 3× downsampling of 1500 Hz */
#define STRIDE_SAMPLES     16   /* run inference every STRIDE_SAMPLES new samples */

/* -------------------------------------------------------------------------
 * Ring buffer  —  [WIN_SAMPLES][EMG_CHANNELS]  float32, z-scored
 *
 * The buffer stores the most recent WIN_SAMPLES frames.
 * write_head points to the *next* position to be written (mod WIN_SAMPLES).
 * ------------------------------------------------------------------------- */
static float  ring_buf[WIN_SAMPLES][EMG_CHANNELS];
static int    write_head   = 0;
static int    samples_since_last_infer = 0;

/* Working buffers reused across inference calls to avoid stack blow-up */
static float  infer_input [WIN_SAMPLES][EMG_CHANNELS];
static float  infer_output[WIN_SAMPLES][N_ANGLES];

/* -------------------------------------------------------------------------
 * Z-score normalisation parameters
 *
 * These must be the *training-set* mean and std — the same values used when
 * building the representative dataset in quantise.py.
 *
 * Replace with the actual values printed by quantise.py when it runs.
 * A convenience: quantise.py prints a C-ready array for you to paste here.
 * ------------------------------------------------------------------------- */
static const float EMG_MEAN[EMG_CHANNELS] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
static const float EMG_STD[EMG_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};

/* -------------------------------------------------------------------------
 * simulate_emg_sample  (placeholder for real ADC driver)
 *
 * Writes one normalised EMG frame into dst[EMG_CHANNELS].
 * Replace this function body with the real ADC read + z-score step.
 * ------------------------------------------------------------------------- */
static uint32_t s_sim_tick = 0u;

static void simulate_emg_sample(float dst[EMG_CHANNELS])
{
    /* Synthetic signal: low-amplitude sine + noise per channel.
     * This exercises the inference pipeline without real hardware. */
    for (int c = 0; c < EMG_CHANNELS; ++c) {
        /* phase offset per channel so channels are decorrelated */
        float phase = (float)(s_sim_tick + c * 7u) * 0.05f;
        float raw   = 0.3f * (float)((int)(phase * 100) % 200 - 100) / 100.0f;
        dst[c] = (raw - EMG_MEAN[c]) / EMG_STD[c];
    }
    s_sim_tick++;
}

/* -------------------------------------------------------------------------
 * push_sample  —  write one frame into the ring buffer
 * ------------------------------------------------------------------------- */
static void push_sample(const float frame[EMG_CHANNELS])
{
    for (int c = 0; c < EMG_CHANNELS; ++c) {
        ring_buf[write_head][c] = frame[c];
    }
    write_head = (write_head + 1) % WIN_SAMPLES;
    samples_since_last_infer++;
}

/* -------------------------------------------------------------------------
 * copy_window  —  materialise the current ring buffer into a linear array
 * ------------------------------------------------------------------------- */
static void copy_window(float out[WIN_SAMPLES][EMG_CHANNELS])
{
    /* The oldest sample is at write_head (the slot about to be overwritten) */
    for (int i = 0; i < WIN_SAMPLES; ++i) {
        int src = (write_head + i) % WIN_SAMPLES;
        for (int c = 0; c < EMG_CHANNELS; ++c) {
            out[i][c] = ring_buf[src][c];
        }
    }
}

/* -------------------------------------------------------------------------
 * log_angles  —  RTT log of the most-recent (last) frame's angles
 * ------------------------------------------------------------------------- */
static void log_angles(const float angles[WIN_SAMPLES][N_ANGLES])
{
    const float *last = angles[WIN_SAMPLES - 1];
    /* Log wrist angle (index 21) and a few key MCP angles as a sanity check */
    LOG_INF("angles | wrist=%.3f idx_mcp=%.3f mid_mcp=%.3f",
            (double)last[21],
            (double)last[4],
            (double)last[7]);
}

/* =========================================================================
 * BENCHMARK MODE — UART-based dataset benchmark
 *
 * When CONFIG_ES327_BENCHMARK_MODE is set the firmware enters a host-driven loop:
 *   1. Wait for SYNC byte (0xAA) on UART0 (J-Link bridges this to one /dev/cu.usbmodem*).
 *   2. Send ACK (0x55) + arena_used (4 B little-endian).
 *   3. For each window:
 *      - Receive 1 byte: 0x01 = window follows, 0xFF = end.
 *      - Receive WIN_SAMPLES*EMG_CHANNELS*4 bytes of float32 EMG data.
 *      - Run inference, measure cycle count.
 *      - Send inference_us (4 B LE) + WIN_SAMPLES*N_ANGLES*4 bytes of angles.
 *   4. On END (0xFF), send summary and restart wait loop.
 * ========================================================================= */
#ifdef CONFIG_ES327_BENCHMARK_MODE

#define BENCH_SYNC   0xAA
#define BENCH_ACK    0x55
#define BENCH_WINDOW 0x01
#define BENCH_DEBUG  0x02
#define BENCH_END    0xFF

#define INPUT_BYTES  (WIN_SAMPLES * EMG_CHANNELS * (int)sizeof(float))
#define OUTPUT_BYTES (WIN_SAMPLES * N_ANGLES     * (int)sizeof(float))

static const struct device *uart_dev;

static void uart_write_bytes(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        uart_poll_out(uart_dev, buf[i]);
    }
}

static int uart_read_bytes(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        int rc;
        while ((rc = uart_poll_in(uart_dev, &buf[i])) == -1) {
            k_busy_wait(10);
        }
        if (rc < -1) {
            return rc;
        }
    }
    return 0;
}

static void uart_write_u32(uint32_t val)
{
    uint8_t buf[4];
    buf[0] = (uint8_t)(val);
    buf[1] = (uint8_t)(val >> 8);
    buf[2] = (uint8_t)(val >> 16);
    buf[3] = (uint8_t)(val >> 24);
    uart_write_bytes(buf, 4);
}

static int benchmark_loop(void)
{
    uint32_t arena = (uint32_t)model_runner_arena_used();

    while (1) {
        LOG_INF("Benchmark: waiting for SYNC (0x%02X) on UART0 …", BENCH_SYNC);

        uint8_t sync = 0;
        while (sync != BENCH_SYNC) {
            uart_read_bytes(&sync, 1);
        }

        uint8_t ack = BENCH_ACK;
        uart_write_bytes(&ack, 1);
        uart_write_u32(arena);
        LOG_INF("Benchmark: ACK sent, arena=%u B", arena);

        uint32_t count = 0;
        uint64_t total_us = 0;
        uint32_t min_us = UINT32_MAX, max_us = 0;
        int      errors = 0;

        while (1) {
            uint8_t cmd;
            uart_read_bytes(&cmd, 1);

            if (cmd == BENCH_END) {
                break;
            }
            if (cmd == BENCH_SYNC) {
                /* Host sent SYNC again (e.g. retry) — resync and continue */
                LOG_WRN("SYNC (0xAA) during benchmark — resending ACK");
                uint8_t ack = BENCH_ACK;
                uart_write_bytes(&ack, 1);
                uart_write_u32(arena);
                continue;
            }
            if (cmd == BENCH_DEBUG) {
                /* Host requests output debug: rank, dim1, dim2, raw[8] */
                int rank, dim1, dim2;
                float raw[8];
                model_runner_get_output_debug(&rank, &dim1, &dim2, raw);
                uart_write_u32((uint32_t)rank);
                uart_write_u32((uint32_t)dim1);
                uart_write_u32((uint32_t)dim2);
                uart_write_bytes((const uint8_t *)raw, sizeof(raw));
                continue;
            }
            if (cmd != BENCH_WINDOW) {
                LOG_WRN("Unexpected cmd 0x%02X, skipping", cmd);
                continue;
            }

            uart_read_bytes((uint8_t *)infer_input, INPUT_BYTES);

            /* Use DWT cycle counter (64 MHz CPU clock) for precise timing */
            volatile uint32_t *DWT_CTRL  = (volatile uint32_t *)0xE0001000;
            volatile uint32_t *DWT_CYCCNT = (volatile uint32_t *)0xE0001004;
            volatile uint32_t *DEMCR      = (volatile uint32_t *)0xE000EDFC;
            *DEMCR |= (1 << 24);   /* enable DWT */
            *DWT_CTRL |= 1;        /* enable CYCCNT */
            *DWT_CYCCNT = 0;       /* reset counter */

            int ret = model_runner_infer(infer_input, infer_output);

            uint32_t cycles = *DWT_CYCCNT;
            uint32_t elapsed_us = (uint32_t)(cycles / 64u);  /* 64 MHz -> us */

            if (ret != MODEL_OK) {
                errors++;
                LOG_ERR("Benchmark: inference %u FAILED (ret=%d) after %u us",
                        count, ret, elapsed_us);
            }

            uart_write_u32(elapsed_us);
            uart_write_bytes((const uint8_t *)infer_output, OUTPUT_BYTES);

            count++;
            total_us += elapsed_us;
            if (elapsed_us < min_us) min_us = elapsed_us;
            if (elapsed_us > max_us) max_us = elapsed_us;

            if (count % 10 == 0) {
                gpio_pin_toggle_dt(&led);
            }
        }

        uint32_t avg_us = count > 0 ? (uint32_t)(total_us / count) : 0;
        LOG_INF("Benchmark done: %u inferences, avg=%u us, min=%u us, max=%u us, errors=%d",
                count, avg_us, min_us, max_us, errors);

        uart_write_u32(count);
        uart_write_u32(avg_us);
        uart_write_u32(min_us);
        uart_write_u32(max_us);
    }
    return 0;
}

#endif /* CONFIG_ES327_BENCHMARK_MODE */

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    int ret;

    /* ---- LED init ---- */
    if (!gpio_is_ready_dt(&led)) {
        LOG_ERR("LED not ready");
        return -1;
    }
    ret = gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    if (ret < 0) {
        LOG_ERR("LED configure failed: %d", ret);
        return ret;
    }

    LOG_INF("ES327 EMG inference firmware starting …");

    /* ---- Model init ---- */
    ret = model_runner_init();
    if (ret != MODEL_OK) {
        LOG_ERR("model_runner_init failed: %d", ret);
        return ret;
    }
    LOG_INF("Model loaded — arena used: %u B", (unsigned)model_runner_arena_used());

#ifdef CONFIG_ES327_BENCHMARK_MODE
    LOG_INF("Build: benchmark mode (waiting for SYNC on UART)");
    /* ---- Benchmark: UART-driven dataset test ---- */
    /* UART0 on nRF52840 DK is wired to the J-Link — one of the two /dev/cu.usbmodem* ports. */
    uart_dev = DEVICE_DT_GET(DT_NODELABEL(uart0));
    if (!device_is_ready(uart_dev)) {
        LOG_INF("Benchmark: UART0 not ready — check CONFIG_SERIAL and board overlay");
        return -1;
    }
    LOG_INF("Benchmark mode — UART ready, waiting for SYNC");
    return benchmark_loop();

#else
    LOG_INF("Build: normal (inference loop)");
    /* ---- Normal sensor-driven inference loop ---- */

    /* Pre-fill ring buffer with zeros so first window is valid */
    float zero_frame[EMG_CHANNELS] = {0};
    for (int i = 0; i < WIN_SAMPLES; ++i) {
        push_sample(zero_frame);
    }
    samples_since_last_infer = 0;

    LOG_INF("Entering inference loop (sample period %d ms, stride %d)",
            SAMPLE_PERIOD_MS, STRIDE_SAMPLES);

    uint32_t infer_count = 0u;

    while (1) {
        /* ---- Acquire one EMG frame ---- */
        float frame[EMG_CHANNELS];
        simulate_emg_sample(frame);
        push_sample(frame);

        /* ---- Run inference every STRIDE_SAMPLES new samples ---- */
        if (samples_since_last_infer >= STRIDE_SAMPLES) {
            samples_since_last_infer = 0;

            copy_window(infer_input);

            ret = model_runner_infer(infer_input, infer_output);
            if (ret != MODEL_OK) {
                LOG_ERR("inference failed: %d", ret);
            } else {
                infer_count++;
                /* Toggle LED to show inference is running */
                gpio_pin_toggle_dt(&led);
                /* Log every 10th inference to avoid flooding RTT */
                if (infer_count % 10u == 0u) {
                    LOG_INF("infer #%u", (unsigned)infer_count);
                    log_angles(infer_output);
                }
            }
        }

        k_msleep(SAMPLE_PERIOD_MS);
    }
#endif /* CONFIG_ES327_BENCHMARK_MODE */

    return 0;
}
