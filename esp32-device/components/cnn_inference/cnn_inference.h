#pragma once

#include "esp_err.h"
#include "cJSON.h"
#include <stdint.h>
#include <stddef.h>

// Fixed input dimensions expected by the model.
#define CNN_INPUT_W   32
#define CNN_INPUT_H   32

#define CNN_SPECIE_LEN 64

// A single classification result.
typedef struct {
    char  specie[CNN_SPECIE_LEN];   // e.g. "APIS_MELLIFERA"
    float confidence;               // 0.0 – 1.0 (softmax probability)
} cnn_detection_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate scratch buffers and prepare the inference engine.
 * Must be called once before cnn_inference_run().
 */
esp_err_t cnn_inference_init(void);

/**
 * Run INT8 CNN inference on a single 64×64 RGB888 patch.
 *
 * @param rgb888     Caller-supplied buffer: CNN_INPUT_H × CNN_INPUT_W × 3 bytes,
 *                   packed RGB888 (R first), row-major.
 * @param targets    cJSON array of {"specie":"...","threshold":0.7}.
 *                   Only classes that appear in this list AND meet their threshold
 *                   are written to out.  Pass NULL to return all classes above 0.5.
 * @param out        Caller-allocated array of cnn_detection_t, capacity max_out.
 * @param max_out    Maximum number of results to write.
 * @param out_count  Set to the number of results written.
 *
 * Returns ESP_OK, ESP_ERR_INVALID_STATE if init was not called,
 * or ESP_ERR_NO_MEM if scratch allocation failed.
 */
esp_err_t cnn_inference_run(const uint8_t   *rgb888,
                             const cJSON     *targets,
                             cnn_detection_t *out,
                             size_t           max_out,
                             size_t          *out_count);

/**
 * Free scratch buffers allocated by cnn_inference_init().
 */
void cnn_inference_deinit(void);

#ifdef __cplusplus
}
#endif
