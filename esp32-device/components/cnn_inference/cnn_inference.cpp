#include "cnn_inference.h"
#include "sdkconfig.h"

#include <math.h>
#include <float.h>
#include <string.h>

#include "esp_log.h"

// ESP-DL headers
#include "dl_model_base.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_image_define.hpp"
#include "dl_tensor_base.hpp"
#include "fbs_loader.hpp"

static const char *TAG = "CNN";

// ─── embedded model binary (rodata mode only) ─────────────────────────────────
// Used when CONFIG_CNN_MODEL_PARTITION is not set.
// Generate lettercnn.espdl with: python training_pipeline/export_espdl.py
// Flash the model partition with:
//   esptool.py write_flash 0x7C0000 training_pipeline/lettercnn.espdl
#ifndef CONFIG_CNN_MODEL_PARTITION
extern const uint8_t lettercnn_espdl_start[] asm("_binary_lettercnn_espdl_start");
extern const uint8_t lettercnn_espdl_end[]   asm("_binary_lettercnn_espdl_end");
#endif

// ─── class labels ─────────────────────────────────────────────────────────────
// Must match the alphabetically-sorted folder names used during training:
// ['A_Mellifera', 'V_Crabro', 'V_V_Nigrithorax', 'V_Vulgaris']
#define CNN_NUM_CLASSES 3
static const char *const CNN_CLASS_NAMES[CNN_NUM_CLASSES] = {
    "APIS_MELLIFERA",
    "VESPA_CABRO",
    "VESPA_VELUTINA_NIGRITHORAX",
};

// ─── runtime state ────────────────────────────────────────────────────────────
static dl::Model                    *s_model       = nullptr;
static dl::image::ImagePreprocessor *s_preprocessor = nullptr;

// ─── helpers ──────────────────────────────────────────────────────────────────

static void softmax(float *x, int n)
{
    float max_v = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > max_v) max_v = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_v); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ─── public API ───────────────────────────────────────────────────────────────

esp_err_t cnn_inference_init(void)
{
#ifdef CONFIG_CNN_MODEL_PARTITION
    // Load the .espdl model from the "model" flash partition (OTA-updatable).
    // Flash with: esptool.py write_flash 0x7C0000 lettercnn.espdl
    s_model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
#else
    // Load the .espdl model from embedded flash rodata.
    s_model = new dl::Model((const char *)lettercnn_espdl_start,
                             fbs::MODEL_LOCATION_IN_FLASH_RODATA);
#endif
    s_model->build(/* max_internal_size= */ 0);

    // Preprocessing: normalise RGB888 to match training Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    // ImagePreprocessor resizes the input image to the model's input shape automatically.
    const std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    const std::vector<float> std  = {0.5f, 0.5f, 0.5f};
    s_preprocessor = new dl::image::ImagePreprocessor(s_model, mean, std);

#ifdef CONFIG_CNN_MODEL_PARTITION
    ESP_LOGI(TAG, "init ok — model loaded from 'model' partition");
#else
    ESP_LOGI(TAG, "init ok — model %u B (rodata)",
             (unsigned)(lettercnn_espdl_end - lettercnn_espdl_start));
#endif
    return ESP_OK;
}

void cnn_inference_deinit(void)
{
    delete s_preprocessor; s_preprocessor = nullptr;
    delete s_model;        s_model        = nullptr;
}

esp_err_t cnn_inference_run(const uint8_t   *rgb888,
                             const cJSON     *targets,
                             cnn_detection_t *out,
                             size_t           max_out,
                             size_t          *out_count)
{
    *out_count = 0;
    if (!s_model || !s_preprocessor) {
        ESP_LOGE(TAG, "not initialised");
        return ESP_ERR_INVALID_STATE;
    }

    // ── 1. Preprocess ─────────────────────────────────────────────────────────
    dl::image::img_t img = {
        .data     = (void *)rgb888,
        .width    = CNN_INPUT_W,
        .height   = CNN_INPUT_H,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888,
    };
    s_preprocessor->preprocess(img);

    // ── 2. Inference ──────────────────────────────────────────────────────────
    s_model->run();

    // ── 3. Read output tensor and dequantise ──────────────────────────────────
    // The model output is INT8 with a power-of-2 exponent:
    //   float_val = int8_val * 2^exponent
    dl::TensorBase *output = s_model->get_output();
    if (!output || output->size < CNN_NUM_CLASSES) {
        ESP_LOGE(TAG, "unexpected output tensor size %d", output ? output->size : -1);
        return ESP_FAIL;
    }

    float logits[CNN_NUM_CLASSES];
    const float scale = powf(2.0f, (float)output->get_exponent());

    if (output->get_dtype() == dl::DATA_TYPE_INT8) {
        const int8_t *data = output->get_element_ptr<int8_t>();
        for (int i = 0; i < CNN_NUM_CLASSES; i++)
            logits[i] = (float)data[i] * scale;
    } else {
        // Float output (model may not be quantised end-to-end)
        const float *data = output->get_element_ptr<float>();
        for (int i = 0; i < CNN_NUM_CLASSES; i++)
            logits[i] = data[i];
    }

    softmax(logits, CNN_NUM_CLASSES);
    for (int i = 0; i < CNN_NUM_CLASSES; i++) {
        ESP_LOGI(TAG, "  [%d] %-26s %.4f", i, CNN_CLASS_NAMES[i], logits[i]);
    }

    // ── 4. Filter by targets list and per-class threshold ────────────────────
    for (int c = 0; c < CNN_NUM_CLASSES && *out_count < max_out; c++) {
        const char *name   = CNN_CLASS_NAMES[c];
        float       thresh = 0.5f;

        if (targets) {
            bool found = false;
            int  n     = cJSON_GetArraySize((cJSON *)targets);
            for (int t = 0; t < n; t++) {
                cJSON *item = cJSON_GetArrayItem((cJSON *)targets, t);
                cJSON *sp   = cJSON_GetObjectItem(item, "specie");
                if (sp && strcmp(sp->valuestring, name) == 0) {
                    cJSON *thr = cJSON_GetObjectItem(item, "threshold");
                    if (thr) thresh = (float)thr->valuedouble;
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        if (logits[c] >= thresh) {
            snprintf(out[*out_count].specie, CNN_SPECIE_LEN, "%s", name);
            out[*out_count].confidence = logits[c];
            (*out_count)++;
        }
    }

    return ESP_OK;
}
