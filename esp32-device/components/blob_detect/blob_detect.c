#include "blob_detect.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "esp_log.h"

static const char *TAG = "BLOB";

// Weighted grayscale: 0.299R + 0.587G + 0.114B via integer arithmetic
static inline uint8_t to_gray(const uint8_t *p)
{
    return (uint8_t)((77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8);
}

// Internal blob state — extends blob_t with the last row it was touched
typedef struct {
    blob_t   bb;
    uint16_t last_row;
} iblob_t;

static void merge_into(iblob_t *dst, const iblob_t *src)
{
    uint16_t x1 = dst->bb.x < src->bb.x ? dst->bb.x : src->bb.x;
    uint16_t y1 = dst->bb.y < src->bb.y ? dst->bb.y : src->bb.y;
    uint16_t x2_d = dst->bb.x + dst->bb.w;
    uint16_t x2_s = src->bb.x + src->bb.w;
    uint16_t y2_d = dst->bb.y + dst->bb.h;
    uint16_t y2_s = src->bb.y + src->bb.h;
    uint16_t x2 = x2_d > x2_s ? x2_d : x2_s;
    uint16_t y2 = y2_d > y2_s ? y2_d : y2_s;
    dst->bb.x = x1; dst->bb.y = y1;
    dst->bb.w = x2 - x1; dst->bb.h = y2 - y1;
    if (src->last_row > dst->last_row) dst->last_row = src->last_row;
}

int blob_detect_rgb888(const uint8_t *rgb888, uint16_t width, uint16_t height,
                       const blob_cfg_t *cfg, blob_t *out, int max_out)
{
    // Pass 1: compute mean brightness so threshold adapts to exposure/lighting
    uint64_t sum = 0;
    size_t npix = (size_t)width * height;
    for (size_t i = 0; i < npix; i++) {
        sum += to_gray(rgb888 + i * 3);
    }
    uint8_t mean = (uint8_t)(sum / npix);
    uint8_t threshold = (uint8_t)((uint32_t)mean * cfg->threshold_pct / 100);
    ESP_LOGI(TAG, "mean=%u  threshold=%u (%u%% of mean)", mean, threshold, cfg->threshold_pct);

    iblob_t blobs[BLOB_MAX];
    int n = 0;

    // Pass 2: scan-line blob detection
    for (uint16_t y = 0; y < height; y++) {
        const uint8_t *row = rgb888 + (size_t)y * width * 3;

        // Walk the row finding dark runs, process each run as it closes
        bool     in_run  = false;
        uint16_t run_x0  = 0;

        for (uint16_t x = 0; x <= width; x++) {
            bool dark = (x < width) && (to_gray(row + x * 3) < threshold);

            if (dark && !in_run) {
                run_x0 = x;
                in_run = true;
            } else if (!dark && in_run) {
                uint16_t run_x1 = x - 1;
                in_run = false;

                // Find all blobs this run overlaps (x and row proximity)
                int match = -1;
                for (int b = 0; b < n; b++) {
                    uint16_t bx0 = blobs[b].bb.x;
                    uint16_t bx1 = blobs[b].bb.x + blobs[b].bb.w;
                    bool x_overlap = (run_x0 <= bx1 + cfg->gap_cols) &&
                                     (run_x1 + cfg->gap_cols >= bx0);
                    bool y_close   = (y <= (uint16_t)(blobs[b].last_row + cfg->gap_rows));

                    if (x_overlap && y_close) {
                        if (match == -1) {
                            match = b;
                        } else {
                            // Two existing blobs connected by this run — merge b into match
                            merge_into(&blobs[match], &blobs[b]);
                            // Remove b by swapping with last
                            if (b < n - 1) blobs[b] = blobs[n - 1];
                            n--;
                            b--;  // recheck this slot after swap
                        }
                    }
                }

                if (match >= 0) {
                    // Expand matched blob to include this run
                    blob_t *bb = &blobs[match].bb;
                    uint16_t bx2 = bb->x + bb->w;
                    uint16_t by2 = bb->y + bb->h;
                    if (run_x0 < bb->x) bb->x = run_x0;
                    if (run_x1 > bx2)   bx2 = run_x1;
                    if (y      > by2)   by2 = y;
                    bb->w = bx2 - bb->x;
                    bb->h = by2 - bb->y;
                    blobs[match].last_row = y;
                } else if (n < BLOB_MAX) {
                    // Start a new blob
                    blobs[n].bb.x      = run_x0;
                    blobs[n].bb.y      = y;
                    blobs[n].bb.w      = run_x1 - run_x0;
                    blobs[n].bb.h      = 0;
                    blobs[n].last_row  = y;
                    n++;
                }
            }
        }
    }

    // Copy blobs meeting minimum size to the output array
    int count = 0;
    for (int b = 0; b < n && count < max_out; b++) {
        if (blobs[b].bb.w >= cfg->min_w && blobs[b].bb.h >= cfg->min_h) {
            out[count++] = blobs[b].bb;
        }
    }
    return count;
}
