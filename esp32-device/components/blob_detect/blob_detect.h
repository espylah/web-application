#pragma once
#include <stdint.h>
#include "sdkconfig.h"

#define BLOB_MAX 16

typedef struct {
    uint16_t x, y, w, h;  // bounding box in image pixels
} blob_t;

typedef struct {
    uint8_t  threshold_pct;  // foreground threshold as % of mean brightness (1-99)
                             // threshold = mean * threshold_pct / 100
                             // Lower → only very dark pixels; higher → catches faint insects
    uint16_t min_w;          // reject blobs narrower than this
    uint16_t min_h;          // reject blobs shorter than this
    uint8_t  gap_rows;       // vertical gap (rows) bridged within one blob
    uint8_t  gap_cols;       // horizontal gap (pixels) bridged within one run
} blob_cfg_t;

// Defaults pulled from Kconfig (override via idf.py menuconfig → Blob Detection)
#define BLOB_CFG_DEFAULT {                          \
    .threshold_pct = CONFIG_BLOB_THRESHOLD_PCT,     \
    .min_w         = CONFIG_BLOB_MIN_W,             \
    .min_h         = CONFIG_BLOB_MIN_H,             \
    .gap_rows      = CONFIG_BLOB_GAP_ROWS,          \
    .gap_cols      = CONFIG_BLOB_GAP_COLS,          \
}

/**
 * Find blobs (connected dark regions) in an RGB888 image.
 *
 * Processes the image in a single row-by-row pass — no large intermediate
 * buffers are required beyond a fixed-size blob list.
 *
 * @param rgb888   Packed RGB888 buffer (3 bytes per pixel, R first)
 * @param width    Image width in pixels
 * @param height   Image height in pixels
 * @param cfg      Detection parameters (use BLOB_CFG_DEFAULT to start)
 * @param out      Caller-allocated array of blob_t, capacity max_out
 * @param max_out  Maximum number of blobs to return
 * @return         Number of blobs written to out
 */
int blob_detect_rgb888(const uint8_t *rgb888, uint16_t width, uint16_t height,
                       const blob_cfg_t *cfg, blob_t *out, int max_out);
