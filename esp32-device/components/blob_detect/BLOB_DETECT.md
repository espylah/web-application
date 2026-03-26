# Blob Detection — How It Works

The `blob_detect` component finds dark objects against a lighter background using a
single two-pass scan of the RGB888 image. No large intermediate buffers are needed —
only a small fixed-size list of active blobs is kept in memory.

---

## Pass 1 — Adaptive Threshold

Before detecting anything, the algorithm computes the **mean brightness** of the
entire image by converting every pixel to grayscale:

```
gray = (77·R + 150·G + 29·B) / 256
```

This is an integer approximation of standard luminance weighting
(0.299R + 0.587G + 0.114B).

The **detection threshold** is set as a percentage of that mean:

```
threshold = mean_brightness × threshold_pct / 100
```

`threshold_pct` defaults to **65**. Any pixel whose brightness falls **below** the
threshold is treated as foreground (insect).

Because the threshold scales with the mean, it remains correct across different
exposures and lighting conditions:

| Scene | mean | threshold (65%) |
|---|---|---|
| Dark / underexposed | 80 | 52 |
| Typical indoor | 140 | 91 |
| Bright / overexposed | 200 | 130 |

A fixed delta (`mean - 50`) would collapse to near-zero in a dark scene, missing
everything. A percentage stays proportional — if an insect is 40% darker than the
background, it is detected regardless of overall exposure.

---

## Pass 2 — Scan-Line Blob Detection

The image is scanned **left-to-right, top-to-bottom**, one row at a time.

### Step 1 — Find dark runs

For each row, the algorithm finds contiguous spans of foreground pixels called
**runs**:

```
Row y:  . . . . ■ ■ ■ ■ . . . ■ ■ . . .
                ^-------^     ^--^
                run A         run B
```

Each run has an `x0` (first dark pixel) and `x1` (last dark pixel).

### Step 2 — Match runs to existing blobs

Each run is compared against every currently tracked blob. A run is matched to a
blob if both conditions are met:

| Condition | Meaning |
|---|---|
| `run_x0 ≤ blob_x1 + gap_cols` | The run starts within `gap_cols` pixels of the blob's right edge |
| `run_x1 + gap_cols ≥ blob_x0` | The run ends within `gap_cols` pixels of the blob's left edge |
| `y ≤ blob.last_row + gap_rows` | The row is within `gap_rows` rows of the last row the blob appeared on |

The **gap tolerances** allow the algorithm to bridge small holes — for example an
insect with a slightly lighter band across its body, or a narrow gap between two
parts of the same insect.

### Step 3 — Update or create blobs

**If a matching blob is found:**
The blob's bounding box is expanded to include the new run, and its
`last_row` is updated to the current row `y`.

```
Before:            After (run extends right edge):
┌──────┐           ┌─────────┐
│ blob │     +  ■■ │  blob   │
└──────┘           └─────────┘
```

**If a run matches two blobs simultaneously:**
The two blobs are merged into one. This handles cases where two nearby insects were
tracked separately but a run bridges them:

```
blob A    blob B        merged
┌────┐    ┌────┐       ┌──────────┐
│    │    │    │  →    │          │
└────┘    └────┘       └──────────┘
          ■■■■■■  (connecting run)
```

**If no matching blob is found:**
A new blob is created starting at the current run position.

---

## Pass 2 complete — Filter by size

After the full image has been scanned, any blob whose bounding box is smaller than
`min_w × min_h` is discarded. This removes single-pixel noise and camera artefacts.

---

## Configuration

All parameters are configurable via `idf.py menuconfig → Blob Detection`:

```c
blob_cfg_t cfg = BLOB_CFG_DEFAULT;   // picks up Kconfig values automatically
```

| Parameter (Kconfig) | Default | Effect of increasing |
|---|---|---|
| `BLOB_THRESHOLD_PCT` | 65 | Catches fainter insects; also more false positives from shadows |
| `BLOB_MIN_W` | 8 px | Larger minimum — removes more noise, may drop small insects |
| `BLOB_MIN_H` | 8 px | Same, in height |
| `BLOB_GAP_ROWS` | 4 rows | Bridges larger vertical gaps; risk of merging adjacent insects |
| `BLOB_GAP_COLS` | 8 px | Bridges larger horizontal gaps; same risk |

---

## Memory usage

| Item | Size |
|---|---|
| Blob list (`BLOB_MAX = 16`) | 16 × 8 bytes = 128 bytes |
| No grayscale image stored | — |
| No pixel label map stored | — |

The only significant memory cost is the RGB888 input buffer itself
(320 × 320 × 3 = **~300 KB** in PSRAM).

---

## Limitations

- **One global threshold** — a large shadow that is only slightly darker than the
  mean can inflate the threshold and cause real insects to be missed. Increasing
  `darkness_delta` helps.
- **Merging is permanent** — once two blobs merge they cannot be split. If insects
  overlap or are very close they will be detected as one blob.
- **Background must be reasonably uniform** — heavily textured or multi-toned
  backgrounds will produce false positives.
