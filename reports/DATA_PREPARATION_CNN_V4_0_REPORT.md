# CNN V4.0 data preparation report

Status: **PASS**. Last phase: complete.

Sources: Al-Emadi binary audio (ae), Drone-detection-dataset audio (dd). Salford is excluded and was not inventoried or processed.

## Execution and fixed protocol

Seed 42; channel 0; 16 kHz; soxr HQ; dd segments are 16,000 samples, non-overlapping float32 WAVs. Al-Emadi is referenced without copying or resampling. No level normalization or dither.

The owner approved the pooled pre-split source-majority audit and the absolute ≤3 dB weak-positive comparison. Band power uses the one-sided SciPy periodogram, boxcar window, no detrending, inclusive 100–4,000 Hz bins and amplitude-1 reference. Zero-energy descriptor values use the existing −240 dB sentinel.

Original Al-Emadi test-use history remains applicable; the new split does not establish previously unseen recordings. No model was loaded or scored; no mel cache, fitted normalization statistics, threshold or runtime package was produced.

## Gates

| Gate | Passed | Observed |
|---|---|---|
| SEED | True | 42 |
| FRONTEND | True | 16 kHz / 1024 / 256 / 64 / 7500 / 7936 / 28 |
| EXISTING_FLAGS | True | ['clip_flags', 'crop_flags', 'eligibility', 'dc_flags', 'groups', 'cleaning'] |
| A2_COUNTS | True | {'unknown': 10372, 'yes_drone': 1332} |
| A2_SUBFAMILIES | True | {'esc': 10000, 'drone': 1332, 'silence': 301, 'other': 71} |
| A2_FORMAT | True | {(16000, 1, 'PCM_16'): 11704} |
| A3_COUNTS | True | {'BACKGROUND': 30, 'DRONE': 30, 'HELICOPTER': 30} |
| A4_LICENSES_READ | True | README/LICENSE read for ae and dd; D031 copied |
| B1_GROUP_COUNTS | True | {'positive': 203, 'esc': 1528} |
| B1_SHORT | True | ['unknown/doing_the_dishes019.wav', 'unknown/silence300.wav'] |
| B1_ZEROS | True | {'clips': 1089, 'contents': 10, 'representatives': 9} |
| B1_KEPT | True | {0: 9291, 1: 1332} |
| C4_ALL_READBACK | True | {'clips': 900, 'all_float32_readbacks_exact': True} |
| D_CROSS_CLASS | True | no float32 identity spans labels |
| D_ZERO_REPRESENTATIVES | True | nine AE representatives; no other retained silence |
| D_NO_DUPLICATES | True | unique retained float32 SHA-256 |
| G-SRC | True | 0.5 |
| RAW_READ_ONLY | True | {'verified_included_source_files': 11794, 'unchanged': 11794} |
| G-GRP | True | one split per source-qualified group |
| G-DUP | True | one split per float32 identity |
| G-FRAC | True | {0: {'val': 0.14229329015281855, 'test': 0.14330533346827246}, 1: {'val': 0.13786764705882354, 'test': 0.13480392156862744}} |
| G-POS | True | {'test': 32, 'train': 168, 'val': 33} |
| G-STR | True | {'test': {'0/ae': 219, '0/dd': 9, '1/ae': 28, '1/dd': 4}, 'train': {'0/ae': 1102, '0/dd': 42, '1/ae': 146, '1/dd': 22}, 'val': {'0/ae': 219, '0/dd': 8, '1/ae': 29, '1/dd': 4}} |
| F_PINNED | True | 68 |
| F_SMOKE | True | 20 training clips per included source, shape (1,64,28), finite |

## inventory counts

```json
{
  "ae": {
    "unknown": 10372,
    "yes_drone": 1332
  },
  "ae_drone_types": {
    "none": 10372,
    "bebop": 666,
    "membo": 666
  },
  "dd": {
    "BACKGROUND": 30,
    "DRONE": 30,
    "HELICOPTER": 30
  }
}
```

## ae cleaning

```json
{
  "input": 11704,
  "retained": 10623,
  "rejections": {
    "exact_zero_duplicate": 1079,
    "too_short": 2
  },
  "zero_representatives": [
    "unknown/1-100032-A-00.wav",
    "unknown/1-100032-A-04.wav",
    "unknown/1-103995-A-302.wav",
    "unknown/silence000.wav",
    "unknown/silence001.wav",
    "unknown/silence003.wav",
    "unknown/silence006.wav",
    "unknown/silence011.wav",
    "unknown/silence299.wav"
  ],
  "positive_groups": 203
}
```

## dd segmentation

```json
{
  "recordings": 90,
  "written_segments": 900,
  "class_counts": {
    "BACKGROUND": 300,
    "DRONE": 300,
    "HELICOPTER": 300
  },
  "discarded_samples": 48,
  "background_median_band_dbfs": -41.065028095956485,
  "weak_positive": 25
}
```

## duplicates

```json
{
  "groups": 17,
  "cross_source_groups": 0,
  "exclusions": {
    "exact_zero_duplicate": 1079,
    "exact_float32_duplicate": 10,
    "too_short": 2
  },
  "retained": 11513
}
```

## source only

```json
{
  "balanced_accuracy": 0.5,
  "class_recalls": {
    "0": 1.0,
    "1": 0.0
  },
  "majority": {
    "ae": 0,
    "dd": 0
  },
  "population": "pooled surviving clips before split; owner-approved",
  "ties": "STOP; no tie-break rule is invented",
  "is_model_evaluation": false
}
```

## split counts

```json
[
  {
    "split": "test",
    "source": "ae",
    "label": 0,
    "clips": 1326,
    "groups": 219,
    "seconds": 1375.7908750000001
  },
  {
    "split": "test",
    "source": "ae",
    "label": 1,
    "clips": 180,
    "groups": 28,
    "seconds": 180.38400000000001
  },
  {
    "split": "test",
    "source": "dd",
    "label": 0,
    "clips": 90,
    "groups": 9,
    "seconds": 90.0
  },
  {
    "split": "test",
    "source": "dd",
    "label": 1,
    "clips": 40,
    "groups": 4,
    "seconds": 40.0
  },
  {
    "split": "train",
    "source": "ae",
    "label": 0,
    "clips": 6639,
    "groups": 1102,
    "seconds": 6818.406937500001
  },
  {
    "split": "train",
    "source": "ae",
    "label": 1,
    "clips": 967,
    "groups": 146,
    "seconds": 967.21025
  },
  {
    "split": "train",
    "source": "dd",
    "label": 0,
    "clips": 420,
    "groups": 42,
    "seconds": 420.0
  },
  {
    "split": "train",
    "source": "dd",
    "label": 1,
    "clips": 220,
    "groups": 22,
    "seconds": 220.0
  },
  {
    "split": "val",
    "source": "ae",
    "label": 0,
    "clips": 1326,
    "groups": 219,
    "seconds": 1375.44475
  },
  {
    "split": "val",
    "source": "ae",
    "label": 1,
    "clips": 185,
    "groups": 29,
    "seconds": 185.224
  },
  {
    "split": "val",
    "source": "dd",
    "label": 0,
    "clips": 80,
    "groups": 8,
    "seconds": 80.0
  },
  {
    "split": "val",
    "source": "dd",
    "label": 1,
    "clips": 40,
    "groups": 4,
    "seconds": 40.0
  }
]
```

## smoke

```json
{
  "clips": 40,
  "split": "train",
  "shape": [
    1,
    64,
    28
  ],
  "finite": true,
  "normalization": "identity placeholders only to exercise canonical tensor interface; no data-fitted statistics computed or persisted"
}
```

## manifest sha256

```json
{
  "full": "2407e2a3f31beebd86134b4423f14a4a3b17b65d64ce9bfaeb01156a064d4a39",
  "train": "6b895c0d6ce420aa7452bed72c06f0db14eb472490ab82bd99de91276ef24d81",
  "val": "6fc5134f8faa3875e088d7f84fdc3d03debb2de0466cb73c5624410e754430e9",
  "test": "11606d8fb9d597d579b53c996980422d2cb8ecec7dc2df9eca776dd775da813c"
}
```

## warnings

```json
[
  {
    "source": "ae",
    "median_level_difference_db": 0.25077677201791104,
    "median_rolloff_difference_hz": 2276.6959616131862,
    "level_warning": false,
    "rolloff_warning": true
  },
  {
    "source": "dd",
    "median_level_difference_db": 1.2071524106633973,
    "median_rolloff_difference_hz": 2140.0,
    "level_warning": false,
    "rolloff_warning": true
  }
]
```

## Evidence and limitations

Detailed CSV/JSON/PNG evidence is in storage/corpora/corpus-002/audits; source license records and environment provenance are beside it. A requirement audit is recorded separately after completion verification.

Al-Emadi was recorded indoors and covers Bebop/Membo. dd has only 30 original recordings per class. Quiet or distant drone segments retain source labels; weak_positive is a descriptor flag, not verified audibility. Calibration and segment-label uncertainty are reasons Salford is excluded.

Rejected clips too short for a canonical crop have crop_all_zero and stats_eligible recorded as not applicable (empty CSV values), rather than fabricated window measurements. No such clip enters a retained split.
