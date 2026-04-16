# Dataset — MIT-BIH Arrhythmia Database

**Source:** PhysioNet — https://physionet.org/content/mitdb/1.0.0/

This document captures the dataset facts the pulsegate architecture depends on. Mix of PhysioNet-published metadata and empirical findings from inspecting records 100, 102, 203, and 232 via `scripts/inspect_record.py`.

---

## 1. Provenance & licensing

- **Publisher:** Massachusetts Institute of Technology (MIT) and Beth Israel Hospital (now Beth Israel Deaconess Medical Center), Boston.
- **Distributed by:** PhysioNet.
- **Version used:** 1.0.0 (the canonical, mature release).
- **License:** Open Data Commons Attribution License v1.0 (ODC-By 1.0).
- **Citation requirement:** standard PhysioNet citation — Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database*. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

## 2. Signal acquisition

- **Sampling rate:** **360 Hz** per channel. (Confirmed across records 100, 102, 203, 232.)
- **Number of channels:** **2** (simultaneous two-lead recording).
- **Lead configuration:** typically **MLII on channel 0** + a precordial lead (V1, V2, V5, etc.) on channel 1. Per-record variation — see §9.
- **ADC resolution:** 11-bit over 10 mV range → ~5 µV per ADC step. The smallest signal change the equipment can represent.
- **Stored as:** float64 in memory (after `wfdb`'s int → mV conversion). Architecture casts to **float32 for transport** — float32's quantization (~0.1 µV) is ~50× finer than the underlying ADC resolution, so no information loss.
- **Voltage range observed:** roughly ±1 to ±4 mV, varies significantly per patient (record 100 std 0.19 mV vs record 203 std 0.50 mV — 2.6× variability).
- **Duration per record:** 30 minutes (650,000 samples = 1,805.6 s).
- **Total records:** **48**.
- **Subjects:** 47 (one subject contributed two recordings).
- **Recording era:** 1975-1979, ambulatory Holter tapes from Beth Israel Hospital.

## 3. File format

Three companion files per record (each `.dat` requires its `.hea`; `.atr` carries the labels):

| Extension | Format | Contents |
|---|---|---|
| `.dat` | Binary, WFDB format 212 (two 12-bit samples per 3 bytes) | Raw signal data for both channels |
| `.hea` | Plain text | Header — fs, n_sig, lead names, units, ADC gain/zero, file format |
| `.atr` | Binary | Annotations: list of `(sample_index, symbol)` pairs |

- **Reader library:** `wfdb` (Python).
- **In-memory shape:** `record.p_signal` is `(N, 2)` float64 array where N ≈ 650,000.
- **Annotation object:** `wfdb.rdann` returns parallel arrays `annotation.sample` (int sample indices) and `annotation.symbol` (single-char strings).

## 4. Annotations

- **Annotation type:** mixed per-beat AND non-beat events in the same `.atr` file. Filtering required.
- **Annotation cadence:** ~1-2 per second on average (matches resting heart rate).
- **Annotated by:** two independent cardiologists per record; disagreements resolved to consensus. This is what makes MIT-BIH the gold standard.
- **Total beat annotations across the database:** ~110,000 (PhysioNet figure; sample records: 100 has 2,274; 203 has 3,108; 102 has 2,192; 232 has 1,816).
- **Annotation is per-beat, not per-channel.** A single sample index marks one beat event in the recording as a whole; both channels recorded that beat simultaneously. The label describes the beat, not a waveform.

## 5. Label space

### Beat symbols (the ones we classify)

Empirically observed across 4 sample records:

| Symbol | Meaning | AAMI class |
|---|---|---|
| `N` | Normal beat | **N** |
| `L` | Left bundle branch block | **N** |
| `R` | Right bundle branch block (seen in 232) | **N** |
| `e` | Atrial escape | **N** |
| `j` | Nodal (junctional) escape (seen in 232) | **N** |
| `A` | Atrial premature (dominant in 232) | **S** |
| `a` | Aberrated atrial premature (seen in 203) | **S** |
| `J` | Nodal (junctional) premature | **S** |
| `S` | Supraventricular premature | **S** |
| `V` | Premature ventricular contraction (444 in 203) | **V** |
| `E` | Ventricular escape | **V** |
| `F` | Fusion of ventricular and normal (1 in 203) | **F** |
| `/` | Paced beat (dominant in 102) | **Q** |
| `f` | Fusion of paced and normal (seen in 102) | **Q** |
| `Q` | Unclassifiable (seen in 203) | **Q** |
| `?` | Unclassifiable | **Q** |

### Non-beat symbols (filtered out before windowing)

| Symbol | Meaning |
|---|---|
| `+` | Rhythm change marker (with `aux_note` naming the new rhythm) |
| `~` | Signal quality change (noise begins/ends) |
| `\|` | Isolated QRS-like artifact |
| `[` `]` | Start/end of ventricular flutter region |
| `"` | Comment |

### AAMI EC57 5-class output space

pulsegate's V1 ECG model outputs one of 5 classes per beat:

- **N** — Normal + bundle branch blocks (origin is normal sinus, electrical conduction may be delayed)
- **S** — Supraventricular ectopic (early beats from atria)
- **V** — Ventricular ectopic (early beats from ventricles — clinically the most important class to catch)
- **F** — Fusion (normal + ventricular blend)
- **Q** — Unknown / unclassifiable / paced

### Class imbalance — confirmed in sample data

MIT-BIH is overwhelmingly N. Per-record class distribution varies wildly:

| Record | N-class % | Other classes |
|---|---|---|
| 100 | 98.5% | 1.5% S |
| 203 | 81% N | 14% V, plus rare F, Q, S |
| 232 | ~98% N (counting `R` as N) | 76% of all beats are `A` (S-class) — exception case |

Implication: **per-class F1 metric is mandatory.** Overall accuracy would mask catastrophic failure on rare classes.

## 6. Splits

### Standard DS1 / DS2 inter-patient split (Chazal et al. 2004)

**DS1 (training, 22 records):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230

**DS2 (testing, 22 records):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

**Excluded (4 records):** 102, 104, 107, 217 — paced beats per AAMI EC57. Records 102 and 104 also lack MLII entirely (channel 0 = V5 in 102), giving two reasons for exclusion.

**Total usable:** 44 of 48 records. No patient appears in both DS1 and DS2.

### Why this specific split

- **Inter-patient:** prevents the model memorizing patient-specific morphology (intra-patient splits inflate scores meaninglessly).
- **Class-balanced:** Chazal et al. hand-curated the split so every AAMI class has enough examples in both halves to produce stable per-class F1.
- **Community-standard:** every ECG paper since 2004 reports against DS1/DS2, making our metrics directly comparable.

### Within-DS1: train/val split

For internal model selection, hold out a few DS1 records as a validation set. Decision deferred to model-training time.

## 7. Windowing decisions (architecture-relevant)

- **Classification unit:** **per-beat**, not per-window.
- **Window size:** ~250 ms before R-peak + ~450 ms after R-peak = ~700 ms total.
- **At 360 Hz:** ~250 samples per window. Refine after testing on real R-peak placements.
- **Window stride:** N/A — windows are R-peak-anchored, one per beat annotation.
- **R-peak alignment:** required. Use the annotation's sample index as the R-peak.
- **Record boundaries:** for the first/last few beats of a record, windows may extend past the recording; pad with edge value or skip those beats.
- **Pre-processing:** **per-beat z-score normalization** (`(window - mean) / std`) to make the model invariant to per-patient voltage scale. Confirmed necessary by the wide voltage-range variation across records.

## 8. Stream message shape (architecture-relevant)

```json
{
  "signal_type": "ecg_beat",
  "request_id": "uuid-...",
  "virtual_patient_id": "42",
  "record_id": "100",
  "beat_index": 1423,
  "r_peak_sample": 512340,
  "samples": [float32, ...],
  "ground_truth": "N",
  "producer_ts": 1234567890.123
}
```

- **Payload size:** ~250 floats × 4 bytes (float32) = **~1 KB samples** + ~100 B metadata = **~1.1 KB per beat message**.
- **Beat rate per virtual patient at 1× playback:** ~1-1.5 msg/sec (matches natural heart rate).
- **Throughput at load-test setting** (22 records × 10 virtual-patient copies × 100× speed) ≈ **~22,000 msg/sec**.

## 9. Known gotchas

- **Records to exclude (AAMI EC57):** 102, 104, 107, 217 (paced beats). Confirmed via record 102 (92.5% `/` annotations).
- **Records lacking MLII on channel 0:** at least 102 (V5/V2 instead) and 104 (V5/V2). Required handling: select MLII channel by name from `record.sig_name`, not by index.
- **Non-beat annotations mixed in:** `+`, `~`, `|`, `[`, `]`, `"` must be filtered out before windowing or computing R-R intervals. Otherwise temporal features include non-physiological tiny intervals (e.g. record 203's spurious 0.028s "R-R" was actually a beat → noise marker gap).
- **Lowercase vs uppercase symbols differ:** `A` vs `a`, `j` vs `J`, `f` vs `F`. Both cases are valid distinct symbols with their own AAMI mapping.
- **Long pauses exist in real data:** record 232 has a 5.87s gap between two annotations — likely a real pathological pause (sinus arrest), not data corruption. Don't filter these out as outliers.
- **Voltage scale varies 2-3× across patients** — per-beat normalization required.

## 10. Open questions

- **Exact window size in samples** — start with ~250 (250 ms before + 450 ms after at 360 Hz), but verify R-peak placement on real beats first. Some literature uses different splits (e.g. 100 ms before + 150 ms after). Refine after first model training.
- **Per-record val split within DS1** — defer to training-script time. Likely 4-5 records held out, or k-fold within DS1.
- **What to do with `Q` class examples that aren't from paced records** — there are 4 in record 203. Keep as `Q`, or drop entirely from the eval? Standard literature drops them. Decide at eval-harness time.
- **Two-channel input** — V1 uses channel 0 (MLII) only. Week 3 upgrade to 2-channel input for the 1D-CNN. Need per-record lead-identity check to avoid concatenating MLII with whatever-channel-1-is.
