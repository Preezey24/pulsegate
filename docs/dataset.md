# Dataset — MIT-BIH Arrhythmia Database

**Source:** PhysioNet — https://physionet.org/content/mitdb/1.0.0/

This document captures the dataset facts that the pulsegate architecture depends on.
Fill in each section after inspecting the data directly. Keep answers terse — this is a
schema reference, not a tutorial.

---

## 1. Provenance & licensing

- **Publisher:**
- **Version used:**
- **License / usage terms:**
- **Citation requirement:**

## 2. Signal acquisition

- **Sampling rate (Hz):**
- **Number of channels / leads:**
- **Lead configuration (e.g. MLII, V1, V5):**
- **ADC resolution (bits) and gain (units/mV):**
- **Signal duration per record:**
- **Total number of records:**
- **Total number of subjects (note: records ≠ subjects):**

## 3. File format

- **On-disk format (e.g. `.dat` + `.hea` + `.atr`):**
- **Reader library:** `wfdb` (Python)
- **What a single record looks like in memory (shape, dtype):**

## 4. Annotations

- **Annotation type (per-beat? per-rhythm? both?):**
- **Annotation symbols present in this dataset:**
- **Annotation cadence (samples between annotations, typical):**
- **Who annotated (cardiologists? automated?):**

## 5. Label space

- **Raw symbol set:**
- **AAMI class mapping (N / S / V / F / Q) — which raw symbols map to which class:**
- **Final class space pulsegate will use:**
- **Per-class counts across the full dataset:**
- **Class imbalance ratio (majority : minority):**

## 6. Splits

- **Standard DS1 / DS2 inter-patient split — record IDs in each:**
- **Train / val / test split pulsegate will use:**
- **Rationale (intra-patient vs inter-patient — and why it matters clinically):**

## 7. Windowing decisions (architecture-relevant)

- **Classification unit: per-beat or per-window?**
- **Window size (samples / ms):**
- **Window stride:**
- **R-peak alignment required?**
- **How are windows constructed at record boundaries?**

## 8. Stream message shape (architecture-relevant)

- **Payload per Redis Streams message:**
- **Estimated payload size (bytes):**
- **Implied throughput at real-time replay (msg/s for one record):**
- **Implied throughput at Nx replay for load testing:**

## 9. Known gotchas

- **Records to exclude (paced beats — 102, 104, 107, 217 per AAMI convention) — confirm:**
- **Records with non-MLII lead 0:**
- **Any noisy or partially annotated records:**

## 10. Open questions

- (list anything ambiguous after first-pass reading — resolve before architecture phase)
