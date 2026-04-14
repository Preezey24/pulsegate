# Architecture

Working document. Will evolve into C4 diagrams (context → container → component) as decisions firm up. Current state: component inventory and data-flow sketch, pre-diagram.

---

## Guiding principles

1. **This is a platform project, not a model project.** The architecture exists to demonstrate streaming infra, eval discipline, and production-readiness patterns. The model is a swappable box.
2. **Mirror a proven real-time classification stack deliberately.** FastAPI + Redis Streams + model-behind-wrapper + K8s readiness probes + golden-dataset eval harness. Each element maps to a specific CV bullet from prior text/image/video classification platform work.
3. **Multi-connector + orchestrator pattern from day one.** Mirrors a prior text/image classification platform. One connector ships first (ECG); the second (PCG) is a Week 4 capstone. But routing, message schema, and shared lib are designed for N connectors from the start.
4. **Share code between training, offline eval, and online serving — but not runtime.** One windowing/feature library imported by three distinct processes. Prevents train/serve skew.
5. **Streaming only where streaming is load-bearing.** Redis for the live demo path. Direct function calls for everything else.

---

## Connectors and the orchestrator

A **Connector** owns everything about a specific signal type:
- Source dataset + loader
- Windowing strategy (beat-centered, sliding, whole-recording)
- Feature extraction
- Model artifact for that signal
- Output schema

The **Orchestrator** is a dispatcher: it reads the `signal_type` field on each incoming message and hands the payload to the matching connector's inference path. No business logic, no A/B routing in V1 — just a switch.

```
                  ┌───────────────────┐
  producer(s) ──▶ │  pulsegate:in     │ (single ingress stream)
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   Orchestrator    │ ── dispatch on signal_type
                  └─────┬───────┬─────┘
                        │       │
                ┌───────▼───┐ ┌─▼──────────┐
                │ECG conn.  │ │PCG conn.   │  (future: EEG, PPG, ...)
                │(MIT-BIH)  │ │(CinC 2016) │
                └─────┬─────┘ └──────┬─────┘
                      │              │
                      ▼              ▼
              pulsegate:ecg:out  pulsegate:pcg:out
```

**Planned connectors:**

| Connector | Signal | Dataset | Window | Task | Status |
|---|---|---|---|---|---|
| ECG | 360 Hz voltage | MIT-BIH Arrhythmia | Beat-centered ~250 samples | 5-class AAMI (N/S/V/F/Q) | V1 (Weeks 1-3) |
| PCG | 2000 Hz audio | PhysioNet/CinC 2016 | Multi-second recording-level | Binary normal/abnormal | Stretch (Week 4) |

**Shared `Connector` protocol (sketch):**

```python
class Connector(Protocol):
    signal_type: str                       # "ecg_beat", "pcg_recording", ...
    def window(self, payload) -> Tensor: ...
    def predict(self, window: Tensor) -> Prediction: ...
```

Each connector exposes the same shape to the orchestrator. The orchestrator never imports a specific connector — it resolves them from a registry keyed on `signal_type`.

---

## Three runtime paths

```
                    ┌──────────────────────────────────┐
                    │   pulsegate-core (shared lib)    │
                    │   - window extraction            │
                    │   - feature engineering          │
                    │   - model load/predict interface │
                    └────────────┬─────────────────────┘
                                 │ imported by ↓
       ┌─────────────────────────┼─────────────────────────┐
       │                         │                         │
       ▼                         ▼                         ▼
┌─────────────┐         ┌──────────────────┐      ┌────────────────┐
│  Training   │         │  Offline eval    │      │ Online serving │
│  (one-shot) │         │  (CI harness)    │      │ (FastAPI+Redis)│
└─────────────┘         └──────────────────┘      └────────────────┘
   DS1 records            DS2 records                 DS2 replayed
   → fit model            → score model               → classify live
   → write artifact       → emit metrics JSON         → emit metrics
                          → gate CI on regression     → serve /classify
```

---

## Components (online serving path — the demo)

### 1. Producer

- Reads a DS2 record from disk (`.dat` + `.atr` via `wfdb`).
- Walks the signal sample-by-sample, maintaining simulated wall-clock.
- At each annotated R-peak, extracts a beat-centered window + metadata.
- Publishes to Redis stream `pulsegate:beats:in`.
- Configurable playback rate: 1x (real-time), 10x, 100x for load testing.

**Message shape (draft):**
```json
{
  "signal_type": "ecg_beat",    // orchestrator dispatches on this
  "record_id": "100",
  "beat_index": 1423,
  "r_peak_sample": 512340,
  "samples": [float, ...],      // length ~250 for ECG; differs per connector
  "ground_truth": "N",          // included for eval; stripped in true prod
  "producer_ts": 1234567890.123
}
```

Connector-specific fields live alongside the common fields. The orchestrator only reads `signal_type`; each connector owns interpretation of the rest.

### 2. Consumer / inference worker (FastAPI)

- Reads from `pulsegate:beats:in` via Redis consumer group.
- Runs `model.predict(window)` → class + confidence.
- Writes result to `pulsegate:beats:out`.
- Exposes `/classify` HTTP endpoint for one-shot use (same model, same features — single code path).
- Exposes `/healthz` (liveness) and `/readyz` (readiness — only green once model is loaded and warm).
- Emits Prometheus metrics on every prediction.

### 3. Result stream + scorer

- Reads `pulsegate:beats:out`.
- Compares prediction to ground truth (available in demo mode because we control the producer).
- Updates running per-class confusion matrix, exposed as Prometheus counters.

### 4. Observability

- **Prometheus metrics** — throughput (msg/s), latency histogram (p50/p95/p99), per-class prediction counts, per-class TP/FP/FN, model load time, queue depth.
- **Grafana dashboard** — panels for each of the above, screenshots go in README.

### 5. K8s wrapper (Week 3)

- Deployment for the inference worker.
- Service + Ingress for FastAPI.
- Readiness probe hitting `/readyz` — pod only receives traffic after model is loaded + warmed with a sample inference. Directly parallels the cold-start fix from prior production classification platform work.
- ConfigMap for model artifact path + replay rate.

---

## Components (offline eval path — the CI harness)

- Loads model artifact.
- Iterates DS2 records in-process (no Redis, no FastAPI).
- Uses the **same** window extraction + predict code as the online path (shared lib).
- Emits `eval_report.json`: per-class precision/recall/F1, macro-F1, confusion matrix, record-level breakdown.
- CI compares to `baseline_report.json` on main. Any per-class F1 drop >2% (threshold TBD) fails the build.

---

## Components (training path)

- Loads DS1 records.
- Extracts windows + labels (shared lib).
- Fits classifier (sklearn W1, 1D-CNN W3).
- Writes artifact to `models/v{N}.pkl` (or `.pt`).
- Prints train-time metrics. No Redis, no FastAPI.

---

## Data contract (the spine)

The shared `pulsegate-core` library owns:

- **`extract_window(signal, r_peak_sample) -> np.ndarray`** — deterministic, identical behavior across all three paths.
- **`Features = extract_features(window) -> dict`** — for classical models.
- **`Model` protocol** — `load(path)`, `predict(window_or_features) -> (label, confidence)`. Swap implementations without touching serving code.

If these three things drift between training and serving, the model silently degrades. They are the most important pieces of code in the project and deserve the tightest tests.

---

## Online/offline parity test

Property we get for free from the shared-lib architecture:

> Running DS2 through the online path (Redis → FastAPI → result stream) and through the offline eval harness should produce **identical predictions, beat for beat.**

Any divergence is a bug in the serving path (windowing difference, feature drift, ordering issue). This is a powerful smoke test and a great CV-worthy bullet: *"online/offline parity testing to catch train/serve skew."*

---

## Open architectural questions

- **Backpressure strategy** when consumer falls behind producer — drop, buffer, shed? (Week 2 task.)
- **Horizontal scale** — multiple consumer pods sharing a Redis consumer group. How do we partition?
- **Model versioning** — how does a new model artifact roll out? Blue/green? Shadow traffic?
- **Secrets / config** — where do model paths, Redis URLs, thresholds live? (ConfigMap + env vars, probably.)
- **Artifact storage** — git-lfs? S3-like blob? For V1, local filesystem is fine.

---

## Next step

1. Finish dataset recon (`docs/dataset.md` populated from actual data).
2. Lock window size + message schema based on data inspection.
3. First C4 diagram: Context level (pulsegate in isolation — who calls it, what does it output).
4. Second C4 diagram: Container level (the three runtime paths, Redis, FastAPI, K8s).
