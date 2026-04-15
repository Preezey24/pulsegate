# Architecture

Working document. Will evolve into C4 diagrams (context → container → component) as decisions firm up. Current state: component inventory and data-flow sketch, pre-diagram.

---

## Guiding principles

1. **This is a platform project, not a model project.** The architecture exists to demonstrate streaming infra, eval discipline, and production-readiness patterns. The model is a swappable box.
2. **Mirror a proven real-time classification stack deliberately.** FastAPI + Redis Streams + model-behind-wrapper + K8s readiness probes + golden-dataset eval harness. Each element maps to a specific CV bullet from prior text/image/video classification platform work.
3. **Multi-connector dispatch pattern from day one.** Mirrors a prior text/image classification platform. One processing connector ships first (ECG); the second (PCG) is a Week 4 capstone. Single Gateway at ingress dispatches to per-type Redis streams — never split per-type. Routing, message schema, and shared lib are designed for N connectors from the start.
4. **Share code between training, offline eval, and online serving — but not runtime.** One windowing/feature library imported by three distinct processes. Prevents train/serve skew.
5. **Streaming only where streaming is load-bearing.** Redis for the live demo path. Direct function calls for everything else.

---

## Serving topology (online path)

Layered ingress in front of per-signal-type worker pools, with a Redis stream pair per signal type for request and response. The Gateway is a single service that owns HTTP, auth, dispatch on `signal_type`, and response correlation — a unified ingress layer with no separate dispatcher service.

```
                    ┌──────────────────────┐
  Internet ───────▶ │  Ingress + TLS       │  (NGINX + cert-manager)
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────────────────┐
                    │  Gateway (single service, HPA)   │
                    │  - auth, rate limit, validate    │
                    │  - assign request_id             │
                    │  - dispatch on signal_type       │  internal switch
                    │  - XADD to matching :in stream   │
                    │  - background XREAD on :out      │
                    │  - resolve future, reply caller  │
                    └───┬──────────────────────────┬───┘
                        │                          │
                        ▼                          ▼
                 ┌────────────┐            ┌────────────┐
                 │ ecg:in     │            │ pcg:in     │  Redis (AOF + replica)
                 └─────┬──────┘            └─────┬──────┘
                       │                          │
                       ▼                          ▼
                 ┌──────────────────────┐   ┌──────────────────────┐
                 │ ECG worker pod       │   │ PCG worker pod       │
                 │  N consumer tasks    │   │  N consumer tasks    │  1 consumer / shard
                 │  (1 per shard owned) │   │  (1 per shard owned) │  shared model
                 │  ONE shared model    │   │  ONE shared model    │  shared context cache
                 │  per-patient cache   │   │  per-patient cache   │  in-process V1
                 └──────┬───────────────┘   └──────┬───────────────┘   sidecar stretch
                        │                         │
                        ▼                         ▼
                 ┌────────────┐            ┌────────────┐
                 │ ecg:out    │            │ pcg:out    │
                 └─────┬──────┘            └─────┬──────┘
                       │                          │
                       └──────────────┬───────────┘
                                      ▼
                           back into Gateway's
                        long-lived XREAD BLOCK task
```

### Layer responsibilities

| Layer | Purpose | Scaling |
|---|---|---|
| **Ingress / LB** (NGINX) | TLS termination, hostname/path routing, L4/L7 LB across Gateway pods. Network-layer only — no application logic. | Few pods |
| **Gateway** (FastAPI) | Auth, `request_id` assignment, payload validation, dispatch on `signal_type` to matching `:in` stream, request/response correlation via in-memory `asyncio.Future` + long-lived `XREAD BLOCK` on `:out` streams. Single service; never split per-type. | HPA on CPU, `minReplicas: 2`, `maxReplicas: 20` |
| **Redis Streams** | Durable (AOF + replica), at-least-once via consumer-group ACK, per-patient order via sharded streams keyed on `hash(virtual_patient_id)`. | Primary + 1 replica |
| **Worker (per signal type)** | Consumer-group reads `:in`, runs model, writes `:out`, XACKs. **Multiple consumer coroutines per pod, one per shard**, sharing a single model instance and a per-patient context cache in-process. Model colocated in-process (V1) or as sidecar container (stretch). | Deployment per type, HPA per type. Shard count is a fixed topology decision; pod count is the HPA dial. |

### Connector (processing-side) — owns one signal type's pipeline

A **Connector** (the processing kind, running in a Worker pod) owns everything about a specific signal type:
- Source dataset + loader
- Windowing strategy (beat-centered, sliding, whole-recording)
- Feature extraction
- Model artifact for that signal
- Output schema

Distinct from the **Gateway** (ingress + dispatcher) which is a single service that routes to the right `:in` stream based on `signal_type`. The Gateway owns HTTP concerns; the processing Connector owns everything about a signal's pipeline.

**Planned processing connectors:**

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

Each worker pod hosts exactly one processing connector (one signal type per Deployment). The Gateway never imports a processing connector — the decoupling is via the stream name.

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

Each processing connector exposes the same shape. The Gateway never imports one — dispatch is by `signal_type` to the matching Redis `:in` stream, and the worker Deployment hosts the connector.

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

### 1. Producer / Load generator

- Runs as a **Kubernetes Job** launched via `scripts/loadgen.sh`. Not a laptop-side script.
- Each pod spawns K async producer tasks, one per **virtual patient**.
- Each producer task reads a DS2 record from disk (baked into the image for V1) and walks through its beats in order.
- **Staggered start offsets** across copies of the same record prevent lockstep emission.
- At each annotated R-peak, extracts a beat-centered window + metadata.
- HTTP POSTs each beat to the Gateway's cluster-internal Service (`http://gateway.pulsegate.svc.cluster.local`). Not directly to Redis, and not through the public Ingress — this exercises the real Gateway → Redis dispatch path.
- Knobs exposed via env vars: `RECORDS`, `COPIES_PER_RECORD`, `SPEED`, `DURATION`.
- Scale levers: **record variety** (22 real DS2) × **virtual-patient fan-out** (copies per record) × **playback speed** (Nx). See design-notes §6 for the throughput math and design-notes §7 for the K8s deployment shape.

**Message shape (draft):**
```json
{
  "signal_type": "ecg_beat",    // Gateway dispatches on this
  "record_id": "100",
  "beat_index": 1423,
  "r_peak_sample": 512340,
  "samples": [float, ...],      // length ~250 for ECG; differs per connector
  "ground_truth": "N",          // included for eval; stripped in true prod
  "producer_ts": 1234567890.123
}
```

Connector-specific fields live alongside the common fields. The Gateway only reads `signal_type` (for dispatch) and `virtual_patient_id` (for the shard hash); the processing connector in the worker owns interpretation of the rest.

### 2. Worker (consumer + colocated model, per signal type)

- One Deployment per signal type (`ecg-worker`, `pcg-worker`, ...).
- Each pod runs **N consumer coroutines**, where N = number of shards owned by this pod.
- Coroutines share:
  - **One model instance** (thread-safe for inference; no duplication).
  - **One per-patient context cache** (in-process dict keyed on `virtual_patient_id`, stores rolling beat history for temporal features like R-R intervals).
- Each coroutine loops: `XREADGROUP` from its shard → `connector.predict(window)` → `XADD {signal}:out` with matching `request_id` → `XACK`.
- Exposes `/healthz` (liveness) and `/readyz` (readiness — only green once model is loaded + warmed with a sample inference).
- Emits Prometheus metrics on every prediction.
- **V1 colocation:** model runs in-process, not as a separate API. No network hop between consumer and model. Rationale: 1D-CNN is tiny (~500k params), GPU not needed, colocation saves inference-path latency.
- **Stretch:** sidecar model container in the same pod, connected via localhost gRPC, for decoupled rollout and language-agnostic model runtimes.

### 3. Result stream + scorer

- Reads `{signal}:out` streams.
- Compares prediction to ground truth (available in demo mode because we control the producer and include `ground_truth` on replayed messages).
- Updates running per-class confusion matrix, exposed as Prometheus counters.
- In production (no ground truth in the message), the scorer is either omitted or runs as a shadow job against a held-out dataset.

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
- **Sharding activation** — V1 ships 1 shard per `:in` stream. What measured throughput triggers turning on sharded streams by `hash(virtual_patient_id)`?
- **Model versioning** — how does a new model artifact roll out? Blue/green via separate worker Deployments + Gateway routing rules? Shadow traffic?
- **Gateway scale-out** — each gateway pod holds its own `request_id → future` map. If ingress sticky-sessions break, a response could arrive at a different pod than the one holding the caller. Solvable via Redis pub/sub fanout of responses to all gateway pods, or by making the gateway strictly sticky.
- **Secrets / config** — where do model paths, Redis URLs, thresholds live? (ConfigMap + env vars, probably.)
- **Artifact storage** — git-lfs? S3-like blob? For V1, local filesystem is fine.

---

## Next step

1. Finish dataset recon (`docs/dataset.md` populated from actual data).
2. Lock window size + message schema based on data inspection.
3. First C4 diagram: Context level (pulsegate in isolation — who calls it, what does it output).
4. Second C4 diagram: Container level (the three runtime paths, Redis, FastAPI, K8s).
