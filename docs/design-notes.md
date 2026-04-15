# Design Notes

Running log of design decisions made during pulsegate. Each entry captures *what* was decided, *why*, and *what alternatives were rejected*. This is the "show your working" doc — future-you (or an interviewer) should be able to read this and understand the reasoning behind every non-obvious choice.

---

## 1. Window strategy — beat-centered, not fixed-sliding

**Decision:** Classify one heartbeat at a time. Each model input is a fixed-size window of ECG samples centered on an R-peak.

**Concrete shape (starting point, to refine after data inspection):**
- ~250 ms before R-peak + ~450 ms after R-peak
- At 360 Hz → ~250 samples per window
- One beat annotation → one label → one training example

**Why beat-centered:**
- MIT-BIH annotations are **per-beat**, attached to the R-peak sample. The labels already align naturally to beat-centered windows.
- The clinical question on this dataset is "what type of beat is this?" (N / S / V / F / Q), not "what rhythm is this stretch?" Beat-centered windows match the clinical unit of decision.
- Standard approach in the MIT-BIH literature — makes our numbers directly comparable to published work.

**Alternative considered: fixed sliding windows** (e.g. 10-sec chunks every 5 sec)
- Rejected because:
  - Doesn't match the per-beat annotation structure — you'd have to aggregate multiple beats into one label, losing information.
  - Better suited to rhythm-level tasks (AFib detection on MIT-BIH AFib DB), not beat classification.
  - Would make results harder to compare to the published benchmark line.

**Message structure on the wire:**

A beat window has two logical parts:
- **Samples** — a flat array of ~250 float32 values (the voltage readings around the R-peak). Only this is fed to the model.
- **Metadata** — a small dict of identifying / tracing fields. Model ignores it; infra uses it for scoring, tracing, observability.

```python
{
  "record_id": "100",
  "beat_index": 1423,
  "r_peak_sample": 512340,
  "samples": [0.023, 0.019, 0.031, ...],   # length ~250, float32
  "ground_truth": "N",                      # eval mode only; stripped in prod
  "producer_ts": 1234567890.123,
}
```

Serialized to JSON initially (simple, debuggable); switch to msgpack/protobuf if throughput demands. Per-message size: ~1 KB signal + ~100 B metadata ≈ **~1.1 KB per beat message.**

**Architectural implication — why beat-per-message shapes load testing:**

The unit of work is a **beat**, not a sample. That choice changes the throughput math dramatically:

- If we classified every sample: 360 samples/sec × patients → **360 k msg/sec for 1000 patients.** Serious streaming system.
- Since we classify per-beat: resting heart = ~1-1.5 beats/sec/patient. One record replayed at 1x = **~1 msg/sec.** A trickle.

Consequence: **real-time replay of one record generates no meaningful load.** The system sits 99.99% idle; no latency or throughput number worth reporting.

To produce CV-worthy numbers we must drive load through two levers:
- **Accelerated playback** (Nx speed): replay one record at 100x, 1000x, 10,000x until consumer lags.
- **Patient parallelism**: replay many records concurrently into the same stream.

Back-of-envelope targets:

| Setup | Approx msg/sec |
|---|---|
| 1 patient × 1x | ~1 (idle) |
| 1 patient × 1000x | ~1,000 |
| 100 patients × 100x | ~10,000 (matches prior production platform peak order of magnitude) |
| 1000 patients × 100x | ~100,000 |

**Load-testing strategy** is therefore: crank playback rate and patient count until the consumer group falls behind, then record p50/p95/p99 at the ceiling. This is the direct consequence of the beat-centered windowing choice.

---

## 2. Train/test split — inter-patient (DS1/DS2), not intra-patient

**Decision:** Use the Chazal et al. 2004 DS1/DS2 split. 22 records for training (DS1), 22 for testing (DS2), 4 paced-beat records excluded (102, 104, 107, 217) per AAMI EC57.

**Split ratio:** ~50/50 by record count (22 / 22 of 44 usable records). Also approximately 50/50 by beat count — DS1/DS2 was hand-curated to balance beat totals and AAMI class representation across both halves. No patient appears in both train and test.

### What "per-patient" means in MIT-BIH

Each MIT-BIH record is **one patient's 30-minute recording.** During those 30 minutes, that patient's heart produces a **mixture of beat types** — mostly normal beats, plus however many arrhythmic beats they had in that window. The cardiologists then labeled **every individual beat** with its own N/S/V/F/Q symbol (~110,000 annotations total across the 48 records).

So record 203 isn't labeled "patient 203 has V arrhythmia." It's labeled beat-by-beat: thousands of N annotations interspersed with V annotations across the same 30 minutes. Most records contain a mix of beat types; some records are heavy on one class (203 is a high-V record; 100 is mostly N). Per-record class distributions are known and published — and are what we'll verify when we first load the data.

### Why inter-patient (not intra-patient)

- **Intra-patient** (shuffle all ~110k beats, split randomly, e.g. 80/20): patient 203's beats appear in *both* train and test. The model sees some of 203's normal and abnormal beats in training, and some more in test — it memorizes 203's ECG morphology and implicitly uses "this looks like patient 203" as a feature. Test scores look great but are meaningless on new patients.
- **Inter-patient** (split by patient): the model never sees a test patient's beats during training. Scores reflect real generalization — the only clinically meaningful question.
- Reported accuracies from intra-patient splits are inflated; using them is a red flag in ECG literature. DS1/DS2 is the community-standard inter-patient split, and naming it explicitly signals to a cardiology-literate reader that we avoided the obvious trap.

### Why this *specific* inter-patient split, not a random 22/22

Splitting by patient is necessary but not sufficient. A random 22/22 split could be unlucky in two ways:

- **Class absence:** if all patients with V beats end up in training, the test set has zero V examples → can't measure V-class F1.
- **Class imbalance between splits:** one split ends up with most of the rare classes (F, S), the other barely any → per-class F1 comparisons are noisy.

Chazal et al. (2004) **hand-picked** DS1 and DS2 specifically so that every AAMI class is represented with enough examples in both halves to produce stable per-class F1 on each side. It's not 22 random patients; it's 22 deliberately chosen ones.

On top of that, DS1/DS2 is the **community-standard** split: virtually every ECG classification paper since 2004 reports against it. Using it makes our numbers directly comparable to published benchmarks. A custom split would satisfy "inter-patient" but not "class-balanced" or "comparable."

### Paced-beat exclusion (102, 104, 107, 217)

These four records contain beats driven by artificial pacemakers rather than the heart's natural electrical system. AAMI EC57 says to exclude them because they're not the classification target — you're testing whether the model can classify *biological* arrhythmias, not detect pacemaker signals.

**DS1 (train):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230

**DS2 (test):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

**Alternative considered: random shuffle of beats (intra-patient)**
- Rejected outright — produces inflated scores that don't reflect generalization to new patients.

**Architectural implication:**
- DS2 records are what get replayed through Redis as "live" streaming data.
- DS2 is also the golden set for the offline eval harness — same bytes, different delivery mechanism. This enables online/offline parity testing.

---

## 3. Evaluation metric — per-class F1, macro-F1 as headline

**Decision:** Report precision, recall, and F1 **per AAMI class** (N, S, V, F, Q). Headline number is **macro-F1** (unweighted mean across the 5 classes).

**Why per-class:**
- MIT-BIH is ~90% N (normal) beats. Overall accuracy is dominated by the majority class — a model predicting "N" for everything scores 90% and misses every dangerous beat.
- Per-class F1 exposes this: the "V" (ventricular ectopic) F1 of the all-N model would be 0, immediately revealing the failure.
- Clinicians care about catching V and S beats specifically; those F1s are the ones that matter.

**Why macro-F1 as headline:**
- Unweighted mean of per-class F1s — refuses to let the majority class dominate.
- One comparable number across experiments and model versions.
- Used as the regression-gate metric in CI: a PR that drops macro-F1 (or any per-class F1) beyond threshold is blocked.

**Definitions (for quick recall):**
- **Precision** (of what I called class X, how many were really X): TP / (TP + FP)
- **Recall** (of all real class X, how many did I catch): TP / (TP + FN)
- **F1:** harmonic mean of precision and recall. 1.0 = perfect, 0 = useless.
- **Macro-F1:** mean of per-class F1s, each class weighted equally regardless of frequency.

---

## 4. Model scope — off-the-shelf, not bespoke

**Decision:** Baseline sklearn classifier (RandomForest or GradientBoosting) on hand-crafted beat features in Week 1. Small 1D-CNN or tiny transformer in Week 3. No custom architectures. No hyperparameter-tuning marathons.

**Why:**
- This is a **platform project**, not a model project. The differentiator is the infra around the model (eval harness, CI gates, streaming pipeline, readiness probes, observability) — directly mirroring prior production real-time classification platform work.
- Model quality only needs to be "good enough that regression gates are meaningful" — i.e. metrics that move detectably when code changes. A 90%-macro-F1 1D-CNN satisfies that. Chasing 95% burns the time budget on the wrong problem.
- Off-the-shelf lets us swap models behind a stable FastAPI interface (the "SLM wrapper" pattern from prior production platform work).

---

## 5. Multi-connector dispatch, single Gateway service (prior-platform parallel)

**Decision:** Architect pulsegate from day one around a **connector + dispatcher** pattern, even though only one connector's workflow ships in the first pass.

- A **processing connector** (ECG worker, PCG worker) owns a specific signal type's windowing + model. One per signal type.
- A single **Gateway** service sits at ingress and handles HTTP, auth, `request_id` assignment, payload validation, **dispatch on `signal_type`** to the matching `:in` Redis stream, and the response-correlation loop that reads `:out` streams and resolves pending caller futures.

**The Gateway is always one service, never split per-type.** The earlier design separated "Gateway" and "Client Connector" as two layers; on review those concerns collapse cleanly into one service. Rationale:
- Dispatch logic is a dict lookup on `signal_type`. Trivial. Not worth its own microservice.
- Gateway concerns (auth, request_id, validation) are identical across signal types. Splitting would duplicate them.
- One service = one Deployment, one HPA, one image, one set of observability dashboards.
- Scales on **load volume**, not on signal-type count. A 10x traffic spike needs more Gateway pods, not new services.

**HPA on the Gateway:**
- CPU utilization, target ~70% average.
- `minReplicas: 2` to survive rolling deploys and single-pod failure.
- `maxReplicas: 20` as a safety ceiling.
- V2 upgrade path: custom metric (p95 latency or queue depth) — CPU is a rough proxy because XADD is I/O-bound and the correlation wait is pure asyncio parking, not CPU.

**Connectors planned:**
1. **ECG connector** (primary, Weeks 1-3): MIT-BIH Arrhythmia. Beat-centered windows. 5-class AAMI beat classification.
2. **PCG connector** (stretch, Week 4): PhysioNet/CinC Challenge 2016 phonocardiogram (heart sound audio). Multi-second windows. Binary normal/abnormal classification.

**Why two connectors on different signals (not two windows on ECG):**
- Two connectors both classifying ECG would be theatre — the Gateway dispatcher would be solving a fake routing problem.
- ECG (360 Hz voltage) vs PCG (2000 Hz audio) differ in every architectural dimension: signal type, sampling rate, window strategy, label space, model task. The Gateway dispatcher is doing genuine work.
- Still coherent narrative ("heart health platform"), not a scattered multi-modal demo.

**Why this pattern specifically — prior-platform parallel:**
- A prior real-time text/image/video classification platform used distinct connectors per payload type (text / image / video), each with its own preprocessing + model, fronted by an Gateway dispatcher that dispatched based on payload type. pulsegate mirrors this: distinct connectors for ECG / PCG, same dispatch pattern.
- CV bullet: *"Designed a multi-connector Gateway dispatcher pattern for biosignal classification, mirroring a production real-time text/image/video classification platform architecture I built previously."*

**Why multi-connector from day one, not retrofit:**
- Retrofitting routing into a single-model codebase always produces uglier seams than designing for it up front.
- The shared `pulsegate-core` library, message schema, and FastAPI interface are trivially extended to support a `signal_type` dispatch now; hard to add later.

**Scope discipline:**
- The Gateway dispatcher stays **dumb in V1** — a switch statement on `signal_type`. No A/B logic, no shadow routing, no weighted traffic splits. Those are stretch goals *after* two connectors exist.
- Connector #2 (PCG) is a Week 4 capstone, not a Week 1 foundation. If time runs out, the bullet is still true ("designed for multi-connector") without the code being half-finished.

**Message schema impact:**
Every stream message gains a `signal_type` field (`"ecg_beat"` | `"pcg_recording"` | ...). The Gateway dispatcher dispatches on it. Each connector publishes to its own output stream so downstream scorers/dashboards can separate them.

**Alternative considered: beat-level vs rhythm-level, both on ECG**
- Rejected because the two paths would share ~80% of their code (same signal, same sampling rate, same dataset family). The Gateway dispatcher would be decorative.

---

## 6. Load simulation strategy — real data, fanned out and accelerated

**Decision:** No synthetic ECG. Generate load by replaying the 22 real DS2 records through many concurrent producer workers, each representing one "virtual patient." Three levers stack:

1. **Record variety** — 22 real DS2 records are the base set of distinct signal sources.
2. **Virtual-patient fan-out** — each record is replayed N times concurrently, with **staggered start offsets** so the N copies aren't in lockstep (e.g. copy 1 at beat 0, copy 2 at beat 300, copy 3 at beat 600). Each copy traverses its record sequentially; variety at the message level comes from the copies being at different points in the recording at any given moment.
3. **Playback speed (Nx)** — each replay emits beats at Nx the natural heart rate.

**Why real data, not synthetic:**
- Real data has genuine beat-morphology variety the model was trained on. Synthetic signals would either be trivial (the model aces them) or adversarial (the model fails in uninteresting ways) — neither useful for load testing.
- Using real DS2 means the same bytes that the offline eval harness scores against are what's flowing through Redis. Enables online/offline parity testing (see §6 in architecture.md).

**Why sequential-within-stream:**
Inter-beat intervals matter clinically and are a feature used by some classifiers. Shuffling beats within a record would destroy that signal and produce unrealistic sequences.

**Why staggered offsets across copies:**
If all 10 copies of record 100 start at beat 0 simultaneously, they emit identical messages in lockstep — volume without variety. Staggering means the 10 copies are each at a different point in the recording at any moment → 10 different beats in flight.

**Circular replay (wrap-around):**
Each producer plays its record beats in order from its `start_offset`, wraps past the last beat back to beat 0, and continues to `start_offset - 1` — then loops again. Two reasons:
- No signal variety is lost (beats before the offset still get played).
- A 30-min record at 100x finishes in ~18s; without looping, producers would go idle almost immediately. Loop-around sustains load for multi-minute tests.

**Wrap-point caveat:** at the wrap boundary, the last beat of the recording and the first beat aren't adjacent in real time, so inter-beat intervals across the seam are meaningless. Beat-level classification doesn't care (each beat is classified independently). But if a future **rhythm-level** connector (PCG or AFib) uses windows that span the seam, those windows should be tagged so the scorer excludes them.

**Throughput math (back-of-envelope):**

Each DS2 record has ~2,000-3,000 beats over 30 minutes (60-90 bpm × 30 min). At 1x playback each stream emits ~1-1.5 beats/sec. Fan-out and speed multiply this:

| Setup | Streams | Per-stream rate | Total msg/sec |
|---|---|---|---|
| 22 records × 1x | 22 | ~1/sec | ~22 |
| 22 × 10 copies × 100x | 220 | ~100/sec | ~22,000 |
| 22 × 100 copies × 100x | 2,200 | ~100/sec | ~220,000 |

**Realism-vs-volume trade:**
Pure duplicates (same record, no perturbation) are fine for measuring infra throughput and latency — the model doesn't care. For robustness testing later, add small per-copy noise or timing jitter to prevent identical payloads. Park as a stretch.

**Producer worker contract (sketch):**
```python
async def producer(record_id, virtual_patient_id, start_offset, speed):
    for beat_window, annotation in replay(record_id, start=start_offset, speed=speed):
        # HTTP POST to Gateway's internal Service (not directly to Redis)
        await http.post(f"{gateway_url}/classify", json={
            "signal_type": "ecg_beat",
            "virtual_patient_id": virtual_patient_id,  # disambiguates shared record replays; drives sharding
            "record_id": record_id,
            "samples": beat_window,
            "ground_truth": annotation,
            "producer_ts": time.time(),
        })
```

Load-gen talks to the Gateway via the cluster-internal Service (e.g. `http://gateway.pulsegate.svc.cluster.local`), not directly to Redis and not through the public Ingress. The Gateway does the XADD to the sharded `:in` stream. This exercises the real end-to-end path (validation → request_id → dispatch → Redis → worker → response correlation) and produces honest latency numbers.

`virtual_patient_id` is essential — without it, the Gateway can't compute the shard key, and the scorer can't tell two virtual patients apart when they replay the same underlying record.

---

## 7. Load generator runs as a K8s workload

**Decision:** The load generator is a **Kubernetes Job** (or Deployment with replicas), driven by a bash/CLI script that spins it up on demand. It POSTs to the Gateway's cluster-internal Service, exercising the full serving path (Gateway dispatch → Redis → worker → Redis → correlator → caller). Direct-to-Redis XADD is a possible alternative for pure infra stress testing but bypasses the Gateway and produces misleading end-to-end numbers; we don't use it.

**Why a K8s workload and not a local script:**
- Mirrors production load-testing patterns — load generation needs to be horizontally scalable too.
- Decouples load volume from the developer laptop: `kubectl scale` to ramp up.
- Demonstrates end-to-end K8s-native operation: producer pods, consumer pods, Redis, all in the cluster.
- CV bullet: *"Kubernetes-native load-generation harness for streaming classification service."*

**Shape:**

```
 ┌────────────────────┐
 │ bash: loadgen.sh   │  → kubectl apply + env-var knobs
 │  --records=22      │    (records, copies, speed, duration)
 │  --copies=10       │
 │  --speed=100       │
 └─────────┬──────────┘
           │
           ▼
 ┌─────────────────────────────┐
 │ K8s Job: pulsegate-loadgen  │
 │  - N replica pods           │
 │  - each pod runs K          │
 │    producer async tasks     │
 │  - virtual-patient-ID space │
 │    partitioned across pods  │
 └─────────┬───────────────────┘
           │ HTTP POST
           ▼
 ┌─────────────────────────────┐
 │ Gateway Service (cluster)   │  → XADD to sharded :in streams
 └─────────┬───────────────────┘
           ▼
     per-signal-type workers → :out → Gateway correlator → caller
```

**Operator experience:**
```
./scripts/loadgen.sh --copies=10 --speed=100 --duration=5m
```
The script builds (or assumes pre-built) a `pulsegate-loadgen` image, renders a Job manifest with the knobs as env vars, applies it, and tails logs. Teardown is either automatic (Job completion) or explicit `kubectl delete job pulsegate-loadgen`.

**What the pod does on start:**
1. Reads env vars for record set, copies, speed, duration.
2. Loads DS2 records from a PVC or baked into the image (MIT-BIH is small enough to bake for V1).
3. Claims its slice of the virtual-patient-ID space (so no patient is produced by two pods).
4. Spawns producer async tasks with staggered start offsets.
5. HTTP POSTs beats to the Gateway's internal Service until duration elapses.

**Why this matters architecturally:**
- The load generator becomes part of the deployable system, not a shell script on the dev's laptop.
- Observability is unified — producer metrics (send rate, drops) come through the same Prometheus pipeline as consumer metrics.
- Pod count, replica scale, and per-pod concurrency become the load-testing knobs.

**Scope discipline:**
- V1: single Job, configured by env vars, one bash script to launch it. No operator, no CRD, no autoscaling.
- Stretch: HPA on consumer queue depth; separate load-shape profiles (spike, ramp, sustained).

---

## 8. Production serving topology — layered ingress, per-type Redis stream pairs

**Decision:** The serving path is layered:

```
Internet → Ingress/LB (NGINX + cert-manager)
        → Gateway (FastAPI: auth, request_id, dispatch on signal_type,
                    response correlation, HPA on CPU)
        → Redis :in stream per signal type
        → Worker (consumer + model per signal type, 1 consumer per pod)
        → Redis :out stream per signal type
        → Gateway response correlator → caller
```

### Ingress / LB vs Gateway — two distinct layers

**Ingress / LB (NGINX or K8s Ingress controller):**
- Network-layer only. TLS termination, L4/L7 load balancing, hostname/path routing, basic IP rate limiting.
- Declarative config (YAML), no application code.
- Stateless. Swap NGINX for Traefik or a cloud LB without touching app code.

**Gateway (stateless FastAPI, single service, HPA):**
- Application-layer. Owns the public API contract.
- Assigns `request_id` on inbound request.
- Validates payloads, runs auth (noop V1, hook present).
- **Dispatches on `signal_type`** to the matching `:in` Redis stream (single service, internal switch; never split per-type).
- Holds the caller's HTTP connection open while awaiting a response on `:out` streams via an in-memory `asyncio.Future`.
- Background task runs a long-lived `XREAD BLOCK` correlator over all `:out` streams; resolves futures on match.
- Application metrics, structured logs, per-route latency.

NGINX is the wrong home for request-ID assignment and future resolution — those are Python/asyncio concerns. Keeping Ingress and Gateway as distinct layers lets the infra concern (TLS/routing) evolve independently from the application concern (correlation).

### Redis durability

- **Persistence:** AOF with `appendfsync everysec`. At most 1 second of in-flight data lost on primary crash. (Redis *is* persistent if configured — no "it's all in RAM" property to work around.)
- **Replication:** one replica via `REPLICAOF`. StatefulSet + PodDisruptionBudget in K8s. No Sentinel or Redis Cluster in V1 — overkill.
- **At-least-once delivery:** Redis Streams consumer groups with `XACK`. Unacked messages on a crashed consumer are reclaimed via `XPENDING` + `XCLAIM`. This is the property that makes "no data lost on worker crash" true.

### Ordering — partition by virtual_patient_id, not topic-wide order

Beat-morphology classification doesn't require global order. But two cases do require per-patient order:
1. **R-R interval features** (time between consecutive beats) — order within a virtual patient is load-bearing.
2. **Rhythm-level classification** (PCG/AFib stretch) — windows span multiple beats.

**Solution:** shard the `:in` stream by `hash(virtual_patient_id) % N`. Each shard is consumed by **at most one consumer** (so no interleaving). Across shards, processing is parallel. This is the Kafka "partition key" pattern implemented on Redis.

**V1 starting point:** 1 shard per signal type, 1 consumer. Sharding is a scale-out lever, turned on when measured throughput demands it. Not premature.

### Kafka considered and rejected

| Property | Redis Streams | Kafka |
|---|---|---|
| Partitioning | Manual (sharded streams) | Native |
| Persistence | AOF + replica | Native, strong |
| Ops complexity | Low (1 process, 1 config) | High (consensus cluster, broker tuning, rebalancing, schema registry, JVM ops) |
| Narrative fit | ✓ matches plan | Would rewrite the stack |
| 100h budget | ✓ | ✗ (15-20h ops learning alone) |

**Rejected on ops cost, not capability.** At ~22k msg/s, Redis is more than sufficient. Kafka is the correct answer at 100k+/sec with multi-DC replication, strict schema governance, or audit-grade retention needs. None of those apply here. Being able to articulate *why* Redis is the right choice — and what would flip the decision to Kafka — is a senior-infra conversation.

### Request/response correlation

Every signal type has a **pair of streams**: `ecg:in` / `ecg:out`, `pcg:in` / `pcg:out`, etc. Correlation is via `request_id`:

- Gateway assigns UUID `request_id` on inbound request, creates `pending[request_id] = future`.
- Gateway XADDs `{request_id, signal_type, ...payload}` to the matching `:in` stream (borrows a pool connection, releases in sub-ms).
- Worker processes, writes `{request_id, prediction, confidence}` to the matching `:out` stream, XACKs the input.
- Gateway's background **response correlator** task runs a long-lived `XREAD BLOCK` over all `:out` streams on a dedicated connection (not from the pool). On each message, it pops the matching future from `pending` and resolves it.
- The original HTTP handler was awaiting that future; resolution wakes the handler; it replies.

**Critical semantics:**
- The Gateway **does not hold a Redis connection** during the wait for a response. The wait happens on a Python `asyncio.Future`. Redis connections are held only for the XADD itself.
- The background correlator owns **one long-lived Redis connection** shared across all responses. It does not draw from the pool.

**Timeouts:** caller-facing `asyncio.wait_for(future, timeout=N)`. On expiry, return HTTP 504. For long-running inference (rhythm-level, future), fall back to `202 Accepted + poll URL`. Messages whose processing times out on the caller side still get ACKed by the worker normally — the worker isn't aware of caller state.

### Ordering invariants — per-patient beat order end to end

Per-patient sequential order is preserved by a chain of five invariants. **Break any one and ordering is lost.** This is the invariant set that must survive refactors.

**1. Producer-side: one producer task per virtual patient.**
Each virtual patient has exactly one producer task in exactly one load-gen pod. That task XADDs beats sequentially via `await`. Virtual-patient-ID space is partitioned across load-gen pods at startup so no patient has two concurrent producers.

**2. Redis stream IDs are monotonic.**
Redis assigns each XADD a strictly-increasing ID. Shard K's stream physically stores patient 42's beats in XADD order.

**3. XREADGROUP returns in stream-ID order.**
Consumers see oldest messages first. Reading from shard K yields beats in the order they were XADDed.

**4. Consumer discipline: sequential await within a shard, concurrent across shards.**

```python
# RIGHT — sequential within a shard
async for batch in xreadgroup_loop(shard=K, count=10):
    for msg in batch:
        await process(msg)    # await blocks this coroutine
        await xack(msg)

# WRONG — breaks order
async for batch in xreadgroup_loop(shard=K, count=10):
    for msg in batch:
        asyncio.create_task(process(msg))   # parallel firing
```

Batch-reading is a network optimization; processing is strictly sequential. While this shard's coroutine awaits inference, the event loop runs *other shards'* coroutines. Parallelism across shards; serial within a shard.

**5. One consumer per shard in the consumer group.**
Redis consumer groups deliver each message to exactly one consumer. Two consumers on the same shard = interleaved delivery = broken order.

Rule: `consumers_in_group ≤ shard_count`, practically `= shard_count`. Shards are the parallelism unit.

**Failover behavior:** unacked messages are reclaimed via `XCLAIM` to another consumer after `min-idle-time`. The replacement reads reclaimed messages in stream-ID order, context cache rebuilds from those messages. First few beats after failover may have shorter R-R history; accepted degradation.

**Shard count is a topology constant.** Changing N at runtime reshuffles the patient→shard map and two consumers could transiently hold different context for the same patient. Treat shard count as changed only via planned maintenance (drain, reconfigure, restart).

### Response routing across multiple Gateway pods

**Problem:** with N Gateway pods, a response on `:out` could be consumed by any pod, but only the pod holding the caller's HTTP connection (and the `pending[request_id]` future) can resolve it.

**Solution (V1): every Gateway pod does independent `XREAD BLOCK` on `:out`, NOT as part of a consumer group.**

- Each pod sees every `:out` message.
- Each pod looks up `request_id` in its local `pending` dict (O(1), ~150 ns).
- The pod holding the future resolves it; other pods discard silently.

Cost at our scale (22k msg/sec, 5 Gateway pods):
- CPU: ~3 ms / sec / pod for lookups (<0.5% of a core). Trivial.
- Network: ~5 MB/s per pod for XREAD bandwidth. Trivial.

**Alternative considered and rejected: single consumer group on `:out` across all Gateway pods.**
- Redis would deliver each response to exactly one pod; if that pod doesn't hold the future, response is lost. Breaks correlation. Don't do this.

**Alternative for larger scale (not V1): dedicated correlator + Redis pub/sub fanout.**
- One service XREADs `:out` once, republishes to a pub/sub channel.
- Gateway pods subscribe; each inspects, resolves locally.
- Wins at 100k+ msg/sec when N pods × XREAD bandwidth becomes costly. Not pulsegate.

### Sticky routing at Ingress

Caller's HTTP connection must land on a specific Gateway pod — that's the pod holding the `pending[request_id]` future. If the connection moves mid-flight, the response can't find its caller.

**V1: Kubernetes Service `sessionAffinity: ClientIP`.**
- Pins a source IP to a specific backing Gateway pod.
- Works for load testing because each load-gen pod has one IP; all its virtual patients route to the same Gateway pod (acceptable — stickiness is pod-to-pod, not per-patient).
- Known limitation: real clients behind NAT share an IP → all route to the same pod, defeating load balancing. Mitigation for production: cookie-based affinity at the Ingress.

**V2 upgrade path:** NGINX Ingress cookie affinity. NGINX sets a `route` cookie on first response naming the chosen Gateway pod; subsequent requests carry the cookie and route to the same pod.

### Failure modes

| Mode | What happens | Behavior |
|---|---|---|
| **Caller disconnects / times out** | HTTP connection closed before future resolves | Gateway cancels future, returns 504 or logs and moves on. Worker still processes and writes to `:out`. Response discarded on arrival (no pending future). Work done, result thrown away — standard async. |
| **Worker crashes mid-processing** | Message pulled from `:in` but not ACKed | Redis consumer group marks "pending". After `min-idle-time`, another consumer claims via `XCLAIM`, processes, ACKs. If Gateway still waiting, future resolves (slow but successful). If Gateway already timed out, falls into the caller-disconnect case. |
| **Gateway pod crashes** | In-memory `pending` map lost with the pod | Caller sees TCP reset. Worker still processes. Response orphaned — other Gateway pods see it on `:out` but find no matching `pending` entry. **Mitigation: sticky Ingress routing in V1**; persistent correlation state in Redis for V2. |
| **Permanently-failing message** | Worker crashes every time on one input (malformed payload) | Retry count tracked per message. After N retries, worker moves it to `ecg:dlq` (Dead Letter Queue) and ACKs the original. Reaper / human inspects DLQ. Prometheus alert on DLQ size. V1: defer. Stretch: implement. |

### Future-proofing for new signal types

Adding a new signal type (e.g. EEG in Week 5) is:
1. Define new streams: `eeg:in`, `eeg:out`.
2. Add `eeg` case to the Gateway's dispatch table.
3. Deploy a new worker Deployment for EEG.
4. No changes to Ingress, existing streams, or existing workers.

This is what "designed for multi-connector" buys. Without per-type stream pairs, every change would require touching a shared pipeline.

### Worker pod composition — multiple consumers per pod sharing one model

**Key insight:** shards-per-pod is flexible, but **one consumer per shard is a hard constraint** (preserves per-patient order). Consumers within a pod share a single model instance in memory — models are thread-safe for inference, so no duplication is needed.

```
Pod
├── ONE Model instance           (shared across consumers; inference is thread-safe)
├── Per-patient context cache    (in-process dict: virtual_patient_id → deque of recent beats)
├── Consumer coroutine 1 → shard 0  ─┐
├── Consumer coroutine 2 → shard 1   │ all share model + cache
├── Consumer coroutine 3 → shard 2   │ each owns exactly one shard
└── Consumer coroutine 4 → shard 3  ─┘
```

**Cost rationale:**
- One 100 MB model per pod, not per consumer. A pod with 4 consumers costs ~100 MB of model memory, not 400 MB.
- HPA scales pods; pod count and per-pod consumer count are independent dials. Shard count is a topology decision, set once based on peak-load math.
- K8s resource limits sized to concurrent inference load: `requests.cpu ≈ 4 × per-inference-cpu + headroom` for a 4-consumer pod.

**Rejected alternative: one consumer per pod.** Over-provisions model memory; unnecessary given model inference is thread-safe.

### Per-patient context cache

When features depend on recent history (R-R intervals, rolling statistics, rhythm-level windows), consumers need access to prior beats from the same patient.

**Pattern: in-process dict in each consumer pod, scoped to the patients in its shards.**

```python
patient_context: dict[str, deque[Beat]] = {}

async def handle_beat(msg):
    pid = msg["virtual_patient_id"]
    history = patient_context.setdefault(pid, deque(maxlen=5))
    rr_interval = msg.ts - history[-1].ts if history else None
    features = extract_features(msg, rr_interval=rr_interval)
    label = model.predict(features)
    history.append(msg)
    await redis.xadd(f"{signal_type}:out", {...})
```

**Why in-process works:** sharding guarantees patient 42 always lands on the same consumer (as long as shard count is stable). The consumer's local cache is always authoritative for its patients.

**Why not Redis-backed:** adds ~1 ms per beat for the round-trip. At 22k msg/sec that's a meaningful CPU + network tax for data that's ephemeral by design.

**Eviction:**
- Size-bound via `deque(maxlen=N)` per patient (N tuned per feature — 5 beats for R-R, more for rhythm-level).
- Idle-timeout sweep every ~5 min drops patients with no recent activity, preventing unbounded growth.

**When this breaks:** shard count changes at runtime (rebalance moves patient 42 to a different consumer). Mitigation: treat shard count as a topology constant, reconfigured only via planned change, not dynamic rebalance. V2 could add a fallback Redis-backed cache on cache miss.

### Sidecar model container — stretch pattern

**V1 / Week 3:** model is a Python import, colocated in-process. Simplest, lowest latency.

**Stretch (Week 4+):** sidecar model container — consumer container + model container in one pod, connected over localhost gRPC or HTTP. Trade-offs:

| Pattern | Pros | Cons |
|---|---|---|
| In-process (V1) | Lowest latency, simplest, one container to observe | Model rollout = consumer rebuild |
| Sidecar (stretch) | Decoupled lifecycles, language-agnostic model runtime (ONNX/Triton), separate resource limits per container | ~200-500μs added per inference (localhost network), two containers to observe, coordinated readiness probes |

Sidecar is a demonstrable production-grade pattern — CV bullet: *"model-serving sidecar for language-agnostic swap-in of optimized runtimes."* V1 stays in-process.

### Coroutines — separated by shard, not by patient

Three hierarchical concurrency boundaries:

```
Pod                    (K8s scheduling unit)
 └── Consumer coroutine   (1 per shard; owns the stream-read + processing loop)
      └── Patient message (transient; processed sequentially within the coroutine)
```

Patient is the **data key** (drives sharding); shard is the **execution boundary** (one coroutine per shard). Gateway handler coroutines are per-request, not per-patient — they exist for the lifetime of one HTTP request and don't retain patient-specific state.

### Connection pools — what they actually do

Every pod that talks to Redis uses a **connection pool** to amortize TCP setup. Important clarifications:

- A pool connection is **borrowed for the duration of a single Redis command** (sub-ms for XADD), then returned. It is NOT held across a request's lifetime.
- **The Gateway does not hold a Redis connection while awaiting the caller's response.** The wait is on an `asyncio.Future` in Python memory; no Redis resources are held.
- The Gateway's **response correlator** uses a single dedicated long-lived connection (not from the pool) doing `XREAD BLOCK` over all `:out` streams.

**Pool multiplexing:** many concurrent coroutines share a small pool of connections. At any instant, the number of in-use connections equals the number of XADDs currently mid-flight, which is tiny because XADDs are sub-ms. A pool of 10-50 connections comfortably serves 1000+ concurrent HTTP requests.

**Pipelining (distinct from multiplexing):** within one coroutine, multiple commands can be batched onto one connection without waiting for intermediate responses. Useful for bulk operations; not used for per-request XADD.

Pool sizing guidance: `max_connections ≈ 2-3 × peak_concurrent_redis_ops_per_pod`. Tune during load testing; look for `connection pool exhausted` errors as the signal to increase.

### Load generator — client topology

Each **virtual patient** is a logical client, but many share one load-gen pod's HTTP connection pool. Implementation:

```
Load-gen pod
├── HTTP connection pool to Gateway (e.g. 20 keep-alive connections)
└── N async producer tasks (one per virtual patient assigned to this pod)
    └── Each task: read next beat from its record → HTTP POST to Gateway
```

- **`virtual_patient_id` is in the payload**, not tied to the TCP connection. Sharding downstream keys on the payload field.
- **No sticky mapping between producer task and HTTP connection** — tasks borrow from the pool freely.
- **When load-gen scales horizontally:** partition the virtual-patient-ID space across load-gen pods at startup so each virtual patient exists in exactly one pod (avoids collision on shards downstream).

---

## 9. Training infra vs. serving infra — shared code, separate runtime

**Decision:** Three runtime paths, one shared feature/windowing library:

| Path | Runtime | Uses Redis? |
|---|---|---|
| Training | Offline, reads MIT-BIH from disk | No |
| Offline eval (golden harness) | CI, in-process | No |
| Online serving (demo) | FastAPI + Redis consumer | Yes |

All three import the **same windowing/feature code** from a shared `pulsegate-core` library. This prevents train/serve skew — the #1 cause of ML systems silently underperforming in production.

**Why no Redis in training or offline eval:**
- Adds latency and complexity without adding realism.
- Discipline: streaming where streaming is load-bearing, direct function calls everywhere else.

**Bonus property this buys us:**
- Online/offline parity testing — running DS2 through both paths should produce identical predictions. Any divergence is a bug in the serving path.
