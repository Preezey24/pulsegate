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

## 5. Multi-connector + orchestrator pattern (prior-platform parallel)

**Decision:** Architect pulsegate from day one around a **connector + orchestrator** pattern, even though only one connector ships in the first pass. A connector owns a specific signal type and its windowing strategy; the orchestrator routes incoming messages to the right connector based on a `signal_type` field.

**Connectors planned:**
1. **ECG connector** (primary, Weeks 1-3): MIT-BIH Arrhythmia. Beat-centered windows. 5-class AAMI beat classification.
2. **PCG connector** (stretch, Week 4): PhysioNet/CinC Challenge 2016 phonocardiogram (heart sound audio). Multi-second windows. Binary normal/abnormal classification.

**Why two connectors on different signals (not two windows on ECG):**
- Two connectors both classifying ECG would be theatre — the orchestrator would be solving a fake routing problem.
- ECG (360 Hz voltage) vs PCG (2000 Hz audio) differ in every architectural dimension: signal type, sampling rate, window strategy, label space, model task. The orchestrator is doing genuine work.
- Still coherent narrative ("heart health platform"), not a scattered multi-modal demo.

**Why this pattern specifically — prior-platform parallel:**
- A prior real-time text/image/video classification platform used distinct connectors per payload type (text / image / video), each with its own preprocessing + model, fronted by an orchestrator that dispatched based on payload type. pulsegate mirrors this: distinct connectors for ECG / PCG, same dispatch pattern.
- CV bullet: *"Designed a multi-connector orchestrator pattern for biosignal classification, mirroring a production real-time text/image/video classification platform architecture I built previously."*

**Why multi-connector from day one, not retrofit:**
- Retrofitting routing into a single-model codebase always produces uglier seams than designing for it up front.
- The shared `pulsegate-core` library, message schema, and FastAPI interface are trivially extended to support a `signal_type` dispatch now; hard to add later.

**Scope discipline:**
- The orchestrator stays **dumb in V1** — a switch statement on `signal_type`. No A/B logic, no shadow routing, no weighted traffic splits. Those are stretch goals *after* two connectors exist.
- Connector #2 (PCG) is a Week 4 capstone, not a Week 1 foundation. If time runs out, the bullet is still true ("designed for multi-connector") without the code being half-finished.

**Message schema impact:**
Every stream message gains a `signal_type` field (`"ecg_beat"` | `"pcg_recording"` | ...). The orchestrator dispatches on it. Each connector publishes to its own output stream so downstream scorers/dashboards can separate them.

**Alternative considered: beat-level vs rhythm-level, both on ECG**
- Rejected because the two paths would share ~80% of their code (same signal, same sampling rate, same dataset family). The orchestrator would be decorative.

---

## 6. Training infra vs. serving infra — shared code, separate runtime

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
