# pulsegate · architecture diagrams

Where the project is now and where it'll be at end of Week 4.
Same skeleton, scaled up — the shape today is the shape at the finish line.

For deeper rationale see `architecture.md` and `design-notes.md`.

---

## Visual language

| Shape | Means |
|---|---|
| `[(  )]` cylinder | persistent store (Redis, Prometheus TSDB, object storage) |
| `{{  }}` hexagon  | long-running process |
| `[/  /]` parallelogram | input / load source |
| `[  ]`  rectangle | UI / read-only viewer |
| ══>  thick arrow | hot data path (per-beat traffic) |
| ─.─>  dashed arrow | control / observability / out-of-band |

---

## Current state · end of Week 2 Step 4 Chunk B

```mermaid
flowchart LR
    P{{"⚡<br/>Producer"}}
    C{{"🧠<br/>Consumer<br/>(RF)"}}
    M{{"📡<br/>Monitor"}}
    R[("🗄️ Redis<br/>:in · :out")]
    PR[("📈 Prometheus")]
    GF["📊 Grafana"]

    P ==> R
    R ==> C
    C ==> R
    M -.-> R

    P -. /metrics .-> PR
    C -. /metrics .-> PR
    M -. /metrics .-> PR
    PR ==> GF

    classDef proc  fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
    classDef store fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#7c2d12
    classDef ui    fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#581c87
    class P,C,M proc
    class R,PR store
    class GF ui
```

**Verified this morning**:
predict = 26.7 ms = 97% of consumer wall time · lag peaks at ~1376 messages
under sustained 91 b/s in vs ~38 b/s out · 68 tests green.

---

## Target state · end of Week 4

```mermaid
flowchart TB
    LG[/"⚙️<br/>Load-gen<br/>(K8s Job)"/]

    GW{{"🚪 Gateway<br/>FastAPI · HPA"}}

    Rin[("🗄️<br/>:in × N shards<br/>(hash by patient)")]
    Rout[("🗄️<br/>:out")]

    W1{{"🧠<br/>ECG worker<br/>1D-CNN"}}
    W2{{"🧠<br/>ECG worker"}}
    W3{{"🧠<br/>PCG worker"}}:::stretch

    Rpcg[("🗄️<br/>pcg :in/:out")]:::stretch

    SM{{"🎯<br/>Sampler"}}:::stretch
    MN[("📦<br/>MinIO")]:::stretch

    PR[("📈<br/>Prometheus")]
    GF["📊 Grafana"]
    EV["✓ Golden eval<br/>(CI gate)"]

    LG ==> GW
    GW ==> Rin
    Rin ==> W1
    Rin ==> W2
    W1 ==> Rout
    W2 ==> Rout
    Rout -.-> GW

    GW -.-> Rpcg
    Rpcg -.-> W3
    W3 -.-> Rpcg

    Rout ==> SM
    SM ==> MN

    GW -. /metrics .-> PR
    W1 -. /metrics .-> PR
    W2 -. /metrics .-> PR
    PR ==> GF
    EV -. DS2 replay .-> Rin

    classDef gw      fill:#dbeafe,stroke:#1d4ed8,stroke-width:3px,color:#1e3a8a
    classDef worker  fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#14532d
    classDef store   fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#7c2d12
    classDef ui      fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#581c87
    classDef input   fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#831843
    classDef stretch fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,stroke-dasharray: 6 4,color:#7f1d1d
    class GW gw
    class W1,W2 worker
    class Rin,Rout,PR store
    class GF,EV ui
    class LG input
```

Dashed red boxes = stretch goals (PCG signal type, sampler, MinIO).

---

## What changes between the two

| | **Now** | **Week 4 target** |
|---|---|---|
| **Scale** | 1 producer · 1 consumer · laptop | N pods, HPA on lag, K8s |
| **Streams** | single `:in` / `:out` | sharded by `hash(patient_id) % N` |
| **Routing** | direct XADD/XREAD | Gateway with `request_id` ↔ Future correlation |
| **Model** | RandomForest baseline (F1 0.37) | 1D-CNN (target ~0.55+) |
| **Signals** | ECG only | ECG + PCG (heart-sound stretch) |
| **Eval** | offline harness | golden DS2 replay through live workers, CI-gated |
| **Sampling** | none | stratified slice → object storage |

What stays identical: **producer → Redis stream → consumer → out-stream**, plus
Prometheus + Grafana on the side. The skeleton today is the skeleton at scale.

---

## CV-bridge one-liner

> Production real-time classification platform — sharded Redis Streams,
> multi-consumer worker pods sharing one model instance, request-correlated
> stateless gateway. Same architecture as my prior text/image/video classification
> platform at ~20k msg/s p99 ~100ms, transposed to ECG arrhythmia detection on
> MIT-BIH (PhysioNet).
