# Decision Log

Running log of non-obvious implementation choices and their rationale. Preserved in the repo rather than buried in commit messages so future readers (including future Rhys) can retrace deliberation, and so the portfolio artefact shows its work — choices were made, alternatives considered, trade-offs acknowledged.

Format: reverse-chronological. Each entry stands alone; cross-reference other docs by section.

---

## 2026-04-21 — Baseline retrained on all of DS1; internal val split abandoned

### Context
First training run (archived as `models/archive/baseline_v0_ds1-split.joblib`, local only — `models/` is gitignored) held out 5 DS1 records (`101, 203, 208, 215, 223`) as an internal validation set. Standard hyperparameter-tuning convention: train on the remaining 17, score on the 5, then evaluate on DS2 in Step C as the final test.

### Problem discovered
Per-record class distribution, computed post-hoc, revealed catastrophic training starvation on rare classes:

| Class | Train count | Val count | Train % |
|---|---|---|---|
| N | 34,637 | 11,211 | 93.0% |
| S | 861 | 83 | 2.3% |
| V | 1,715 | 2,073 | 4.6% |
| **F** | **26** | **388** | **0.07%** |
| **Q** | **0** | **8** | **0.00%** |

Record 208 (held out as val) alone contained **372 of 388 total F-beats in val** — 14× more F-class beats than the entire 17-record training set combined. Q-class had zero training examples; `model.classes_` had only 4 entries (`F, N, S, V`), so the model's output vocabulary literally excluded Q.

Val macro-F1: 0.362. Per-class: N=0.962, V=0.846, **S=0.000, F=0.000, Q=0.000**.

### Decision
Retrain on all 22 DS1 records. Drop the internal val split entirely. Evaluate on DS2 in Step C.

### Rationale
1. **Not tuning hyperparameters.** The internal val split exists to compare variants during hyperparameter search. We picked hyperparameters from `design-notes.md` §12 with no sweep. Internal val was solving a problem we didn't have.
2. **DS2 is the canonical test set.** Chazal et al. (2004) established DS1/DS2 as the community-standard inter-patient split. Step C's DS2 eval is the authoritative metric; internal DS1 val adds no information for the headline number.
3. **Recovers training data on rare classes.** Adding records 208, 223 back to training brings F-class training count from 26 → ~400, enables Q-class predictions, and substantially strengthens S-class coverage.
4. **Cleaner portfolio narrative.** Baseline macro-F1 on the canonical DS2 test set is the number that belongs on the CV and in the blog post. Internal val F1 would have been noise.

### Alternatives considered
- **Stratified k-fold CV within DS1** — technically cleanest. Every record appears as both train and val across folds. Ruled out on engineering cost for V1 scope; could revisit if we need per-class stability intervals.
- **Beat-level stratified split (not patient-level)** — would cause patient leakage between train and val, inflating metrics. Ruled out on correctness.
- **Hand-pick a better val split** — fixing the rare-class concentration by moving 208 to training and choosing different val records. Ruled out on grounds of "tuning the split for the metric is almost cheating" and "still doesn't change the DS2 eval."

### What's archived
- `models/archive/baseline_v0_ds1-split.joblib` — first-run model trained on 17 DS1 records. Local only (gitignored). Reloadable via `joblib.load` for regression comparison if needed.
- This decision doc captures the deliberation; the physical artefact isn't committed but the story is.

### Lessons
- When hand-picking val records, **always inspect per-record class distribution first**. Rare-class concentration in specific records is the dominant training-starvation risk, not overall class balance.
- `class_weight='balanced'` scales split gains during training but can't conjure examples from zero. Zero training rows of any class → class is silently dropped from the model's `classes_` vocabulary, not merely hard to predict.
- For small imbalanced datasets, the "always hold out an internal val set" convention can cost more than it buys. Check the assumption against the dataset's actual shape.

### Related
- `design-notes.md` §12 — RandomForest baseline rationale (claim of 80-90% macro-F1 was from the literature; actual number for our setup likely 55-70% and heavily gated by minority-class F1).
- `dataset.md` §5 — AAMI class distribution; makes the imbalance visible but doesn't predict the starvation pattern.
