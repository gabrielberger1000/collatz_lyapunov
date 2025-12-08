# Progress Log

## Current Best

| Metric | Value | Run |
|--------|-------|-----|
| Trajectory Violations | 14/429 (3.3%) | Run 7 |
| Overall Pass % | 99.68% | Run 7 |
| Growth Weighted Pass % | 99.95% | Run 7 |

---

## Run 7 — 2024-12-08 (Lookahead 100) ⭐ BEST

**Config:**
```bash
modal run --detach train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 50000 --start-seeds 0 --lookahead 100"
```

**Model:** 4,589,570 parameters | **Platform:** Modal T4 GPU | **Lookahead:** 100

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.170 | 90.09 | 90.31 | 98.53 | 98.36 | 127/429 (29.6%) | |
| 10000 | 0.062 | 92.57 | 92.80 | 98.68 | 98.58 | 64/429 (14.9%) | 111 ✓, 27 ✓ |
| 15000 | 0.038 | 99.37 | 99.26 | 99.92 | 99.89 | 21/429 (4.9%) | 703 down to 1 |
| 20000 | 0.039 | 99.60 | 99.51 | 99.92 | 99.90 | 17/429 (4.0%) | **703 ✓** |
| 25000 | 0.039 | 99.66 | 99.58 | 99.98 | 99.97 | 18/429 (4.2%) | |
| 40000 | 0.034 | 99.68 | 99.60 | 99.96 | 99.95 | 16/429 (3.7%) | |
| 45000 | 0.031 | 99.68 | 99.60 | 99.96 | 99.95 | 14/429 (3.3%) | Best |

**Final Trajectory Results:**
| Seed | Violations | Steps | Worst Margin | Status |
|------|------------|-------|--------------|--------|
| 111 | 0 | 24 | — | ✓ |
| 27 | 0 | 41 | — | ✓ |
| 703 | 0 | 62 | — | ✓ **SOLVED** |
| 26623 | 1 | 113 | -0.06 | Almost! |
| 626331 | 13 | 189 | -0.41 | ✗ |

**Observations:**
- **Massive improvement** — 14 violations vs 58 in Run 6 (76% reduction)
- **703 completely solved** — 0 violations (was 8 in Run 6)
- **26623 nearly solved** — only 1 violation with tiny margin (-0.06)
- **626331 greatly improved** — 13 violations vs 32 (59% reduction)
- **99.68% overall pass rate** — up from 97.91%
- **Confirms hypothesis**: lookahead matters enormously, no ceiling observed yet

**Lookahead Progression:**
| Lookahead | Violations | 703 | 26623 | 626331 |
|-----------|------------|-----|-------|--------|
| 10 | 64/429 | 11 | 16 | 37 |
| 20 | 58/429 | 8 | 18 | 32 |
| 100 | **14/429** | **0** | **1** | **13** |

---

## Run 6 — 2024-12-07 (Lookahead 20)

**Config:**
```bash
modal run --detach train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 50000 --start-seeds 0 --lookahead 20"
```

**Model:** 4,425,730 parameters | **Platform:** Modal T4 GPU | **Lookahead:** 20

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.041 | 89.85 | 89.67 | 97.35 | 97.15 | 108/429 (25.2%) | 111 ✓ |
| 10000 | 0.027 | 92.60 | 92.36 | 99.22 | 99.10 | 91/429 (21.2%) | 27 ✓ |
| 15000 | 0.011 | 97.09 | 96.53 | 99.36 | 99.17 | 74/429 (17.2%) | |
| 20000 | 0.011 | 97.81 | 97.31 | 99.63 | 99.53 | 63/429 (14.7%) | |
| 25000 | 0.006 | 98.01 | 97.50 | 99.76 | 99.68 | 60/429 (14.0%) | |
| 50000 | 0.013 | 97.94 | 97.43 | 99.82 | 99.75 | 58/429 (13.5%) | Plateaued |
| 85000 | 0.015 | 97.93 | 97.42 | 99.82 | 99.75 | 58/429 (13.5%) | |
| 100000 | — | 97.91 | 97.40 | 99.82 | 99.75 | 58/429 (13.5%) | Final |

**Final Trajectory Results:**
| Seed | Violations | Steps | Worst Margin | Status |
|------|------------|-------|--------------|--------|
| 111 | 0 | 24 | — | ✓ |
| 27 | 0 | 41 | — | ✓ |
| 703 | 8 | 62 | -0.50 | ✗ |
| 26623 | 18 | 113 | -0.79 | ✗ |
| 626331 | 32 | 189 | -1.17 | ✗ |

**Observations:**
- **Lookahead 20 helped!** 58 violations vs 64 in Run 5
- **626331 improved most** — 37 → 32 violations, margins -1.50 → -1.17
- **703 also improved** — 11 → 8 violations
- **26623 mixed** — slightly more violations (16 → 18) but better margins (-1.30 → -0.79)
- **Solved 27 faster** — epoch 8k vs 18k in Run 5

**Comparison to Run 5:**
| Metric | Run 5 (LA=10) | Run 6 (LA=20) | Change |
|--------|---------------|---------------|--------|
| Traj violations | 64/429 | **58/429** | ✓ -9% |
| Overall pass % | 97.22% | **97.91%** | ✓ +0.7% |
| 703 violations | 11 | **8** | ✓ -27% |
| 626331 violations | 37 | **32** | ✓ -14% |
| 626331 worst margin | -1.50 | **-1.17** | ✓ +22% |

---

## Run 5 — 2024-12-06 (No Mining)

**Config:**
```bash
modal run --detach train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 50000 --start-seeds 0"
```

**Model:** 4,405,250 parameters | **Platform:** Modal T4 GPU | **Mining:** Disabled

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.034 | 92.33 | 91.82 | 98.88 | 98.67 | 108/429 (25.2%) | 111 ✓ |
| 10000 | 0.048 | 91.87 | 91.61 | 98.97 | 98.83 | 118/429 (27.5%) | |
| 15000 | 0.018 | 95.28 | 94.62 | 99.28 | 99.11 | 92/429 (21.4%) | |
| 20000 | 0.010 | 96.41 | 95.84 | 99.54 | 99.44 | 80/429 (18.6%) | 27 ✓ |
| 25000 | 0.011 | 96.95 | 96.38 | 99.62 | 99.52 | 75/429 (17.5%) | |
| 50000 | 0.019 | 97.22 | 96.69 | 99.76 | 99.70 | 64/429 (14.9%) | Best |
| 75000 | 0.022 | 97.22 | 96.69 | 99.76 | 99.70 | 64/429 (14.9%) | Plateaued |
| 100000 | — | 97.22 | 96.70 | 99.77 | 99.71 | 64/429 (14.9%) | Final |

**Final Trajectory Results:**
| Seed | Violations | Steps | Worst Margin | Status |
|------|------------|-------|--------------|--------|
| 111 | 0 | 24 | — | ✓ |
| 27 | 0 | 41 | — | ✓ |
| 703 | 11 | 62 | -0.52 | ✗ |
| 26623 | 16 | 113 | -1.30 | ✗ |
| 626331 | 37 | 189 | -1.50 | ✗ |

**Observations:**
- **Mining was hurting, not helping!** Disabling it gave best results across the board
- **Much lower loss** — 0.02 vs 0.06-0.08 with mining
- **More stable training** — no destabilization spikes
- **Best trajectory violations** — 64 vs 73-80 in previous runs
- **Higher pass rates** — 97.22% overall, 99.77% growth

**Comparison to previous runs:**
| Metric | Run 3 (mining) | Run 4 (T4+mining) | Run 5 (no mining) |
|--------|----------------|-------------------|-------------------|
| Traj violations | 80/429 | 73/429 | **64/429** |
| Overall pass % | 95.68% | 95.68% | **97.22%** |
| Growth pass % | 98.72% | 98.62% | **99.77%** |
| Final loss | ~0.06 | ~0.08 | **~0.02** |

---

## Run 4 — 2024-12-03 (Multi-step T4 Constraint)

**Config:**
```bash
modal run --detach train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 50000 --start-seeds 0 --use-t4 --mine-negatives --mine-interval 10000"
```

**Model:** 4,405,250 parameters | **Platform:** Modal T4 GPU | **Constraints:** T1=True, T4=True

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.059 | 91.77 | 91.42 | 99.05 | 98.88 | 131/429 (30.5%) | |
| 10000 | 0.198 | 92.90 | 91.88 | 93.71 | 93.01 | 127/429 (29.6%) | 111 ✓ |
| 15000 | 0.032 | 94.94 | 94.35 | 99.24 | 99.00 | 96/429 (22.4%) | |
| 20000 | 0.084 | 96.28 | 95.69 | 99.34 | 99.13 | 81/429 (18.9%) | 27 ✓ |
| 25000 | 0.041 | 95.63 | 94.93 | 98.66 | 98.28 | 75/429 (17.5%) | Best traj |
| 50000 | 0.080 | 95.66 | 94.97 | 98.64 | 98.24 | 74/429 (17.2%) | |
| 80000 | 0.082 | 95.68 | 95.00 | 98.64 | 98.25 | 71/429 (16.6%) | Best traj |
| 100000 | — | 95.68 | 95.00 | 98.62 | 98.22 | 73/429 (17.0%) | Final |

**Final Trajectory Results:**
| Seed | Violations | Steps | Worst Margin | Status |
|------|------------|-------|--------------|--------|
| 111 | 0 | 24 | — | ✓ |
| 27 | 0 | 41 | — | ✓ |
| 703 | 11 | 62 | -0.25 | ✗ |
| 26623 | 19 | 113 | -0.82 | ✗ |
| 626331 | 43 | 189 | -1.22 | ✗ |

**Observations:**
- **T4 constraint reduced violation count** — 73 vs 80 in Run 3
- **But margins got worse on hardest seeds** — 626331 worst margin -1.22 vs -0.66 in Run 3
- **703 improved significantly** — only 11 violations with -0.25 worst margin
- **Tradeoff**: fewer violations but less confident predictions

**Comparison to Run 3:**
| Metric | Run 3 | Run 4 | Change |
|--------|-------|-------|--------|
| Traj violations | 80/429 | 73/429 | ✓ -9% |
| Worst margin (703) | -0.66 | -0.25 | ✓ +62% |
| Worst margin (26623) | -0.50 | -0.82 | ✗ -64% |
| Worst margin (626331) | -0.66 | -1.22 | ✗ -85% |

---

## Run 3 — 2024-12-03 (Faster Curriculum, Modal)

**Config:**
```bash
modal run --detach train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 50000 --start-seeds 0 --mine-negatives --mine-interval 10000"
```

**Model:** 4,405,250 parameters | **Platform:** Modal T4 GPU

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.046 | 91.79 | 91.40 | 98.54 | 98.37 | 117/429 (27.3%) | |
| 10000 | 0.152 | 91.37 | 90.87 | 96.04 | 95.75 | 131/429 (30.5%) | Mining spike (milder) |
| 15000 | 0.026 | 95.82 | 95.16 | 99.41 | 99.23 | 86/429 (20.0%) | 111 ✓ |
| 20000 | 0.067 | 96.36 | 95.74 | 99.56 | 99.43 | 69/429 (16.1%) | 27 ✓, best traj |
| 50000 | 0.061 | 95.70 | 95.00 | 98.73 | 98.33 | 80/429 (18.6%) | All seeds exhausted |
| 100000 | — | 95.68 | 94.97 | 98.72 | 98.31 | 80/429 (18.6%) | Final |

**Final Trajectory Results:**
| Seed | Violations | Steps | Worst Margin | Status |
|------|------------|-------|--------------|--------|
| 111 | 0 | 24 | — | ✓ |
| 27 | 0 | 41 | — | ✓ |
| 703 | 12 | 62 | -0.66 | ✗ |
| 26623 | 21 | 113 | -0.50 | ✗ |
| 626331 | 47 | 189 | -0.66 | ✗ |

**Observations:**
- **Faster curriculum helped** — 80 violations vs 87 in Run 2, worst margins improved from ~1.0 to ~0.5-0.7
- **Best trajectory result at epoch 20k** — 69 violations (16.1%), then slightly regressed
- **Mining spike milder** — only went to 30.5% violations (vs 71.3% in Run 2)
- **All 49 seeds exhausted by epoch 50k** — plateaued after that
- **Still stuck on 703, 26623, 626331** — but margins are tighter

**Comparison to Run 2:**
| Metric | Run 2 | Run 3 | Change |
|--------|-------|-------|--------|
| Traj violations | 87/429 | 80/429 | ✓ -8% |
| Worst margin (703) | -0.93 | -0.66 | ✓ +29% |
| Worst margin (26623) | -1.08 | -0.50 | ✓ +54% |
| Worst margin (626331) | -1.09 | -0.66 | ✓ +39% |

---

## Run 2 — 2024-12-02 (New Eval Framework)

**Config:**
```bash
python collatz.py \
    --epochs 100000 \
    --layers "2048,1024,1024,512,512,512,256,128,64" \
    --curriculum --ramp-len 100000 --start-seeds 0 \
    --mine-negatives --mine-interval 10000 \
    --eval-csv collatz_eval_50k.csv \
    --eval-interval 5000 --quick-eval-interval 1000
```

**Model:** 4,405,250 parameters | **Time:** 108 minutes on M1 Mac (MPS)

| Epoch | Loss | Pass% | Wtd% | Growth% | GrWtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|--------|-----------|-------|
| 5000 | 0.047 | 92.95 | 92.46 | 99.05 | 98.91 | 111/429 (25.9%) | |
| 10000 | 0.640 | 84.29 | 84.26 | 91.70 | 91.40 | 306/429 (71.3%) | Mining spike |
| 15000 | 0.039 | 95.33 | 94.59 | 99.40 | 99.21 | 82/429 (19.1%) | Recovered |
| 20000 | 0.079 | 95.65 | 94.95 | 99.45 | 99.26 | 77/429 (17.9%) | Best traj |
| 30000 | 0.056 | 95.32 | 94.53 | 98.88 | 98.48 | 81/429 (18.9%) | Traj27 ✓ |
| 50000 | 0.055 | 95.42 | 94.57 | 98.58 | 98.10 | 87/429 (20.3%) | Plateaued |
| 75000 | 0.072 | 95.41 | 94.56 | 98.55 | 98.05 | 88/429 (20.5%) | |
| 100000 | — | 95.43 | 94.57 | 98.53 | 98.02 | 87/429 (20.3%) | Final |

**Final Trajectory Results:**
| Seed | Violations | Steps | Status |
|------|------------|-------|--------|
| 111 | 0 | 24 | ✓ |
| 27 | 0 | 41 | ✓ |
| 703 | 15 | 62 | ✗ |
| 26623 | 23 | 113 | ✗ |
| 626331 | 49 | 189 | ✗ |

**Observations:**
- **Plateaued around epoch 40k** — no meaningful improvement from 40k to 100k
- **Mining destabilized at 10k** — injecting 50 hard negatives caused temporary collapse (71% violations). Consider gentler mining (fewer samples, more frequent)
- **Easy seeds solved early** — 111 and 27 passed by epoch 26k-30k
- **Hard seeds stuck** — 703, 26623, 626331 never improved past ~15/24/49 violations
- **Growth steps are easy!** — 98-99% pass rate on growth steps, but only 95% overall. The model struggles more on *decay* steps, which is surprising
- **Worst violations ~1.1** — the stuck trajectories all have worst margins around -1.0 to -1.1

---

## Run 1 — 2024-12-02 (Old Eval Framework, Baseline)

**Config:**
```bash
python collatz.py \
    --epochs 100000 \
    --layers "2048,1024,1024,512,512,512,256,128,64" \
    --curriculum --ramp-len 100000 --start-seeds 0
```

**Final Results (old eval):**
- Trajectory 27: 0 violations / 41 steps ✓
- Random 10k: 97.33% pass rate

**Notes:** This run used the old validation method (only traj27 + random samples). Not directly comparable to Run 2, but shows the model can pass traj27 without mining.

---

## Ideas to Try

### Completed
- [x] **Faster curriculum** — `--ramp-len 50000` ✓ Helped (Run 3)
- [x] **Multi-step constraints** — `--use-t4` Mixed results (Run 4)
- [x] **Disable mining** — ✓ Significant improvement (Run 5)
- [x] **Lookahead 20** — ✓ Helped (Run 6)
- [x] **Lookahead 100** — ✓ **Major breakthrough!** (Run 7)

### Future Directions (if continuing)
- [ ] **Lookahead 200** — would it solve 626331?
- [ ] **Interpretability** — what features does the model actually use?
- [ ] **Failure analysis** — where exactly in trajectories do violations occur?
- [ ] **Feature ablation** — which inputs matter most?

### Deprioritized
- [ ] Adaptive beta
- [ ] Gentler mining
- [ ] Smaller model

---

## Key Findings

**Lookahead is the key variable (Run 7):** Increasing from 20 to 100 steps reduced violations from 58 to 14 — a 76% improvement. 703 is now completely solved, 26623 has only 1 violation. This strongly suggests that constructing a Lyapunov function requires knowing many steps of the future trajectory.

**No ceiling observed:** The progression LA=10→20→100 shows continued improvement with no sign of diminishing returns. The information horizon for V(n) extends at least 50-100 steps into the future.

**Implication for closed-form solutions:** If V(n) truly requires ~100 steps of lookahead, this explains why no simple closed-form Lyapunov function has been found. Any such function would need to encode long-range trajectory information, not just local properties of n.

**Mining was harmful (Run 5):** Disabling mining gave a big improvement. Simpler training is better.

**T4 Multi-step constraint (Run 4):** Mixed results, not clearly helpful.

---

## Notes

### Hard Seeds Analysis

The test seeds span trajectory difficulty:

| Seed | Length | Max Excursion | Excursion Ratio | Best Run | Status |
|------|--------|---------------|-----------------|----------|--------|
| 111 | 25 | 3,077 | 27.7× | Run 3+ | ✓ Solved |
| 27 | 42 | 9,232 | 114.0× | Run 3+ | ✓ Solved |
| 703 | 63 | 83,501 | 118.8× | **Run 7 (0 viol)** | ✓ **Solved** |
| 26623 | 114 | 35,452,673 | 1,331.7× | Run 7 (1 viol, -0.06) | Almost! |
| 626331 | 190 | 2,407,427,729 | 3,843.7× | Run 7 (13 viol, -0.41) | ✗ Stuck |

Lookahead 100 solved 703 completely and nearly solved 26623. The remaining challenge is 626331 with its 190-step trajectory — even 100 steps of lookahead only covers ~53% of its future.

### Mining Destabilization (Deprioritized)

Mining was too aggressive in Runs 2-4:
- 50 hard negatives injected every 10k epochs
- Caused destabilization spikes (especially Run 2: 25.9% → 71.3% violations)
- Run 5 showed best results without mining

May revisit with gentler settings:
- Less frequent (`--mine-interval 25000`)
- Fewer samples (would need code change)
- Start after curriculum exhausts (`--mine-start-epoch 50000`, would need code change)

### The Plateau Problem

All runs plateau after exhausting training seeds (~50k epochs):

| Run | Final Violations | Final Pass % | Lookahead | Notes |
|-----|------------------|--------------|-----------|-------|
| Run 2 | 87/429 | 95.43% | 10 | Mining + slow curriculum |
| Run 3 | 80/429 | 95.68% | 10 | Mining + fast curriculum |
| Run 4 | 73/429 | 95.68% | 10 | Mining + T4 |
| Run 5 | 64/429 | 97.22% | 10 | No mining |
| Run 6 | 58/429 | 97.91% | 20 | No mining |
| Run 7 | **14/429** | **99.68%** | **100** | **No mining** ⭐ |

Clear trend: lookahead is the dominant factor. Increasing from 10→100 reduced violations by 78%.