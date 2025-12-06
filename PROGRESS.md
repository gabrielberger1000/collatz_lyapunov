# Progress Log

## Current Best

| Metric | Value | Run |
|--------|-------|-----|
| Trajectory Violations | 73/429 (17.0%) | Run 4 |
| Growth Weighted Pass % | 98.31% | Run 3 |
| Margin Min (Growth) | -0.52 | Run 3 |

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

### High Priority
- [x] **Faster curriculum** — `--ramp-len 50000` ✓ Helped (Run 3)
- [x] **Multi-step constraints** — `--use-t4` ✓ Fewer violations but worse margins on hardest seeds (Run 4)
- [ ] **Disable mining** — compare to baseline without mining disruption
- [ ] **Adaptive beta** — `--target-type adaptive` to learn drift coefficient

### Medium Priority
- [ ] `--use-t4 --use-t8` — both multi-step constraints together
- [ ] Smaller model — current one may be overfitting
- [ ] Larger lookahead — `--lookahead 20`

### Investigate
- [ ] Why are decay steps failing more than growth steps?
- [ ] What's special about seeds 703, 26623, 626331 that makes them hard?
- [ ] Why did T4 help 703 but hurt 626331?

---

## Mixed Results

**T4 Multi-step constraint (Run 4):** Reduced total violations (73 vs 80) but made worst-case margins worse on the hardest seeds. The constraint seems to spread the "effort" differently — 703 got much better while 626331 got worse. Not a clear win.

---

## Notes

### Hard Seeds Analysis

The test seeds span trajectory difficulty:

| Seed | Length | Max Excursion | Excursion Ratio | Best Run | Status |
|------|--------|---------------|-----------------|----------|--------|
| 111 | 25 | 3,077 | 27.7× | Run 3+ | ✓ Solved |
| 27 | 42 | 9,232 | 114.0× | Run 3+ | ✓ Solved |
| 703 | 63 | 83,501 | 118.8× | Run 4 (11 viol, -0.25) | ✗ Stuck |
| 26623 | 114 | 35,452,673 | 1,331.7× | Run 4 (19 viol, -0.82) | ✗ Stuck |
| 626331 | 190 | 2,407,427,729 | 3,843.7× | Run 3 (47 viol, -0.66) | ✗ Stuck |

The pattern: longer trajectories with higher excursion ratios remain unsolved. Note that 703 responds better to T4 constraint while 626331 does not.

### Mining Destabilization

Run 2 (epoch 10k): Mining found 3088 violations and added 50 hard negatives. This caused:
- Trajectory violations spiked from 25.9% → 71.3%
- Pass rate dropped from 93% → 84%
- Took 5k epochs to recover

Run 3 (epoch 10k): Much milder spike — 27.3% → 30.5% violations. The faster curriculum may have helped the model be more robust to hard negatives.

### The Plateau Problem

All runs plateau after exhausting training seeds:
- Run 2: Plateaued at epoch 40k
- Run 3: Plateaued at epoch 50k
- Run 4: Plateaued at epoch 50k

Final metrics barely move after plateau:
- Pass rate: ~95.7% ± 0.1%
- Trajectory violations: 73-80/429
- Growth margin min: ~-0.5 to -0.65

The hard trajectories (703, 26623, 626331) remain unsolved. Interestingly, different approaches help different seeds:
- T4 constraint helped 703 (margins improved from -0.66 to -0.25)
- T4 constraint hurt 626331 (margins worsened from -0.66 to -1.22)

This suggests these seeds have structurally different challenges.