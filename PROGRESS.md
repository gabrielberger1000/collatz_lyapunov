# Progress Log

## Current Best

| Metric | Value | Run |
|--------|-------|-----|
| Growth Weighted Pass % | 98.02% | Run 2 |
| Test Trajectory Violations | 87/429 (20.3%) | Run 2 |
| Margin Min (Growth) | -0.40 | Run 2 |

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
- [ ] **Faster curriculum** — `--ramp-len 50000` (model plateaued, maybe introduce hard seeds faster)
- [ ] **Gentler mining** — `--mine-interval 5000 --mine-count 20` (avoid destabilization)
- [ ] **Disable mining** — compare to baseline without mining disruption

### Medium Priority
- [ ] `--target-type adaptive` — let model learn drift coefficient
- [ ] `--use-t4 --use-t8` — multi-step constraints for trajectory consistency
- [ ] Smaller model — current one may be overfitting
- [ ] Larger lookahead — `--lookahead 20`

### Investigate
- [ ] Why are decay steps failing more than growth steps?
- [ ] What's special about seeds 703, 26623, 626331 that makes them hard?

---

## Failed Experiments

(None yet — document what doesn't work)

---

## Notes

### Hard Seeds Analysis

The test seeds span trajectory difficulty:

| Seed | Length | Max Excursion | Excursion Ratio | Status |
|------|--------|---------------|-----------------|--------|
| 111 | 25 | 3,077 | 27.7× | ✓ Solved |
| 27 | 42 | 9,232 | 114.0× | ✓ Solved |
| 703 | 63 | 83,501 | 118.8× | ✗ Stuck |
| 26623 | 114 | 35,452,673 | 1,331.7× | ✗ Stuck |
| 626331 | 190 | 2,407,427,729 | 3,843.7× | ✗ Stuck |

The pattern: longer trajectories with higher excursion ratios remain unsolved.

### Mining Destabilization

At epoch 10k, mining found 3088 violations and added 50 hard negatives. This caused:
- Trajectory violations spiked from 25.9% → 71.3%
- Pass rate dropped from 93% → 84%
- Took 5k epochs to recover

The mined samples were likely very different from the training distribution, causing catastrophic interference.

### The Plateau Problem

From epoch 40k onwards, metrics barely moved:
- Pass rate: 95.42% ± 0.02%
- Trajectory violations: 87-89 / 429
- Growth margin min: -0.40

The model is stuck in a local minimum. The hard trajectories (703, 26623, 626331) have consistent ~1.1 margin violations that aren't improving.