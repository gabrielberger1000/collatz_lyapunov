# Progress Log

## Current Best

| Metric | Value | Run |
|--------|-------|-----|
| Growth Weighted Pass % | — | — |
| Test Trajectory Violations | — | — |
| Margin Min | — | — |

---

## Run 1 — 2024-12-02

**Config:**
```bash
python collatz.py \
    --epochs 100000 \
    --layers "2048,1024,1024,512,512,512,256,128,64" \
    --curriculum --ramp-len 100000 --start-seeds 0 \
    --eval-csv collatz_eval_50k.csv
```

**Model:** 9 layers, ~15M parameters

| Epoch | Loss | Pass% | Wtd% | Growth% | Growth Wtd% | Traj Viol | Notes |
|-------|------|-------|------|---------|-------------|-----------|-------|
| 5000 | | | | | | | |
| 10000 | | | | | | | |
| 15000 | | | | | | | |
| 20000 | | | | | | | |
| 25000 | | | | | | | |
| 30000 | | | | | | | |

**Observations:**
- (Notes about what's working, what's not, ideas to try)

---

## Ideas to Try

- [ ] Enable `--use-t4` and `--use-t8` for multi-step constraints
- [ ] Try `--target-type adaptive` to learn drift coefficient
- [ ] Experiment with different architectures
- [ ] Increase lookahead beyond 10
- [ ] Add more residue class features (mod 243, etc.)

## Failed Experiments

(Document what didn't work so you don't repeat it)

---

## Notes

### Hard Seeds Analysis

The test seeds span trajectory difficulty:

| Seed | Length | Max Excursion | Excursion Ratio |
|------|--------|---------------|-----------------|
| 111 | 25 | 3,077 | 27.7× |
| 27 | 42 | 9,232 | 114.0× |
| 703 | 63 | 83,501 | 118.8× |
| 26623 | 114 | 35,452,673 | 1,331.7× |
| 626331 | 190 | 2,407,427,729 | 3,843.7× |

Run `python collatz.py --analyze-seeds` for full analysis.
