# Collatz Lyapunov Function Learner

A neural network approach to learning Lyapunov functions for the Collatz conjecture. This is an experimental/educational project — not a serious attempt to prove the conjecture, but an exploration of how far gradient-based learning can get on a famously hard problem.

## Background

### The Collatz Conjecture

The Collatz conjecture states that for any positive integer n, the sequence defined by:

```
T(n) = n/2       if n is even
T(n) = 3n + 1    if n is odd
```

eventually reaches 1. Despite its simple statement, this remains unproven since 1937.

### Syracuse Form

We use the "Syracuse" (or "shortcut") formulation that tracks only odd numbers:

```
T(n) = (3n + 1) / 2^k
```

where k is the 2-adic valuation of (3n+1) — i.e., the number of times you can divide by 2. This compresses the trajectory and focuses on the "interesting" steps.

### Lyapunov Functions

A Lyapunov function V(n) is a function satisfying:

1. V(n) > 0 for all n > 1
2. V(T(n)) < V(n) for all n > 1

If such a function exists and can be proven to work for all n, it would prove the Collatz conjecture — the sequence must terminate because V strictly decreases and is bounded below.

The challenge: during a Collatz trajectory, n often *increases* before eventually decreasing (e.g., 27 → 41 → 31 → 47 → ...). A Lyapunov function must decrease at every step even when n itself grows, encoding information about "future potential."

### Our Approach

We train a neural network to approximate V(n) using:

- **Feature engineering**: log(n), residue classes mod 3/9/27/81, lookahead k-values, and drift from expected behavior
- **Ranking loss**: Enforce V(n) > V(T(n)) + margin
- **Anchor loss**: Guide V toward a target scale based on ln(n) + β·drift
- **Curriculum learning**: Gradually introduce harder seed trajectories
- **Hard negative mining**: Find and focus on failure cases

## Installation

```bash
pip install torch numpy
```

## Usage

### 1. Generate Evaluation Set (once)

```bash
python collatz_eval_gen.py --samples 50000 --output collatz_eval_50k.csv --seed 42
```

This creates a fixed evaluation set that should never be used for training.

### 2. Train

Basic training:
```bash
python collatz.py --epochs 50000 --eval-csv collatz_eval_50k.csv
```

Full configuration (example):
```bash
python collatz.py \
    --epochs 100000 \
    --layers "2048,1024,1024,512,512,512,256,128,64" \
    --curriculum --ramp-len 100000 --start-seeds 0 \
    --mine-negatives --mine-interval 10000 \
    --eval-csv collatz_eval_50k.csv \
    --eval-interval 5000 \
    --quick-eval-interval 1000
```

### 3. Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--layers` | Hidden layer dimensions | `128,128,128` |
| `--lookahead` | Steps to look ahead for features | `10` |
| `--curriculum` | Gradually introduce hard seeds | off |
| `--mine-negatives` | Enable hard negative mining | off |
| `--eval-csv` | Path to fixed evaluation CSV | none |
| `--use-t4`, `--use-t8` | Multi-step constraints | off |
| `--target-type` | `fixed` or `adaptive` beta | `fixed` |

Run `python collatz.py --help` for full options.

## Evaluation Metrics

### Trajectory Metrics
- **Violations**: Steps where V(T(n)) ≥ V(n) along held-out seed trajectories
- Test seeds span difficulty spectrum: `[111, 27, 703, 26623, 626331]`

### Population Metrics (from eval CSV)
- **Pass rate**: % of samples where V(n) > V(T(n))
- **Weighted pass rate**: Pass rate weighted by difficulty (w_expansion)
- **Growth pass rate**: Pass rate on growth steps only (T(n) > n) — the hard cases

### Margin Distribution
- **margin = V(n) - V(T(n))**: Should be positive
- Track mean, median, min, and 5th percentile
- Negative margins indicate violations; small positive margins indicate fragility

## Files

| File | Description |
|------|-------------|
| `collatz.py` | Main training script |
| `collatz_eval_gen.py` | Generates fixed evaluation datasets |
| `PROGRESS.md` | Experiment log and results |

## Theory Notes

### Why This Is Hard

The fundamental obstacle is that Collatz trajectories can grow arbitrarily large before collapsing. For example:
- 27 reaches a maximum of 9,232 (342× its starting value) before descending to 1
- 837,799 reaches ~2.4 billion (2,974× start) over 524 steps

A Lyapunov function must "know" at n=27 that the upcoming growth to 9,232 will eventually be paid back by decay. This requires encoding subtle information about the trajectory's future behavior.

### The log₂(3) Threshold

Each Syracuse step multiplies by 3 and divides by 2^k. On average, k ≈ 2, so:
- Expected change: ×3 / 4 = ×0.75 (decay)
- Critical ratio: log₂(3) ≈ 1.585

When cumulative k-values fall below this threshold, the trajectory grows. Our "drift" feature measures deviation from expected behavior:
```
drift = (lookahead × log₂(3)) - Σ(k_values)
```

Positive drift indicates faster-than-average growth.

### What Success Would Look Like

A "perfect" model would have:
- 0 violations on all test trajectories
- 100% pass rate (weighted and unweighted)
- Positive margins everywhere, with reasonable minimum

In practice, we're exploring how close gradient descent can get and what patterns emerge.

## License

MIT

## References

- Lagarias, J. C. (1985). The 3x + 1 problem and its generalizations. *The American Mathematical Monthly*, 92(1), 3-23.
- Tao, T. (2019). Almost all orbits of the Collatz map attain almost bounded values. *arXiv:1909.03562*.
