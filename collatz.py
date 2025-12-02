"""
Collatz Lyapunov Function Trainer v3

Trains a neural network to approximate a Lyapunov function V(n) for the
Collatz (Syracuse) conjecture, attempting to satisfy V(T(n)) < V(n) for all n > 1.

Enhancements over v2:
- Fixed evaluation set loaded from CSV
- Weighted accuracy metrics using w_expansion
- Separate tracking of growth steps
- Quick eval option for frequent lightweight checks
- Comprehensive margin distribution analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import math
import os
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict

# =============================================================================
# CONSTANTS
# =============================================================================

LOG2_3 = 1.5849625007211563  # math.log2(3) - the critical growth/decay ratio
DEFAULT_BETA = 0.35          # Default drift coefficient
MAX_K_CLIP = 10              # Cap on k-values for normalization
LOG10_SCALE = 40.0           # Normalization divisor for log10(n)
DRIFT_SCALE = 10.0           # Normalization divisor for drift

# Known "hard" seeds with long trajectories or large excursions
HARD_SEEDS = [
    27, 31, 41, 47, 54, 55, 71, 73, 91, 97,
    103, 111, 127, 155, 159, 171, 231, 237,
    267, 303, 319, 351, 359, 391, 447, 479,
    607, 703, 871, 1161, 2223, 2463, 2919, 3711,
    6171, 10971, 13255, 17647, 23529, 26623, 34239,
    35655, 52527, 77031, 106239, 142587, 156159, 216367,
    230631, 410011, 511935, 626331, 837799, 1117065
]

# Default held-out test seeds (stratified by trajectory length, max excursion within tier)
DEFAULT_TEST_SEEDS = [111, 27, 703, 26623, 626331]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""
    
    # Architecture
    layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    lookahead: int = 10
    
    # Constraints (which future steps to enforce)
    use_t1: bool = True
    use_t4: bool = False
    use_t8: bool = False
    
    # Target/Loss configuration
    target_type: str = "fixed"  # "fixed" or "adaptive"
    split_loss: bool = False    # Apply adaptive beta only to hard seeds
    
    # Training hyperparameters
    batch_size: int = 1024
    max_bits: int = 128
    lr: float = 1e-3
    margin: float = 0.05
    epochs: int = 20001
    
    # Hard seed / curriculum settings
    hard_ratio: float = 0.25
    num_hard_seeds: int = -1  # -1 means use all
    curriculum: bool = False
    start_seeds: int = 30
    ramp_len: int = 25000
    
    # Hard negative mining
    mine_negatives: bool = False
    mine_interval: int = 5000
    mine_count: int = 50
    
    # Test/train split for trajectory validation
    test_seeds: List[int] = field(default_factory=lambda: DEFAULT_TEST_SEEDS.copy())
    
    # Evaluation settings
    eval_csv: Optional[str] = None  # Path to fixed eval CSV
    eval_interval: int = 5000       # Full eval every N epochs
    quick_eval_interval: int = 1000 # Quick eval every N epochs (0 to disable)
    
    # Anchor decay schedule
    decay_start: int = 1000
    decay_len: int = 5000
    min_anchor: float = 0.0
    
    # I/O
    load_checkpoint: Optional[str] = None
    save_model: str = "collatz_model.pth"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainConfig':
        """Construct config from parsed command-line arguments."""
        layers = [int(x) for x in args.layers.split(',')]
        
        # Parse test seeds if provided
        if args.test_seeds:
            test_seeds = [int(x) for x in args.test_seeds.split(',')]
        else:
            test_seeds = DEFAULT_TEST_SEEDS.copy()
        
        return cls(
            layers=layers,
            lookahead=args.lookahead,
            use_t1=args.use_t1,
            use_t4=args.use_t4,
            use_t8=args.use_t8,
            target_type=args.target_type,
            split_loss=args.split_loss,
            batch_size=args.batch_size,
            max_bits=args.max_bits,
            lr=args.lr,
            margin=args.margin,
            epochs=args.epochs,
            hard_ratio=args.hard_ratio,
            num_hard_seeds=args.num_hard_seeds,
            curriculum=args.curriculum,
            start_seeds=args.start_seeds,
            ramp_len=args.ramp_len,
            mine_negatives=args.mine_negatives,
            mine_interval=args.mine_interval,
            mine_count=args.mine_count,
            test_seeds=test_seeds,
            eval_csv=args.eval_csv,
            eval_interval=args.eval_interval,
            quick_eval_interval=args.quick_eval_interval,
            decay_start=args.decay_start,
            decay_len=args.decay_len,
            min_anchor=args.min_anchor,
            load_checkpoint=args.load_checkpoint,
            save_model=args.save_model,
        )


# =============================================================================
# SYRACUSE / COLLATZ FUNCTIONS
# =============================================================================

def get_valuation(n: int) -> int:
    """Return the 2-adic valuation of n (number of trailing zeros in binary)."""
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1


def syracuse_step(n: int) -> Tuple[int, int]:
    """
    Perform one Syracuse step: n -> (3n+1) / 2^k where k is the 2-adic valuation.
    
    Returns:
        (next_odd, k): The next odd number in the sequence and the valuation k.
    """
    if n <= 1:
        return 1, 0
    
    if n % 2 == 0:
        while n % 2 == 0:
            n //= 2
        return n, 0
    
    x = 3 * n + 1
    k = get_valuation(x)
    return x >> k, k


def get_future_step(n: int, steps: int) -> int:
    """Return the value after `steps` Syracuse iterations."""
    curr = n
    for _ in range(steps):
        curr, _ = syracuse_step(curr)
        if curr <= 1:
            return 1
    return curr


def compute_trajectory_stats(n: int, lookahead: int) -> Tuple[List[int], float]:
    """
    Compute k-values and drift for lookahead steps starting from n.
    """
    k_vals = []
    curr = n
    cumulative_k = 0
    
    for _ in range(lookahead):
        x = 3 * curr + 1
        k = get_valuation(x)
        k_vals.append(k)
        cumulative_k += k
        curr = x >> k
    
    drift = (lookahead * LOG2_3) - cumulative_k
    return k_vals, drift


def get_trajectory(n: int) -> List[int]:
    """Generate the full Syracuse trajectory from n to 1."""
    traj = [n]
    curr = n
    while curr > 1:
        curr, _ = syracuse_step(curr)
        traj.append(curr)
    return traj


# =============================================================================
# TRAJECTORY ANALYSIS
# =============================================================================

@dataclass
class TrajectoryStats:
    """Statistics for a single seed's trajectory."""
    seed: int
    length: int
    max_val: int
    excursion_ratio: float
    growth_steps: int
    growth_ratio: float
    max_consecutive_growth: int


def analyze_trajectory(seed: int) -> TrajectoryStats:
    """Compute comprehensive statistics for a seed's trajectory."""
    curr = seed
    traj = [curr]
    max_val = curr
    growth_steps = 0
    max_consecutive_growth = 0
    consecutive_growth = 0
    
    while curr > 1:
        prev = curr
        curr, _ = syracuse_step(curr)
        traj.append(curr)
        max_val = max(max_val, curr)
        
        if curr > prev:
            growth_steps += 1
            consecutive_growth += 1
            max_consecutive_growth = max(max_consecutive_growth, consecutive_growth)
        else:
            consecutive_growth = 0
    
    return TrajectoryStats(
        seed=seed,
        length=len(traj),
        max_val=max_val,
        excursion_ratio=max_val / seed,
        growth_steps=growth_steps,
        growth_ratio=growth_steps / len(traj),
        max_consecutive_growth=max_consecutive_growth,
    )


def analyze_seeds(seeds: List[int] = None) -> List[TrajectoryStats]:
    """Analyze all seeds and return sorted by trajectory length."""
    if seeds is None:
        seeds = HARD_SEEDS
    stats = [analyze_trajectory(s) for s in seeds]
    stats.sort(key=lambda x: x.length)
    return stats


def print_seed_analysis(seeds: List[int] = None):
    """Print detailed analysis of hard seeds."""
    stats = analyze_seeds(seeds)
    
    print("=" * 95)
    print(f"{'Seed':>10} {'Length':>8} {'Max Val':>15} {'Excursion':>12} "
          f"{'Growth%':>8} {'MaxConsec':>10}")
    print("=" * 95)
    
    for s in stats:
        print(f"{s.seed:>10} {s.length:>8} {s.max_val:>15,} {s.excursion_ratio:>11.1f}x "
              f"{s.growth_ratio*100:>7.1f}% {s.max_consecutive_growth:>10}")
    
    print("=" * 95)


# =============================================================================
# EVALUATION DATASET
# =============================================================================

@dataclass
class EvalDataset:
    """
    Fixed evaluation dataset loaded from CSV.
    
    Attributes:
        n: List of input values
        t1: List of Syracuse successors T(n)
        is_growth: Boolean array, True where t1 > n
        sigma_n: List of stopping times
        w_expansion: Array of difficulty weights
    """
    n: List[int]
    t1: List[int]
    is_growth: np.ndarray
    sigma_n: List[int]
    w_expansion: np.ndarray
    
    @classmethod
    def from_csv(cls, path: str) -> 'EvalDataset':
        """Load evaluation dataset from CSV file."""
        n_list = []
        t1_list = []
        is_growth_list = []
        sigma_list = []
        w_list = []
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_list.append(int(row['n']))
                t1_list.append(int(row['t1']))
                is_growth_list.append(int(row['is_growth']) == 1)
                sigma_list.append(int(row['sigma_n']))
                w_list.append(float(row['w_expansion']))
        
        return cls(
            n=n_list,
            t1=t1_list,
            is_growth=np.array(is_growth_list, dtype=bool),
            sigma_n=sigma_list,
            w_expansion=np.array(w_list, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        return len(self.n)
    
    def summary(self) -> str:
        """Return summary string."""
        n_growth = self.is_growth.sum()
        return (f"EvalDataset: {len(self):,} samples, "
                f"{n_growth:,} growth ({100*n_growth/len(self):.1f}%), "
                f"w_expansion: [{self.w_expansion.min():.2f}, {self.w_expansion.max():.2f}]")


# =============================================================================
# BATCH DATA STRUCTURE (for training)
# =============================================================================

@dataclass
class CollatzBatch:
    """Container for a batch of training samples."""
    
    n: List[int]
    t1: List[int]
    t4: List[int]
    t8: List[int]
    log_n: List[float]
    drift: List[float]
    weights: List[float]
    
    def to_tensors(
        self,
        device: torch.device,
        lookahead: int,
        extract_fn: Callable
    ) -> dict:
        """Convert batch to model-ready tensors."""
        return {
            'n_vec': extract_fn(self.n, device, lookahead),
            't1_vec': extract_fn(self.t1, device, lookahead),
            't4_vec': extract_fn(self.t4, device, lookahead),
            't8_vec': extract_fn(self.t8, device, lookahead),
            'log_t': torch.tensor(self.log_n, dtype=torch.float32, device=device).unsqueeze(1),
            'drift_t': torch.tensor(self.drift, dtype=torch.float32, device=device).unsqueeze(1),
            'weights': torch.tensor(self.weights, dtype=torch.float32, device=device).unsqueeze(1),
        }


# =============================================================================
# SAMPLER
# =============================================================================

class CollatzSampler:
    """Handles sampling of training data with support for hard seeds and curriculum."""
    
    def __init__(
        self,
        all_hard_seeds: List[int],
        max_bits: int,
        lookahead: int
    ):
        self.all_hard_seeds = list(all_hard_seeds)
        self.active_seeds = list(all_hard_seeds)
        self.max_bits = max_bits
        self.lookahead = lookahead
        self.mined_negatives: List[int] = []
    
    def set_active_seeds(self, count: int) -> None:
        """Set how many hard seeds are active (for curriculum learning)."""
        self.active_seeds = self.all_hard_seeds[:count]
    
    def set_curriculum_progress(self, progress: float, start_seeds: int) -> None:
        """Update active seeds based on curriculum progress."""
        progress = min(1.0, max(0.0, progress))
        total = len(self.all_hard_seeds)
        count = int(start_seeds + (total - start_seeds) * progress)
        self.active_seeds = self.all_hard_seeds[:count]
    
    def add_mined_negatives(self, negatives: List[int]) -> None:
        """Add hard negatives found during training."""
        self.mined_negatives = list(set(self.mined_negatives + negatives))
        if len(self.mined_negatives) > 1000:
            self.mined_negatives = self.mined_negatives[-1000:]
    
    def _sample_from_hard_trajectory(self) -> Optional[int]:
        """Sample a random odd value from a random hard seed's trajectory."""
        pool = self.active_seeds + self.mined_negatives
        
        if not pool:
            seed = 27
        else:
            seed = random.choice(pool)
        
        if seed in self.mined_negatives and random.random() < 0.3:
            return seed if seed % 2 == 1 else seed + 1
        
        traj = []
        curr = seed
        for _ in range(300):
            if curr <= 1:
                break
            if curr % 2 != 0:
                traj.append(curr)
            curr, _ = syracuse_step(curr)
        
        return random.choice(traj) if traj else None
    
    def _sample_random_odd(self) -> int:
        """Sample a random odd number with uniform bit-length distribution."""
        k = random.randint(2, self.max_bits - 4)
        val = random.randint(2**k, (2**(k + 1)) - 1)
        if val % 2 == 0:
            val += 1
        return val
    
    def _create_sample(self, val: int, w_growth: float, w_decay: float) -> dict:
        """Create a single sample with all required fields."""
        t1 = get_future_step(val, 1)
        t4 = get_future_step(val, 4)
        t8 = get_future_step(val, 8)
        
        _, drift = compute_trajectory_stats(val, self.lookahead)
        ln_n = math.log(max(1, val))
        
        weight = w_growth if t1 > val else w_decay
        
        return {
            'n': val,
            't1': t1,
            't4': t4,
            't8': t8,
            'log_n': ln_n,
            'drift': drift,
            'weight': weight,
        }
    
    def get_batch(self, batch_size: int, hard_ratio: float) -> CollatzBatch:
        """Generate a batch of training samples."""
        samples = []
        num_hard = int(batch_size * hard_ratio)
        
        while len(samples) < num_hard:
            val = self._sample_from_hard_trajectory()
            if val is not None:
                samples.append(self._create_sample(val, w_growth=20.0, w_decay=2.0))
        
        while len(samples) < batch_size:
            val = self._sample_random_odd()
            samples.append(self._create_sample(val, w_growth=10.0, w_decay=1.0))
        
        return CollatzBatch(
            n=[s['n'] for s in samples],
            t1=[s['t1'] for s in samples],
            t4=[s['t4'] for s in samples],
            t8=[s['t8'] for s in samples],
            log_n=[s['log_n'] for s in samples],
            drift=[s['drift'] for s in samples],
            weights=[s['weight'] for s in samples],
        )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(n_list: List[int], device: torch.device, lookahead: int) -> torch.Tensor:
    """
    Extract neural network input features from a list of integers.
    
    Features:
        - log10(n) / 40: Normalized magnitude
        - n mod 3, 9, 27, 81: Residue classes (normalized)
        - drift: Deviation from expected log2(3) behavior
        - k_1, ..., k_lookahead: Normalized valuation sequence
    """
    features = []
    
    for n in n_list:
        log_val = math.log10(max(1, n)) / LOG10_SCALE
        
        mod3 = (n % 3) / 2.0
        mod9 = (n % 9) / 8.0
        mod27 = (n % 27) / 26.0
        mod81 = (n % 81) / 80.0
        
        k_vals, drift = compute_trajectory_stats(n, lookahead)
        drift_norm = drift / DRIFT_SCALE
        k_feats = [min(k, MAX_K_CLIP) / float(MAX_K_CLIP) for k in k_vals]
        
        row = [log_val, mod3, mod9, mod27, mod81, drift_norm] + k_feats
        features.append(row)
    
    return torch.tensor(features, dtype=torch.float32, device=device)


# =============================================================================
# MODEL
# =============================================================================

class CollatzFeatureNet(nn.Module):
    """
    Neural network that predicts a Lyapunov-like energy value for each input.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        self.drift_param = nn.Parameter(torch.tensor(DEFAULT_BETA))
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.1))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + 1e-6


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class LyapunovLoss(nn.Module):
    """Combined loss for Lyapunov function training."""
    
    def __init__(
        self,
        margin: float,
        use_t1: bool = True,
        use_t4: bool = False,
        use_t8: bool = False
    ):
        super().__init__()
        self.rank_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.mse_loss = nn.MSELoss()
        self.use_t1 = use_t1
        self.use_t4 = use_t4
        self.use_t8 = use_t8
    
    def forward(
        self,
        v_n: torch.Tensor,
        v_t1: torch.Tensor,
        v_t4: torch.Tensor,
        v_t8: torch.Tensor,
        weights: torch.Tensor,
        target: torch.Tensor,
        anchor_weight: float
    ) -> torch.Tensor:
        ones = torch.ones_like(weights)
        loss = torch.tensor(0.0, device=v_n.device)
        
        if self.use_t1:
            loss = loss + (self.rank_loss(v_n, v_t1, ones) * weights).mean()
        if self.use_t4:
            loss = loss + self.rank_loss(v_n, v_t4, ones).mean() * 0.5
        if self.use_t8:
            loss = loss + self.rank_loss(v_n, v_t8, ones).mean() * 0.2
        
        loss = loss + self.mse_loss(v_n, target) * anchor_weight
        
        return loss


# =============================================================================
# SCHEDULERS
# =============================================================================

class AnchorScheduler:
    """Manages the decay of anchor loss weight during training."""
    
    def __init__(self, decay_start: int, decay_len: int, min_anchor: float):
        self.decay_start = decay_start
        self.decay_len = decay_len
        self.min_anchor = min_anchor
    
    def get_weight(self, epoch: int) -> float:
        if epoch < self.decay_start:
            return 1.0
        progress = min(1.0, (epoch - self.decay_start) / self.decay_len)
        return self.min_anchor + (1.0 - self.min_anchor) * (0.001 ** progress)


# =============================================================================
# TARGET COMPUTATION
# =============================================================================

def compute_target(
    log_t: torch.Tensor,
    drift_t: torch.Tensor,
    model: CollatzFeatureNet,
    config: TrainConfig
) -> torch.Tensor:
    """Compute anchor target values."""
    if config.target_type == 'adaptive':
        if config.split_loss:
            split = int(config.batch_size * config.hard_ratio)
            target = torch.zeros_like(log_t)
            target[:split] = log_t[:split] + model.drift_param * drift_t[:split]
            target[split:] = log_t[split:] + DEFAULT_BETA * drift_t[split:]
        else:
            target = log_t + model.drift_param * drift_t
    else:
        target = log_t + DEFAULT_BETA * drift_t
    
    return target


# =============================================================================
# HARD NEGATIVE MINING
# =============================================================================

def find_hard_negatives(
    model: CollatzFeatureNet,
    device: torch.device,
    lookahead: int,
    sampler: CollatzSampler,
    n_samples: int = 10000,
    n_return: int = 100
) -> Tuple[List[int], Dict]:
    """Find samples where the model fails most badly."""
    model.eval()
    
    batch = sampler.get_batch(n_samples, hard_ratio=0.5)
    
    n_vec = extract_features(batch.n, device, lookahead)
    t1_vec = extract_features(batch.t1, device, lookahead)
    
    with torch.no_grad():
        v_n = model(n_vec).squeeze()
        v_t1 = model(t1_vec).squeeze()
        margins = v_n - v_t1
    
    worst_idx = torch.argsort(margins)[:n_return]
    worst_n = [batch.n[i] for i in worst_idx.cpu().numpy()]
    
    stats = {
        'n_violations': int((margins <= 0).sum()),
        'worst_margin': float(margins.min()),
        'mean_margin': float(margins.mean()),
    }
    
    return worst_n, stats


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_on_trajectories(
    model: CollatzFeatureNet,
    device: torch.device,
    lookahead: int,
    test_seeds: List[int]
) -> Dict:
    """
    Evaluate model on held-out trajectory seeds.
    
    Returns dict with per-seed violations and totals.
    """
    model.eval()
    results = {}
    
    total_violations = 0
    total_steps = 0
    
    for seed in test_seeds:
        traj = get_trajectory(seed)
        feats = extract_features(traj, device, lookahead)
        
        with torch.no_grad():
            energies = model(feats).cpu().numpy().flatten()
        
        violations = 0
        worst_margin = float('inf')
        for i in range(len(energies) - 1):
            margin = energies[i] - energies[i + 1]
            if margin <= 0:
                violations += 1
                worst_margin = min(worst_margin, margin)
        
        results[f'traj_{seed}_violations'] = violations
        results[f'traj_{seed}_length'] = len(traj) - 1
        results[f'traj_{seed}_worst'] = worst_margin if violations > 0 else None
        
        total_violations += violations
        total_steps += len(traj) - 1
    
    results['traj_total_violations'] = total_violations
    results['traj_total_steps'] = total_steps
    results['traj_violation_rate'] = total_violations / max(1, total_steps)
    
    return results


def evaluate_on_dataset(
    model: CollatzFeatureNet,
    device: torch.device,
    lookahead: int,
    eval_data: EvalDataset,
    batch_size: int = 4096
) -> Dict:
    """
    Evaluate model on fixed evaluation dataset.
    
    Returns comprehensive metrics including weighted accuracy and margin distribution.
    """
    model.eval()
    
    all_margins = []
    n_samples = len(eval_data)
    
    # Process in batches to handle large eval sets
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        n_batch = eval_data.n[start:end]
        t1_batch = eval_data.t1[start:end]
        
        n_vec = extract_features(n_batch, device, lookahead)
        t1_vec = extract_features(t1_batch, device, lookahead)
        
        with torch.no_grad():
            v_n = model(n_vec).cpu().numpy().flatten()
            v_t1 = model(t1_vec).cpu().numpy().flatten()
        
        margins = v_n - v_t1
        all_margins.append(margins)
    
    margins = np.concatenate(all_margins)
    weights = eval_data.w_expansion
    is_growth = eval_data.is_growth
    
    # Compute pass/fail
    passes = margins > 0
    
    results = {}
    
    # Overall metrics
    results['pass_rate'] = 100 * np.mean(passes)
    results['weighted_pass_rate'] = 100 * np.sum(weights * passes) / np.sum(weights)
    results['n_total'] = n_samples
    
    # Growth step metrics
    n_growth = is_growth.sum()
    results['n_growth'] = int(n_growth)
    
    if n_growth > 0:
        growth_passes = passes[is_growth]
        growth_weights = weights[is_growth]
        growth_margins = margins[is_growth]
        
        results['growth_pass_rate'] = 100 * np.mean(growth_passes)
        results['growth_weighted_pass_rate'] = 100 * np.sum(growth_weights * growth_passes) / np.sum(growth_weights)
        results['growth_margin_mean'] = float(np.mean(growth_margins))
        results['growth_margin_min'] = float(np.min(growth_margins))
    
    # Margin distribution (all samples)
    results['margin_mean'] = float(np.mean(margins))
    results['margin_median'] = float(np.median(margins))
    results['margin_min'] = float(np.min(margins))
    results['margin_5pct'] = float(np.percentile(margins, 5))
    
    return results


def quick_eval(
    model: CollatzFeatureNet,
    device: torch.device,
    lookahead: int,
    sampler: CollatzSampler,
    n_random: int = 1000
) -> str:
    """
    Quick lightweight evaluation.
    
    Returns a single-line summary string.
    """
    model.eval()
    
    # Check trajectory 27
    traj = get_trajectory(27)
    feats = extract_features(traj, device, lookahead)
    with torch.no_grad():
        energies = model(feats).cpu().numpy().flatten()
    
    traj27_violations = sum(1 for i in range(len(energies) - 1) if energies[i + 1] >= energies[i])
    traj27_status = "✓" if traj27_violations == 0 else f"✗({traj27_violations})"
    
    # Check random samples
    batch = sampler.get_batch(n_random, hard_ratio=0.0)
    n_vec = extract_features(batch.n, device, lookahead)
    t1_vec = extract_features(batch.t1, device, lookahead)
    
    with torch.no_grad():
        v_n = model(n_vec).cpu().numpy().flatten()
        v_t1 = model(t1_vec).cpu().numpy().flatten()
    
    random_pass = 100 * np.mean(v_n > v_t1)
    
    return f"[Quick] Traj27: {traj27_status} | Random {n_random}: {random_pass:.1f}% pass"


def print_full_eval(
    traj_results: Dict,
    dataset_results: Optional[Dict],
    test_seeds: List[int],
    epoch: int
):
    """Print formatted full evaluation results."""
    
    print("\n" + "=" * 70)
    print(f"EVALUATION (Epoch {epoch})")
    print("=" * 70)
    
    # Trajectory results
    print("\n[Held-Out Trajectories]")
    for seed in test_seeds:
        violations = traj_results[f'traj_{seed}_violations']
        length = traj_results[f'traj_{seed}_length']
        worst = traj_results.get(f'traj_{seed}_worst')
        
        status = "✓" if violations == 0 else "✗"
        worst_str = f"  (worst: {worst:.4f})" if worst is not None else ""
        print(f"  {seed:>8}: {status} {violations:3d} violations / {length:3d} steps{worst_str}")
    
    total_v = traj_results['traj_total_violations']
    total_s = traj_results['traj_total_steps']
    rate = traj_results['traj_violation_rate']
    print(f"\n  Total: {total_v} violations / {total_s} steps ({100*rate:.1f}%)")
    
    # Dataset results (if available)
    if dataset_results:
        print(f"\n[Population Eval - {dataset_results['n_total']:,} samples]")
        print(f"  Overall:     {dataset_results['pass_rate']:.2f}% pass    "
              f"{dataset_results['weighted_pass_rate']:.2f}% weighted")
        
        if 'growth_pass_rate' in dataset_results:
            print(f"  Growth only: {dataset_results['growth_pass_rate']:.2f}% pass    "
                  f"{dataset_results['growth_weighted_pass_rate']:.2f}% weighted  "
                  f"(n={dataset_results['n_growth']:,})")
        
        print(f"\n[Margin Distribution]")
        print(f"  All:    mean={dataset_results['margin_mean']:.4f}  "
              f"median={dataset_results['margin_median']:.4f}  "
              f"min={dataset_results['margin_min']:.4f}  "
              f"5th_pct={dataset_results['margin_5pct']:.4f}")
        
        if 'growth_margin_mean' in dataset_results:
            print(f"  Growth: mean={dataset_results['growth_margin_mean']:.4f}  "
                  f"min={dataset_results['growth_margin_min']:.4f}")
    
    print("=" * 70 + "\n")


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(config: TrainConfig) -> None:
    """Main training loop."""
    
    device = get_device()
    print(f"Training on: {device}")
    print(f"Constraints: T1={config.use_t1}, T4={config.use_t4}, T8={config.use_t8}")
    print(f"Target type: {config.target_type} | Split loss: {config.split_loss}")
    print(f"Architecture: {config.layers}")
    print(f"Hard negative mining: {config.mine_negatives}")
    print(f"Test seeds (held out): {config.test_seeds}")
    
    # Load evaluation dataset if provided
    eval_data = None
    if config.eval_csv:
        if os.path.exists(config.eval_csv):
            eval_data = EvalDataset.from_csv(config.eval_csv)
            print(f"Eval dataset: {eval_data.summary()}")
        else:
            print(f"WARNING: Eval CSV not found: {config.eval_csv}")
    else:
        print("No eval CSV provided (use --eval-csv for population metrics)")
    
    print()
    
    # Compute training seeds (exclude test seeds)
    all_train_seeds = [s for s in HARD_SEEDS if s not in config.test_seeds]
    if config.num_hard_seeds != -1:
        all_train_seeds = all_train_seeds[:config.num_hard_seeds]
    
    print(f"Training seeds: {len(all_train_seeds)} available")
    
    # Initialize model
    input_dim = 6 + config.lookahead
    model = CollatzFeatureNet(input_dim, config.layers).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Load checkpoint if specified
    if config.load_checkpoint and os.path.exists(config.load_checkpoint):
        try:
            model.load_state_dict(torch.load(config.load_checkpoint, map_location=device))
            print(f"Loaded checkpoint: {config.load_checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Initialize sampler
    sampler = CollatzSampler(
        all_hard_seeds=all_train_seeds,
        max_bits=config.max_bits,
        lookahead=config.lookahead
    )
    
    # Initialize loss, optimizer, schedulers
    loss_fn = LyapunovLoss(
        margin=config.margin,
        use_t1=config.use_t1,
        use_t4=config.use_t4,
        use_t8=config.use_t8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2000
    )
    anchor_scheduler = AnchorScheduler(
        config.decay_start, config.decay_len, config.min_anchor
    )
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        
        # Update curriculum if enabled
        if config.curriculum:
            progress = epoch / config.ramp_len
            sampler.set_curriculum_progress(progress, config.start_seeds)
        
        # Hard negative mining
        if config.mine_negatives and epoch > 0 and epoch % config.mine_interval == 0:
            hard_negs, mine_stats = find_hard_negatives(
                model, device, config.lookahead, sampler,
                n_samples=5000, n_return=config.mine_count
            )
            sampler.add_mined_negatives(hard_negs)
            print(f"  [Mining] Found {mine_stats['n_violations']} violations, "
                  f"worst margin: {mine_stats['worst_margin']:.4f}, "
                  f"added {len(hard_negs)} negatives")
        
        # Get batch and convert to tensors
        batch = sampler.get_batch(config.batch_size, config.hard_ratio)
        tensors = batch.to_tensors(device, config.lookahead, extract_features)
        
        # Forward pass
        v_n = model(tensors['n_vec'])
        v_t1 = model(tensors['t1_vec'])
        v_t4 = model(tensors['t4_vec'])
        v_t8 = model(tensors['t8_vec'])
        
        # Compute target and anchor weight
        target = compute_target(
            tensors['log_t'], tensors['drift_t'], model, config
        )
        anchor_weight = anchor_scheduler.get_weight(epoch)
        
        # Compute loss and update
        loss = loss_fn(
            v_n, v_t1, v_t4, v_t8,
            tensors['weights'], target, anchor_weight
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)
        
        # Quick eval
        if config.quick_eval_interval > 0 and epoch % config.quick_eval_interval == 0:
            quick_str = quick_eval(model, device, config.lookahead, sampler)
            beta = model.drift_param.item()
            n_seeds = len(sampler.active_seeds)
            n_mined = len(sampler.mined_negatives)
            mined_str = f" | Mined: {n_mined}" if config.mine_negatives else ""
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f} | "
                  f"Beta: {beta:.4f} | Seeds: {n_seeds}{mined_str} | {quick_str}")
        
        # Full eval
        if epoch > 0 and epoch % config.eval_interval == 0:
            traj_results = evaluate_on_trajectories(
                model, device, config.lookahead, config.test_seeds
            )
            
            dataset_results = None
            if eval_data is not None:
                dataset_results = evaluate_on_dataset(
                    model, device, config.lookahead, eval_data
                )
            
            print_full_eval(traj_results, dataset_results, config.test_seeds, epoch)
            
            # Save checkpoint
            checkpoint_path = f"{config.save_model}_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    traj_results = evaluate_on_trajectories(
        model, device, config.lookahead, config.test_seeds
    )
    
    dataset_results = None
    if eval_data is not None:
        dataset_results = evaluate_on_dataset(
            model, device, config.lookahead, eval_data
        )
    
    print_full_eval(traj_results, dataset_results, config.test_seeds, config.epochs - 1)
    
    torch.save(model.state_dict(), config.save_model)
    print(f"Saved final model: {config.save_model}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a neural Lyapunov function for the Collatz conjecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Architecture
    arch = parser.add_argument_group("Architecture")
    arch.add_argument("--layers", default="128,128,128",
                      help="Comma-separated hidden layer dimensions")
    arch.add_argument("--lookahead", type=int, default=10,
                      help="Number of lookahead steps for features")
    
    # Constraints
    const = parser.add_argument_group("Constraints")
    const.add_argument("--use-t1", action="store_true", default=True,
                       help="Enforce V(n) > V(T(n))")
    const.add_argument("--use-t4", action="store_true",
                       help="Enforce V(n) > V(T^4(n))")
    const.add_argument("--use-t8", action="store_true",
                       help="Enforce V(n) > V(T^8(n))")
    
    # Target/Loss
    loss_grp = parser.add_argument_group("Loss Configuration")
    loss_grp.add_argument("--target-type", default="fixed", choices=["fixed", "adaptive"],
                          help="How to compute anchor target")
    loss_grp.add_argument("--split-loss", action="store_true",
                          help="Apply adaptive beta only to hard seeds")
    
    # Training
    train_args = parser.add_argument_group("Training")
    train_args.add_argument("--batch-size", type=int, default=1024)
    train_args.add_argument("--max-bits", type=int, default=128,
                            help="Maximum bit length for random samples")
    train_args.add_argument("--lr", type=float, default=1e-3,
                            help="Learning rate")
    train_args.add_argument("--margin", type=float, default=0.05,
                            help="Margin for ranking loss")
    train_args.add_argument("--epochs", type=int, default=20001)
    
    # Hard seeds / Curriculum
    curriculum = parser.add_argument_group("Curriculum Learning")
    curriculum.add_argument("--hard-ratio", type=float, default=0.25,
                            help="Fraction of batch from hard trajectories")
    curriculum.add_argument("--num-hard-seeds", type=int, default=-1,
                            help="Number of hard seeds to use (-1 = all)")
    curriculum.add_argument("--curriculum", action="store_true",
                            help="Enable curriculum learning")
    curriculum.add_argument("--start-seeds", type=int, default=30,
                            help="Number of seeds to start curriculum with")
    curriculum.add_argument("--ramp-len", type=int, default=25000,
                            help="Epochs to ramp up to full seed set")
    
    # Hard negative mining
    mining = parser.add_argument_group("Hard Negative Mining")
    mining.add_argument("--mine-negatives", action="store_true",
                        help="Enable hard negative mining")
    mining.add_argument("--mine-interval", type=int, default=5000,
                        help="Epochs between mining runs")
    mining.add_argument("--mine-count", type=int, default=50,
                        help="Number of hard negatives to add per mining run")
    
    # Test/train split
    split = parser.add_argument_group("Train/Test Split")
    split.add_argument("--test-seeds", type=str, default=None,
                       help="Comma-separated list of held-out test seeds "
                            f"(default: {DEFAULT_TEST_SEEDS})")
    
    # Evaluation
    eval_grp = parser.add_argument_group("Evaluation")
    eval_grp.add_argument("--eval-csv", type=str, default=None,
                          help="Path to fixed evaluation CSV (from collatz_eval_gen.py)")
    eval_grp.add_argument("--eval-interval", type=int, default=5000,
                          help="Epochs between full evaluations")
    eval_grp.add_argument("--quick-eval-interval", type=int, default=1000,
                          help="Epochs between quick evals (0 to disable)")
    
    # Anchor decay
    anchor = parser.add_argument_group("Anchor Schedule")
    anchor.add_argument("--decay-start", type=int, default=1000,
                        help="Epoch to start anchor decay")
    anchor.add_argument("--decay-len", type=int, default=5000,
                        help="Epochs over which to decay anchor")
    anchor.add_argument("--min-anchor", type=float, default=0.0,
                        help="Minimum anchor weight after decay")
    
    # I/O
    io_args = parser.add_argument_group("Input/Output")
    io_args.add_argument("--load-checkpoint", default=None,
                         help="Path to checkpoint to resume from")
    io_args.add_argument("--save-model", default="collatz_model.pth",
                         help="Path to save final model")
    
    # Utility commands
    util = parser.add_argument_group("Utilities")
    util.add_argument("--analyze-seeds", action="store_true",
                      help="Print seed analysis and exit")
    
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    
    if args.analyze_seeds:
        print_seed_analysis()
        return
    
    config = TrainConfig.from_args(args)
    train(config)


if __name__ == "__main__":
    main()
