"""
Collatz Evaluation Dataset Generator

Generates a fixed evaluation set for measuring Lyapunov function quality.
Outputs n, T(n), growth flag, trajectory stats, and difficulty weight.

Usage:
    python collatz_eval_gen.py --samples 50000 --output collatz_eval_50k.csv
"""

import csv
import random
import math
import argparse


def get_valuation(n: int) -> int:
    """Return the 2-adic valuation of n (number of trailing zeros in binary)."""
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1


def syracuse_step(n: int) -> tuple:
    """
    Perform one Syracuse step: n -> (3n+1) / 2^k
    Returns (next_odd, k)
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


def get_collatz_stats(n: int, limit: int = 200000) -> tuple:
    """
    Run full Collatz trajectory for n.
    Returns: (stopping_time, max_value)
    """
    curr = n
    max_val = n
    steps = 0
    
    while curr > 1 and steps < limit:
        if curr % 2 == 0:
            curr //= 2
        else:
            curr = 3 * curr + 1
        
        max_val = max(max_val, curr)
        steps += 1
    
    return steps, max_val


def generate_eval_dataset(num_samples: int, output_file: str, max_bits: int = 128, seed: int = None):
    """
    Generate evaluation dataset with Syracuse successor and difficulty weights.
    
    Output columns:
        n: Starting value (odd)
        t1: Syracuse successor T(n)
        is_growth: 1 if t1 > n, else 0
        sigma_n: Total stopping time
        max_n: Maximum value in trajectory
        w_expansion: Difficulty weight = log10(max(1, max_n/n)) + 1
    """
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    print(f"Generating {num_samples} evaluation samples...")
    print(f"Max bits: {max_bits}")
    
    data = []
    seen = set()
    
    n_growth = 0
    
    while len(data) < num_samples:
        # Sample log-uniformly over bit lengths
        k = random.randint(2, max_bits)
        val = random.randint(2**(k-1), (2**k) - 1)
        
        # Ensure odd
        if val % 2 == 0:
            val += 1
        
        if val in seen:
            continue
        seen.add(val)
        
        # Compute Syracuse successor
        t1, _ = syracuse_step(val)
        is_growth = 1 if t1 > val else 0
        n_growth += is_growth
        
        # Compute full trajectory stats
        sigma_n, max_n = get_collatz_stats(val)
        
        # Compute expansion weight
        if val == 0:
            w_expansion = 1.0
        else:
            w_expansion = math.log10(max(1.0, max_n / val)) + 1.0
        
        data.append({
            'n': val,
            't1': t1,
            'is_growth': is_growth,
            'sigma_n': sigma_n,
            'max_n': max_n,
            'w_expansion': w_expansion,
        })
        
        if len(data) % 10000 == 0:
            print(f"  Generated {len(data):,} / {num_samples:,} "
                  f"({100*n_growth/len(data):.1f}% growth steps)")
    
    # Summary stats
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Growth steps:  {n_growth:,} ({100*n_growth/len(data):.1f}%)")
    print(f"  Decay steps:   {len(data) - n_growth:,} ({100*(len(data)-n_growth)/len(data):.1f}%)")
    
    weights = [d['w_expansion'] for d in data]
    print(f"  w_expansion:   min={min(weights):.2f}, max={max(weights):.2f}, "
          f"mean={sum(weights)/len(weights):.2f}")
    
    # Write CSV
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 't1', 'is_growth', 'sigma_n', 'max_n', 'w_expansion'])
        
        for d in data:
            writer.writerow([
                d['n'],
                d['t1'],
                d['is_growth'],
                d['sigma_n'],
                d['max_n'],
                f"{d['w_expansion']:.4f}"
            ])
    
    print(f"Done. Saved to {output_file}")
    print(f"\nTo use in training:")
    print(f"  python collatz.py --eval-csv {output_file} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fixed evaluation dataset for Collatz Lyapunov training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="collatz_eval_50k.csv",
                        help="Output CSV file")
    parser.add_argument("--max-bits", type=int, default=128,
                        help="Maximum bit length for sampled numbers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (use different seed for train data)")
    
    args = parser.parse_args()
    generate_eval_dataset(args.samples, args.output, args.max_bits, args.seed)
