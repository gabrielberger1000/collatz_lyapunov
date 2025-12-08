#!/usr/bin/env python3
"""
Generate shifted k-sequences for the Syracuse map.

The shifted k-sequence for odd n is (k1-2, k2-2, k3-2, ...) where ki is the
2-adic valuation of (3 * T^{i-1}(n) + 1).

Properties:
- E[k] = 0 (centered)
- At n=1: sequence is [0] (terminates)
- Values are >= -1
"""

import argparse
import json


def v2(n: int) -> int:
    """Return the 2-adic valuation of n (number of trailing zeros in binary)."""
    if n == 0:
        return 0
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return count


def shifted_k_sequence(n: int, max_len: int) -> list:
    """
    Generate the shifted k-sequence for odd n.
    
    Stops when:
    - max_len terms are generated, OR
    - n reaches 1 (outputs one final 0)
    """
    if n == 1:
        return [0]
    
    seq = []
    curr = n
    
    for _ in range(max_len):
        if curr == 1:
            seq.append(0)
            break
        
        x = 3 * curr + 1
        k = v2(x)
        seq.append(k - 2)
        curr = x >> k  # divide by 2^k
    
    return seq


def main():
    parser = argparse.ArgumentParser(
        description="Generate shifted k-sequences for the Syracuse map"
    )
    parser.add_argument("n", type=int, help="Input number (or max if --range)")
    parser.add_argument("--range", action="store_true",
                        help="Generate sequences for all odd numbers from 1 to n")
    parser.add_argument("--max-len", type=int, default=1000,
                        help="Maximum sequence length (default: 1000)")
    
    args = parser.parse_args()
    
    if args.range:
        # All odd numbers from 1 to n
        for i in range(1, args.n + 1, 2):
            seq = shifted_k_sequence(i, args.max_len)
            print(f"{i}: {json.dumps(seq)}")
    else:
        # Single number
        n = args.n
        if n < 1:
            print("Error: n must be >= 1")
            return
        if n % 2 == 0:
            print(f"Warning: {n} is even, using {n+1}")
            n = n + 1
        seq = shifted_k_sequence(n, args.max_len)
        print(f"{n}: {json.dumps(seq)}")


if __name__ == "__main__":
    main()
