"""v1b world: A->B->C->A->B. The clean two-back test for the complementary-systems split.

The final return to B (swap 4) comes after TWO intervening contexts (C, then A), and B is NOT the
first context learned. v1 (A->B->C->A->C) only had a two-back return to A, confounded with the
first-context weakness (arc 07) and, at a 10 s clock, with memory eviction. See
experiments_integration.md.

Reuses v1's runner unchanged (run_v1_seed.py stays frozen). This module only swaps the schedule
constant before calling it, so the rig, seeds' construction and output format are identical to v1.

Usage: run_v1b_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v1_schedule_data"))
import run_v1_seed as v1

v1.PHASE_CORR_BLOCKS = [0, 1, 2, 0, 1]   # A, B, C, A (two back, first context), B (two back, not first)

if __name__ == '__main__':
    v1.run_v1_seed(int(sys.argv[1]), sys.argv[2])
    print(f"v1b seed {sys.argv[1]} -> {sys.argv[2]}")
