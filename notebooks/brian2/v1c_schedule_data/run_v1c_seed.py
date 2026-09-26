"""v1c world: A->B->C->A->C->B. Tests the handoff horizon rule (experiments_integration.md).

The rule under test: memory must bridge only the time from the substrate RELEASING a context to its
return, not the context's whole absence, because the substrate reactivates (refreshes) a context in
memory at the moment it releases it. Here B is held during C (one-back), released at C->A
(refreshed then), and returns only after A AND C, so release-to-return is 2000 s. At a 10 s clock
that's beyond memory's 150-step (1500 s) horizon. In v1b the same interval was 1000 s and B
survived.

Reuses v1's frozen runner; only the schedule constant differs (6 phases x 1000 s).
Usage: run_v1c_seed.py <seed> <out.json.gz>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v1_schedule_data"))
import run_v1_seed as v1

v1.PHASE_CORR_BLOCKS = [0, 1, 2, 0, 2, 1]   # A, B, C, A, C, B
v1.PHASE_DURATIONS_S = [1000.0] * 6

if __name__ == '__main__':
    v1.run_v1_seed(int(sys.argv[1]), sys.argv[2])
    print(f"v1c seed {sys.argv[1]} -> {sys.argv[2]}")
