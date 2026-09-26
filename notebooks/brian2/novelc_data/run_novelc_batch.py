"""Novel-C batch: 8 seeds at N=7, 13mV/1.5 (matches the N=7 nonstationary run this contrasts with).
Writes one JSON per seed into this dir. Sequential; each seed logs progress."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_novelc_seed import run_novelc_seed

SEEDS = list(range(32000, 32008))
INHIB_REF = 13.0
GAP = 1.5
N_POST = 7

if __name__ == '__main__':
    t0 = time.time()
    for s in SEEDS:
        out = HERE / f"novelc_n7_reliable_13_1p5_seed{s}.json"
        print(f"=== seed {s} start (elapsed {time.time()-t0:.0f}s) ===", flush=True)
        run_novelc_seed(s, INHIB_REF, GAP, str(out), n_post=N_POST)
    print(f"=== BATCH DONE, {len(SEEDS)} seeds, {time.time()-t0:.0f}s total ===", flush=True)
