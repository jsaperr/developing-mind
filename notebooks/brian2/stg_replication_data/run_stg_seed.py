"""Replication of the v1b / v1c results at the OTHER operating point, strong_tight_gate
(10mV reference, gap_scale 1.0). It's bistable at N=3 (about 50/50 differentiate/converge) and
the one point with genuine ongoing identity churn, so it's a harder test of whether one-back
retention, the two-back pass and the handoff horizon rule are properties of the architecture or of
13mV/1.5.

Reuses v1's frozen runner (v1_schedule_data/run_v1_seed.py); only module constants are overridden
before the call: the operating point, the schedule and (for v1c) the phase durations. The 30-input
rig was calibration-checked at this point before the batch (N=7, 300 s: 16.9-18.3 Hz, w_total
9.69-10.24; bar 3-20 Hz / ~9.5-10.5; PASS).

Usage: run_stg_seed.py <v1b|v1c> <seed> <out.json.gz>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v1_schedule_data"))
import run_v1_seed as v1

SCHEDULES = {"v1b": [0, 1, 2, 0, 1], "v1c": [0, 1, 2, 0, 2, 1]}

if __name__ == '__main__':
    sched, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    v1.INHIB_REF_MV = 10.0
    v1.GAP_SCALE = 1.0
    v1.PHASE_CORR_BLOCKS = SCHEDULES[sched]
    v1.PHASE_DURATIONS_S = [1000.0] * len(SCHEDULES[sched])
    v1.run_v1_seed(seed, out)
    print(f"stg {sched} seed {seed} -> {out}")
