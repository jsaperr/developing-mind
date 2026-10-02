"""Scores the Modal fidelity check (predictions P1-P4 in modal_ensemble_check.py and the arc-06 entry,
committed before results were read). Three-way comparison per seed:
  modal vs saved   modal_ensemble_check_out/modal_seed<N>.json vs apre005_ensemble_data/apre_ensemble_seed<N>.json
  local vs saved   modal_ensemble_check_out/local_seed2001.json (today's laptop env, frozen run_single_seed.py)
  modal vs local   seed 2001, separates version drift since July from Modal itself

Usage: modal_ensemble_compare.py   (any env with numpy)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "modal_ensemble_check_out"
REF = HERE / "apre005_ensemble_data"


def load(p):
    return json.loads(Path(p).read_text())


def cmp(a, b):
    wa = np.array(a["final_corr_w"] + a["final_uncorr_w"]); wb = np.array(b["final_corr_w"] + b["final_uncorr_w"])
    ga, gb = np.array(a["group_mean_gap"]), np.array(b["group_mean_gap"])
    same_len = len(ga) == len(gb)
    return dict(bit=bool(np.array_equal(wa, wb) and same_len and np.array_equal(ga, gb)),
                w_diff=float(np.abs(wa - wb).max()), gap_diff=float(np.abs(ga - gb).max()) if same_len else float("nan"))


def main():
    rows = []
    print("seed  target  wall(s)  post_rate modal/saved   min_gap modal/saved     max_overlap  | modal vs saved: bit, max|dw|, max|dgap|")
    for s in range(2001, 2009):
        m, r = load(OUT / f"modal_seed{s}.json"), load(REF / f"apre_ensemble_seed{s}.json")
        c = cmp(m, r)
        rows.append((s, m, r, c))
        print(f"{s}  {m['target']:6s}  {m['wall_elapsed']:7.0f}  {m['post_rate']:.4f}/{r['post_rate']:.4f}   "
              f"{m['min_group_mean_gap']:.5f}/{r['min_group_mean_gap']:.5f}   {m['max_overlap_fraction']:.2f}/{r['max_overlap_fraction']:.2f}"
              f"  | {c['bit']}, {c['w_diff']:.3g}, {c['gap_diff']:.3g}")
    loc = OUT / "local_seed2001.json"
    if loc.exists():
        l = load(loc); r = load(REF / "apre_ensemble_seed2001.json"); m = load(OUT / "modal_seed2001.json")
        print(f"\nlocal (today's laptop env), seed 2001: status {l.get('status')}, wall {l.get('wall_elapsed', float('nan')):.0f} s")
        for name, (a, b) in (("local vs saved", (l, r)), ("modal vs local", (m, l))):
            c = cmp(a, b)
            print(f"  {name:15s} bit-identical {c['bit']}, max|dw| {c['w_diff']:.3g}, max|dgap| {c['gap_diff']:.3g}")
    walls = [m['wall_elapsed'] for _, m, _, _ in rows]
    stats_ok = all(18.7 <= m['post_rate'] <= 18.9 and -0.022 <= m['min_group_mean_gap'] <= -0.007 and 0.68 <= m['max_overlap_fraction'] <= 1.0
                   for _, m, _, _ in rows)
    print(f"\nP1 bit-identical to saved: {sum(c['bit'] for *_, c in rows)}/8")
    print(f"P2 statistics in the stated ranges for all 8: {stats_ok}")
    print(f"P4 wall_elapsed in 560-1100 s: {sum(560 <= w <= 1100 for w in walls)}/8 (range {min(walls):.0f}-{max(walls):.0f} s)")


if __name__ == "__main__":
    main()
