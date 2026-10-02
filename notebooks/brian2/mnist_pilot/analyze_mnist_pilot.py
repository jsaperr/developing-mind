"""Scores MP-P1..P4, MC-1, MC-2 (predictions in modal_mnist_pilot.py, written before launch) on
mnist_pilot_seed<S>.json.gz. Written before any result was read. Labels are used only here.

Per image: response r = spike counts during the 350 ms presentation (per neuron). Rectified fingerprint
f = unit(sum_j max(r_j - mean r, 0) w_j), with w = the weight snapshot at the start of that period (label period:
after train; test period: after label). Strength = || centered, unnormalized f ||.

Usage: analyze_mnist_pilot.py   (conda env; writes mnist_pilot_summary.json and receptive-field figures)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.hopfield.episodic_dormant import DormantGatedMemory
from src.integration.interface import unit

KNOWN, HELD = [0, 1, 2, 3], [4, 5]


def load_all():
    runs = []
    for f in sorted(HERE.glob("mnist_pilot_seed*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("status") == "completed":
            runs.append(r)
    return runs


def fingerprints(r, period, wkey):
    C = np.array(r["counts"], dtype=float); y = np.array(r["labels"]); sp = np.array(r["split"])
    W = np.array(r["weights_after"][wkey])
    sel = np.where(sp == period)[0]
    F, S = [], []
    for i in sel:
        v = np.clip(C[:, i] - C[:, i].mean(), 0, None) @ W
        S.append(float(np.linalg.norm(v - v.mean()))); F.append(unit(v))
    return sel, np.array(F), np.array(S), y[sel], C[:, sel]


def main():
    from subset_means import mean_images
    runs = load_all()
    print(f"seeds: {len(runs)}")
    means = mean_images()                                                     # class -> (196,) mean known image
    agg = {k: [] for k in ("spec", "rf", "acc", "sep_ok", "mc1_rise", "mc1_gap", "mc2_ratio", "assign_counts")}
    for r in runs:
        y = np.array(r["labels"]); sp = np.array(r["split"]); C = np.array(r["counts"], dtype=float)
        lab = sp == "label"
        resp = np.array([[C[j, lab & (y == c)].mean() for c in KNOWN] for j in range(r["n_post"])])   # (N, 4)
        order = np.sort(resp, axis=1)
        spec = float(np.mean(order[:, -1] >= 1.5 * np.maximum(order[:, -2], 1e-9)))
        assign = np.array(KNOWN)[resp.argmax(1)]
        W = np.array(r["weights_after"]["test"])
        rf_ok = []
        for j in range(r["n_post"]):
            cors = {c: np.corrcoef(W[j], means[c])[0, 1] for c in KNOWN}
            rf_ok.append(max(cors, key=cors.get) == assign[j])
        test = sp == "test"
        tk = test & np.isin(y, KNOWN)
        pred = []
        for i in np.where(tk)[0]:
            score = {c: C[assign == c, i].mean() if (assign == c).any() else -1 for c in KNOWN}
            pred.append(max(score, key=score.get))
        correct = np.array(pred) == y[tk]
        acc = float(correct.mean())
        sel, F, S, yy, _ = fingerprints(r, "test", "label")
        km = np.isin(yy, KNOWN)
        Fk, yk, Sk = F[km], yy[km], S[km]
        within = {c: np.mean([Fk[a] @ Fk[b] for a in np.where(yk == c)[0] for b in np.where(yk == c)[0] if a < b]) for c in KNOWN}
        sep = all(min(within[a], within[b]) > np.mean([Fk[i] @ Fk[k] for i in np.where(yk == a)[0] for k in np.where(yk == b)[0]])
                  for ia, a in enumerate(KNOWN) for b in KNOWN[ia + 1:])
        q = np.quantile(Sk, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(Sk, q)
        accq = [float(correct[bins == b].mean()) if (bins == b).any() else np.nan for b in range(5)]
        rise = bool(np.all(np.diff(accq) >= -0.05))
        gap = accq[4] - accq[0]
        _, Ftr, _, _, _ = fingerprints(r, "train", "train")
        m = DormantGatedMemory(dim=196, gap_scale=0.1)
        for x in torch.tensor(Ftr, dtype=torch.float32):
            m.step(x, steady=True, changing=False)
        bank = torch.stack([p for p, w, s in m.character()]) if m.character() else torch.zeros(1, 196)
        _, Flab, _, _, _ = fingerprints(r, "label", "train")
        best_lab = (torch.tensor(Flab, dtype=torch.float32) @ bank.T).max(1).values.numpy()
        radius = np.percentile(best_lab, 5)
        best_test = (torch.tensor(F, dtype=torch.float32) @ bank.T).max(1).values.numpy()
        strange_known = float(np.mean(best_test[km] < radius)); strange_held = float(np.mean(best_test[~km] < radius))
        ratio = strange_held / max(strange_known, 1e-9)
        for k, v in (("spec", spec), ("rf", float(np.mean(rf_ok))), ("acc", acc), ("sep_ok", sep), ("mc1_rise", rise),
                     ("mc1_gap", gap), ("mc2_ratio", ratio), ("assign_counts", np.bincount(assign, minlength=4)[KNOWN].tolist())):
            agg[k].append(v)
        print(f"seed {r['seed']}: specialized {spec:.0%} | RF matches class {np.mean(rf_ok):.0%} | acc {acc:.1%} | "
              f"separation {sep} | strength-quintile acc {[round(a, 2) for a in accq]} | strange known {strange_known:.0%} "
              f"held {strange_held:.0%} (ratio {ratio:.1f}) | neurons per class {np.bincount(assign, minlength=4)[KNOWN].tolist()} | "
              f"memory entries {len(m.character())} | mean spikes/image {C.sum(0).mean():.0f}")
    print(f"\nMP-P1 specialized >= 60%: mean {np.mean(agg['spec']):.0%}")
    print(f"MP-P2 RF matches class >= 60%: mean {np.mean(agg['rf']):.0%}")
    print(f"MP-P3 separation in every pair: {sum(agg['sep_ok'])}/{len(runs)} seeds")
    print(f"MP-P4 accuracy >= 50%: mean {np.mean(agg['acc']):.1%} (min {min(agg['acc']):.1%})")
    print(f"MC-1 rise in {sum(agg['mc1_rise'])}/{len(runs)} seeds; weakest-vs-strongest gap mean {np.nanmean(agg['mc1_gap']) * 100:.0f} points")
    print(f"MC-2 held-out strange ratio >= 2: mean {np.mean(agg['mc2_ratio']):.1f}")
    json.dump({k: [float(x) if not isinstance(x, list) else x for x in v] for k, v in agg.items()},
              open(HERE / "mnist_pilot_summary.json", "w"), indent=1)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
