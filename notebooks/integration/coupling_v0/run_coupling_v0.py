"""Coupling toy v0 (system_contract.md section 9, step 4): offline, one-way substrate -> memory.

Saved Brian2 runs are replayed into the existing, validated memory code (src/hopfield: two-layer
w_fast/w_char + episodic grow-and-prune, ambiguity-gated retrieval). There's no live coupling and
no feedback into the substrate (Q3 is out of scope for v0). Jasper gave the explicit go
(2026-09-25).

WORLDS (existing runs, N=7, 13mV/1.5):
  A->B->A        nonstationary_data/nonstat_n7_reliable_13_1p5_*   (2 blocks, 20 inputs)
  A->B->C        novelc_data/novelc_n7_reliable_13_1p5_*           (3 blocks, 30 inputs)
  A->B->C->A->C  v1_schedule_data/v1_n7_reliable_13_1p5_*          (3 blocks; if present)

CLOCK (Q2): one memory step = one readout window of W seconds, W in {10, 50}. Windows that
straddle a swap are dropped (same as S1).

QUERY ARMS (dim = n_pre; every query is centered and unit-normalized):
  substrate  H = sum_j (r_j - mean r) w_j, the S1 readout that carries context identity.
             Label-free.
  clean      prototype of the TRUE current context (the block indicator, centered and normalized)
             plus Gaussian noise. The noise is set per seed so the mean settled cosine to the
             prototype matches the substrate arm's. It switches instantly at a swap. A control
             that uses labels: the memory layer's own baseline.
  blended    normalize((1-f) s_prev + f s_cur + same noise), where f is the substrate's own
             per-window mixing fraction (f = b/(a+b), with a and b the positive parts of the
             substrate query's projections onto s_prev and s_cur). This reproduces the
             substrate's ambiguity time course with clean ingredients (the other session's
             control). Substrate vs blended isolates substrate-specific structure (e.g. 2-back
             components) from ambiguity alone.

MEMORY: EpisodicMemory with the validated EPISODIC_OPERATING_POINT (k=0.5, w_fast_max=10),
beta=4, staleness_threshold=150 steps, gap_scale_evict=20, strength_bonus=10, all unchanged.
gap_scale is re-grounded the way it was originally grounded (0.6 x the empirical top1-top2
similarity-gap median, experiments.md). Pass 1 runs the substrate arm with the old 0.1534 and
records the gaps the gate actually sees. Pass 2, the reported run, uses 0.6 x that median for ALL
arms, so the arms differ only in their queries.

CREATION RULE (PROVISIONAL; contract Q4). The episodic layer never had one: every validated
notebook added patterns by oracle. Here a new entry is created from the query when its best
cosine to every stored pattern is below THETA=0.5, midway between S1's within-context (about 0.95)
and between-context (at most 0.35) similarities. The same rule applies in every arm. Sensitivity
at THETA 0.3 and 0.7 is run on the substrate arm only, descriptively. Separately, src
retrieve_gated indexes sorted_sims[1] and so fails with a single stored pattern. The toy handles
the 1-pattern case in its own wrapper (winner = that pattern, weight 1) and leaves src/ alone.

SCORING (labels used only here): every entry is tagged with the true context at the step it was
created. The system's "reported context" at a step is its retrieval winner's tag. Per swap:
  latency  steps until reported == true context for 3 consecutive steps
  stale    steps after the swap on which reported == the PREVIOUS context
  reuse    (returns only) whether the first correct winner is an entry created in an earlier phase
Also: overall accuracy, entries created and evicted, alive-count trajectory.

PREDICTIONS ON RECORD (written before this script was first run):
  TOY-P1 (contract P1): after NOVEL swaps at W=10 s, the substrate arm reports the previous
         context on more steps than the clean arm (median over swaps, per world). There's no
         directional prediction for substrate vs blended; that comparison is the attribution test.
  TOY-P2 (contract P2, refined by arc 05 / arc 07): at a ONE-back return (A->B->A swap 2, v1 swap
         4) at W=10 s, the substrate arm reports the returning context within 1 step in >= 6/8
         seeds, because holders' rate jump flips H before any weight moves, and it does so by
         REUSING the old entry. At the TWO-back return (v1 swap 3), it doesn't in >= 6/8 seeds.
B1-B5 are otherwise descriptive (contract section 4), with no pass bars.

REVISIONS AFTER THE FIRST RUN (the predictions above are unchanged). Both are fixes to the
controls and scoring, not to the system under test. The first run's output is kept in
first_run_output.txt.
  1. Label leak in scoring. Entries were tagged with the TRUE context at their creation step, so
     an entry created at a novel swap counted as "correct" by construction, even when its content
     was mostly the old context (the contaminated-proxy failure, principles.md #4). Entries are
     now tagged by CONTENT: the context prototype they're most similar to.
  2. Unfair control noise. Clean/blended noise was matched to the substrate's cosine to the
     prototype, using i.i.d. noise per step. The substrate's deviation is mostly a stable offset
     (consecutive windows correlate at >= 0.95), so that gave the controls far MORE step-to-step
     jitter than the substrate (they created 14-27 duplicate entries vs the substrate's 8). Noise
     is now matched to the substrate's consecutive-window cosine: same jitter, none of the
     systematic distortion.
Also added: "other" = steps reporting a context that's neither the current nor the previous one.

Usage: run_coupling_v0.py   (conda env; writes coupling_v0_summary.json and coupling_v0_*.png)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "brian2" / "interface_readout"))
from src.brian2_stdp.results_io import load_result
from src.hopfield.episodic import EpisodicMemory
from analyze_s1_readout import windows

B2 = ROOT / "notebooks" / "brian2"
WORLDS = [
    ("ABA", B2 / "nonstationary_data", "nonstat_n7_reliable_13_1p5_seed*.json"),
    ("ABC", B2 / "novelc_data", "novelc_n7_reliable_13_1p5_seed*.json"),
    ("ABCAC", B2 / "v1_schedule_data", "v1_n7_reliable_13_1p5_seed*.json.gz"),
]
CLOCKS = [10, 50]
ARMS = ["substrate", "clean", "blended"]
THETA = 0.5
THETA_SENS = [0.3, 0.7]
OLD_GAP_SCALE = 0.1534
NAMES = "ABC"


def unit(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def prototypes(n_pre, block_size):
    return [unit(np.r_[np.zeros(b * block_size), np.ones(block_size), np.zeros(n_pre - (b + 1) * block_size)])
            for b in range(n_pre // block_size)]


def load_world(ddir, pattern):
    out = []
    for f in sorted(glob.glob(str(ddir / pattern))):
        d = load_result(f)
        if d.get('status') != 'completed':
            continue
        n = d['n_post']; si = np.array(d['syn_i']); sj = np.array(d['syn_j'])
        n_pre = int(si.max()) + 1
        w = np.array(d['weight_trace'])
        wm = np.zeros((n, n_pre, w.shape[1])); wm[sj, si] = w
        r = np.array(d['spike_rate_bins'], dtype=float)
        out.append((d, wm, r, n_pre, d.get('block_size', 10)))
    return out


def build_streams(d, wm, r, n_pre, bs, W, rng):
    """Per-window true context, previous context, and one query per arm."""
    ps = [0.0] + list(d['swap_times_s'])
    blocks = d['phase_corr_blocks']
    ws = windows(d, wm, r, ps, W)
    protos = prototypes(n_pre, bs)
    ph = np.array([x[0] for x in ws]); settled = np.array([x[1] for x in ws])
    true = np.array([blocks[p] for p in ph]); prev = np.array([blocks[p - 1] if p else -1 for p in ph])
    sub = np.array([unit(x[4]) for x in ws])
    c_sub = np.mean([sub[i] @ protos[true[i]] for i in range(len(ws)) if settled[i]])
    # match the substrate's STEP-TO-STEP jitter (consecutive settled windows, same phase), not its
    # distance from the prototype. For i.i.d. noise, E[cos(q_i, q_i+1)] ~ 1/(1 + d*sigma^2).
    t1 = np.mean([sub[i] @ sub[i + 1] for i in range(len(ws) - 1)
                  if settled[i] and settled[i + 1] and ph[i] == ph[i + 1]])
    sigma = np.sqrt(max(1 / max(t1, 1e-3) - 1, 0) / n_pre)
    clean = np.array([unit(protos[true[i]] + sigma * rng.standard_normal(n_pre)) for i in range(len(ws))])
    blend, fs = [], []
    for i in range(len(ws)):
        if prev[i] < 0:
            blend.append(clean[i]); fs.append(1.0); continue
        a = max(sub[i] @ protos[prev[i]], 0.0); b = max(sub[i] @ protos[true[i]], 0.0)
        f = b / (a + b) if a + b > 0 else 0.5
        fs.append(f)
        blend.append(unit((1 - f) * protos[prev[i]] + f * protos[true[i]] + sigma * rng.standard_normal(n_pre)))
    return dict(ph=ph, true=true, prev=prev, settled=settled, c_sub=c_sub, t1=t1, sigma=sigma, f=np.array(fs),
                protos=np.array(protos), q={"substrate": sub, "clean": clean, "blended": np.array(blend)})


def step_memory(mem, q):
    """Retrieve and update, handling the 1-pattern case that src retrieve_gated can't. Returns
    (winner_index, gap or None)."""
    if len(mem.patterns) == 1:
        w_fast, w_char = mem._update_fn(torch.tensor(mem.w_fast), torch.tensor(mem.w_char), torch.tensor([1.0]))
        mem.w_fast, mem.w_char = w_fast.tolist(), w_char.tolist()
        mem.staleness[0] = 0
        return 0, None
    sims = torch.stack(mem.patterns) @ q
    top = torch.sort(sims, descending=True).values
    winner, _, _ = mem.retrieve_and_update(q)
    return winner, float(top[0] - top[1])


def run_memory(stream, arm, gap_scale, theta):
    Q = torch.tensor(stream['q'][arm], dtype=torch.float32)
    mem = EpisodicMemory(dim=Q.shape[1], gap_scale=gap_scale)
    tag, born_phase = {}, {}
    reported, winner_ids, alive, gaps = [], [], [], []
    created = 0
    for s in range(len(Q)):
        q = Q[s]
        if not mem.patterns or float((torch.stack(mem.patterns) @ q).max()) < theta:
            mem.add_pattern(q.clone(), s)
            tag[mem.ids[-1]] = int(np.argmax(stream['protos'] @ q.numpy()))   # tag by CONTENT
            born_phase[mem.ids[-1]] = int(stream['ph'][s])
            created += 1
        wi, gap = step_memory(mem, q)
        if gap is not None:
            gaps.append(gap)
        wid = mem.ids[wi]
        winner_ids.append(wid); reported.append(tag[wid])
        mem.prune_step(s)
        alive.append(len(mem.patterns))
    return dict(reported=np.array(reported), winner_ids=winner_ids, born_phase=born_phase,
                alive=np.array(alive), gaps=gaps, created=created, evicted=len(mem.eviction_log))


def score(stream, res):
    true, prev, ph, rep = stream['true'], stream['prev'], stream['ph'], res['reported']
    swaps = [i for i in range(1, len(ph)) if ph[i] != ph[i - 1]]
    per = []
    for k, i0 in enumerate(swaps):
        i1 = swaps[k + 1] if k + 1 < len(swaps) else len(ph)
        lat = None
        for i in range(i0, i1 - 2):
            if np.all(rep[i:i + 3] == true[i]):
                lat = i - i0; break
        stale = int(np.sum(rep[i0:i1] == prev[i0]))
        other = int(np.sum((rep[i0:i1] != prev[i0]) & (rep[i0:i1] != true[i0])))
        reuse = None
        if lat is not None and true[i0] in set(true[:i0].tolist()):
            wid = res['winner_ids'][i0 + lat]
            reuse = bool(res['born_phase'][wid] < ph[i0])
        per.append(dict(swap=k + 1, frm=int(prev[i0]), to=int(true[i0]),
                        returning=bool(true[i0] in set(true[:i0].tolist())), latency=lat, stale=stale,
                        other=other, reuse=reuse))
    return dict(acc=float(np.mean(rep == true)), swaps=per)


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    summary = {}
    for world, ddir, pat in WORLDS:
        runs = load_world(ddir, pat)
        if not runs:
            print(f"[{world}] no completed runs, skipped"); continue
        for W in CLOCKS:
            streams = [build_streams(d, wm, r, n_pre, bs, W, rng) for d, wm, r, n_pre, bs in runs]
            # pass 1: ground gap_scale from the substrate arm's own gaps
            g1 = np.concatenate([run_memory(s, "substrate", OLD_GAP_SCALE, THETA)['gaps'] for s in streams])
            gs = 0.6 * float(np.median(g1))
            key = f"{world}_W{W}"
            summary[key] = dict(gap_scale=gs, gap_median=float(np.median(g1)),
                                c_sub=float(np.mean([s['c_sub'] for s in streams])),
                                t1=float(np.mean([s['t1'] for s in streams])), arms={})
            for arm in ARMS:
                res = [run_memory(s, arm, gs, THETA) for s in streams]
                sc = [score(s, x) for s, x in zip(streams, res)]
                summary[key]['arms'][arm] = dict(
                    acc=[x['acc'] for x in sc], swaps=[x['swaps'] for x in sc],
                    created=[x['created'] for x in res], evicted=[x['evicted'] for x in res],
                    alive_max=[int(x['alive'].max()) for x in res], alive_end=[int(x['alive'][-1]) for x in res])
                if W == 10 and arm in ("substrate", "blended", "clean"):
                    summary[key]['arms'][arm]['trace_seed0'] = dict(
                        true=streams[0]['true'].tolist(), reported=res[0]['reported'].tolist(),
                        alive=res[0]['alive'].tolist(), f=streams[0]['f'].tolist())
            summary[key]['theta_sens'] = {}
            for th in THETA_SENS:
                res = [run_memory(s, "substrate", gs, th) for s in streams]
                sc = [score(s, x) for s, x in zip(streams, res)]
                summary[key]['theta_sens'][str(th)] = dict(acc=[x['acc'] for x in sc], created=[x['created'] for x in res])
            report(key, summary[key])
    json.dump(summary, open(HERE / "coupling_v0_summary.json", "w"), indent=1)
    figure(summary)


def report(key, s):
    print(f"\n===== {key}: gap median {s['gap_median']:.3f} -> gap_scale {s['gap_scale']:.4f} "
          f"(old 0.1534); substrate settled cos to true prototype {s['c_sub']:.2f}, "
          f"consecutive-window cos {s['t1']:.3f}")
    for arm, a in s['arms'].items():
        print(f"  {arm:9s} acc {np.mean(a['acc']):.3f}  created {np.mean(a['created']):.1f}  "
              f"evicted {np.mean(a['evicted']):.1f}  alive max/end {np.mean(a['alive_max']):.1f}/{np.mean(a['alive_end']):.1f}")
    n_sw = len(s['arms']['substrate']['swaps'][0])
    for k in range(n_sw):
        sw0 = s['arms']['substrate']['swaps'][0][k]
        line = f"  swap {k + 1} {NAMES[sw0['frm']]}->{NAMES[sw0['to']]}{' (return)' if sw0['returning'] else ' (novel)'}:"
        for arm, a in s['arms'].items():
            lats = [x[k]['latency'] for x in a['swaps']]
            stale = [x[k]['stale'] for x in a['swaps']]
            other = [x[k]['other'] for x in a['swaps']]
            reuse = [x[k]['reuse'] for x in a['swaps'] if x[k]['reuse'] is not None]
            line += (f"\n      {arm:9s} lat {lats}  stale {stale}  other-ctx {sum(other)}"
                     + (f"  reuse {sum(reuse)}/{len(reuse)}" if sw0['returning'] else ""))
        print(line)
    print("  theta sensitivity (substrate): " + "  ".join(
        f"theta={th}: acc {np.mean(v['acc']):.3f} created {np.mean(v['created']):.1f}" for th, v in s['theta_sens'].items()))


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in summary if k.endswith("_W10")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.6 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        arms = summary[key]['arms']
        t = arms['substrate']['trace_seed0']
        x = np.arange(len(t['true']))
        ax.step(x, np.array(t['true']) + 0.0, where='post', color='k', lw=2.5, label='true context')
        for off, arm, col in ((0.12, 'substrate', 'C3'), (-0.12, 'blended', 'C0'), (0.24, 'clean', 'C2')):
            ax.step(x, np.array(arms[arm]['trace_seed0']['reported']) + off, where='post', color=col, lw=1, label=f'{arm} reported')
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(list(NAMES))
        ax.set_title(f"{key}: seed 0, reported vs true context per 10 s step", fontsize=9)
        ax2 = ax.twinx(); ax2.plot(x, t['alive'], color='grey', lw=0.7, ls=':'); ax2.set_ylabel('entries alive', fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='upper left', ncol=4)
    fig.tight_layout(); fig.savefig(HERE / "coupling_v0_traces.png", dpi=80)
    print("\nsaved coupling_v0_traces.png")


if __name__ == '__main__':
    main()
