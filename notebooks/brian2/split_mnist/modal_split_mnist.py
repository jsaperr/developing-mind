"""Split-MNIST two-back (generalization plan step 4b): the two-back test on real data, on the reconciled substrate.
PREPARED 2026-10-02, NOT LAUNCHED (Jasper reviews first).

Schedule: digit sets {0,1} -> {2,3} -> {4,5} -> {0,1} -> {2,3}, 5 x 1000 s (2000 images per phase, 350 ms on + 150 ms
blank, 14x14). A return shows NEW images of the same digits; every image is used once (split_subset.npz, built by
prep_split() from the local torchvision MNIST train split, git-ignored; the seed shuffles within each phase).
Substrate: the reconciled one, not adopted as a default:
- winner-take-all 20 mV (= v_thresh - v_rest), ambiguity gate off;
- leaky fair-share threshold (step and tau from the gain map; default 0.05 mV / 100 s);
- causal gain control (tau_g 5 ms, mnist_sim.build_input_temporal; no image boundaries used);
- N=40, budget 30, Apre 0.005, dt 0.2 ms, Brian2 seeded.
Recording (as in the life run): every 1 s, D(t) = sum |w(t) - w(t-60)| from a ring buffer gives the change flags
(interface.changing_per_second's rule), plus 10 s block means of all weights, 10 s rates, and per-image spike counts.
Compact output: rectified/contrast fingerprints at W = 10, 50; change flags; D; rates10; assignments every 50 s (nearest
digit-set prototype = unit(mean image of the set)); per-image counts and labels. Raw npz on the Volume (life-run pattern).

PREDICTIONS (draft, to be committed before launch; fingerprint check 2026-10-02: on this substrate, 10 s windows of the SAME
digit set agree at only ~0.75 (5th pct ~0.45), below memory's 0.8 radius and 0.9 steadiness gate; between sets 0.27-0.47):
  SM-P1 (the plan's pass bar) default memory at W=50 names the returning {0,1} correctly in more than half of its settled
        checks in >= 6/8 seeds, with absorption 0. Same for the {2,3} return.
  SM-P2 at W=10 the same check FAILS (<= 3/8): within-set noise keeps the steadiness gate shut and makes most checks NOVEL.
        If so, that's the clock question (Q2) on real data: the clock must be long enough to average a context's samples.
  SM-P3 substrate: before {0,1} returns (two back), <= 30% of neurons are nearest {0,1} (flat retention on this substrate
        gave 12-27% per past context with six click contexts).
  SM-P4 the change signal fires within 300 s after >= 3 of the 4 switches in >= 6/8 seeds; the activity signal (familiar-
        switch rule) after all 4.
  MC-3 (plan): at switches to a never-seen set, at least one NOVEL check before the first correct naming; at the {0,1}
        return, no NOVEL before the first correct naming; >= 6/8 seeds, W=50.

Cost: 8 seeds x 5000 s simulated, about 25 min wall each, about $0.25 on Modal (or free on the laptop, 8 parallel).
Prepare data (local, once):  python notebooks/brian2/split_mnist/modal_split_mnist.py prep
Launch (repo root):          python -m modal run --detach notebooks/brian2/split_mnist/modal_split_mnist.py
Collect:                     python -m modal run notebooks/brian2/split_mnist/modal_split_mnist.py --collect
"""
import io
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MN = HERE.parent / "mnist_pilot"
sys.path.insert(0, str(HERE.parent))
from modal_common import IMAGE, RAW_MOUNT, RAW_VOLUME  # noqa: E402

SUBSET = MN / "data" / "split_subset.npz"
SETS = [(0, 1), (2, 3), (4, 5), (0, 1), (2, 3)]
PER_PHASE = 2000
SEEDS = list(range(55000, 55008))
TP, TAU_THETA, TAU_G = 0.05, 100.0, 0.005
N_POST, BUDGET = 40, 30.0

if SUBSET.exists():
    image = (IMAGE.add_local_file(str(HERE.parent / "modal_common.py"), "/root/modal_common.py")
             .add_local_file(str(MN / "mnist_sim.py"), "/root/mnist_sim.py")
             .add_local_file(str(SUBSET), "/root/split_subset.npz"))
else:
    image = IMAGE
app = modal.App("developing-mind-split-mnist")


def prep_split():
    """Fixed images for every seed: per phase, PER_PHASE train-split images of its digit set, disjoint across phases."""
    import numpy as np
    import torchvision
    tr = torchvision.datasets.MNIST(str(MN / "data"), train=True, download=False)
    X = tr.data.numpy().astype(np.float32).reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).round().astype(np.uint8)
    y = tr.targets.numpy()
    rng = np.random.default_rng(2026)
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(6)}
    xs, ys = [], []
    for s in SETS:
        idx = []
        for c in s:
            idx += [pools[c].pop() for _ in range(PER_PHASE // len(s))]
        idx = np.array(idx)
        xs.append(X[idx]); ys.append(y[idx])
    np.savez_compressed(SUBSET, x=np.stack(xs), y=np.stack(ys))
    print("wrote", SUBSET, np.stack(xs).shape)


def rel(seed, kind):
    return f"split_mnist/{kind}/split_seed{seed}." + ("json.gz" if kind == "compact" else "npz")


@app.function(image=image, cpu=1.0, memory=8192, timeout=6 * 3600, max_containers=16, volumes={RAW_MOUNT: RAW_VOLUME},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_split(seed: int, phase_images: int = PER_PHASE) -> str:
    import time
    import numpy as np
    from brian2 import SpikeMonitor, defaultclock, mV, ms, network_operation, prefs, run, second, start_scope, seed as b2_seed
    import mnist_sim as S
    import modal_common as MC
    from src.brian2_stdp import network as N
    from src.integration.interface import unit

    z = np.load("/root/split_subset.npz" if Path("/root/split_subset.npz").exists() else str(SUBSET))
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(z["x"].shape[1])[:phase_images] for _ in SETS]
    seq_x = np.concatenate([z["x"][p][perms[p]].reshape(phase_images, -1) for p in range(len(SETS))]).astype(float)
    seq_y = np.concatenate([z["y"][p][perms[p]] for p in range(len(SETS))])
    total_ref = float(seq_x[:phase_images].sum(1).mean())            # the first phase's mean ink: the gain set point
    idx, t, starts = S.build_input_temporal(seq_x, S.MAX_RATE, rng, total_ref, TAU_G)
    period = S.PRESENT_S + S.REST_S
    total = len(seq_x) * period
    phase_s = phase_images * period

    start_scope()
    defaultclock.dt = 0.2 * ms
    b2_seed(seed)
    inh = float((N.v_thresh - N.v_rest) / mV)
    pre, post, syn, inhib, share = S.build_adaptive(N_POST, idx, t * second, 0.005, inh * mV, 1e9, n_pre=196,
                                                    w_init=BUDGET / 196, target_total=BUDGET, theta_plus=TP * mV,
                                                    tau_theta=TAU_THETA * second, fair_share=True)
    si, sj = np.array(syn.i[:]), np.array(syn.j[:])
    n_syn, n_sec, n_blk = len(si), int(total), int(total // 10)
    ring = np.zeros((61, n_syn), np.float32)
    D = np.full(n_sec, np.nan, np.float32)
    blocks = np.zeros((n_blk, n_syn), np.float32)
    acc = np.zeros(n_syn, np.float64)
    state = {"s": 0}

    @network_operation(dt=1 * second, when="start")
    def sample():
        s = state["s"]
        if s >= n_sec:
            return
        w = np.asarray(syn.w[:], np.float32)
        ring[s % 61] = w
        if s >= 60:
            D[s] = float(np.abs(w - ring[(s - 60) % 61]).sum())
        acc[:] += w
        if s % 10 == 9 and s // 10 < n_blk:
            blocks[s // 10] = acc / 10.0; acc[:] = 0.0
        state["s"] = s + 1

    sp = SpikeMonitor(post)
    t0 = time.time()
    run(total * second)
    st, sidx = np.array(sp.t / second), np.array(sp.i[:])
    rates10 = np.zeros((N_POST, n_blk), np.float32)
    np.add.at(rates10, (sidx, np.minimum((st // 10).astype(int), n_blk - 1)), 1.0)
    rates10 /= 10.0
    img = np.minimum((st // period).astype(int), len(seq_x) - 1)
    on = (st - starts[img]) < S.PRESENT_S
    counts = np.zeros((N_POST, len(seq_x)), np.int16)
    np.add.at(counts, (sidx[on], img[on]), 1)
    flags = np.ones(n_sec, bool)                                       # interface.changing_per_second's rule on D
    for s_ in range(60, n_sec):
        hist = D[max(60, s_ - 900):s_ - 60]; hist = hist[np.isfinite(hist)]
        if len(hist) < 100:
            continue
        med = np.median(hist); flags[s_] = D[s_] > med + 3.0 * 1.4826 * np.median(np.abs(hist - med))

    def wm_block(b0, b1):
        m = np.zeros((N_POST, 196), np.float32); m[sj, si] = blocks[b0:b1].mean(0); return m
    protos = np.array([unit(np.asarray(seq_x[np.isin(seq_y, s_)].mean(0))) for s_ in [(0, 1), (2, 3), (4, 5)]])
    fp = {}
    for W in (10, 50):
        nb = W // 10; qh, qc = [], []
        for b in range(0, n_blk - nb + 1, nb):
            r = rates10[:, b:b + nb].mean(1).astype(float); m = wm_block(b, b + nb).astype(float)
            qh.append(unit(np.clip(r - r.mean(), 0, None) @ m)); qc.append(unit((r - r.mean()) @ m))
        fp[f"W{W}"] = dict(rect=np.round(np.array(qh), 4).tolist(), contrast=np.round(np.array(qc), 4).tolist())
    assign = [[int(np.argmax(protos @ unit(wm_block(b, b + 5)[j].astype(float)))) for j in range(N_POST)] for b in range(0, n_blk, 5)]
    buf = io.BytesIO()
    np.savez_compressed(buf, blocks=blocks, rates10=rates10, D=D, counts=counts, syn_i=si, syn_j=sj, labels=seq_y,
                        spike_t=st.astype(np.float32), spike_i=sidx.astype(np.int16))
    p = Path(RAW_MOUNT) / rel(seed, "raw"); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(buf.getvalue())
    compact = dict(status="completed", seed=seed, n_post=N_POST, n_pre=196, sets=[list(s_) for s_ in SETS], phase_s=phase_s,
                   phase_images=phase_images, total_s=total, tp=TP, tau_theta=TAU_THETA, tau_g=TAU_G, budget=BUDGET,
                   substrate="wta20 + leaky fair-share + causal gain", target=prefs.codegen.target, wall_elapsed=time.time() - t0,
                   fingerprints=fp, change_flags=flags.astype(int).tolist(), D=np.round(np.nan_to_num(D, nan=-1), 3).tolist(),
                   rates10=np.round(rates10, 2).tolist(), assign_50s=assign, protos=np.round(protos, 5).tolist(),
                   labels=seq_y.tolist(), counts=counts.tolist(), raw_path=rel(seed, "raw"))
    q = Path(RAW_MOUNT) / rel(seed, "compact"); q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(MC.gz_bytes(compact))
    RAW_VOLUME.commit()
    return rel(seed, "compact")


@app.local_entrypoint()
def main(collect: bool = False):
    import os
    import subprocess
    if collect:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        got = 0
        for s in SEEDS:
            dst = HERE / f"split_seed{s}.json.gz"
            if not dst.exists():
                subprocess.run([sys.executable, "-m", "modal", "volume", "get", "developing-mind-raw", rel(s, "compact"), str(dst)],
                               capture_output=True, env=env)
            got += dst.exists()
        print(f"== {got}/{len(SEEDS)} collected"); return
    calls = [run_split.spawn(s) for s in SEEDS]
    print("spawned", len(calls), "detached split-MNIST runs", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "prep":
    prep_split()
