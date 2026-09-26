# Brian2 log: Experiment B, N-scaling (2026-07-23)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-07-23 — Experiment B step 4: N=7 at full 5000s — extends Test A's finding, not a contradiction. Reliable competition still means fast, permanent settling, not genuine reorganization

**Data:** `notebooks/brian2/n_scaling_data/step4_n7_5000s/` (4 seed JSONs, `run_n_scaling_seed.py`
reused with `n_post=7, duration_s=5000`). Seeds picked to span the range observed at 600s
(top_tier_size and rank-swap range): 22016, 22017, 22019, 22020.

**Question, reframed by web before this ran:** Test A found that at 13mV/1.5's N=3, the
swap-count "richness" metric was measuring noise-trading between two already-decided winners, not
genuine reorganization — the laggard was permanently excluded within ~80s and never returned. Does
N=7 (this thread's step-4 target, picked for its tier-legibility minimum) show real reorganization
(an excluded neuron genuinely re-entering the leading tier later in the run), or does it extend
the same fast-permanent-exclusion pattern to a larger N?

**Explicit instruction from web, applied here:** don't lead with `full_rank_swap_count` or any
swap-derived richness number. Check directly, per seed, whether excluded/lower-tier neurons ever
re-enter the top tier well past the initial settling period, using `compute_tiers` over time.

**A false positive caught and corrected before reporting, not after — worth documenting the
mistake itself, not just the fix:** a first pass (comparing the top tier at t=0-200s against the
top tier after t=1000s) found what looked like real re-entry in 2 of 4 seeds (22017, 22019) —
neurons absent from the very first window appeared in the top tier later. Before reporting that as
a genuine reorganization finding, re-checked with finer time resolution (250s windows across the
full run) and found the "re-entry" was an artifact of the comparison window, not the phenomenon:
in both seeds, the FULL final top-tier composition was already reached by t≈250-500s — those
neurons weren't excluded and later re-admitted, they just took slightly longer than others to join
the initial settling process, still well within it. After t≈250s, top-tier composition in **all
four seeds** — not just these two — never changed again for the remaining 95% of the run (4750 of
5000s). Exactly the kind of thing a two-point comparison misses and a full trajectory catches; the
project's own "distrust a single check" principle catching itself in the act again.

**Result: extends Test A's finding cleanly, does not contradict it.** All 4 seeds settle to a
final, permanent top-tier composition by t≈250s and hold it unchanged for the rest of the 5000s
run — zero genuine reorganization in any seed tested. What varies across seeds is hierarchy
*shape*, not temporal dynamics: 22016 and 22020 settle to a single clear leader (top tier size 1),
22017 and 22019 settle to 5-of-7 neurons tied at the top (only 2 genuinely excluded) — consistent
with, and a direct confirmation of, the real shape-variance already measured in the 600s N-scaling
curve (N=7's std_top_tier_frac=0.23, real but the lowest of any N tested). The settling itself is
just faster in absolute terms here than Test A's N=3 (t≈250s vs t≈3-81s) relative to nothing —
both are fast and permanent relative to the full 5000s duration; N=7 is not meaningfully slower or
more prone to reorganizing than N=3, contrary to what a hopeful reading of the swap-count numbers
might have suggested.

**Verdict, matching web's explicitly stated framing:** reliable competition at 13mV/1.5 achieves
its reliability through fast, permanent structural settling, full stop, at every N tested so far
(3 and 7). The genuine identity-persists-through-churn reorganization found at `strong_tight_gate`
appears to be a property of that more marginal, near-bifurcation operating point specifically, not
something that survives at settings tuned for high reliability. This doesn't threaten any
MNIST-adjacent use of the mechanism (permanent, reliable specialization is exactly what that needs)
— it specifically narrows where the philosophically-central "identity through churn" result is
known to hold: the original bistable zone, not yet shown to generalize to reliable operating
points. Whether a genuinely reliable-and-reorganizing setting exists anywhere in the untested space
between the coin-flip zone and 13mV/1.5 remains open, not closed by this result.

---

## 2026-07-23 — Experiment B step 3 follow-up: is N=10's shape reversal real or n=8 noise? Partially real — it's a gradual creep from a real minimum at N=7, not a sharp cliff

**Data:** `notebooks/brian2/n_scaling_data/run_n_scaling_followup.py`, 24 new seed JSONs (N=8, N=9
new; N=10 gets 8 more, bringing it to 16 total). Per web's Gate 2 response: the original curve's
N=10 point (n=8) showed the highest tier-shape variability of any N tested — before treating that
as real rather than the same kind of n=8 sampling noise that made `strong_tight_gate` read 1/8
against an established ~50% rate, get more seeds at N=10 and fill in N=8/N=9 to see whether the
reversal from N=7 is a sharp cliff right at 10 or a gradual creep starting earlier. Explicitly
not a new gate-worthy design change — same 600s scale, same batch pattern, no new metrics.

**Result: the original N=10 reading was somewhat inflated by n=8, but the underlying signal is
real — it's just a gradual creep, not a sharp cliff, with a genuine minimum at N=7, not a
monotonic decline:**

| N | n (differentiating) | mean top-tier frac | std top-tier frac | frac. majority-tied |
|---|---|---|---|---|
| 3 | 7 | 0.57 | 0.15 | 0.71 |
| 5 | 7 | 0.34 | 0.18 | 0.29 |
| 7 | 8 | **0.27** | 0.23 | **0.12** |
| 8 | 8 | 0.39 | 0.29 | 0.25 |
| 9 | 7 | 0.41 | 0.22 | 0.29 |
| 10 | 15 (n=16 total, one now disorder) | 0.42 | 0.29 | 0.33 |

With the doubled sample, N=10's mean top-tier fraction drops from the original 0.51 to 0.42, and
its majority-tied fraction drops from 50% to 33% — the most alarming part of the original reading
*does* wash out with more data, the same direction as `medium`'s and `strong_tight_gate`'s prior
small-n corrections. But it doesn't wash out to nothing: N=10's mean (0.42) and std (0.29) both
stay clearly above N=7's (0.27, 0.23), and — this is the actual new information — N=8 and N=9 sit
at essentially the same elevated level (0.39, 0.41) as N=10, not somewhere between N=7 and N=10.
That rules out "sharp cliff exactly at N=10" cleanly: the curve is `0.57 → 0.34 → 0.27 → 0.39 →
0.41 → 0.42`, a real valley bottoming out at N=7 with a gradual, not sudden, rise on the other
side. One seed in the expanded N=10 batch also landed as genuine `disorder` for the first time
anywhere in this curve (a single occurrence, not treated as a trend on its own).

**Verdict, direct answer to web's question:** neither "real cliff" nor "pure n=8 noise" — the
honest picture is a genuine local minimum in tier-shape variability/legibility at N=7, with a
real (smaller than first estimated) gradual widening on both sides of it. N=7 remains the
strongest legibility candidate in the tested range, now on a firmer footing (n=8, consistent
with N=5's neighboring point) rather than looking like an isolated best-of-four score. Reporting
this back to web before deciding step 4's target, per its own instruction not to self-approve
that choice.

---

## 2026-07-23 — Experiment B step 3: population-competition N-scaling curve — reliability holds through N=10, hierarchy shape doesn't move monotonically

**Data:** `notebooks/brian2/n_scaling_data/run_n_scaling_seed.py` (parametrized twin of
`run_competitive_seed.py`, applies `scale_inhib_for_n`), `run_n_scaling_sweep.py` (orchestrator),
`analyze_n_scaling.py`, 32 per-seed result JSONs, `n_scaling_analysis.json`. Per web's Experiment
B design, cleared at Gate 1 (see prior entries for the normalization design and the
`classify_hierarchy`/`compute_tiers`/`full_rank_swap_count` metrics, validated against all 128
N=3 sweep results before use — 127/128 agreement, one real, inspected disagreement).

**Question:** does the competitive-population mechanism (reliable differentiation, preserved
richness, legible hierarchy shape) hold up as `n_post` grows, and where — if anywhere — does it
start breaking down? Explicitly not a search for "the best N" — output is a curve describing
current limits, not a recommended size.

**Design:** `n_post` ∈ {3, 5, 7, 10}, same winning combo (13mV-reference/gap_scale=1.5) with
per-connection `inhib_strength` normalized via `scale_inhib_for_n` so total per-neuron inhibitory
drive stays comparable across `n_post`, 8 seeds per point (32 runs), 600s calibration scale, same
batch approach as the original 4×4 grid.

**Falsification criteria, stated before running (three-way, per web's Gate 1 instruction):** (a)
HOLDS CLEANLY — ≥80% differentiate, richness present, at most 1 disorder seed; (b) EARLY STRAIN —
differentiation 50-80%, richness thinning, or 2-3 disorder seeds; (c) BREAKDOWN — disorder becomes
the dominant differentiating-seed outcome, or differentiation collapses below 50%.

**Result on reliability/disorder — clean, no breakdown found in the tested range:**

| N | differentiate | converge | disorder | reliability | mean rank-swaps | verdict |
|---|---|---|---|---|---|---|
| 3 | 7/8 | 1 | 0 | 0.88 | 71.4 | holds cleanly |
| 5 | 7/8 | 1 | 0 | 0.88 | 225.1 | holds cleanly |
| 7 | 8/8 | 0 | 0 | 1.00 | 410.6 | holds cleanly |
| 10 | 8/8 | 0 | 0 | 1.00 | 531.5 | holds cleanly |

Disorder never appeared at any tested N — the leadership-consistency check (`classify_hierarchy`'s
top-tier-set stability across sub-windows) stayed clean throughout. `full_rank_swap_count` rises
with N, but this is expected and not itself evidence of richness improving — more neurons means
combinatorially more possible pairwise rank changes even from pure noise (directly consistent with
the caveat from the same day's Test A entry: swap-count metrics conflate genuine reorganization
with noise-level trading among near-tied non-winners, and that conflation gets worse, not better,
as N grows).

**A gap in my own automated verdict, caught before reporting, not after:** the table above says
"holds cleanly" at every single N — technically true on the reliability/disorder axis, but that
verdict never looked at hierarchy *shape*, and manually inspecting the raw tier data (`top_tier_size`
from `compute_tiers`) found something the verdict logic completely missed:

| N | mean top-tier fraction | std top-tier fraction | frac. seeds majority-tied |
|---|---|---|---|
| 3 | 0.57 | 0.15 | 0.71 |
| 5 | 0.34 | 0.18 | 0.29 |
| 7 | 0.27 | 0.23 | 0.12 |
| 10 | 0.51 | **0.33** | 0.50 |

Hierarchy shape is **not monotonic in N**: legibility (a small, clear top tier rather than most
neurons tied together) *improves* from N=3 to N=7 — top-tier fraction drops from 0.57 to 0.27, and
the fraction of seeds where a majority of neurons tie for the top spot drops from 71% to 12% — then
**reverses sharply at N=10**, jumping back to N=3-like tier fractions with the highest variability
of any N tested (std=0.33, roughly double every other point). Concretely: at N=10, half the
seeds keep a small, legible top tier (1-2 neurons), the other half collapse toward 8-9 of the 10
neurons tying together with only 1-2 genuinely excluded — a real bimodal split in what
"differentiate" even means at that seed, invisible to both the reliability metric (still counts as
differentiate either way) and the disorder detector (a huge tied group is still a *stable* top
tier, just not a legible one).

**Candidate explanation, explicitly flagged as a hypothesis, not settled** (per web's Gate 1
instruction to consider this): `scale_inhib_for_n` normalizes *mean* total inhibitory drive per
neuron, not its variance — more, smaller kicks smooth out relative to fewer, larger ones at
matched mean. This predicts *monotonically increasing* smoothing with N, which does not by itself
explain why N=7 looks *more* legible than N=3 and N=5 before N=10 reverses course — a purely
monotonic smoothing story doesn't fit a curve that improves then reverses. The mean-vs-variance
gap is a plausible contributing factor at N=10 specifically, not a full explanation for the shape
of the whole curve. Real, unresolved complexity here, not glossed over.

**Scope, stated precisely:** 8 seeds/point is the same scale as the original grid, and the
original grid's own experience (the `strong_tight_gate` anchor landing at 1/8 in a fresh sample
against an established ~50% rate) is a direct reminder that single-point statistics at n=8 carry
real sampling noise — the N=10 shape-variability finding is a real, reportable signal worth
surfacing, not yet a robust distributional claim.

**Verdict:** reliability and freedom from genuine disorder hold cleanly through N=10, the full
tested range — no breakdown on that axis. Hierarchy-shape legibility is a separate, non-monotonic
story: improves through N=7, then shows real strain (elevated variability, a bimodal legible/tied
split) at N=10. Reporting the full curve, both axes, to web at Gate 2 rather than reporting only
the reliability table's clean "holds cleanly" read, which would have hidden the more interesting
finding.

---

