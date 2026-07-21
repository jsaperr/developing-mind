# Experiments Log — ESN phase

Echo State Network work — a new, third tool alongside the Hopfield/two-layer-memory work
(`experiments.md`) and the Brian2/SNN work (`experiments_brian2.md`). Split out at the same
natural phase boundary as those two: different technique, different questions, not mixed into
either existing log.

## 2026-07-21 — Stage 1: memory-capacity characterization — grounding the two-layer decay constants

**Notebook:** `notebooks/esn/esn_memory_capacity.ipynb`. Shared reservoir code:
`src/esn/reservoir.py` (`build_reservoir`, `run_reservoir`), `src/esn/memory_capacity.py`
(`memory_capacity`, standard Jaeger-2001-style formulation).

**Question (per Jasper + web's message):** every decay constant in the two-layer Hopfield
memory (`decay_fast=0.02`, `consolidation_rate=0.01`, `decay_char=0.0005`) was picked by feel
and validated empirically after the fact — there's no principled sense of whether 0.02 is a
fast forgetting rate or a slow one, in any real, comparable sense. Build a minimal Echo State
Network, measure a real memory-capacity/spectral-radius tradeoff directly (not guessed), and
check where the existing decay constants land on that curve as a rough sanity-check comparison.

**Design:** fixed random recurrent reservoir (n_units=300, dense), leaky-integrator update
(`leak_rate=0.3`, fixed), spectral radius as the one tunable knob (rescale the recurrent weight
matrix so its largest-magnitude eigenvalue equals the target exactly). No training on reservoir
weights ever — only a linear readout (closed-form ridge regression) gets fit, and only for the
memory-capacity measurement itself. Standard task: feed a random scalar input stream, fit a
linear readout per lag `k` (1 to 30) to reconstruct the input from `k` steps back from the
current reservoir state, memory capacity per lag = held-out R², total = sum across lags. Swept
`spectral_radius` across {0.1, ..., 1.5} (15 points), 5 independent seeds per point (avoid a
single-draw noise artifact looking like a real peak — same discipline as the STDP seed work).

**Falsification criteria, stated before running:** a sane result is a real peaked or monotonic
tradeoff curve that responds to spectral radius — non-flat, non-degenerate (memory doesn't
vanish at every setting, doesn't blow up at every setting either). A flat or unresponsive curve
means something is wrong with the reservoir construction, not that ESNs don't work as a
technique — root-cause before concluding anything about the technique itself.

**Result: real, non-degenerate response — passes the falsification check cleanly.** Total
memory capacity ranges from 3.59 (spectral_radius=0.1) up to a broad plateau of 5.1-5.75 across
spectral_radius=0.4-1.2, peaking at spectral_radius=1.1 (MC=5.75), then dropping off toward
spectral_radius=1.5 (MC=4.23) — a 37% spread across the sweep, smooth and multi-seed-averaged,
not a single noisy point. Shape matches the standard ESN "edge of stability" picture (memory
rises toward spectral_radius≈1 then degrades beyond it), with the actual degradation
onset shifted a bit past 1.0 — expected for a leaky-integrator reservoir, since the leak rate
itself already contributes independent damping/stability beyond the raw recurrent spectral
radius. Per-lag capacity curves (both the 1-30 sweep and an extended 1-150 check at the best
spectral radius) decay smoothly toward zero, not degenerate spikes or flat lines — visually
confirmed via the full curves, not just the summary numbers.

**Effective memory horizon at the best-performing spectral radius (1.1, n_units=300): ~37
steps** (first lag after which per-lag capacity stays below a small threshold, 0.02).

**Comparison against the two-layer decay constants — not a rigorous unit conversion, a
sanity-check order-of-magnitude comparison only** (the two-layer update rules aren't pure
exponential decay — headroom terms, multiplicative modulation — and "one ESN reservoir step"
isn't defined to equal "one two-layer memory step" in any principled way). Expressing each
constant's approximate characteristic timescale as `~1/rate` steps:

| constant | value | approx. timescale | vs. reservoir's ~37-step horizon |
|---|---|---|---|
| `decay_fast` | 0.02 | ~50 steps | beyond |
| `consolidation_rate` | 0.01 | ~100 steps | beyond |
| `decay_char` | 0.0005 | ~2000 steps | far beyond |

**All three sit beyond this reservoir's measured memory horizon — including `decay_fast`, the
fastest of the three.** A genuinely informative result, not a null one: the two-layer memory's
persistence timescales, even its "fast" layer, are systematically longer than what a modestly-
sized (300-unit), generic, untrained reservoir echoes for free at its best-performing operating
point. This is consistent with why the project needed a dedicated fast/slow consolidation
mechanism rather than relying on passive substrate dynamics — at this reservoir scale, passive
echo memory alone doesn't reach where the two-layer memory's own constants operate. Whether a
larger reservoir (more units) would close this gap, or whether it's a structurally different
kind of memory regardless of size, is not tested here.

**Verdict: stage 1 complete, results reviewed, everything is sane.** Not proceeding to stage 2
(the episodic-layer context-signal follow-up) — explicitly deferred pending a separate,
specific go-ahead per the original design message. Reporting and stopping here.
