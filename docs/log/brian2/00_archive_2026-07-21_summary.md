# Brian2 log: Archived: consolidated STDP/SNN summary as of 2026-07-21 (superseded)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-07-21 — STDP/SNN arc: consolidated summary

Closing summary for the Brian2/STDP thread — every entry below is kept in full as the detailed
record; this is a synthesis on top, not a replacement. Fifteen-plus entries deep now (runaway →
two rejected hypotheses → amplitude-vs-frequency reframe → long-run stability check →
population extension → shared-input competition → bistability → n=7 seed expansion →
positional-bias resolution) — read this for the throughline, follow the links below for
mechanism and data.

**What's validated:**

- **The core claim holds, robustly, at every scale tested.** STDP alone, zero supervision,
  learns to differentiate correlated from uncorrelated input from spike-timing statistics —
  confirmed at a single postsynaptic neuron, at an independent-replicate population (N=5), and
  at a genuine shared-input competitive population (N=3, lateral inhibition). Not a
  single-neuron artifact, not an artifact of any one network configuration (see the "First STDP
  experiment", "Population extension", and "Shared-input population competition" entries).
- **The runaway problem and its fix.** Unconstrained STDP produces positive-feedback lock-in —
  higher weight → more reliable postsynaptic spikes → more potentiation, both groups saturate
  together before any real signal develops. Homeostatic synaptic scaling (a `(summed)` Brian2
  variable rescaling each postsynaptic neuron's own incoming weights toward a fixed target)
  contains this cleanly without eliminating the differentiation signal — confirmed via total
  weight sum and postsynaptic firing rate, not just individual-synapse saturation (see
  "Homeostatic synaptic scaling" entry).
- **The stability finding, precisely stated.** Individual synapses never truly converge —
  reversal frequency is essentially constant (~140-500/synapse, scaled by duration) regardless
  of learning rate, at every `Apre` tested. `Apre` controls excursion *amplitude*, not switching
  *frequency* — this is the STANDING EXPLANATION, don't re-derive it (see "Is the Apre
  instability a threshold or a gradient" entry). Despite that, group-level differentiation stays
  genuinely bounded away from collapse over timescales far exceeding naive diffusion estimates,
  in the large majority of realizations tested — genuine-but-probabilistic stability, not
  absolute (see the 8-seed/5000s ensemble entry).
- **The multi-neuron result — the strongest one, and the actual payoff of this arc.** A stable
  population-level signal ("someone represents the correlated pattern") holds even when
  individual identity churns underneath it, across a real n=7 sample in a genuine shared-input
  competitive population: ranging from near-permanent individual-level lock-in, through
  one-time reorganization (two distinct sub-mechanisms), to seeds that never lock into a
  hierarchy at all yet still produce a rock-solid population signal — the strongest instance of
  this property found. This directly validates the population-readout stability definition in
  `principles.md`, tested (not just extrapolated) at the multi-neuron scale that decision was
  originally missing (see "Seed expansion (n=2 -> n=7)" entry).
- **The bimodal winner-take-most structure.** "Group-mean gap stays bounded" was always
  accurate but coarser than reality — final weights are bimodal (synapses cluster near the clip
  boundaries, not the middle), and roughly a third of the "winning" correlated group's own
  synapses land at the floor, behaving like losers. Confirmed independently across two
  structurally different datasets (see the bimodal-distribution entry).

**What's still open:**

- **The bimodal structure's timing dynamics are unknown.** None of the early ensemble/
  population runs saved full per-synapse time traces (only aggregates, finals, and reversal
  counts) — so *when* a synapse commits to its final side, and whether "defectors" flip back
  and forth or settle early, isn't answerable without a rerun. Future long runs should save at
  least a subsampled per-synapse trace (see the design note in `src/brian2_stdp/network.py`).
- **Whether the ~50/50 bistability rate at `strong_tight_gate` holds at other operating
  points is untested.** Only one inhibition-strength/gap_scale combination has been
  characterized in any depth; whether other combinations show the same bifurcation structure,
  a different split ratio, or no bistability at all is a real, unstarted follow-up.
- **The n=7 seed-expansion result is a typology, not a distribution** — the relative frequency
  of each individual-identity regime (near-permanent lock-in / one-time reorganization / never
  locks in) is observed at this sample size, not established as a stable rate. Stated explicitly
  in `principles.md` rather than implied.

---

