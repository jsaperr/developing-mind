# Experiment Plan: does the integration story generalize beyond the one rig?

**Status:** PLANNED, not run. Written 2026-09-28 at Jasper's request, while the long world ran.
Nothing here has been launched.
**Phase:** Brian2 substrate + integration replay. It re-runs existing pipelines on new substrate
settings; no new mechanism.

---

## One-line question

Do the two substrate findings the whole integration layer rests on hold outside the rig they
were found in?
- **One-back retention:** the substrate holds exactly the just-departed context.
- **The re-learning window:** a label-free, detectable ~200 s of weight change after every world
  change.

## Why now

Everything integration-side was found on one skeleton: N=7, 30 inputs, contexts of 10, fixed rates,
p_share 0.9, 1000 s phases. It was replicated at two operating points, but every world shares the
skeleton. The memory-side results are checked with clean-input controls and mostly follow from the
rules' own arithmetic, so they should transfer. The substrate side is a single point of support.
If a bigger network holds three contexts back, or re-learns in 50 s, memory's job and the tuned
numbers change:
- the 60 s / 900 s change-signal windows;
- the 0.8 radius;
- the clock and the horizon.

The plan's job is to find out which results are **logic** (should survive) and which are
**numbers** (may need re-grounding), before building live coupling on top.

## What counts as "generalizes" (fixed now, before any data)

For each new setting, run the same pipeline used so far, and score:

1. **Retention depth k:** holders of the incoming context at 1-, 2-, 3-back returns (the v1b-style
   metric). *Generalizes* = k is small, sharp and consistent across seeds (≥ 7/8 agree). k ≠ 1 is
   not a failure: it changes memory's job size, and gets recorded as the new number.
2. **Change signal:** 60 s weight displacement above its causal trailing threshold. *Generalizes* = it
   fires after ≥ 90% of changes and in < 5% of settled windows, at the same k=3 robust-SD rule. If
   the window is much shorter or longer, re-ground L and trail from the data, and flag it.
3. **Readout:** settled H+ cosine to the true prototype ≥ 0.9; between-context cosine below 0.8
   (the radius) for disjoint contexts.
4. **End to end:** the promoted `GatedEpisodicMemory` (adopted rule) with label-free grounding
   (gap_scale procedure), unchanged. Two-back genuine recognition at W=50 ≥ 7/8 (the v1b result),
   absorption 0.

Any re-tuned number gets logged as "re-grounded at setting X", never silently changed.

## Step 0: free, on existing data (do first; no runs)

- **Change-triggered or slow release?** checkb has 300 s filler phases. Does a core's holder count
  (e.g. B's) drop at the *first* following change, even when that phase is only 300 s? If B is
  released at F2's start, release is triggered by the change. If B is still held through a 300 s
  filler, release is slow, and a short-phase world would hold more than one back. This is an open
  Brian2 item and it decides what step 1 is really testing.
- **Residue accumulation over 20 phases:** the long world (running now) answers it directly (LW-P1).
- **Retainer fraction vs rig:** already known to be rig-dependent (37% in the 20-input rig, 55% in
  the 30-input rig at N=7). Step 2 extends this; nothing new to mine.

## Step 1: phase length (highest priority)

- **Why:** one-back could be "the change releases the older context" or "1000 s happens to be the
  release time". Those predict different things at other phase lengths, and the horizon/rehearsal
  logic depends on which it is.
- **Design:** v1b schedule (A B C A B), same rig, N=7, 13mV/1.5. Phase lengths 300 s and 3000 s
  (1000 s exists).
- **Watch:** at 300 s, is the substrate settled before the change? Substrate settling is ~300 s, so
  read "settled" carefully.
- **Cost:** 300 s is 1500 s total (~5 min per batch); 3000 s is 15,000 s total (~50 min for 8 seeds
  in parallel).

## Step 2: network size

- **Design:** v1b schedule, same rig, N ∈ {5, 10, 15} (7 exists), inhibition via
  `scale_inhib_for_n` (as in arc 03).
- **Question:** does retention depth grow with N (more neurons, room to keep more contexts), and does
  the retainer fraction stay put?
- **Watch:** arc 03 found a non-monotonic hierarchy with its minimum at N=7, so N=7 may be atypical.
- **Cost:** about 16 min per batch at N=7; bigger N is somewhat slower (dispatch-bound). Roughly
  1 hour for all three.

## Step 3: more inputs, more contexts

- **Design:** 60 inputs, 6 disjoint blocks of 10, N=7 and N=15. A schedule with 1-, 2-, 3- and 4-back
  returns.
- **Question:** with more contexts than neurons can each own, does one-back still hold, and does the
  readout still separate contexts?
- **Needs a calibration step first** (target_total and w_init for 60 inputs, as `calibrate_novelc.py`
  did for 30).
- **Cost:** calibration ~10 min plus a batch ~30-60 min.

## Optional, only if 1-3 hold

- **Weaker correlation** (p_share 0.6 vs 0.9): does the readout degrade gracefully?
- **A third operating point.**

## Order, cost, and where to run

Step 0 (free) → 1 → 2 → 3. That's about 3-4 hours of laptop time in total, at 8 concurrent jobs.
Steps 2-3 at larger N are the natural first use of the WWU HTCondor cluster if the laptop gets
tight.

## Conventions (as for every run so far)

- New `run_*_seed.py` / `run_*_batch.py` reusing the frozen v1 runner via module-attribute
  overrides. Old scripts untouched.
- `.json.gz` via `results_io`, seeds in fresh ranges (43000+), manifest rows in
  `notebooks/brian2/README.md`.
- **Predictions go in each script's docstring before launch.** Current leanings (not yet
  predictions):
  - release is change-triggered, so one-back holds at 300 s and 3000 s;
  - k stays 1 at larger N, but with more retainers per context;
  - the change-signal window shortens with shorter phases only if settling does.
- Results go to `docs/log/brian2/` (substrate) and `experiments_integration.md` (end to end).
