# Experiment Plan: does the integration story generalize beyond the one rig, and onto real data?

**Status:** IN PROGRESS. Steps 0 and 1 are done (2026-09-28/29). Steps 2, 3 and 4 (MNIST) are
planned, not run. Written 2026-09-28; MNIST folded in as step 4 on 2026-09-29.
**Phase:** Brian2 substrate + integration replay. Steps 0-3 re-run existing pipelines on new
synthetic settings. Step 4 is the first real data (MNIST), and needs a few new input and substrate
pieces (listed there).

---

## One-line question

Do the two substrate findings the whole integration layer rests on hold outside the rig they
were found in, and does the substrate + memory system do something meaningful on real data?
- **One-back retention:** the substrate holds exactly the just-departed context.
- **The re-learning window:** a label-free, detectable ~200 s of weight change after every world
  change.

## Why this order

Each step changes one thing, so when something breaks we know what broke it. MNIST changes
everything at once (rate coding, natural overlap, ten classes, a hundred times the connections), so
it comes last, after the synthetic steps have told us which variable does what.

| step | what changes | isolates |
|---|---|---|
| 0 | nothing (existing data) | change-triggered vs slow release |
| 1 | phase length | timescale |
| 2 | network size N | capacity |
| 3 | number of contexts and inputs | more contexts than neurons can each own |
| 4 | real data (MNIST) | rate coding, natural overlap, 10 classes, scale |

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
**numbers** (may need re-grounding), before building live coupling on top. MNIST then asks the
question the framework (v5, XII) sets for the substrate: does structure precipitate from real
experience through local plasticity alone?

## What counts as "generalizes" (fixed now, before any data)

For each new setting, run the same pipeline used so far, and score:

1. **Retention depth k:** holders of the incoming context at 1-, 2-, 3-back returns (the v1b-style
   metric). *Generalizes* = k is small, sharp and consistent across seeds (≥ 7/8 agree). k ≠ 1 is
   not a failure: it changes memory's job size, and gets recorded as the new number. (Step 1 showed
   overlap gives less than one back; that's also a valid k.)
2. **Change signal:** 60 s weight displacement above its causal trailing threshold. *Generalizes* = it
   fires after ≥ 90% of changes and in < 5% of settled windows, at the same k=3 robust-SD rule. If
   the window is much shorter or longer, re-ground L and trail from the data, and flag it.
3. **Readout:** settled H+ cosine to the true prototype ≥ 0.9; between-context cosine below 0.8
   (the radius) for disjoint contexts.
4. **End to end:** the default memory (`DormantGatedMemory`, radius 0.8), unchanged, with the
   previous `GatedEpisodicMemory` run alongside. Label-free grounding (gap_scale procedure).
   Two-back genuine recognition at W=50 ≥ 7/8 (the v1b result), absorption 0.

Any re-tuned number gets logged as "re-grounded at setting X", never silently changed.

**Tripwire (2026-09-28, `framework_drift.md` item 10):** if any gate number (radius 0.8, the 60 s / 900 s
change window, stability tau 0.9, k=3) has to be re-tuned by hand for a new setting, flag it. That's evidence
we're routing around the thalamus. The fix is to make that number self-set from the system's own statistics.
Step 4 is the most likely place for it to fire; expect it and treat it as information.

## Step 0: free, on existing data

**DONE 2026-09-28** (`docs/log/brian2/05_nonstationary_correlation.md`). Re-assignment is
change-triggered (never between changes). No residue build-up and no readout drift over the 20-phase
long world. "Exactly one back" turned out to be schedule-dependent, which step 1 then traced to
overlap.

## Step 1: phase length

**300 s arms DONE 2026-09-28** (arc 05 log):
- disjoint contexts at 300 s give one-back, just like 1000 s;
- at 50% overlap, a new context captures most of the population and the just-departed keeps ~37%
  (disjoint: 86-96%), the same at 300 and 1000 s.
- So overlap, not phase length, sets the split.

**3000 s arm DONE 2026-09-29:** identical to 300 and 1000 s (B keeps 93% at C's arrival, A released to
0.5 by +300 s, two-back incoming 0.12-0.38, flat between changes). **Step 1 is complete:** phase length
(300-3000 s) doesn't change the substrate; overlap does.

Original 3000 s text: the 3000 s arm (disjoint, v1b schedule, N=7, 13mV/1.5). It checks the other
direction: does a long phase harden the retainers so much that the next change can't release them?
- **Cost:** 15,000 s total, ~50 min for 8 seeds in parallel.

## Step 2: network size

- **Top size, open (2026-09-29):** 30 was a loose judgment call (Diehl & Cook's ~10 neurons per class, scaled
  down loosely). 40 is proposed so the top size equals the MNIST pilot at their ratio (10 per class x 4
  classes). Jasper to pick. Seed Brian2 (`brian2.seed(seed)`) in these runs so they're reproducible.
- **Design:** v1b schedule, same 30-input rig, N ∈ {5, 10, 15, 30} (7 exists), inhibition via
  `scale_inhib_for_n` (as in arc 03). N=30 is added as the bridge to step 4, whose pilot needs
  20-30 neurons.
- **Question:** does retention depth grow with N (more neurons, room to keep more contexts), and does
  the retainer fraction stay put?
- **Watch:** arc 03 found a non-monotonic hierarchy with its minimum at N=7, so N=7 may be atypical.
- **Cost:** about 16 min per batch at N=7; bigger N is somewhat slower (dispatch-bound). Roughly
  1-1.5 hours for all four.

## Step 3: more contexts, more inputs (the synthetic control for step 4)

- **Design:** 60 inputs, 6 disjoint blocks of 10, N=7 and N=30. A schedule with 1-, 2-, 3- and 4-back
  returns.
- **Question:** with more contexts than neurons can each own, does one-back still hold, and does the
  readout still separate contexts?
- **Why keep it now that MNIST is in the plan:** it separates "more contexts" from "real data". If
  step 4 breaks and step 3 doesn't, the cause is rate coding or natural overlap, not the context
  count.
- **Needs a calibration step first** (target_total and w_init for 60 inputs, as `calibrate_novelc.py`
  did for 30).
- **Cost:** calibration ~10 min plus a batch ~30-60 min.

## Step 4: MNIST, the first real data

**Why MNIST:**
- The framework names it as the calibration step (v5, XII): unsupervised STDP, no labels, no loss.
- There's a published baseline with the same ingredients as our substrate (Diehl & Cook 2015: STDP,
  weight normalization, lateral inhibition): neurons' weights come to look like digits, with 82-95%
  accuracy depending on size.
- Digits share many pixels, so it's natural overlap. That's exactly where our substrate holds less
  than one back and memory had its weakest case (70% overlap).
- The split schedule is our two-back test on real data, and it's close to split-MNIST, the standard
  continual-learning benchmark.

**What's new (hand-built rules, not hand-set numbers):**
- **Rate-coded input:** each pixel is a wire whose Poisson click rate follows its brightness
  (max ~64 Hz, as in Diehl & Cook). Each image is shown for ~350 ms, then ~150 ms of rest. The
  information is in rates, not synchrony. H+ uses rates and weights, so it should carry over.
- **Adaptive threshold:** each neuron's firing threshold rises a little every time it fires and
  slowly relaxes. Diehl & Cook needed it so a few neurons don't win every image. It's a homeostatic
  rule, in the same family as our weight budget. Add it only if the pilot shows a few neurons
  monopolizing, and log it.
- **What a "context" is:** a set of digit classes shown together. The memory still only sees
  fingerprints; labels are used only for scoring.

**4a. Pilot (an afternoon; laptop).**
- **Setup:** 14×14 images (196 inputs), 4 digit classes, N=20-30, a few hundred images per class.
- **Questions:**
  - Do neurons specialize by digit without labels (scored afterwards by the class each neuron
    responds to most)?
  - Do their weights look like digits?
  - Does the H+ fingerprint separate the classes (within-class vs between-class cosine)?

**4b. Split-MNIST two-back (the headline experiment).**
- **Schedule:** {0,1} → {2,3} → {4,5} → {0,1} → {2,3}, 14×14, N≈30.
- **Substrate:** how much of {0,1} is still held at its two-back return (expected: little, given
  natural overlap).
- **Memory:** replayed into the default memory and the previous one. Is {0,1} recognized at its
  return, with NOVEL and TRANSITIONAL doing their jobs?
- **Scored against the synthetic results.**

**4c. Full MNIST (cluster).**
- **Setup:** 28×28 (784 inputs), N=100-400, the full training set.
- **Scoring:** the standard label-assignment classification score, to compare with the literature.
  This is a calibration of the substrate, not a goal in itself.

**Pass criteria for step 4 (fixed now, before any MNIST run; refined in each script's docstring
before launch):**
- Pilot: ≥ 60% of neurons respond most to one class (by a clear margin); within-class fingerprint
  cosine above between-class cosine for every class pair.
- Split-MNIST: the returning {0,1} is named correctly in more than half of its settled checks in
  ≥ 6/8 seeds (default memory, W=50), with absorption 0.
- Full MNIST: a classification score in the published range for its size (a sanity check, not a
  pass/fail on the architecture).

**Mood-map checks (added 2026-09-29, before any MNIST data; see `experiments_integration.md`,
the mood map).** The three label-free signals (re-learning flag, memory match, fingerprint strength)
carried to real data, at two nested timescales:
- **image level:** one memory check per image, from the ~350 ms presentation;
- **context level:** 10-50 s windows, as now.

Every threshold is self-set from the system's own statistics (the tripwire rule). Any that needs hand
re-tuning gets flagged.
- **MC-1, calibration (4a):**
  - Image-level strength is the unnormalized H+ over the image's presentation.
  - Bin test images into strength quintiles (quintiles of the run's own strength distribution) and
    score each bin with the standard label-assignment accuracy (labels used only for scoring).
  - *Pass:* accuracy rises across the quintiles, and the weakest quintile is at least 15 points below
    the strongest.
  - Expected failure mode, to report not hide: confidently wrong digits (a 4 written like a 9).
    Strength can't catch those, as in the toy worlds.
- **MC-2, novelty (4a):**
  - Train on 4 classes, then present 2 held-out classes (and the 4 known ones, interleaved).
  - An image is "strange" when its best memory match is below the self-set radius (the 5th
    percentile of known-class best-match similarity during training).
  - *Pass:* held-out images are strange at least 2x as often as known-class images.
- **MC-3, lost / returning on split-MNIST (4b):** context-level states around each class-set switch.
  *Pass,* in ≥ 6/8 seeds:
  - at switches to a never-seen set, at least one "strange" check before the first "home";
  - at the return to {0,1}, no "strange" before "home" (it goes lost → returning → home).
- **Also recorded:** which thresholds (radius, change window, strength cutoff) had to be re-grounded
  for MNIST, and whether the self-set versions coped.

**Practicalities:**
- **Data:** MNIST isn't on this machine. It would be downloaded through torchvision (installed) from
  the standard mirror, ~11 MB. **Ask Jasper before downloading.**
- **Backend:** MNIST is a new thread with nothing earlier to match, so `cpp_standalone` (~90x
  faster) is allowed there. Run one small pilot seed in both backends first and log the comparison
  (arc 06 found they diverge at `strong_tight_gate`).
- **Compute:**
  - 4a and 4b run on the laptop, at 196 inputs × 30 neurons ≈ 5,900 connections.
  - 4c goes to the WWU HTCondor cluster: 784 × 100+ ≈ 78,000+ connections, and full MNIST is
    ~30,000 simulated seconds per pass.

## Optional, only if 1-3 hold

- **Weaker correlation** (p_share 0.6 vs 0.9): does the readout degrade gracefully?
- **A third operating point.**

## Order, cost, and where to run

Step 1 (3000 s) → 2 → 3 → 4a → 4b → 4c.
- Steps 1-3: about 3 hours of laptop time at 8 concurrent jobs.
- 4a-4b: an afternoon each, including building the input encoder and (if needed) the adaptive
  threshold.
- 4c: the natural first use of the cluster.

**Shortcut if MNIST is the priority:** 4a only needs step 2's N=30 arm to show the substrate behaves
at that size, so the minimum path is step 2 (N=30) → 4a → 4b. Steps 1 (3000 s) and 3 can run
alongside or after.

## Conventions (as for every run so far)

- New `run_*_seed.py` / `run_*_batch.py` reusing the frozen v1 runner via module-attribute
  overrides, where possible. MNIST needs its own runner (new input builder and, if needed, the
  adaptive threshold); it goes in its own `notebooks/brian2/mnist_*` directory and `src` stays
  untouched until something is promoted with tests.
- `.json.gz` via `results_io`, seeds in fresh ranges (45000+ for steps 1-3, 50000+ for MNIST),
  manifest rows in `notebooks/brian2/README.md`.
- **Predictions go in each script's docstring before launch.** Current leanings (not yet
  predictions):
  - one-back holds at 3000 s;
  - k stays 1 at larger N, but with more retainers per context;
  - on MNIST, neurons specialize and weights look like digits, but the substrate holds little of
    old classes (natural overlap), so memory does most of the carrying, and the 0.8 radius
    probably needs re-grounding (tripwire).
- Results go to `docs/log/brian2/` (substrate) and `experiments_integration.md` (end to end). MNIST
  likely gets its own arc file (`08_mnist.md`).
