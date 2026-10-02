# Brian2 log: MNIST, the first real data (generalization plan step 4)

Index: `experiments_brian2.md`. Plan: `experiment_plan_generalization.md`, step 4. Newest entries first.

## 2026-10-02 — Dose and leak tests: ONE substrate now passes both worlds. A LEAKY fair-share threshold (proportional, not integral, homeostasis) with a ~50-100 s memory keeps every digit (65-75%) and the click world's memory (8/8); the leak is a single dial trading 'spread the work' against 'hold a sustained context'

**Why:** the factorial (below) found the fair-share threshold habituates sustained contexts away.
**Data:** all on Modal, paired seeds (click 53000-53007 / 53100-53107; MNIST 70100-70107):
- `notebooks/brian2/unified_substrate/dose_tp*_n*_seed*.json.gz` and `leak_tau*_n*_seed*.json.gz` (compact; raw on the Volume);
- `notebooks/brian2/mnist_pilot/mnist_dose_tp*_e10_seed*.json.gz`, `mnist_leak_tau*_seed*.json.gz`.

**Scripts** (each pre-registered and committed before launch):
- `modal_dose.py` (D-P1..P5);
- `modal_leak.py` (L-P1..P4; the follow-up L-P5, L-P6 was added after tau 10/100/1000 and before tau 30/50);
- `analyze_dose.py`, `analyze_leak.py`, `analyze_leak_flicker.py`.

**Outputs:** `dose_output.txt`, `leak_output.txt`, `leak_flicker_output.txt`, `mnist_dose_*_output.txt`, `mnist_leak_*_output.txt`.

**Infrastructure notes:**
- The dose test's blocking driver was killed by the local background-task limit with 21 MNIST results uncollected.
- Those were relaunched detached, writing to the Volume (`--mnist-detached`; identical jobs).
- Lesson in CLAUDE.md.

**Dose (slower integral fair-share; MNIST given 10 passes):**

| | click N=7 | click N=40 | MNIST (10 passes) |
|---|---|---|---|
| speed x1 (v4) | -0.18, 0/8 | 0.06, 0/8 | **82.8%** every digit (1 pass: 79.2%) |
| x10 slower | 0.89, 8/8 (readout fades 0.86 -> 0.20 by 900 s) | 0.76, 7/8 | 75.6%, every digit |
| x100 slower | 0.97, 8/8: PASS | 0.83, 6/8 (fades 0.84 -> 0.64) | 53.3%, no diversity |

- **Predictions:**
  - **D-P1 (x10 passes the click world): REFUTED.**
  - **D-P2 (x100 passes): REFUTED at N=40.**
  - **D-P3 (MNIST x10 >= 70%): CONFIRMED** (75.6%).
  - **D-P4 (x100 < 65%): CONFIRMED** (53.3%).
  - **D-P5 (x1 with 10 passes >= 75%): CONFIRMED** (82.8%, the best MNIST result yet).
- **Reading:** slowing only DELAYS habituation. Integral fair-share (no decay) drives every sustained rate difference to
  exactly zero, eventually.

**Leak (fair-share with decay tau, Diehl & Cook's 0.05 mV step; MNIST 1 pass, v4's conditions):**

| tau | click flicker N=7 / N=40 | click memory (B two-back) | MNIST accuracy | every digit >= 3 neurons | novelty |
|---|---|---|---|---|---|
| 10 s | 0% / 1.3% | 8/8, 8/8 | 45.0% | 0/8 (one digit takes ~35) | 1.8x |
| 30 s | 1.9% / 2.4% | 8/8, 8/8 | 54.3% | 4/8 | 3.0x |
| 50 s | 5.3% / 6.5% | 8/8, 8/8 | 64.3% | **8/8** | 3.1x |
| 100 s | 21% / 14% | 8/8, 8/8 | **74.8%** | **8/8** | 3.2x |
| 1000 s | 60% / 60% (readout ~0) | 1/8, 1/8 | 79.8% | 8/8 | 2.6x |
| no decay (v4) | (readout ~0) | 0/8, 0/8 | 79.2% | 8/8 | 2.5x |

**Flicker** = the share of settled 10 s windows whose rectified readout has cosine < 0.5 to the current context. The
pre-registered settled MEDIAN stays 0.97-1.00 for tau 10-100 s.

- **Predictions:**
  - **L-P1 (tau 10 s passes the click world, no habituation): CONFIRMED** (0.98 / 0.98; 900 s into a phase 0.98 / 0.95).
  - **L-P2 (tau 1000 s fails): CONFIRMED.**
  - **L-P3 (tau 10 s keeps MNIST diversity): REFUTED.** A 10 s fairness memory (~20 images) can't spread digits.
  - **L-P4 (tau 1000 s ~ v4): CONFIRMED** (79.8%).
  - **L-P5 (tau 30 s passes both): REFUTED on MNIST** (4/8, 54%); the click half held (< 5% flicker).
  - **L-P6 (tau 50 s passes both, flicker in between): MOSTLY.** Flicker 5-6.5% (in between), click 8/8; MNIST covers
    every digit in 8/8 but reaches 64.3% against L-P3's 65% bar.
- **Flicker is post-hoc, labelled.**
  - At tau 100 s, settled medians stay high (0.97-0.98), but 14-21% of 10 s windows briefly point AWAY from the current
    context (~8-12 dips per 700 s).
  - Adaptation makes the winners sit out and others take over for a moment: the alternation ingredient the ambiguity
    experiment found missing (arc 05, "no rivalry without adaptation").
  - Memory still recognizes 8/8; its steadiness gate and the 0.8 radius ride through the dips.

**Reading:**
- **The reconciliation exists, and it's a timescale.** Fair-share's job (spreading the work across inputs) needs a memory
  of who has been winning lasting a few hundred inputs (MNIST: ~50-100 s, 100-200 images). Integration needs that
  memory to be short relative to a sustained context (click world: ~100 s or less against 1000 s phases).
- **The leak is one dial between those.** Short = clean sustained representation, no diversity. Long = diversity,
  then flicker, then habituation (integral = guaranteed erasure).
- **At tau 50-100 s, one substrate passes both worlds.** The substrate there is: winner-take-all (20 mV, gate off),
  gain control, and the fair-share threshold with decay.
- **Biologically familiar:** a leaky, mean-preserving threshold is spike-frequency adaptation with fair sharing. The
  trade-off is the familiar one between categorization (spread representations) and sustained attention (fatigue).
- **Hand-chosen doses (tripwire spirit).** tau is picked from a sweep. It isn't adopted. The principled next step is to
  set the leak from the system's own statistics, e.g. relative to how fast its input changes.
- **Not yet done:**
  - the life-length world on this substrate;
  - temporal (not per-image) gain control;
  - whether the flicker matters for anything downstream.

## 2026-10-02 — Substrate factorial: the FAIR-SHARE THRESHOLD alone breaks the click world (winner-take-all alone is harmless), and the mechanism is habituation: rate homeostasis erases a sustained context within ~2-5 minutes. MNIST still needs both rules

**Why:** the unified-substrate check changed two rules at once. This tests each alone, on both worlds.
**Data:** `notebooks/brian2/unified_substrate/fact_{ctrl,fair,wta}_n{7,40}_seed*.json.gz` (48 click runs, paired seeds with
the unified check) and `notebooks/brian2/mnist_pilot/mnist_fact_{norm_fair,wta_norm}_seed*.json.gz` (16, paired with v1-v4),
all on Modal.
- **Scripts:** `modal_factorial.py` (predictions F-P1..P5, before launch), `analyze_factorial.py` (before results), outputs
  `factorial_output.txt`, `mnist_fact_*_output.txt`.
- **ctrl cell:** the old substrate through the notebook builder, verified bit-identical to `src`'s builder before launch.

| click world (pass = readout >= 0.9 AND B named at two-back >= 7/8) | N=7 | N=40 |
|---|---|---|
| ctrl (gated, no threshold = old substrate) | 0.974, 8/8: PASS | 0.985, 8/8: PASS |
| **fair-share alone** (gated) | **0.462, 0/8: fail** | **0.108, 0/8: fail** |
| winner-take-all alone | 0.969, 8/8: PASS | 0.902, 8/8: PASS |
| both (unified check, 2026-10-01) | -0.180, 0/8 | 0.055, 0/8 |

| MNIST (gain control on) | vote accuracy | specialized | classes with >= 3 neurons |
|---|---|---|---|
| neither (v2 norm) | 46.7% | 0% | ~1 |
| fair-share alone | 47.1% | 0% | ~1-2 (class 1 takes 31-39 of 40) |
| winner-take-all alone | 28.7% | 2% | 1 (class 0 takes 37-38) |
| both (v4) | 79.2% | 65% | 4, every seed |

- **F-P1 (ctrl passes): CONFIRMED.**
- **F-P2 (fair-share alone passes the click world): REFUTED.** It is the culprit.
- **F-P3 (winner-take-all alone passes): CONFIRMED.** At N=40 only ~40% of neurons stay active, and the change signal
  fires 50-69% (fewer neurons re-wire per change); the readout and memory still pass.
- **F-P4 (MNIST, winner-take-all alone: monopolized, < 60%): CONFIRMED** (28.7%, one digit wins).
- **F-P5 (MNIST, fair-share alone: no specialization, < 65%): CONFIRMED** (0%, 47.1%).
- **The predicted "interaction" reading is wrong:** one rule (fair-share) is enough to break integration, and MNIST needs
  it (with winner-take-all) anyway.
- **Mechanism (post-hoc, labelled): habituation.** Readout cosine to the current context, and rate CV across neurons, at
  10 / 30 / 60 / 120 / 300 / 600 / 900 s into phases 2-5 (seed means):
  - ctrl N=40: readout -0.09 / 0.04 / 0.75 / 0.96 / 0.98 / 0.98 / 0.98; CV ~0.20 throughout.
  - fair-share N=40: readout 0.17 / 0.44 / **0.57** / 0.25 / 0.03 / -0.10 / -0.03; CV 0.30 -> **0.04**.
  - Fair-share is rate homeostasis. It drives every neuron to the same firing rate. The readout is "who fires above
    average", so once a context lasts longer than the equalization time (~2-5 min at Diehl & Cook's 0.05 mV), the
    representation is erased. The network responds to the change (it reads the new context faster than ctrl at 30-60
    s) and then habituates.
  - On MNIST, images last 0.35 s, so equalization works ACROSS images: that's what spreads neurons over digits.
- **Consequence:** the conflict is a timescale conflict. Homeostasis must be slower than the experiences the readout
  should represent, and fast enough to spread neurons during development. Dose test next (`modal_dose.py`, same day).

## 2026-10-01 — Unified-substrate check: the substrate MNIST needed (winner-take-all + fair-share threshold) BREAKS the click world's integration properties: old contexts aren't released, the change signal misses changes, the readout collapses, memory recognizes nothing. Synthetic and real data currently need different substrates

**Data:** `notebooks/brian2/unified_substrate/unify_n{7,40}_seed*.json.gz` (16 runs, Modal).
- **Format:** compact files, the first real use of the data policy (10 MB here; raw 1 s files on the Volume
  `developing-mind-raw` under `unified_substrate/`).
- **World:** v1b A B C A B, 5 x 1000 s, disjoint 30-input rig.
- **Substrate:** MNIST v4's: WTA 20 mV with the gate off, plus the fair-share threshold. Gain control is irrelevant for
  rate-matched input. The gates were left at their defaults.
- **Scripts:** `modal_unify.py` (predictions U-P1..P4, before launch) and `analyze_unify.py` (before results). Output
  `unify_output.txt`.

| | old substrate (step 2), N=7 / 40 | **unified substrate, N=7 / 40** |
|---|---|---|
| U-P1 late rate, neurons active | ~15 Hz / ~3 Hz, all | 13.2 / 3.7 Hz, 100%: **CONFIRMED** |
| B keeps at C's arrival | 85% / 87% | 88% / 79% |
| incoming before the two-back returns (A / B, share of N) | 9%/5%, 4%/6% | **43%/27%, 55%/23%** |
| U-P3 change signal fired / on when settled | 100% / 0.6-1.6% | **78% / 0.9%, 59% / 0.4%** |
| settled readout to its true prototype | 0.97 / 0.99 | **-0.18 / 0.06** |
| U-P4 B named at two-back, W=50 | 8/8 / 8/8 | **0/8 / 0/8** |

- **U-P1 (no silencing): CONFIRMED.** The fair-share threshold fixed what it was meant to fix.
- **U-P2 (one back survives): REFUTED.** B is kept, but old contexts are NOT released: 23-55% of the population still
  holds a context two back just before it returns.
- **U-P3 (change signal at its defaults, tripwire #3): REFUTED.** It catches only 59-78% of changes. **The tripwire
  fires here,** though the cause looks like the substrate, not the gate's numbers.
- **U-P4 (readout >= 0.9, memory >= 7/8): REFUTED.** The readout collapses (the settled fingerprint doesn't point at
  the current context), so memory recognizes nothing (0/8).
- **Likely mechanism (a hypothesis, not tested):**
  - In a sustained world, the fair-share threshold LOWERS the thresholds of quiet neurons, the ones tuned to old
    contexts.
  - Under winner-take-all, whoever crosses threshold first silences the rest, so those low-threshold, wrongly tuned
    neurons can win on background input.
  - Losers don't learn, so they keep stale tuning. The readout then reflects which neurons win, not what the input
    is.
  - This is the risk recorded in U-P2's docstring, worse than feared. MNIST's brief presentations with rests hide it.
- **Consequence:**
  - **Don't adopt the MNIST substrate as THE substrate yet,** and don't build split-MNIST (4b) on it.
  - The old substrate passes everything integration-related but can't learn real digits; the new one learns digits
    but breaks integration.
  - Reconciling them is the open problem. Candidates: keep the ambiguity gate with stronger inhibition; a slower or
    bounded fair-share; competition that lets losers learn a little; temporal gain control.

## 2026-10-01 — MNIST pilot v4: a fair-share threshold (no ratchet) beats the published adaptive threshold: 79% accuracy, every digit represented in every seed, confidence and novelty working; the published one silences any continuously running network

**Why v4:** on the continuously running click world, Diehl & Cook's adaptive threshold silenced the network.
- **Click world:** within ~300 s the threshold was up ~35 mV and the rate was 0 Hz at N=7 (0.05 Hz at N=40).
- **Cause:** it's a one-way ratchet (+0.05 mV per spike, decay 1e4 s). It works in their train-then-freeze setting,
  which the framework doesn't have.
- **Replacement, the fair-share threshold:** each spike raises its own neuron's theta by 0.05 mV (their constant) and
  lowers every neuron's theta by 0.05/N, so the sum is conserved. No new number. Busy neurons get harder to fire, quiet
  ones easier: homeostatic intrinsic plasticity without a setpoint. A local check kept the click world firing steadily.

**Data:** `notebooks/brian2/mnist_pilot/mnist_v4_wta_norm_fair_seed70100-70107.json.gz` (8 runs, Modal, paired with
v1-v3). Script `modal_mnist_v4.py` (predictions before launch); scored with `analyze_mnist_pilot.py <prefix> v3`.
Output `mnist_v4_output.txt`.

| | v3 wta_norm_adapt | **v4 wta_norm_fair** |
|---|---|---|
| vote accuracy (classes 0-3, chance 25%) | 65.1% | **79.2%** (76-84%) |
| silent test images | 0-1% | **0%** |
| neurons per class | every class >= 3 | **4-16, every class, every seed** |
| specialized / weight match | 74% / 67% | 65% / 72% |
| fingerprint separation (every pair) | 6/8 seeds | **8/8** |
| strength quintile gap / rises | +30 / 5 of 8 | +21 / **8 of 8** |
| held-out strange ratio | 2.0x | **2.5x** |

- **V4-P1 (keeps v3's result: specialization >= 60%, every class >= 3 in >= 6/8, accuracy >= 60%): CONFIRMED** (65%, 8/8,
  79.2%).
- **V4-P2 (confidence gap >= 15 in >= 6/8; novelty >= 1.5): CONFIRMED** (6/8 by the gap bar, rising 8/8; novelty 2.5x).
- **Reading:**
  - The ratchet slowly priced neurons out of firing (~10 spikes/image). Fair-share keeps all of them in play (~40-50
    spikes/image) while still spreading the work, and it's the only one of the two that suits an always-on system.
  - The substrate now has three added rules: gain control (per-image, see the caveat), winner-take-all (derived 20 mV,
    ambiguity gate off), and the fair-share threshold.
- **Stopping point for accuracy (Jasper, 2026-10-01): MNIST is calibration, and 79% at 40 neurons is enough.** Further
  rule changes should be judged by the framework-relevant tests (continuity, memory), not accuracy.
- **Caveat on gain control:** it normalizes per IMAGE, which uses the image boundaries as an oracle. A continuous system
  would adapt to its recent input over time. To fix before split-MNIST.

## 2026-10-01 — MNIST pilot v3: with derived winner-take-all inhibition added, the substrate learns real digits (74% specialized, every digit represented, 65% accuracy), and response strength becomes a working confidence signal. The pilot bar is met

**Data:** `notebooks/brian2/mnist_pilot/mnist_v3_{wta_adapt,wta_norm_adapt}_seed70100-70107.json.gz` (16 runs, Modal;
the same seeds as v1/v2, paired).
- **Rule added:** winner-take-all lateral inhibition, derived rather than tuned. Each competitor spike pushes a
  neuron down by its full threshold distance (v_thresh - v_rest = 20 mV), ambiguity gate off. The adaptive
  threshold is v2's (Diehl & Cook's constants).
- **Scripts:** `modal_mnist_v3.py` (predictions V3-P1..P6 and the silent-image scoring definitions, before launch);
  `analyze_mnist_pilot.py <prefix> v3` (v1 scoring regression-checked unchanged). Output `mnist_v3_output.txt`.

| | v1 | v2 norm_adapt | **v3 wta_norm_adapt** | v3 wta_adapt (raw) |
|---|---|---|---|---|
| specialized neurons | 0% | 96% | **74%** | 71% |
| weight map matches its class | 30% | 88% | **67%** | 64% |
| neurons per class (0/1/2/3) | ~25/0/4/11 | ~1/37/0/2 | **~13/12/8/7, every class >= 3 in 8/8** | ~25/0/8/7 |
| vote accuracy, silent = wrong (chance 25%) | 53.9% | 42.5% | **65.1%** (58-73%) | 47.7% (68.0% responsive-only) |
| silent test images | 0% | many | **0-1%** | 27-33% |
| strength: strongest vs weakest quintile | −51 points (inverted) | n/a | **+30** (rises in 5/8) | **+39** (rises in 8/8) |
| held-out 4/5 strange vs known | 1.8x | n/a | **2.0x** (1.4-2.6) | 1.2x |

- **V3-P1 (every class >= 3 neurons in >= 6/8, both arms): HALF.**
  - wta_norm_adapt: 8/8, a real division of labour across digits.
  - wta_adapt (raw input): "1" still gets 0 neurons in every seed, so the ink bias survives without gain control.
- **V3-P2 (the full pilot bar in wta_norm_adapt): CONFIRMED, at the margin.** Specialization 74%, weight match 67%,
  accuracy 65.1% (bar 65%; 58-73% by seed).
- **V3-P3 (raw input loses to abstentions): CONFIRMED.** 27-33% of test images get no response (faint digits can't
  beat the risen thresholds), so accuracy falls to 47.7%, although its answers, when it gives them, are as good (68%).
- **V3-P4 (confidence, >= 15-point gap in >= 6/8 seeds): CONFIRMED** (6/8; mean +30 points). v1's inversion is gone.
  Once competition is sparse, a strong response really means a confident, usually correct call.
- **V3-P5 (novelty >= 2x): CONFIRMED, exactly at the bar** (mean 2.04x; 5/8 seeds >= 2.0x).
- **V3-P6 (fingerprint separation in >= 6/8): CONFIRMED** (6/8).
- **The substrate needed three rules for real data, none of them a hand-tuned number:**
  1. per-image gain control (the input's own mean total);
  2. Diehl & Cook's adaptive threshold (their published constants);
  3. winner-take-all inhibition (derived from the neuron's own voltages).

  The integration gates (radius, change window) were not touched; MC-2's radius was self-set from the data.
  **Flag:** the synthetic-world results (one-back, the change signal, the memory results) were all found on the OLD
  substrate (gentle, gated inhibition, no adaptive threshold). Before building split-MNIST (4b) on the new one,
  check that those integration-critical properties survive on it.

## 2026-10-01 — MNIST pilot v2: gain control flips the ink bias toward sparse digits; the adaptive threshold creates sharp specialization (91-96%, digit-like weights) but nearly every neuron picks the same digit. The missing piece is diversity, not specialization

**Data:** `notebooks/brian2/mnist_pilot/mnist_v2_{norm,adapt,norm_adapt}_seed70100-70107.json.gz` (24 runs, Modal).
The same seeds as v1 (paired: same images, order and noise). Script `modal_mnist_v2.py` (predictions V2-P1..P5
before launch); scored with v1's scorer and criteria (`analyze_mnist_pilot.py <prefix>`). Output
`mnist_v2_output.txt`.

| arm | specialized | weights match assigned class | vote accuracy | neurons per class 0/1/2/3 (typical) | spikes/image (test) |
|---|---|---|---|---|---|
| v1 (none) | 0% | 30% | 53.9% | ~25/0/4/11 | ~720 |
| norm (gain control) | 0% | 10% | 46.7% | ~0/36/0/3 | ~910 |
| adapt (threshold) | **91%** | 77% | 28.1% | ~39/0/0/1 | 9-10 |
| norm_adapt (both) | **96%** | **88%** | 42.5% | ~1/37/0/2 | 10-11 |

- **V2-P1 (norm removes the ink confound, every class gets >= 3 neurons): REFUTED.** Equal total input per image
  packs a thin "1" into a few very bright pixels, so the bias flips to 1s (32-39 of 40 neurons).
- **V2-P2 (adapt sparsifies but keeps the ink bias): CONFIRMED.** Spikes/image drop from ~720 to ~22 (9-10 in
  test), and 36-40 neurons pick "0".
- **V2-P3 (norm_adapt best: specialization >= 40%, weight match >= 50%, accuracy >= 65%): HALF.** Specialization
  96% and weight match 88% are confirmed. Accuracy 42.5% (25-60%) is REFUTED: almost every neuron specializes on the
  SAME digit.
- **V2-P4 / P5 (MC-1, MC-2): undefined in the adapt arms.** With ~10 spikes per image, many test images get no
  response at all, so strength is 0, the quintiles degenerate (NaN), and the novelty match is degenerate (0%).
  Where responses exist, the strongest quintile is 88-100% accurate in norm_adapt, a hint that strength is a real
  confidence signal once ink is out of the way. Not a scored claim.
- **Reading:**
  - v1's failure had two layers: an input confound (ink), and under it a competition too weak to give different
    neurons different jobs.
  - The adaptive threshold solves specialization, but with our gentle, ambiguity-gated inhibition (~0.67 mV per
    spike at N=40) nothing stops all neurons converging on the class that wins first.
  - Diehl & Cook's strong winner-take-all inhibition is the missing rule.
- **Next (v3):** winner-take-all inhibition as a derived rule, not a tuned number. Each competitor spike pushes a
  neuron down by its full threshold distance (v_thresh - v_rest = 20 mV), with the ambiguity gate off. Arms: Diehl &
  Cook's recipe (raw input + adaptive threshold + WTA), and the same with gain control. The scorer gets explicit
  handling for zero-response images, defined before v3 data.

## 2026-10-01 — MNIST pilot v1: the unchanged substrate learns how much ink a digit has, not its shape (0% specialization, no neuron for "1", inverted confidence), though its fingerprint still separates classes and the population vote reaches 54%

**Data:** `notebooks/brian2/mnist_pilot/mnist_pilot_seed70100-70107.json.gz` (8 seeds, Modal, Brian2 seeded).
- **Images:** 14x14 MNIST (196 inputs), classes 0-3 learned, 4-5 held out (`prep_mnist.py`; MNIST itself is
  git-ignored).
- **Encoding:** rate coding at up to 63.75 Hz, 350 ms per image + 150 ms rest.
- **Network:** N=40, the unchanged competitive substrate, weight budget 30 (`calibrate_mnist.py`: the smallest
  budget passing the pre-stated 3-20 Hz bar).
- **Timeline:** 1200 train, 200 label, 600 test images. Plasticity is never switched off.
- **Scripts:** `mnist_sim.py` (core), `modal_mnist_pilot.py` (predictions MP-P1..P4, MC-1, MC-2, written before
  launch, with the caution that specialization was uncertain), `analyze_mnist_pilot.py` (before results). Output
  `mnist_pilot_output.txt`.
- **One false start:** the first Modal launch failed on an import before running (helper not shipped into the
  image); fixed.

| | result |
|---|---|
| MP-P1 neurons specialized (top class >= 1.5x second) | **0%** in every seed (pred. >= 60%): REFUTED |
| MP-P2 weight map matches the assigned class's mean image | 30% (18-40%) (pred. >= 60%): REFUTED |
| MP-P3 fingerprint within-class > between-class, every pair | 7/8 seeds: CONFIRMED in 7 of 8 |
| MP-P4 vote accuracy on classes 0-3 | 53.9% (48-59%; chance 25%; pred. >= 50%): CONFIRMED, barely |
| MC-1 strength quintiles vs accuracy | **inverted** in 8/8: strongest quintile 10-28% vs weakest 53-78% (gap −51 points): REFUTED |
| MC-2 held-out digits strange vs known | 1.8x (0.8-5.3x) (pred. >= 2x): REFUTED |

- **What happened:** neurons track **ink, not shape.**
  - Rate coding makes a digit's total input proportional to its ink, so the network's response is dominated
    by how bright an image is.
  - Neuron assignments: about 20-30 neurons for "0" (the most ink), 2-7 for "2", 8-19 for "3", and **0 for "1"
    in every seed** (the least ink).
  - Firing grew during learning, from ~270 spikes/image at calibration to ~720: weights concentrate on bright
    pixels, drive rises, and the gentle competition can't contain it. Every neuron fires for every image.
  - The strength signal (MC-1) measures ink too, hence the inversion: bright digits drive the strongest responses
    and are the most confused.
- **What still works:** the rectified fingerprint uses *relative* activity, so it separates classes (7/8), and
  the population vote reaches 54% with no specialized neuron at all.
- **Why the toy worlds never showed this:** every synthetic context was rate-matched by design (only timing
  differed). Real images differ in total intensity, a feature the substrate had never met.
- **Next (v2):** two hand-built rules, both standard.
  - Per-image input normalization (gain control: every image delivers the same total input rate, set to the
    training set's own mean).
  - Diehl & Cook's adaptive threshold (a neuron's threshold rises with each spike and decays slowly; their
    published constants).
  - Tested as separate arms so their effects separate.

