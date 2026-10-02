# Brian2 log: MNIST, the first real data (generalization plan step 4)

Index: `experiments_brian2.md`. Plan: `experiment_plan_generalization.md`, step 4. Newest entries first.

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

