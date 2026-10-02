# Brian2 log: Non-stationary correlation: world swaps A->B->A, N=3 and N=7 (2026-09-25)

Entries moved verbatim from `experiments_brian2.md` on 2026-09-25 (no wording changed). Index: `experiments_brian2.md`.

## 2026-09-29 — Ambiguity (both contexts at once): the substrate holds both readings with zero switching, and the two groups' activity jitters around a tie as white noise, with no rivalry

**Data:** `notebooks/brian2/ambiguity_data/`, seeds 47000-47007, 8/8 completed (935 s wall).
- **World:** A(1000) B(1000) AB(2000) A(1000) s, disjoint 30-input rig, N=7, 13mV/1.5.
- **AB:** wires 1-10 synced to one hidden rhythm and wires 11-20 to a second, independent one, both at
  full strength. The builder is local (checked: within-group sync 0.61-0.64, between groups 0.09,
  i.e. chance; rates matched).
- **Code:** predictions AM-P1..P6 were in `run_ambig_seed.py` and the scorer `analyze_ambig.py` was
  committed, both before any result. The post-hoc checks are labelled in the same file. Outputs
  `ambig_output.txt`, `ambig_posthoc_output.txt`.

- **AM-P1 (coexistence, no capture): CONFIRMED, exactly.** A 3.00 and B 4.00 holders from AB's start
  to its end, with zero change.
- **AM-P2 (no rivalry in tuning): CONFIRMED.** 0 preference switches per seed in 1700 s of settled AB.
  With no fatigue mechanism, nothing drives a Necker-style flip.
- **AM-P3 (AB onset is a small event): CONFIRMED.** The displacement peak is 1.16 vs 3.95 after B's
  arrival (8/8).
- **Post-hoc, activity rather than tuning:**
  - In 76% of settled 10 s windows, the neurons firing above average all belong to ONE of the two
    groups. The groups compete in activity even though their tuning never moves.
  - That competition is white noise, not rivalry. The A-minus-B activity difference has an
    autocorrelation time of 1 s and "dominance episodes" of 4.9 s, the same as a control (two
    random halves of the neurons during pure A: 1 s, 4.8 s).
  - Real perceptual rivalry needs something like adaptation to produce episodes. This substrate
    doesn't have it, the same ingredient the MNIST plan flags (adaptive threshold).

## 2026-10-01 — Generalization step 2, network size: from 5 to 40 neurons, one-back, the ~half retainer fraction, the change signal at its hand-set defaults, the readout and memory recognition all hold unchanged

**Data:** `notebooks/brian2/n_scaling_v1b_data/`, N in {5, 7, 10, 15, 40} x 8 seeds (51000-51407), 40/40 completed
**on Modal** (cleared by the arc-06 fidelity checks). Brian2 seeded.
- **World:** the v1b world A B C A B (5 x 1000 s), disjoint 30-input rig, 13 mV/1.5 normalized with
  `scale_inhib_for_n`. Mirrors `run_v1_seed.py` with N as the only variable.
- **N=7 arm:** included so every size runs under one seeded protocol.
- **Top size 40:** the MNIST pilot size (Jasper's go).
- **Scripts:** `modal_nscale.py` (predictions NS-P1..P5 in its docstring, committed before launch) and
  `analyze_nscale.py` (committed before results). Output `nscale_output.txt`.
- **Modal wall per run:** 561 s (N=5) to 936 s (N=40). About $0.70 in total.

| N | B keeps at C's arrival | incoming before two-back returns (A / B, share of N) | A retainers at end of B | change fired / on when settled | settled readout | B named at two-back, W=50 / W=10 |
|---|---|---|---|---|---|---|
| 5 | 79% | 5% / 8% | 52% | 100% / 0.4% | 0.962 | 8/8 / 8/8 |
| 7 | 85% | 9% / 5% | 54% | 100% / 1.6% | 0.974 | 8/8 / 8/8 |
| 10 | 94% | 5% / 0% | 57% | 100% / 0.8% | 0.978 | 8/8 / 8/8 |
| 15 | 80% | 6% / 7% | 50% | 100% / 0.6% | 0.983 | 8/8 / 8/8 |
| 40 | 87% | 4% / 6% | 53% | 100% / 0.6% | 0.987 | 8/8 / 8/8 |

- **NS-P1 (one back at every N): CONFIRMED.** B keeps 79-94% of its holders at C's arrival. A context two
  back holds 0-9% of the population just before it returns.
- **NS-P2 (retainer fraction roughly size-independent, 35-65%): CONFIRMED, tightly.** 50-57% at every N. More
  neurons means proportionally more retainers per context, still one back (the leaning, now measured).
- **NS-P3 (the change signal at its hand-set defaults, no re-tuning): CONFIRMED.** It fires after 100% of swaps
  and is on for 0.4-1.6% of settled seconds at every N. **The tripwire did not fire:** L 60 s, trail 900 s and
  k 3 work from 5 to 40 neurons.
- **NS-P4 (readout): CONFIRMED.** The settled fingerprint is 0.962-0.987 to its prototype, rising slightly with N.
  Between-context cosines are about −0.45, far below the 0.8 radius.
- **NS-P5 (memory, B two-back at W=50 ≥ 7/8): CONFIRMED** (8/8 at every N). Also 8/8 at W=10 and for A's two-back
  return.
- **Reading:**
  - Network size (5-40) is not a variable for this substrate on disjoint contexts. Everything the integration
    layer relies on holds unchanged, with the same hand-set gate numbers.
  - The only size-dependent quantity is readout quality, which improves with N.
  - Together with step 1 (phase length 300-3000 s is not a variable either; overlap is), the substrate's
    behaviour is robust to the two scale knobs tested.
  - N=40 behaves fine, so the MNIST pilot size is cleared.

## 2026-09-29 — Rest: with no synchrony at all for 3000 s, the substrate holds its tuning exactly (zero re-assignment), barely blurs, and doesn't register rest as a change

**Data:** `notebooks/brian2/rest_data/`, seeds 46000-46007, 8/8 completed (1315 s wall).
- **World:** A(1000) B(1000) C(1000) REST(3000) B(1000) s, the disjoint 30-input rig, N=7, 13mV/1.5.
- **REST:** every wire independent at the same 20 Hz. Only the timing structure goes away; rates are
  matched (checked on the builder).
- **Code:** the input builder is local (`run_rest_seed.py`), `src` untouched. Predictions RE-P1..P5
  were in `run_rest_seed.py` before launch; the scorer `analyze_rest.py` was committed before any
  result. Output `rest_output.txt`.
- **Why:** framework v5, section I ("the resting state isn't nothing"). Every earlier run fed a
  context the whole time.

| mean holders (A B C) | end of C | rest +300 s | rest +1500 s | end of rest | B back +150 s |
|---|---|---|---|---|---|
| | 0.50 2.75 3.75 | 0.50 2.75 3.75 | 0.50 2.75 3.75 | 0.50 2.75 3.75 | 0.50 3.12 3.38 |

- **RE-P1 (tuning survives rest): CONFIRMED, exactly.**
  - Zero change in any holder count over 3000 s of rest, and B still has 2.75 holders when it
    returns.
  - With no new context to recruit them, no neuron changes allegiance.
  - Note that B returning after rest takes back only ~0.4 neurons from C (3.75 → 3.38). Rest was
    not a change: B came back as a one-back context, not a two-back one.
- **RE-P2 (selectivity blurs by ≥ 10% of its excess over uniform): narrowly REFUTED** (9%). The
  preferred-block share goes 0.861 → 0.813 over 3000 s. The blurring is real but slow.
- **RE-P3 (rest is a smaller event than a change): CONFIRMED.**
  - The 60 s weight displacement peak at rest onset is 0.83, against 3.25 after C's arrival (8/8
    seeds).
  - The re-learning flag was on **0%** of the first 300 s of rest.
  - The substrate doesn't treat losing all structure as a world change. Only a new structure
    triggers re-assignment.
- The memory side (ghosts, rehearsal during rest) is in `experiments_integration.md` (2026-09-29).

## 2026-09-29 — Generalization step 1, 3000 s arm: one-back holds after 3000 s holds too; 300, 1000 and 3000 s phases are near-identical, and nothing moves between changes over 2700 s of stable world

**Data:** `notebooks/brian2/long_phase_data/`, seeds 45000-45007, 8/8 completed (2755 s wall). v1b
schedule A B C A B, 3000 s phases, the same disjoint 30-input rig, N=7, 13mV/1.5 (v1's frozen runner,
durations overridden). Predictions LP-P1/P2 were in `run_long3000_seed.py`, and the scorer
`analyze_long3000.py` was committed, both before any result. Output `long3000_output.txt`.

| holders (A B C) | 300 s | 1000 s | **3000 s** |
|---|---|---|---|
| swap 2 (→C), before | 3.50 3.50 0.00 | 3.62 3.38 0.00 | 3.50 3.38 0.12 |
| swap 2, +150 s | 0.88 3.00 3.12 | 0.38 3.25 3.38 | 0.75 3.12 3.12 |
| swap 3 (→A, two back), before | 0.50 2.88 3.62 | 0.25 3.25 3.50 | 0.38 2.88 3.75 |
| swap 4 (→B, two back), before | 4.00 0.75 2.25 | 4.25 0.38 2.38 | 3.88 0.12 3.00 |

- **LP-P1 (one-back holds at 3000 s): CONFIRMED.**
  - B keeps 93% of its holders at C's arrival.
  - A, held through 3000 s of B, is down to 0.50 by +300 s.
  - Incoming holders at the two-back returns are 0.38 and 0.12.
  - Long holds don't harden retainers against release.
- **LP-P2 (nothing moves between changes): CONFIRMED.** Over each 2700 s stable stretch (from +300 s to
  the next swap), no context's mean holder count changes by more than 0.38. That's the quiet the
  re-learning signal relies on.
- **Reading:** across a 10x range of phase length (300-3000 s), the disjoint-context substrate does the
  same thing: an even split at the first change, the older context released at the next change, and
  flat in between. Phase length is not a variable for this substrate; overlap is (step 1's overlap
  arm). Step 1 is complete.

## 2026-09-28 — Generalization step 1, overlap arm: with 50% overlapping contexts a new context captures most of the population at 300 s and 1000 s alike, so the just-departed keeps only ~37% (disjoint: 86-96%). Overlap, not phase length, sets the split

**Data:** `notebooks/brian2/short_phase_data/short300ov50_*`, seeds 44000-44007, 8/8 completed (295 s
wall): the ov50 world (v1b schedule, pairwise 50% overlapping input sets) with 300 s phases. Scored
together with the existing ov50 1000 s runs (`set_worlds_data`, 38000-38007) by
`analyze_short_ov50.py` (step 0's holder metric), output `short_ov50_output.txt`. Predictions SO-P1/P2
were in `run_short_ov50_seed.py`, committed before launch and before the 1000 s runs were scored
with this metric.

| world | incoming holders +150 s (swap 1 / swap 2) | just-departed at swap 2: before → +150 s |
|---|---|---|
| ov50, 300 s | 4.75 / 4.38 | 6.00 → 2.25 (**38%**) |
| ov50, 1000 s | 4.88 / 4.38 | 6.62 → 2.38 (**36%**) |
| disjoint, 300 s | 3.25 / 3.12 | 3.50 → 3.00 (86%) |
| disjoint, 1000 s | 3.12 / 3.38 | 3.38 → 3.25 (96%) |

- **SO-P1 (the incoming context takes ≥ 5 of 7 within 150 s, under overlap): REFUTED narrowly, in the
  predicted direction** (4.4-4.9, vs 3.1-3.4 disjoint). By the end of its phase it holds 6.0-6.6 of 7.
- **SO-P2 (the just-departed keeps < 60% of its holders): CONFIRMED** (36-38%, vs 86-96% disjoint), at
  both phase lengths.
- **Reading:** with overlapping contexts there's no even split and no stable retainer group. Each new
  context captures most of the population, presumably because neurons tuned to the previous context
  already respond to the shared inputs. So the substrate holds *less* than one full context back.
  300 s and 1000 s are indistinguishable, so **overlap, not phase length, sets how much the
  substrate retains.** This is also what step 0 saw in checkb, whose fillers overlap the cores.
- **For memory:** nothing breaks. Memory already carries whatever the substrate drops (ov50 two-back
  recognition 8/8 at W=50), and rehearsal is keyed to changes, which still trigger all
  re-assignment. It sharpens `principles.md`'s caveat: "one back" is the disjoint-context case.

## 2026-09-28 — Generalization step 1, short-phase arm: with 300 s disjoint phases the substrate is one-back, almost exactly as at 1000 s, so step 0's deviation came from overlap, not phase length

**Data:** `notebooks/brian2/short_phase_data/`, seeds 43000-43007, 8/8 completed (305 s wall). v1b
schedule A B C A B, 300 s phases, the same 30-input disjoint 3-block rig, N=7, 13mV/1.5 (v1's frozen
runner, durations overridden). Predictions SP-P1/P2 were in `run_short_seed.py` before launch.
Scored by `analyze_short_phase.py` (step 0's holder metric), output `short_phase_output.txt`.

| holders (A B C) | 300 s phases | 1000 s (v1b) |
|---|---|---|
| swap 2 (→C), before | 3.50 3.50 0.00 | 3.62 3.38 0.00 |
| swap 2, +150 s | **0.88** 3.00 3.12 | 0.38 3.25 3.38 |
| swap 3 (→A, two back), before | 0.50 2.88 3.62 | 0.25 3.25 3.50 |
| swap 4 (→B, two back), before | 4.00 0.75 2.25 | 4.25 0.38 2.38 |

- **SP-P1 (one-back holds with short disjoint phases): CONFIRMED on 2 of 3 parts.**
  - B (just departed) keeps 86% of its holders at C's arrival.
  - The incoming context has 0.50 and 0.75 holders at the two-back returns (≤ 1).
  - The third part missed: A's release is slightly slower, 0.88 holders at +150 s against the stated
    ≤ 0.5 (v1b: 0.38). It's down to 0.50 by +300 s.
- **SP-P2 (each phase's own context holds ≥ 4 of 7 by its end): REFUTED, but the threshold was
  mis-set.** Under one-back a new context gets about half the population by design (the rest keep
  the just-departed one). 300 s phases reach 3.4-4.0, the same as 1000 s phases (3.4-4.3). Every
  phase is learned to the 1000 s level within 300 s.
- **Reading:** phase length (300 vs 1000 s) barely changes anything for disjoint contexts. Step 0's
  deviation (checkb: the just-departed filler dropped, older core residue kept) came from what
  differs there: **input overlap**, with fillers sharing 3-5 inputs with the cores. Which part of
  overlap matters (the shared inputs themselves, or overlap with *several* cores at once) is open.
  That's the natural next arm: v1b at 300 s with 50% overlap, a direct comparison to ov50 at 1000 s.

## 2026-09-28 — Generalization step 0 + the long world: re-assignment is change-triggered (never between changes), but "exactly one back" is schedule-dependent: after a short, overlapping filler the substrate drops the just-departed filler and keeps the older core's residue; no drift and no residue build-up over 20 phases

**Why:** `experiment_plan_generalization.md`, step 0. Every earlier one-back measurement used 1000 s
phases, where "the next change" and "1000 s after departure" coincide. checkb's 300 s fillers
break that tie. Holders of X at t = neurons whose 50 s mean weight vector is closest to X's
prototype (among all the world's contexts). Script: `notebooks/brian2/set_worlds_data/analyze_release_timing.py`
(predictions RT-P1/P2 in its docstring before the first run; `post_hoc()` is labelled and was added
after), output `release_timing_output.txt`.

**Pre-registered (checkb cores, 300 s fillers): both REFUTED.**
- **RT-P2 (a core keeps >= 60% of its holders through the 300 s filler):** it keeps only **12-33%**
  (A 7.0 -> 0.9, B 6.0 -> 1.5, C 5.8 -> 1.9). In v1b's 1000 s disjoint phases, A keeps 52% (7.0 -> 3.6).
- **RT-P1 (the rest released within 300 s of becoming two back):** there's no further drop
  (83-100% of the remainder stays, then persists to the end of the run).

**What happens instead (post-hoc, who holds what around each change):**
- **All re-assignment happens within ~150 s of a change, and holder counts are flat between changes,
  in both worlds.** Release is triggered by the change, not slow. This part of step 0's question is
  answered, and it's what the rehearsal and horizon logic needs.
- **Which context loses neurons isn't "the older one" in general.**
  - v1b (1000 s, disjoint): the new context recruits the older context's retainers (A 3.6 -> 0.4
    at C's arrival) and the just-departed keeps its own. That's one-back.
  - checkb: each filler takes 5-6 of 7 neurons within 150 s. The next core then recruits almost
    entirely from the filler's learners (F1 6.1 -> 0.8 at B's arrival), while the older core's
    ~1-2 residue neurons survive every later change.
  - So after a short, overlapping filler, the substrate drops the just-departed context and keeps
    older residue.
- **Candidate causes, not separated:** the filler's short phase (300 s learners haven't hardened);
  its partial overlap with the cores (each shares 3-5 inputs with a core, so core-tuned neurons are
  the easiest to recruit); how many neurons each group holds when the next change arrives.
- **Consequence:** "exactly one context back" is what this substrate does under long, disjoint phases,
  not a general law. `principles.md` gets a scope caveat. For memory it changes little: memory
  already carries whatever the substrate doesn't, and rehearsal is keyed to changes, which *are*
  what triggers re-assignment.

**The long world** (`notebooks/brian2/long_world_data/`, seeds 42000-42007, 8/8 completed, 3735 s
wall. 20 x 1000 s phases, contexts A, B, C disjoint, D 70% overlap with A, E 50% with A and B, each
visited 4 times. Readout: `notebooks/integration/memory_limits/long_world_readout.py`, predictions
written before any result):
- **LW-P1 (one-back holds and the residue doesn't accumulate): CONFIRMED.** At returns more than one
  back: 0.82 holders per seed, 1.30 in the first half and 0.57 in the second (declining, not
  accumulating). This answers the open item for a 20-phase schedule with returns. An all-novel
  schedule is still untested. (This schedule has no one-back returns; there are no immediate
  repeats.)
- **LW-P2 (no drift): CONFIRMED.** The settled rectified readout at a context's 4th visit vs its 1st
  (~15,000 s apart) has cosine 0.956-0.962 for all five contexts. The substrate's picture of a
  context doesn't drift over this span. That's the drift requirement for the dormant character
  store, met here.

## 2026-09-25 — v1 world (A→B→C→A→C): one-back retention confirmed within seeds, the residue doesn't grow over four changes, and a residue neuron recognizes a 2-back return

**Data:** `notebooks/brian2/v1_schedule_data/`, seeds 33000-33007, 8/8 completed (1280 s wall,
8 concurrent). `.json.gz` via `results_io`. Scripts: `run_v1_seed.py`, `run_v1_batch.py`,
`analyze_v1.py`; figure `v1_tuned_gaps.png`. Same 30-input 3-block rig as novel-C (N=7, 13mV/1.5
reference, target_total 10), 5 phases x 1000 s, correlated block A→B→C→A→C. Input from
`spikes.build_multiblock_phase_input` (tested bit-identical to novel-C's builder). The predictions
were in `analyze_v1.py` before launch.

**Why this schedule:** swap 3 (C→A) returns to a context two back and swap 4 (A→C) to one back,
so one-back retention predicts recognition at swap 4 and not at swap 3 in the *same seeds*. It
also gives S1's same-rig lull test (arc 07).

**Predictions on record, and outcomes:**
- **V1-P1 (at swap 3, at most 1 neuron per seed already holds A; at swap 4, 2-4 hold C): CONFIRMED.**
  Holders of the incoming pattern: 0.25 per seed at swap 3 (1 neuron in each of 2 seeds) vs 3.0 at
  swap 4 (2-4 in every seed). Higher at swap 4 in 8/8. The trajectories match the novel-C reading
  at every change. The neurons holding the *older* pattern release it and learn the new one (e.g.
  the B-holders drop B at the C→A swap), while the just-departed pattern is kept.
- **V1-P2 (holders speed up +4 to +8 Hz at swap 4): CONFIRMED.** +4.67 to +8.28 Hz, all 8 seeds.
  The scalar dip is -0.60 Hz at swap 4 vs -3.27 Hz at swap 3.
- **V1-P3 (H lull test, same rig: A4 closest to A1, C5 closest to C3, each in at least 6/8):
  CONFIRMED, 8/8 and 8/8.** Mean H similarity A4-A1 is +0.33 (per seed +0.24 to +0.47) against
  -0.44/-0.55 to B2/C3. C5-C3 is +0.66. The first context is still the weakest match, as arc 07
  found.

**Descriptive:**
- **The lock-in residue doesn't accumulate over four changes.** Neurons holding neither the
  current nor the previous context at each phase end: 0, 0, 2, 0, 0 (2 neurons in 2/8 seeds at
  the end of phase 3, both holding A). **Caveat:** A came back in phase 4, which turns A-holders
  back into "current", so this schedule can't test accumulation without returns. That needs a
  schedule of all-novel contexts (more blocks).
- **The residue isn't pure dead weight.** In the two seeds with an A-holding residue neuron
  (33004, 33005), that neuron hardened (about 0.95) through B and C, sped up at A's two-back
  return (+5.5, +6.2 Hz), and made A instantly available. It's n=2, so a lead, not a finding. But it
  is the only route to 2-back recognition the substrate has shown.
- Commit latency to the incoming pattern (non-holders): 65 / 77 / 88 / 138 s median at swaps 1-4.
  The 2144 s maximum after swap 2 is a neuron that never took C in phase 3 and took it only when C
  returned in phase 5. The latency window isn't bounded by the phase end.
- w_total [9.51, 10.42], healthy through four swaps.

**Net:** one-back retention is now shown within seeds, not just across rigs. The substrate keeps
the just-departed context and releases older ones, with a small, so far non-accumulating residue
that can occasionally carry an older context. Anything older than one back has to be held
elsewhere. For what the memory layer does with this, see `experiments_integration.md`.

## 2026-09-25 — Novel-C (A→B→C, C never seen): held patterns are released at the next change (one-back retention, a small permanent residue), and recognition lives in which neurons speed up, not in the population dip

**Data:** `notebooks/brian2/novelc_data/`, seeds 32000-32007, all completed. Scripts:
`calibrate_novelc.py`, `run_novelc_seed.py`, `run_novelc_batch.py` (32000-32001 sequential,
32002-32007 in parallel; deterministic per seed), `analyze_novelc.py`, figure
`novelc_n7_reliable_13_1p5.png`. There are three disjoint presynaptic blocks of 10 (30 inputs),
one correlated per phase. Phases are A→B→C at 1000 s each; C has never been correlated before.
`target_total` is held at 10 (w_init=1/3), so total synaptic drive matches the 20-input rig, and
`gmax` is unchanged. Calibration was checked before the batch against a bar stated in advance
(3-20 Hz, w_total ~9.5-10.5): 15.0/17.3/16.0 Hz, w_total [9.74, 10.20], pass. N=7, 13mV/1.5
reference with the same inhibition normalization as the N=7 A→B→A runs, dt=0.2 ms, Cython,
I_pert=0. **This is not the same rig as A→B→A.** The correlated fraction is 1/3 here, not 1/2, so
absolute counts aren't directly comparable across the two. Plain `.json` output (written before the
`results_io` convention was known). `analyze_novelc.py`'s main readout was fixed before any result
was looked at. Its `post_hoc()` section was added after inspecting the trajectory figure and is
labeled as such.

**Question.** In A→B→A the held pattern returned, so retention paid off by construction. Does a
held pattern stay held when it never returns (a capacity tax), or is it released? This run is
also the control for the previous entry's "smaller dip on return" reading.

**1. Neither pure reading: one-back retention, with a small lock-in residue.** Every seed learns
A in phase 1 (AAAAAAA in all 8).
- **Swap 1 (A→B):** 31 of 56 neurons keep A and 25 move to B within 44-148 s (median 65 s). That's
  55% retaining, against about 37% in the 20-input rig, **so the retainer fraction depends on the
  rig; "about a third" isn't a constant.**
- **Holds are flat, not slowly decaying.** The A-holders' A tuned-gap loss across phase 2 has a
  median of 0.044 (above 0.1 in 8 of 31, max 0.21). It looks like a stable hold with occasional
  slow erosion (seed 32006 is the clearest), not decay back to uncommitted.
- **Swap 2 (B→C):** 28 of 31 A-holders release A and commit to C (median 77 s). The neurons that
  learned B mostly *keep* B. So at the end of phase 3, stale neurons are 3.1 of 7 on average (45%),
  almost all holding the just-departed B (22 of 56), with the older A mostly gone.
- **Permanent residue:** 3 neurons in 3 of 8 seeds (32000 n0, 32002 n3, 32007 n0) never release A,
  across 2000 s of its absence. None of them erode. One hardens further (A gap 0.82 → 0.96). This
  is genuine lock-in in a minority, about 5% of neurons per additional change.
- **Reading:** the population holds roughly the *most recently departed* context and releases
  older ones when the next change comes. Across 3 phases it doesn't accumulate stale patterns,
  apart from the residue. Two things are untested: whether the residue accumulates over longer
  schedules, and whether release is triggered by the change or would happen anyway after more than
  1000 s (that needs a longer phase 2). **One-back has a sharp, testable consequence.** In
  A→B→C→A, the return to A should *not* be recognized instantly, because A was released at the
  B→C swap. That's unlike A→B→A.

**2. The on-record swap-signal prediction was refuted for the net dip, as the confound flagged in
the previous entry predicted.** The swap-2 dip is also smaller when the incoming pattern is new:
-5.13 Hz at swap 1 vs -2.16 Hz at swap 2 (paired Wilcoxon p=0.008), a ratio of 0.42 against 0.37
for the return in A→B→A. **So the smaller population dip at the A→B→A return was mostly swap order
and population state, not familiarity.** At swap 2, only the neurons tuned to the departing pattern
lose synchronous drive (-6.04 Hz here, -6.33 Hz in A→B→A), and fewer neurons are tuned to it.

**3. The per-neuron recognition signal survives its control.** Per seed, the mean first-10 s rate
change of neurons holding A at swap 2 is +4.66 to +7.69 Hz when A returns (A→B→A) and -0.91 to
+2.14 Hz when it doesn't (A→B→C). That's perfect separation, Mann-Whitney p=0.00016 (per seed,
8 vs 8, cross-rig). The trackers' drop is about the same in both rigs, which anchors the scale.
**Recognition is carried by *which* neurons speed up, a vector, not by the population's summed rate,
a scalar.** A scalar surprise readout can't tell a return from a new pattern. A per-neuron readout
can.

**4. The slow tail doesn't need coverage.** Swap-2 commits to C include two at 692 and 722 s
(seeds 32000 and 32003). Both are A-holders releasing A late, and nobody held C. So "the returned-to
pattern is already covered" (the previous entry's correlation) can't be the general cause of slow
commits. That seed-level correlation still stands as a correlation, but its interpretation is
weakened. An alternative, untested: slow commits come from neurons releasing a long-held pattern.

**5.** Homeostatic scaling held through both swaps at 30 inputs: w_total [9.62, 10.47], target 10.

**What this changes elsewhere.**
- For `principles.md` (still Jasper's call), the finding is now "one-back retention with a small
  lock-in residue". That isn't plain "division of labour", and it isn't "capacity tax".
- For the interface (`system_contract.md` S3/S4), the substrate hands up the previous context,
  held by about 40-55% of units until the next change, plus a small permanent residue. Its change
  signal is context-blind as a scalar and context-specific as a vector. Any use of it for episodic
  check (b) has to read per-neuron activity.

## 2026-09-25 — N=7 follow-up analyses: the slow return is tied to how many neurons already hold the pattern, and the population's firing rate distinguishes a new pattern from a returning one

**Data:** the existing N=7 runs (`nonstat_n7_reliable_13_1p5_*`, seeds 31000-31007). No new
simulation. Scripts: `analyze_slow_tail.py`, `analyze_swap_signal.py` in
`notebooks/brian2/nonstationary_data/`. Both are exploratory analyses of data that was already
saved, with no prediction recorded beforehand, so read them as descriptive at n=8.

**1. The slow swap-2 tail goes with coverage, not with competition among returners.** At swap 2
(back to A), each neuron is either a *holder* (already on A, a swap-1 retainer), a *returner*
(on B, climbs back to A's gap 0.3), or a *refuser* (on B, never returns). This reproduces the
62-977 s tail: 15 returners across 8 seeds. The seed is the honest unit, because returners in one
seed share its holder count and pooling them is pseudoreplication. Per seed, every seed with 3
holders has a slower slowest-returner than every seed with 2 (3-holder: 304, 512, 713, 976,
977 s; 2-holder: 149, 158, 267 s). That's perfect separation, Mann-Whitney p=0.018, the minimum
possible for 5 vs 3. The alternative, that returners slow each other, gets no support (pooled
latency vs number of co-returners: Spearman rho=-0.19, p=0.51). **Mechanism not localized:** the
obvious candidate is the ambiguity-gated inhibition, but the gate readings during each climb
(g_ij from the saved r traces) don't back it. Returners are slightly *more* rate-close to
co-returners (mean g 0.63) than to holders (0.56), and the one nominally significant gate
correlation (latency vs g to co-returners, rho=-0.55, p=0.043, pseudoreplicated) points the wrong
way for a simple suppression story. So "more of the population already holding the pattern goes
with a slower return for the rest" is a seed-level fact. That the gate causes it is untested.
Consequence: the instant phase-3 recovery and the slow tail aren't separate observations. The
retention that keeps A available coincides with the others being slow to rejoin it.

**2. The substrate emits an observable signal at a world change.** The phase-aligned gap can't
show this, because it flips sign at a swap by construction (the reference block changes; weights
are continuous, max |dw| of any synapse across a 4 s window at a swap = 0.106). Firing rate can,
and the network could in principle read it. Population rate in the first 10 s after a swap drops
by 3.27 Hz (16.7 to 13.4, sd 1.74, n=16 swaps). It recovers over about 300 s (13.2, 13.8, 14.7,
15.6 Hz at 10-30, 30-60, 60-120, 120-300 s). The same windows at non-swap times (t=500, 1500,
2500, n=24) change by -0.13 Hz (sd 1.24): Mann-Whitney p=4e-6. Mechanistically, the inputs are
rate-matched, so this is lost synchrony. Neurons tuned to the old block keep receiving it at the
same rate, but no longer as a synchronous volley.

**3. The dip is smaller for a returning pattern than for a new one, and the retainers are why.**
Paired per seed: first-10 s dip -4.76 Hz at swap 1 (to B, never seen) vs -1.77 Hz at swap 2
(back to A). The return dip is smaller in 8/8 seeds (Wilcoxon p=0.0078, the minimum for n=8), and
the 120 s integrated deficit is -439 vs -177 Hz·s. Decomposed at swap 2, every neuron holding A
*speeds up* (+5.83 Hz, n=21, range +4.4 to +7.8) and every neuron on B slows down (-6.33 Hz,
n=35, range -9.2 to -3.8), with no overlap. The population response to a change therefore splits
into a mismatch part (neurons tuned to what just left) and a recognition part (neurons still
holding what just came back). **Confound, not removed here:** swap 2 is later than swap 1 and hits
a population that's already split (2-3 neurons still on A), whereas swap 1 hits one committed
entirely to A. The decomposition shows the smaller dip comes mechanically from the holders. It
doesn't show that "familiar" in general produces a small dip. The novel-C runs (swap 2 goes to a
never-seen block, seeds 32000-32007, in progress) are the control. They predict a novel-sized
dip at swap 2 and no speed-up in the neurons still holding A.

**Why this might matter beyond this arc (a proposal, not a finding).** Episodic check (b) is
blocked because the memory layer can't tell "abandoned" from "between its periods of relevance".
Retainers are neurons that keep a pattern through its lull and fire up when it returns. That's a
candidate context signal generated by the substrate, not designed in. Whether a downstream layer
can use it is an interface question (see `system_contract.md`), and it's only meaningful if
novel-C confirms the familiarity reading.

## 2026-09-25 — Non-stationary correlation (world swaps A→B→A): the population always tracks, but a subset keeps the old pattern (1 of 3 at N=3, 2-3 of 7 at N=7) — and the on-record prediction was not supported

**Data:** `notebooks/brian2/nonstationary_data/` (16 seed JSONs, seeds 30000-30007 at 13mV/1.5 and
30100-30107 at strong_tight_gate; `run_nonstationary_seed.py`, `run_nonstationary_batch.py`,
`analyze_nonstationary.py`; figures `nonstationary_*.png`). Per `experiment_plan_nonstationary_stdp.md`.
New glue in `src/` with tests: `spikes.build_phase_switching_input`,
`metrics.phase_aligned_per_neuron_gap`. One continuous run per seed, never reset: 3 phases x 1000s,
correlated block A (presynaptic 0-9) → B (10-19) → A, rates matched across blocks and phases, N=3,
dt=0.2ms, Cython backend, I_pert=0, everything else identical to prior competitive runs. Full
per-synapse w(t) at 1s, r(t), w_total(t), 1s spike bins saved. Exploratory: trajectories inspected
first, claims after. 16/16 completed, 0 failures, ~18 min. n=8 per point, so counts below are
descriptive, not distributional estimates.

**Prediction on record before running (from the plan): 13mV/1.5 (suppression-based) tracks the
swap FAST, strong_tight_gate (dominance-based) LAGS. Not supported.** Re-learning latency (first
time a tracking neuron's phase-aligned gap reaches 0.3 after a swap, excluding neurons that were
already positive because they retained the returning pattern): median 92s (13mV/1.5, n=24) vs 82s
(strong_tight_gate, n=29), Mann-Whitney p=0.50. If anything the reliable point had more slow
re-learners (>=100s: 10/24 vs 5/29, Fisher p=0.069, suggestive at best), the opposite direction
from the prediction. Worth noting this also does not repeat the perturbation-testing ranking
(there 13mV/1.5 was the easier one to flip by direct weight injection): under input change the two
points look alike at the population level.

**What actually happened — a division of labour, in both points, in nearly every swap.**
- Phase 1: all three neurons learn A within ~100-200s (phase-aligned gap 0.55-0.8; some tier
  structure, e.g. 0.56 vs 0.75).
- Each swap: every neuron's phase-aligned gap drops to about -0.6 to -0.8 at once (weights
  unchanged, world inverted), then within ~50-100s the neurons that re-learn the new pattern climb
  to ~0.77. Typically two of three do; some are staggered much later (up to 374s at 13mV/1.5, one
  545s outlier at strong_tight_gate).
- **Exactly one neuron does not re-learn: it keeps the old pattern and its weights harden toward
  saturation** (in most cases the aligned gap drifts from about -0.8 toward -1.0, i.e. old-pattern
  weights to the ceiling, new-pattern weights to the floor; a few stay near -0.77). At 13mV/1.5 this is 16/16 swaps (exactly one retainer
  every time); at strong_tight_gate 13/16, the other 3 being slots where all three neurons re-learned
  (seeds 30103 and 30107 at swap 1, 30106 at swap 2).
- **The retainer is a member of the previous phase's top tier:** 15/16 (13mV/1.5) and 13/13
  (strong_tight_gate) of the single-retainer swaps, against about 9/16 and 7.3/13 expected by chance
  given tier sizes. **Correction made after a cleaner check:** that top-tier count is the weaker
  test where tiers are wide (strong_tight_gate has many near-ties). Comparing previous-phase gaps
  at swap 1 only (where every neuron starts positive, so the comparison is not biased by the
  neuron that held the returning pattern): 13mV/1.5 retainers 0.752 vs trackers 0.607 (Mann-Whitney
  p=0.0004, clear); strong_tight_gate 0.753 vs 0.698 (p=0.34, not distinguishable at n=6 vs 18).
  So "the most entrenched neuron is the one that does not follow the world" is supported at
  13mV/1.5 and only suggestive at strong_tight_gate. (`analyze_retainers.py`; the swap-2 version of
  this comparison is deliberately not reported, it is biased.)
- Retainers are NOT silenced: late-phase rate ~12-15 Hz vs ~18 Hz for the trackers. They keep firing
  on their old (now uncorrelated) inputs while holding the old weights.
- **Consequence at the return swap (phase 3):** whenever a retainer existed in phase 2, the returning
  pattern was already represented, so recovery was instant for that neuron: 8/8 at 13mV/1.5, 6/8 at
  strong_tight_gate (both misses are the two seeds where no neuron retained A through phase 2). The
  neuron that tracked B in phase 2 then becomes the new retainer of B. So after two swaps the
  population holds both patterns at once.
- Homeostatic scaling survived the swaps: `w_total` stayed within 9.6-10.4 (target 10) in every seed,
  max deviation 0.40 within 60s of a swap. The swap is not a new failure mode for it.

**What this does and doesn't say.**
- At the population level the system is NOT locked in: in 16/16 seeds at both points the
  correlated-input representation was re-acquired after each swap within minutes. The
  fast-permanent-settling seen at 13mV/1.5 under stationary input (Test A, N-scaling step 4) does not
  mean it cannot follow a changing world; settling was permanence of a hierarchy under fixed input,
  not inability to relearn.
- At the neuron level the most-entrenched neuron does lock in, and that is what supplies memory of
  the earlier pattern. Stability and plasticity end up divided between neurons rather than traded
  off within one — nothing in the mechanism was designed to do this.
- Not a savings claim: the instant recovery is retained weights, and the slope-versus-cold-start
  comparison the plan asked for was not computed. Descriptive at n=8 per point; no test of why a
  neuron becomes the retainer beyond prior strength.

**N=7 follow-up (13mV/1.5 reference, inhibition scaled with `scale_inhib_for_n` to 4.33 mV
per connection as in the N-scaling runs; seeds 31000-31007, same 3x1000s protocol, 8/8 completed,
~9 min).** Question: does "exactly one retainer" hold at larger N, or scale with population?
- **It is not one, and it does not scale linearly either: 2 or 3 of 7 neurons keep the old pattern
  at every swap** (7 swaps with 2, 9 with 3, none with 0, 1 or more than 3; mean 2.56 of 7 = 37%,
  vs 1 of 3 = 33% at N=3). Roughly a constant fraction of the population, not a fixed count of one
  and not everyone. Tight across seeds.
- Same selection rule: at swap 1 the retainers were the stronger neurons beforehand (previous-phase
  gap 0.772 vs 0.661 for the trackers, n=21 vs 35, Mann-Whitney p<0.0001). Same firing signature:
  retainers ~12.2 Hz vs ~17.9 Hz, active but slower.
- Instant recovery at the return swap: 2-3 neurons already held the returning pattern in 8/8 seeds,
  and that number equals the seed's swap-1 retainer count every time. The old pattern is carried
  intact by the retainers through the whole intervening phase.
- Re-learning speed: median 84s, similar to N=3 (92s/82s). But the slow tail is much longer at N=7:
  18/50 re-learners took >=100s and the slowest 977s (nearly the whole phase). It is concentrated at
  swap 2: all swap-1 re-learners at N=7 finished within 181s, whereas swap-2 re-learners (only 1-2
  neurons per seed) took 62-977s, median ~210s. The few neurons that must move a second time are the
  slow ones. (Swap-2 re-learners were also slower at N=3, e.g. 239-374s, but the effect is larger
  here.)
- Not established: why the fraction is about a third (e.g. how it depends on the inhibition
  normalization or on N beyond 3 and 7), and only the 13mV/1.5-equivalent point was run at N=7. The
  swap-2 slowness is descriptive; nothing here tests its cause. n=8.

---

