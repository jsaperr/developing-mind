# Experiments Log

Hopfield / two-layer-memory / episodic-layer work (this file). Brian2/SNN phase work is in
`experiments_brian2.md` — split out at the phase boundary since it's a genuinely different
skill (differential equations, not tensor ops) with different failure modes.

## 2026-07-19 — Episodic layer (v1-v5): consolidated summary

Closing summary for the episodic-layer thread — the five entries below (v1 through v5) are
kept in full as the detailed record; this is a synthesis on top, not a replacement.

**What's validated:**

- **Pruning works.** Variable-size `X` creates and evicts entries correctly — 87-90% of
  created patterns get pruned across the natural-stream runs, memory size stays bounded
  (10-40 alive, oscillating, never runaway), no monopolization at any point (v1).
- **The eviction gate is coverage-fixed.** v1's gate only engaged when 2+ candidates were
  eligible simultaneously (1.7% of real evictions — structurally idle almost always). v2
  replaced the peer-comparison gate with one anchored to distance past the eviction threshold
  itself (`staleness_over`), which applies to any number of eligible candidates. Result: 100%
  of evictions now gated (mean `g_evict=0.879`).
- **The gate's core claim is validated against a properly calibrated falsification test.** A
  genuinely consolidated entry (`w_char` in the 5-8 range, matching real sustained
  revisitation, not a short burst) gets real, proportionate, *bounded* protection from
  eviction — 14.5x the staleness margin of a fresh, unconsolidated entry — but is not rescued
  indefinitely; the gate's influence visibly collapses (`g_evict` dropping toward 0) as
  staleness becomes unambiguous, and the entry is still eventually evicted (v3). This is a
  second, structurally different confirmation of "strength breaks ties, never overrides a
  clear call," after the retrieval-side result — evidence it's a transferable design law for
  this architecture, not a one-off fix for one softmax.

**What's open, and why it's a real stopping point, not an unfinished patch:**

- **Check (b) — does graded primacy/recency ordering survive variable-size X — is
  unresolved.** Not because the gate is broken: a core phase-dominant pattern (pattern 1) was
  evicted 106 steps before its own dominant phase began, but the gate behaved exactly as
  designed given the information available to it (an unconsolidated, unambiguously-stale
  entry gets no protection — correct behavior in isolation) (v4).
- **The actual gap is upstream of the gate and more fundamental:** the system has no way to
  distinguish "this pattern is quiet because it's been abandoned" from "this pattern is quiet
  because it's between its own periods of relevance." A creation-anchored grace window (v5)
  was tried as a stopgap and *correctly diagnosed as solving the wrong axis* — mathematically,
  not just empirically, since the failing pattern wasn't new (created at step 0, same as
  everything else), it was old-but-mid-lull. No value of a creation-timer parameter fixes
  this without turning it into something structurally different from what was proposed
  ("protect core patterns specifically" rather than "protect new entries").
- **This is genuinely blocked on architecture that doesn't exist yet** — some notion of
  context/phase-awareness, so the system can tell "over" from "between beats." Not attempting
  further patches (fake phase-signal, or shrinking the test until it stops testing anything
  real) — both would be worse than leaving this open and clearly documented. **Before any
  future attempt at check (b), first check whether context/phase-awareness has been built
  elsewhere in the system (e.g. as part of the metacognitive layer or curiosity signal) — if
  not, it's not ready to be re-attempted yet.**

---

## 2026-07-19 — Episodic layer v5: grace-window bandaid doesn't touch this failure — wrong axis, not a weak value

**Notebook:** `notebooks/hopfield/episodic_layer_v5_grace_window.ipynb`

**Fix applied (Jasper's, via Claude web chat, explicitly framed as a stopgap, not principled
infrastructure):**

```
grace_window = 100  # steps since creation; arbitrary, not derived from data
eligible_for_eviction = (staleness > eviction_threshold) and (steps_since_creation >= grace_window)
```

An entry younger than `grace_window` since creation cannot be evicted regardless of staleness
or `w_char` — not scored at all. Explicitly labeled: not principled, not related to the future
working-memory sub-brain, just meant to unblock testing check (b).

**Re-ran the exact v4 schedule (3 core patterns, phases `[0,1,2,0]`, 400 steps each, same
filler stream, same seed) with only the grace window added. Result: identical to v4, down to
the exact step number.** Core pattern 1 was evicted at step 294, same as before — the grace
window changed nothing.

**Root cause, confirmed mathematically, not just observed empirically:** all 3 core patterns
are created at step 0 — same as each other. `grace_window=100` means their protection expires
at step 100. Pattern 1's actual eviction happened at step 294 (194 steps *after* its grace
period already ran out) and its own dominant phase doesn't even start until step 400 (300
steps after grace expires). **The grace window protects against being judged too soon after
creation. Pattern 1's problem is a completely different axis: it's not new, it's old but
legitimately under-visited during a period that isn't actually "yet" for it.** A
creation-anchored timer cannot address a problem that isn't about creation time at all — this
isn't a case of the bandaid being set too small (e.g. `grace_window=100` being insufficient
and needing to be, say, 350); it's structurally the wrong variable. Any `grace_window` value
that would happen to cover step 294 would have to extend well past pattern 1's *entire* first
off-phase (400 steps), at which point it stops being a "new entry" grace period in any
meaningful sense and just becomes "don't evict core patterns for their first ~2 phases" —
different from what was specified and proposed.

**Not tuned to force a pass** — per instruction, reporting this plainly rather than treating
"same failure" as license to bump `grace_window` until it happens to clear 294 by coincidence,
which would be curve-fitting to one seed's specific eviction step, not fixing the actual gap.

**Verdict:** the stopgap as specified doesn't unblock check (b). The real distinction the
architecture needs isn't "how long has this existed" but something closer to "is this pattern
currently between two periods of relevance, or has it simply never mattered" — which is much
closer to the "schedule-aware staleness" direction already flagged as a future, more
principled option (knowing *why* something's quiet, not just *that* it is) — not something a
flat creation-timer can express. Check (b) remains open and un-remediated.

---

## 2026-07-19 — Episodic layer check (b): FAILS — a core pattern got evicted before its own dominant phase

**Notebook:** `notebooks/hopfield/episodic_layer_v4_check_b.ipynb`

**Question:** the one item left open since v1 — does the graded primacy/recency `w_char`
ordering from the fixed-X phase experiments (`two_layer_sweep.ipynb`) survive once X is
variable-size with eviction happening alongside it? Reused the exact fixed-X schedule (3 core
patterns, dominant probability 0.7, phases `[0, 1, 2, 0]`, 400 steps each), unchanged mechanism
(v2's coverage-fixed gate, v3's validated math), plus a modest one-off filler stream
(`new_pattern_prob=0.05`) so there's genuine eviction traffic alongside the phase structure.

**New risk stated explicitly before running, per the message:** a phase-dominant pattern gets
almost no querying during phases where it isn't dominant (only its diluted share of the
minor-probability pool) — long enough, and its own staleness could cross the eviction
threshold before its next phase even arrives, something that structurally couldn't happen in
fixed-X where nothing ever left. Failure criteria stated up front: monopolization returning,
or a core pattern evicted before its phase completes.

**Result: this is the failure case, confirmed directly, not inferred.**

- **Core pattern 1 was evicted at step 294** — during phase 1 (dominant=pattern 0), **106
  steps before phase 2 (pattern 1's own dominant phase, starting at step 400) even began.**
  It never got a chance to consolidate — its `w_char` was still 1.0-1.7 at eviction (visible
  directly in the `w_char` plot: pattern 1's curve simply stops at step 294, mid-climb, while
  patterns 0 and 2 both complete their full staged growth curves through their phases).
- **Root cause:** during phase 1, pattern 1 had only its share of the 25%
  "other"-probability pool, split across however many patterns happened to be alive at each
  moment (growing from 3 to ~14-15 by step 294 due to the filler stream) — roughly a 2%/step
  visit probability by that point, an expected ~50-step gap between visits, but with real
  variance: a rough estimate puts a non-trivial (order 10-15%) chance of at least one unlucky
  150+-step silent gap occurring somewhere within a 400-step off-phase. Not a bizarre fluke —
  a real, structurally-expected failure mode once eviction and phase structure coexist.
- **Consequence for the original question:** the 3-way graded-ordering comparison is
  confounded — pattern 1 simply isn't there to compare. What's left: pattern 0 (first +
  returned) ended at `w_char=8.527`, pattern 2 (most recent dominant phase) at `w_char=7.263`
  — consistent with the expected primacy/recency shape as far as it goes, but this is not a
  validated instance of the full result, since one of the three data points is missing because
  it got pruned, not because it lost a fair strength competition.
- No monopolization among the 15 survivors (top/second `w_char` ratio 1.17) and memory size
  stayed bounded (3 → ~39 → 15) — those parts are fine, but moot given the core failure.

**Verdict: FAILED, per the criteria stated in advance — not spun as a partial success.** The
eviction gate itself worked exactly as designed in this scenario (pattern 1 had no `w_char` to
protect it, so it was evicted once unambiguously stale, same correct behavior validated in
v3) — the *design gap* is upstream of the gate: nothing in the current write/promotion logic
gives a pattern any protection, credit, or "it'll matter later" status before it's actually
been consolidated. The gate did its job correctly on a pattern that, by the only signal this
system currently has (frequency), looked exactly like every other never-consolidated one-off.
**Not remediated this round** — per instruction, not quietly widening the staleness threshold
or adding special protection to make this pass without saying so. This is a real, open
architectural question (something like a probationary period, or a lower bar during a
pattern's "youth", or accepting this as expected behavior for a system with no foresight)
that needs an explicit decision, not a unilateral fix.

---

## 2026-07-19 — Episodic layer v3: falsification test redesigned and passed cleanly

**Notebook:** `notebooks/hopfield/episodic_layer_v3_falsification.ipynb`

**Why this round exists:** v2 confirmed the eviction-gate *coverage* fix (100% of evictions
gated, mean `g_evict=0.879`), but the gate's actual claim — elevated strength delays an
ambiguous eviction without rescuing an unambiguous one — was still unconfirmed, because v2's
hand-designed test was miscalibrated (A's 20-visit burst only reached `w_char=1.129`, far
below the ~8 real core patterns reach; the "fresh" contrast pattern D got accidentally
over-consolidated). Coverage-fix code untouched this round — only the test itself changed.

**Redesign:** (1) burst A with **continuous, dynamically-checked** revisitation — loop until
`w_char` actually crosses 6.0, not a fixed visit count; (2) keep B genuinely weak (1 visit);
(3) ambiguous-tie phase, A and B go stale together; (4) introduce D **fresh, right at the
decisive-comparison point** (not pre-existing/drifted), light visits only (2), so it stays
low-`w_char`; (5) **explicit separation check before running the decisive comparison** — abort
and report rather than proceed on contaminated inputs if A and D aren't clearly separated.

**Falsification criteria, stated up front:** validated if A is eventually evicted once
`staleness_over` clears a healthy margin despite high `w_char` (proportionate, bounded delay,
not indefinite immunity); falsified if A survives anomalously long past D's eviction point,
or is never evicted within the window.

**Results — clean pass, first attempt this time:**

- Burst: 227 continuous visits reached `w_char[A]=6.002` (target >=6.0, core-pattern range).
- Ambiguous tie: **B evicted at step 383, A survived** (staleness reset to 11 — A got revisited
  incidentally near the end of this phase — `w_char` had grown further to 7.378 during the
  wait, via ongoing consolidation). Correct, expected outcome.
- Separation check: A `w_char`=7.386 vs. D `w_char`=1.000 (>7x, well clear of the 2x
  requirement) — no test-design failure this time, proceeded to the decisive comparison.
- **Decisive comparison:** D evicted at `staleness_over=2` (`w_char=1.165`, essentially no
  protection — score crossed zero almost immediately past threshold). **A evicted at
  `staleness_over=29`** (`w_char=7.937`, `g_evict=0.408` at the moment of eviction) — **a
  14.5x larger margin than D's**, real and measurable, but bounded: A did NOT survive
  indefinitely. By the time A was evicted, `g_evict` had already decayed from ~1.0 down to
  0.41 — strength's influence was visibly collapsing as staleness became unambiguous, exactly
  the intended shape.

**Verdict: validated, cleanly, per the criteria stated in advance.** The eviction gate does
what it was built for — a genuinely consolidated entry gets real, proportionate protection
from eviction while its staleness is still arguably ambiguous (14.5x the margin a fresh entry
gets), but that protection is bounded and collapses once staleness is no longer ambiguous, not
indefinite immunity for anything with high `w_char`. Combined with v2's coverage-fix result
(gate engaged at 100% of evictions, not just rare ties), the episodic layer's eviction
mechanism is now fully validated end to end: it prunes (check a, from v1), doesn't monopolize
or explode (check c, from v1), is gated at essentially every decision (v2), and the gate
demonstrably changes outcomes in the direction intended (check d, this round). Still open,
unchanged from v1/v2: whether graded primacy/recency survives variable-size X (check b) — the
natural-stream schedule still lacks the phase structure to test that, not attempted this round
since scope was strictly the falsification test redesign.

---

## 2026-07-19 — Episodic layer v2: coverage-gap fix works cleanly; my falsification test scenario didn't

**Notebook:** `notebooks/hopfield/episodic_layer_v2.ipynb`

**Fix applied (Jasper's, via Claude web chat):** v1's eviction gate only engaged when 2+
candidates were eligible simultaneously (1.7% of real evictions). Fix: gate against distance
past the eviction threshold itself, not distance to a peer — applies uniformly whether there's
1 eligible candidate or several, closing the "field of one" gap:

```
staleness_over = (staleness - eviction_threshold).clamp(min=0)
g_evict = 1 / (1 + staleness_over / gap_scale_evict)
eviction_score_i = staleness_over_i - g_evict_i * strength_bonus * (w_char_i - 1)
# evict the eligible candidate with the highest positive eviction_score; if none is positive, evict nobody this step
```

**Coverage-fix metric — the primary ask, and it's a clean, unambiguous win:** re-ran the exact
same natural-stream schedule (4000 steps, same seed). **100% of evictions (173/173) now have
`g_evict > 0.1` at eviction time**, vs. 1.7% under v1 — mean `g_evict` across all evictions was
0.879. The gate is no longer structurally idle; strength gets real, measurable influence at
essentially every eviction decision now, not just the rare multi-candidate tie. Total evicted
dropped slightly (173/197, 87.8%, vs. v1's 178/197, 90.4%) — consistent with genuine (if often
small) protection now applying broadly instead of only in rare ties. Checks (a) and (c) still
hold cleanly: pruning still works, no monopolization (top/second `w_char` ratio 1.01), bounded
memory size, ceilings never approached. Check (b) is still the same open item as v1 — core
patterns (final `w_char` 8.02-8.12) survive and consolidate fine but remain statistically
indistinguishable from each other, since the natural-stream schedule still has no phase
structure to test graded primacy/recency against.

**Falsification test — re-run, but it still didn't produce a clean answer, for a different,
now well-diagnosed reason:** after the "ambiguous tie" phase (A gets a 20-visit burst, B gets
one visit, both go stale together), **both A and B were gone**, not just B as intended —
essentially the same surface outcome as v1, but the mechanism is now understood precisely
rather than being another instance of the same bug:

- A's 20-visit burst only raised `w_char` to 1.129 — nowhere near the ~8 the core patterns in
  the natural-stream run reach after thousands of steps of real, sustained revisitation.
  `w_char` growth is deliberately slow by design (`decay_char=0.0005`,
  `consolidation_rate=0.01`) — a 20-step burst is far too short to produce a strongly
  differentiated entry. At `w_char=1.129`, `strength_bonus=10`, the maximum protection
  (`10*(1.129-1)=1.29`, at `g_evict=1`) is overwhelmed by `staleness_over` within 1-2 steps of
  crossing the threshold — real but negligible protection, not a meaningful delay.
- Separately, D (meant to be a "barely stale, low-consolidation" contrast pattern) was queried
  on 370 of the 400 steps in the second phase before finally being left alone — enough sustained
  revisitation to build its own `w_char` up to 8.386, essentially as consolidated as the core
  patterns, not the low-strength contrast it was supposed to be.
- Net effect: A never survived into the second phase to reach the intended A-vs-D comparison
  at all (confirmed directly — A's staleness plot has zero data points, evicted before the
  second phase's loop even started).

**This is a test-design flaw on my part, not a mechanism failure** — the burst intensity and
D's exposure schedule were miscalibrated relative to how slowly `w_char` actually grows by
design, not evidence the gate doesn't work. Not redesigning and re-running a third time
unilaterally in this pass; flagging it as still open rather than forcing a cleaner-looking
result, consistent with not tuning after seeing an unfavorable outcome.

**Verdict:** the coverage-gap fix itself is validated cleanly and was the primary ask — mission
accomplished on that front. The specific hand-designed falsification scenario needs a redesign
(a much longer/more realistic consolidation burst for the "strong" entry, and a genuinely fresh
contrast entry introduced right at the comparison point rather than pre-aged through heavy
querying) before it can actually demonstrate the "elevated strength doesn't rescue an
unambiguously stale entry" case cleanly. Not attempted this round.

---

## 2026-07-19 — Episodic layer v1 (Option B): pruning works, but the ambiguity gate has an uncovered edge case

**Notebook:** `notebooks/hopfield/episodic_layer_v1.ipynb`

**Question:** per Jasper's message and `principles.md`, first build of the episodic layer —
`X` becomes variable-size, new patterns append as stand-in experience, frequency
(`w_fast`/`w_char`) is still the only write/promote signal. Retrieval reuses the exact
validated mechanism unchanged (`k=0.5, w_fast_max=10, gap_scale=0.1534` — one of the strong
points from the ambiguity-gated retrieval sweep, not re-tuned here). Pruning built
ambiguity-gated from the start: eviction eligibility is raw staleness
(`steps_since_last_retrieval_win`, an unmediated counter) crossing a threshold (150); among
the top-2 most-stale eligible candidates, `w_char` is only allowed to break the tie when
their staleness gap is small (`g_evict = 1/(1+gap/gap_scale_evict)`), not override a clearly
unambiguous case.

**Falsification criteria stated up front:** gate validated if an elevated-`w_char`,
unambiguously-most-stale entry still gets evicted despite its strength; falsified if a naive
strength-weighted-but-ungated policy would make a *different* (worse) call on the same state.

### Natural-stream run (4000 steps, 3 "core" recurring patterns + a stream of one-off patterns)

- **Check (a) — pruning works:** 197 patterns created, 178 evicted (90.4%), 19 alive at end.
  Unpromoted low-frequency entries do get pruned rather than sitting forever.
- **Check (c) — no explosion/monopolization:** `w_char` range [1.0, 8.18] (ceiling 10, never
  approached), `w_fast` range [1.01, 3.07] (ceiling 10). Top/second `w_char` ratio 1.01 — no
  monopolization, bounded memory size oscillating 10-26 alive patterns throughout, no runaway.
- **Check (b) — partial, and honestly a test-design gap, not a mechanism failure:** the 3 core
  patterns all survived with high, comparable `w_char` (8.18/8.09/8.12) and didn't destabilize
  while surrounded by heavy eviction traffic — genuinely useful confirmation. But they came out
  essentially indistinguishable from each other (no primacy/recency grading), because the
  natural-stream schedule draws uniformly among the 3 core patterns with no staged-phase
  structure (unlike the fixed-X phase experiments that produced the original graded-ordering
  result). This test wasn't built with the phase structure needed to reproduce that finding —
  it only shows core patterns *can* consolidate together without exploding, not that graded
  primacy/recency survives the transition to variable-size X. Open item, not answered here.
- **Notable, unplanned finding:** of 178 evictions, only 3 (1.7%) were "ambiguous"
  (multi-candidate tie-break) — 175 were single-candidate cases. Checking eligibility every
  step means entries essentially never queue up together; the tie-breaking logic almost never
  actually engages in organic conditions. This directly set up the failure below.

### Check (d) — the designed falsification test: FAILED to demonstrate the intended comparison

Hand-designed scenario: pattern A gets a consolidation burst (elevated `w_char`), pattern B
gets one visit, both go stale together (ambiguous tie, `w_char` should break it and keep A) —
then A is left stale much longer while fresh pattern D only goes stale a little (unambiguous
gap — does A's `w_char` wrongly rescue it?).

**What happened instead:** after the ambiguous-tie phase, *both* A and B were gone
(`"A alive: False, B alive: False"`) — not just B as intended. By the time the decisive
A-vs-D comparison ran, A had "already evicted earlier," so both the gated and naive-ungated
policies returned `None` (nothing left to compare) — the test produced no evidence either way,
not a clean pass or fail.

**Root cause, confirmed via an isolated minimal repro (not just inferred):** a lone eligible
candidate — no second candidate to compare against — falls through `prune_step`'s
`len(eligible)==1` branch, which evicts **unconditionally, with no ambiguity gating at all**.
Repro: single high-`w_char` entry (3.02, well above baseline) with a fresh distractor kept
around so staleness could actually advance — evicted the instant its staleness crossed the
threshold (step where `staleness=151`), `w_char` doing nothing to protect it. This is exactly
what happened to A: once B was evicted (leaving A as the sole eligible candidate), A was
evicted on the very next check, before the scenario ever reached the intended A-vs-D
comparison.

**This is a real design gap, not a coding bug** — `prune_step` does exactly what it's written
to do. The gap is architectural: the eviction-ambiguity gate, as specified, only ever activates
when 2+ candidates are eligible simultaneously. Retrieval never has this problem (the softmax
is always over every currently-stored pattern, inherently multi-candidate), but eviction can
legitimately face "a field of one" — an entry that's the only one currently past threshold —
and the gate as designed provides **zero protection** in that case, regardless of how
consolidated the entry is. Combined with the 1.7% ambiguous-eviction rate above: **as
implemented, the eviction gate does not actually mitigate familiarity-biased pruning in the
common case.** Per the falsification criteria stated up front, this doesn't cleanly satisfy
either outcome — it's neither validated nor cleanly falsified, because the intended comparison
never got to run. Not patched and re-run to force a cleaner result; reporting as found, per
`principles.md`'s own "how to fail correctly" guidance (root-cause before reframing scope,
which is exactly what the isolated repro was for).

**Open item for next round, not attempted here:** the gate needs to also apply when there's a
single eligible candidate — e.g., gate eviction against a fixed "how stale is stale enough to
evict outright regardless of strength" ceiling, only falling back to unconditional eviction
once staleness clears that higher bar, rather than the threshold-crossing moment itself always
being unconditional. Not implemented this round — flagging for explicit discussion rather than
picking a fix unilaterally, matching how v1 rate-modulation's fix was proposed and approved
before building, not improvised.

---

## 2026-07-19 — Ambiguity-gated retrieval bias: decouples savings from content-match

**Notebook:** `notebooks/hopfield/two_layer_ambiguity_gated.ipynb`

**Question:** v2's rate-modulated savings fix works but exposed a real tradeoff — settings
that produce genuine savings also degrade content-match (down to 21-40%, vs. 85-94% at
`k=0`). Per Jasper's message, tested option (c) from the very first two-layer failure
analysis (never tried until now): gate how much strength (`w_fast`/`w_char`) gets to
influence retrieval by how ambiguous the raw content similarities already are, rather than
picking a fixed point on the tradeoff.

**Mechanism:** `gap = top1_similarity - top2_similarity` (computed from raw content
similarity alone, before any strength term); `g = 1 / (1 + gap/gap_scale)` — small gap
(ambiguous) → `g` near 1 (strength fully active), large gap (clear content winner) → `g` near
0 (content dominates); `biased = beta*similarities + g*(log(w_fast) + char_weight*log(w_char))`.
`w_fast`/`w_char` update rules are **unchanged from v2** — the gate only touches retrieval
bias, not the learning dynamics. Two outcomes stated up front: (A) gating decouples the two
effects cleanly, or (B) content-match still degrades meaningfully even with gating, meaning
the tradeoff is fundamental to this mechanism family, not fixable by bias-shaping alone.

**`gap_scale` grounded in real data, not guessed:** sampled 2000 queries with the same
generation scheme used throughout this series; empirical top1-top2 gap median = 0.2557
(mean 0.2507, std 0.1229, 10th/90th percentile 0.079/0.412). Swept `gap_scale` at
0.2x/0.6x/2x/6x the median (0.0511, 0.1534, 0.5115, 1.5344) — spans roughly an order of
magnitude centered on real data.

**Tested at the three v2 settings that had both genuine savings and degraded content-match:**

| k | w_fast_max | gap_scale | min content-match | mean g | p1 slope | p4 slope | savings real? |
|---|---|---|---|---|---|---|---|
| 0.5 | 10 | 0.0511 | **95.5%** | 0.215 | 0.0257 | 0.0266 | Yes (thin) |
| 0.5 | 10 | 0.1534 | **94.5%** | 0.421 | 0.0267 | 0.0286 | Yes |
| 0.5 | 10 | 0.5115 | 83.0% | 0.687 | 0.0282 | 0.0312 | Yes |
| 0.5 | 10 | 1.5344 | 61.8% | 0.862 | 0.0293 | 0.0325 | Yes |
| 1.0 | 10 | 0.0511 | **95.5%** | 0.215 | 0.0271 | 0.0266 | **No** (only failure) |
| 1.0 | 10 | 0.1534 | **95.5%** | 0.421 | 0.0283 | 0.0287 | Yes (thin) |
| 1.0 | 10 | 0.5115 | 89.0% | 0.687 | 0.0300 | 0.0315 | Yes |
| 1.0 | 10 | 1.5344 | 76.2% | 0.862 | 0.0312 | 0.0334 | Yes |
| 1.0 | 20 | 0.0511 | **95.7%** | 0.215 | 0.0305 | 0.0356 | Yes |
| 1.0 | 20 | 0.1534 | **95.2%** | 0.421 | 0.0320 | 0.0386 | Yes |
| 1.0 | 20 | 0.5115 | 83.5% | 0.687 | 0.0341 | 0.0429 | Yes |
| 1.0 | 20 | 1.5344 | 61.5% | 0.862 | 0.0357 | 0.0460 | Yes |

For reference, v2 ungated at these same three points: min content-match was 40.3%/54.3%/37.7%
respectively (all with savings real, but with clearly degraded retrieval).

**Result: this is Outcome A, cleanly.** Content-match recovers to 83-96% across 11 of 12
gated combinations (vs. 21-54% ungated) — at the most aggressive gating (`gap_scale=0.0511`,
mean `g=0.215`, strength suppressed to ~22% influence on average), content-match actually
**exceeds** the original `k=0` un-modulated baseline (85-94%) at two of the three test points
(95.5%, 95.7%). Savings survives in 11 of 12 combinations — only one combination
(`k=1.0, w_fast_max=10, gap_scale=0.0511`) flipped to a (very thin) non-savings result
(0.0271 vs 0.0266, essentially noise-level). Phase-2 transient sanity check (gated vs.
ungated at `k=0.5, w_fast_max=10, gap_scale=0.1534`) shows the same basic decaying shape,
just far smaller in magnitude (peak ~0.2 vs ~1.0) — nothing structurally new introduced.

**Honest pattern in the data, not cherry-picked:** there's a clear, monotonic trade inside the
gated regime itself — as `gap_scale` increases (weaker gating, `mean_g` rising from 0.215 to
0.862), content-match decreases monotonically back toward the ungated baseline, while the
savings margin (phase4 slope minus phase1 slope) generally widens. So gating doesn't eliminate
the underlying tension, it relocates it to a new, much more favorable knob — `gap_scale` now
lets content-match be pushed close to (or above) baseline while retaining most of the savings
signal, rather than forcing a hard binary choice between "high savings, ~30% content-match"
and "no savings, ~90% content-match" as v2 did.

**Verdict:** validates the "strength breaks ties, not clear mismatches" design principle from
the original two-layer failure analysis. This is a genuinely good architectural result, not a
tuning trick — gating the *retrieval-bias application*, not the update rule, decoupled a
property of learning (savings) from a property of retrieval (content fidelity), which is
exactly what the mechanism was supposed to do. Not picking a "final" `gap_scale`/`k`/`w_fast_max`
combination here — that's still a deliberate downstream choice, not something to default into
from this table alone.

---

## 2026-07-19 (v2) — Rate-modulated savings, fixed: contained, but savings trades off against content-match

**Notebook:** `notebooks/hopfield/two_layer_rate_modulated_savings_v2.ipynb`

**Fix applied (direct instruction from Jasper):** give `w_fast` the same headroom treatment
`w_char` already has, plus an explicit cap on the rate-modulation multiplier itself rather
than relying on it being indirectly bounded through `w_char_max`:

```
multiplier = (1 + k * (w_char - 1)).clamp(max=max_multiplier)   # max_multiplier=3.0
increment_fast_effective = increment_fast * multiplier
fast_headroom = (w_fast_max - w_fast).clamp(min=0) / w_fast_max
w_fast = w_fast + decay_fast * (1 - w_fast) + increment_fast_effective * retrieval_weights * fast_headroom
```

**Step 1 — containment check (`k=0` vs `k=0.5`, `w_fast_max=10`, same 5-phase schedule):**

- Phase-1 peak `w_fast[0]` (where the v1 runaway actually fired): k=0 → 3.52, k=0.5 → 6.15.
  Both well clear of the `w_fast_max=10` ceiling — no runaway. (v1's k=0.5 hit 26.3 here.)
- Content-match recovered strongly: k=0.5 → 81.2/40.3/69.2/82.5/82.0% across the 5 phases
  (vs. v1's collapsed 78.8/**7.8/6.2**/67.0/**9.3**%). Also notable: the `k=0` baseline itself
  improved with `fast_headroom` added (86.5/72.5/87.7/89.5/94.2%, vs. the pre-fix `k=0`
  baseline's 81.2/44.7/78.0/81.7/86.0%) — capping `w_fast`'s ceiling moderates `log(w_fast)`'s
  contribution to the retrieval bias even with no rate modulation active, a positive side
  effect worth noting, not something being claimed as the point of this fix.
- Ordering/monopolization checked on **both** layers this time (v1's `w_char`-only check is
  what let the `w_fast` runaway go undetected): all ratios 1.03-1.75, nowhere near the 5x
  threshold, on both `w_char` and `w_fast`.
- **Genuine savings appeared for the first time in this whole series:** k=0.5 phase-4 raw
  slope (0.0321) > phase-1 cold-start raw slope (0.0302) — the actual unconfounded comparison,
  not the steps-to-90% metric that was contaminated every previous time. `k=0` control still
  shows no savings (0.0148 vs 0.0282), consistent with everything found before — rate
  modulation is doing real work here, not noise.

**Step 2 — 3x3 sweep, `k` in {0.25, 0.5, 1.0} x `w_fast_max` in {5, 10, 20}, `w_char_max=10`
fixed:**

| k | w_fast_max | min content-match | char ratio | fast ratio | p1 slope | p4 slope | savings real? |
|---|---|---|---|---|---|---|---|
| 0.25 | 5  | 70.5% | 1.10 | 1.33 | 0.0226 | 0.0151 | No |
| 0.25 | 10 | 34.3% | 1.08 | 1.33 | 0.0292 | 0.0267 | No |
| 0.25 | 20 | 13.2% | 1.22 | 1.57 | 0.0331 | 0.0331 | No |
| 0.5  | 5  | 68.8% | 1.05 | 1.44 | 0.0232 | 0.0165 | No |
| 0.5  | 10 | 40.3% | 1.03 | 1.75 | 0.0302 | 0.0321 | **Yes** |
| 0.5  | 20 | 21.0% | 1.03 | 1.94 | 0.0343 | 0.0434 | **Yes** |
| 1.0  | 5  | 79.8% | 1.04 | 1.48 | 0.0244 | 0.0179 | No |
| 1.0  | 10 | 54.3% | 1.02 | 1.93 | 0.0321 | 0.0343 | **Yes** |
| 1.0  | 20 | 37.7% | 1.02 | 2.48 | 0.0369 | 0.0474 | **Yes** |

**No monopolization anywhere in the grid** (max ratio 2.48, vs. the 5x threshold) — the two
headroom caps do not fight each other into reopening lock-in at any tested combination. The
"does a low `w_fast_max` bottleneck `w_char`'s own growth" concern was checked directly
(pattern 1's own `w_char` peak across `w_fast_max` = 5.61/6.11/5.45 mean at wfm=5/10/20) and
isn't strongly supported — no clear monotonic starvation effect.

**But a real tradeoff showed up that wasn't the thing being specifically checked for:**
content-match degrades monotonically as `w_fast_max` increases, *independent of whether
monopolization triggers* — at `k=0.25, w_fast_max=20` content-match drops to 13.2% with the
ratio metric still reading "not monopolized" (1.22). And genuine savings only appears in the
`k >= 0.5, w_fast_max >= 10` corner of the grid — exactly the region where content-match is
also at its most degraded (21-40%, well below the ~85-94% `k=0` baseline, though nowhere near
v1's ~7% collapse). **Savings and clean content-match retrieval are in tension across this
parameter space, not independent.** The monopolization ratio (top/second `w_char` or `w_fast`)
is not a sufficient proxy for retrieval-fidelity health on its own — it stayed low everywhere
even where content-match was clearly degraded.

**Verdict:** the `w_fast` headroom + capped-multiplier fix does what it was built to do —
contains the runaway, no lock-in reopened, monopolization ruled out across the swept grid.
Genuine (slope-based, not scale-contaminated) savings is achievable, a first for this series.
But it isn't a clean, free win: the settings that produce real savings (`k>=0.5`,
`w_fast_max>=10`) are the same settings where content-match is meaningfully worse than the
unmodulated baseline. This is a tradeoff to weigh deliberately, not a solved problem — no
parameter choice from this grid should be adopted as "the" setting without that tradeoff being
an explicit decision, not a default.

---

## 2026-07-18 — Rate-modulated savings: lock-in reopened through a new pathway

**Notebook:** `notebooks/hopfield/two_layer_rate_modulated_savings.ipynb`

**Question:** savings was confirmed absent on the plain two-layer mechanism — `w_char` only
ever affected retrieval bias, never acquisition rate. Per Jasper's message, tested a rate
modulation fix on the validated (fixed-X) substrate, isolated from the episodic-layer build:

```
increment_fast_effective = increment_fast * (1 + k * (w_char - 1))
w_fast = w_fast + decay_fast * (1 - w_fast) + increment_fast_effective * retrieval_weights
```

`k=0.5`, first pass. Explicit named risk going in: high `w_char` → faster `w_fast` growth →
faster `w_char` growth (via the existing consolidation term) → even faster `w_fast` growth —
a second feedback loop the `w_char_max` ceiling does nothing to bound, since it caps `w_char`'s
value, not the rate at which `w_fast` can run away before `w_char` saturates.

**Setup:** both `k=0` (control) and `k=0.5` (test) run in the same notebook, identical
seed/schedule (5 phases: pattern 0, 1, 2, 0-return, pattern-3 cross-pattern control),
identical `w_char_max=10`/`consolidation_rate=0.01` from the validated sweep, so the
comparison is apples-to-apples. Three outcomes were stated before running anything: (A) real
savings, ordering and content-match both hold vs. the `k=0` control; (B) lock-in reopened —
ordering breaks or content-match degrades or `w_fast` shows anomalous front-loading, even if
final numbers look fine; (C) no effect, savings still absent.

**Results — this is Outcome B, unambiguously, and worse than the original bug:**

- `w_fast[0]` (k=0.5) climbed from 1.0 to **26.3 within phase 1 alone** (the very first
  400-step dominant phase) — nowhere near the old self-limiting plateau of ~5.2 that `k=0`
  produces. Phase 4 then starts already at 25.9 and stays essentially flat (peak 27.3) because
  there's nothing left to reacquire — it never actually left.
- Content-match rate (the direct lock-in diagnostic) collapsed across every non-pattern-0
  phase: phase 2 dropped from 44.7% (k=0) to **7.8%** (k=0.5); phase 3 from 78.0% to **6.2%**;
  phase 5 (the cross-pattern control, previously unaffected at 86.0%) collapsed to **9.3%**.
  These numbers are comparable to or worse than the original naive-`w_char` bug
  (7-8% content-match) that the saturating headroom fix was built to solve — rate modulation
  reopened essentially the same failure mode through `w_fast` instead of `w_char`.
- `w_char`-based ordering check alone did **not** catch this cleanly (top/second ratio 3.14,
  under the 5x monopolization threshold) — this is exactly why the message insisted on
  checking content-match directly rather than trusting the existing saturation to protect
  things: `w_char` stayed bounded as designed, but `w_fast` — which was never given an
  equivalent ceiling — is what blew up, and the softmax bias reads both terms, so a runaway
  in either one is sufficient to break retrieval fidelity.
- The front-loading check (binned 25-step slopes within phase 4 specifically) did **not** show
  acceleration — slopes decreased monotonically (0.0206→0.0136→0.0156→0.0015→0.0048→0.0015).
  That's because the runaway had already fully happened during phase 1, not phase 4 — the loop
  fires on the very first dominant phase, as soon as `w_char[0]` starts climbing off baseline.
  Worth noting as a methodology lesson: the check built to catch "front-loading in the return
  phase" wasn't looking in the right place; the damage was done three phases earlier.
- Savings itself, measured honestly: k=0.5 phase-4 raw slope (0.0147) is *lower* than k=0's
  phase-4 slope (0.0206) — rate modulation made the genuine acquisition-rate measure of
  savings worse, not better. The steps-to-90% metric still showed an apparent "speedup"
  (108 steps vs 319) but that's the same scale-contamination trap as before, now more extreme:
  phase 4's net increase is trivial (1.45) because it starts 25.9 out of a ~27 ceiling.

**Verdict:** rate-modulated savings as implemented (`k=0.5`, no ceiling on `w_fast`) is unsafe
— it reopens lock-in through a pathway the existing `w_char` saturation fix does nothing to
address, and does so more severely than the bug it was meant to fix, while not even producing
a genuine savings signal (raw slope got worse, not better). **Per the conditional in the
original task, did not proceed to the parameter sweep** — the mechanism didn't hold clean, so
sweeping it across `(w_char_max, consolidation_rate)` would only characterize a broken
mechanism, not validate a working one.

**Root cause, for whoever picks this up next:** `w_fast` has no equivalent of `w_char`'s
`headroom` term — nothing caps how large it can grow, and the rate-modulation multiplier
(`1 + k*(w_char-1)`, ranging up to 5.5x at `w_char_max=10`) is applied to an otherwise-unbounded
increment. A `w_fast`-side saturation term (or a cap on the modulation multiplier itself, or
a smaller `k`) would be the natural next thing to try — not attempted here since this was
scoped as characterization of the first-pass `k=0.5` design, not a tuning pass.

---

## 2026-07-18 (resolved) — seed check: phase-2 multi-bump transient is noise, not oscillation

**Notebook:** `notebooks/hopfield/two_layer_seed_check.ipynb`

**Open item being closed:** the phase-2 misretrieval transient (from the diagnostics run
below) showed a non-monotonic multi-bump shape — peak at step 78, dip, secondary bump around
step 180-200 — flagged as "possibly real competitive oscillation or single-seed noise, not
distinguished." Per Jasper's message, this needed resolving before starting the episodic-layer
build, as a characterization-only task (no fix, no tuning).

**Setup:** same baseline params validated in the sweep (`w_char_max=10`,
`consolidation_rate=0.01`), same pattern set (seed=42 for X). Reran phases 1-2 only (pattern 0
dominant, then pattern 1 dominant — phase 2 dynamics don't depend on anything after it) across
10 different simulation seeds (0-9), overlaid the windowed misretrieval-rate curves, and ran
`scipy.signal.find_peaks` (prominence >= 0.15, ignoring the first 20 steps since that's just
the curve starting near its max) to detect secondary bumps in each seed's curve.

**Results:**

- All 10/10 seeds showed at least one secondary bump — but bump count per seed ranged from 1
  to 6, and first-secondary-bump position ranged from **step 21 to step 307** (mean 110.3,
  std 89.4 — a standard deviation nearly as large as the mean, out of a 400-step phase).
- No clustering: positions are scattered across almost the entire phase width, not concentrated
  around any particular step.
- The overlay plot shows a consistent *broad* decay envelope across all 10 seeds (all reach
  near-zero misretrieval by roughly step 300-380, corroborating the earlier ~250-350-step
  fixed-duration finding), but the specific bump timing/shape riding on top of that envelope is
  seed-dependent noise, not a repeatable signature.

**Verdict — per the falsification criteria set in advance: this is noise, not real competitive
oscillation.** A genuine dynamical signature (e.g., `w_char[1]` crossing through the range
where it actively contests `w_char[0]` for softmax dominance at a roughly fixed point in the
consolidation trajectory) would show bump positions clustering at a similar absolute step
across seeds. They don't — the spread (21-307) covers nearly the entire phase. Per the
resolution rule set in the task: **dropped from findings language.** The multi-bump shape
should not be described as a finding going forward. The one thing that *does* survive from the
original diagnostics run is the fixed-duration transient result itself (~250-350 absolute
steps, independent of phase length) — that's corroborated again here by the consistent decay
envelope across seeds, and stays in the findings as-is.

**Status: resolved. No further action queued on this item.**

---

## 2026-07-18 — Savings metric fix, phase-2 transient characterization, parameter sweep

**Notebooks:** `notebooks/hopfield/two_layer_diagnostics.ipynb` (tasks 1-2), `notebooks/hopfield/two_layer_sweep.ipynb` (task 3)

**Context:** three follow-ups on the saturating-headroom fix below, relayed by Jasper from a
separate Claude web chat: (1) the original savings check was contaminated by residual
`w_fast` elevation, not real evidence of relearning; (2) the phase-2 misretrieval transient
(56.6% vs 13.6% in phase 3) needed measuring before deciding if it's a bug; (3) the corrected
result's graded ordering came from one arbitrary parameter pair and needed a robustness check.

### Task 1 — savings via slope comparison, not threshold

Replaced the absolute-threshold check with three rate-matched (70%-dominance) climb
comparisons: pattern 0's phase-1 climb (true cold start), pattern 0's phase-4 climb (return),
and a new phase 5 where pattern 3 becomes dominant for the first time ever (cross-pattern
cold-start control, ruling out pattern-0-specific idiosyncrasies).

- **steps to reach 90% of the curve's own net increase:** phase1=209, phase4=175, phase5=308.
  By this metric, phase 4 "wins" — reaches its own plateau faster than either cold start.
- **raw initial slope (linear fit, first 50 steps):** phase1=0.0360, phase4=0.0206, phase5=0.0118.
  By this metric, phase 4 is *slower* than its own true cold start (phase 1) and only faster
  than the cross-pattern control (phase 5).
- The overlay plot makes the disagreement visible directly: phase 1's curve starts lower but
  is visibly steeper, and crosses above phase 4's curve around step ~130-140, well before
  phase 4 plateaus.

**Why the metrics disagree:** phase 4 starts already elevated (2.01 vs phase 1's 1.02), so its
net increase to reach the same plateau (~4.5) is smaller (2.53 vs 4.26) — 90% of a smaller
target is mechanically easier to reach in fewer steps even at an unchanged or slower rate.
The steps-to-90% metric is contaminated by this scaling effect, not fully clean either.

**Honest verdict — no clean "savings confirmed":** the underlying reacquisition *rate*
(slope) does not show real savings — if anything it's slower on return than the original
cold start. The only thing "faster" about phase 4 is that it needs a shorter absolute climb
because it never fully lost its prior gains, which is really just re-stating the persistence
result from the corrected run, not a distinct relearning-speed finding. This should not be
reported as "savings" without that caveat attached.

### Task 2 — phase-2 transient characterization (measurement only)

- **Within-phase shape:** phase 2's misretrieval rate is *not* a simple decay — it starts at
  0.85, briefly rises to a peak of 0.95 around step 78, dips, has a secondary bump near
  step 180-200 (up to ~0.75-0.9), a smaller tertiary bump near step 280-320, and only settles
  near 0 by step ~350-400. Phase 3 shows the same multi-bump shape at much smaller amplitude
  (peak 0.50 at step 32, settles faster). Single-seed run — these bumps could be reproducible
  dynamics (competitive oscillation while `w_char[1]` catches up to `w_char[0]`) or
  trajectory-specific noise; not distinguished without repeated-seed averaging.
- **Phase-length scaling:** re-ran at phase_len=200 and phase_len=800 (same phase pattern).
  Aligned by *absolute* step count, all three curves (200/400/800) overlap closely for the
  first ~150-250 steps and resolve to near-zero by roughly the same absolute step count,
  independent of total phase length — the 800-length run's curve is flat and resolved for its
  entire second half. Aligned by *fraction of phase elapsed*, the curves clearly do not
  overlap (they'd need to for length-proportional scaling). **Conclusion: the transient is a
  roughly fixed-duration relaxation in absolute steps (~250-350), not something that scales
  with phase length.** This is measured, not tuned — no fix applied.

### Task 3 — parameter sweep: is the graded ordering robust?

Swept `(w_char_max, consolidation_rate)` across baseline (10, 0.01) plus 4 one-at-a-time
variants: low/high ceiling (5, 20) and slow/fast consolidation (0.005, 0.02).

- **Ordering held at every setting:** final `w_char` order was `[pattern0, pattern2, pattern1,
  {3,4}]` (primacy of pattern 0 + recency of pattern 2, the two never-dominant patterns lowest)
  across all 5 parameter pairs, including both ceiling extremes and both rate extremes.
- **No monopolization anywhere:** top/second-place ratio ranged 1.10-2.42 across the grid (vs.
  the >5x threshold used to flag monopolization, and vs. the naive unbounded version's ratio of
  ~13x). The high-ceiling variant (20) came closest to breaking the graded shape (ratio 2.42,
  pattern 0 reaching 15.18) but still preserved the same relative ordering.
- **Verdict:** the qualitative shape (graded, non-monopolizing persistence) is not an artifact
  of the one parameter pair used in the original corrected run — it holds across a 4x range on
  the ceiling and a 4x range on the consolidation rate. This is now a reasonably trustworthy
  qualitative result, not a "worked once" fluke.

**Overall status:** task 2 and task 3 are now clean. Task 1 (savings) is properly measured but
does *not* support a clean "savings" claim — that framing should be dropped or explicitly
caveated in any future write-up rather than repeated as a finding.

---

## 2026-07-17 — Naive two-layer (fast episodic / slow character) strength split

**Notebook:** `notebooks/hopfield/two_layer_consolidation.ipynb`

**Question:** the single-layer basin-strength experiments in `hopfield_exploration.ipynb`
found a stability-plasticity tension no single decay constant escapes (decay=0.01 → full
erasure of early patterns; decay=0.001 → permanent primacy lock-in). Does splitting strength
into a fast layer (`w_fast`, decay=0.02, driven directly by retrieval) and a slow layer
(`w_char`, decay=0.0005, only nudged via a consolidation term when `w_fast` is elevated) —
the fix flagged as "not yet implemented" — actually resolve this?

**Setup:** same 5-pattern/dim=64 setup as `hopfield_exploration.ipynb`. Replayed the existing
3-phase developmental arc (attention dominated by pattern 0, then 1, then 2, 400 steps each)
plus an added 4th phase returning attention to pattern 0, to test for savings/relearning vs.
the two known failure modes. Retrieval bias combined both layers: `log(w_fast) + log(w_char)`.

**Results:**

- `w_char[0]` grew monotonically and unboundedly across all 1600 steps (final value 47.1),
  never caught by `w_char[1]` or `w_char[2]` even during their own dedicated 400-step
  dominant phases (peaks of only 3.4 and 3.4 respectively). `w_fast[0]` never fully
  receded during phases 2-3 either (stayed >5 throughout).
- Diagnostic (does the retrieval mechanism's winner match the environment's actual query?):
  phase 1 (querying pattern 0) — 80.5% match, expected. Phase 2 (querying pattern 1) — only
  **7.8%** match; pattern 0 won 100% of the non-matching steps. Phase 3 (querying pattern 2) —
  **6.2%** match, pattern 0 won 100% of non-matching steps again.
- "Savings" check (steps to re-cross `w_fast>=1.5` on return to pattern 0) showed 0 steps —
  but this isn't relearning savings, it's an artifact: `w_fast[0]` never dropped below
  threshold in the first place, because retrieval kept getting hijacked back to it during
  phases 2-3.

**Verdict — this is a worse failure mode than the one it was meant to fix, not a fix:**

1. Once `w_char[0]` grew large enough, `log(w_char[0])` in the softmax bias overpowered the
   actual content-similarity term, so the retrieval mechanism stopped being content-addressed
   at all during phases 2-3 — it kept "retrieving" pattern 0 regardless of which pattern the
   query content actually pointed at. That misretrieval then fed `w_fast[0]` more increments,
   which fed `w_char[0]` more consolidation: an unbounded positive feedback loop, not a
   stability-plasticity balance.
2. This is a **sharper** failure than the original decay=0.001 single-layer result. The
   single-layer case at least let queries retrieve correctly (the failure was that early
   strength never decayed enough to be caught by later patterns' *strength*); here the early
   pattern actively hijacks retrieval away from patterns that should clearly win on content
   match alone.
3. **Root cause is architectural, not a tuning issue:** feeding `w_char` into the same
   softmax competition as content-similarity, with unbounded growth and comparable weight to
   `w_fast`, guarantees this outcome for any consolidation_rate > decay_char once strength
   diverges once. Fixing this needs either (a) `w_char` decoupled from the retrieval
   competition entirely — a readout/observational layer, not a driver — or (b) `w_char`
   bounded (saturating, not `(1-w_char)`-toward-unbounded-target) so it can influence but
   never dominate the content term, or (c) retrieval driven by content similarity alone, with
   strength layers only biasing *which near-ties* get broken, not overriding clear mismatches.

**Next steps (not started, flagging for discussion):** re-run with `w_char` excluded from the
retrieval softmax (pure observer/consolidation-only role) to see if persistence-without-lock-in
is recoverable once the feedback loop is architecturally cut.

---

## 2026-07-17 (correction) — saturating consolidation growth, same notebook

**Notebook:** `notebooks/hopfield/two_layer_consolidation.ipynb` (same notebook, `update_two_layer` patched)

**What changed:** Jasper diagnosed the actual bug from the numbers above — the consolidation
increment `consolidation_rate * (w_fast - 1).clamp(min=0)` was flat, independent of `w_char`'s
current size, and `decay_char * (1 - w_char)` only scales with distance from 1, not with how
unsustainable the value already is, so nothing capped growth once it started compounding
(textbook unbounded-Hebbian-growth). Fix: an Oja's-rule-style saturating `headroom` term —
`headroom = (w_char_max - w_char).clamp(min=0) / w_char_max`, multiplied into the consolidation
increment, with `w_char_max = 10.0`. `w_char` stays fully in the retrieval softmax (not
decoupled) — growth just can't run away anymore.

**Corrected results:**

- Content-match rate recovered substantially: phase 2 (querying pattern 1) went from 7.8% → 44.7%;
  phase 3 (querying pattern 2) went from 6.2% → **78.0%**.
- Plasticity is now real: pattern 1 and pattern 2 reach `w_char` peaks of 5.06 and 5.94 during
  their own phases, comparable to pattern 0's own peak (7.15/8.53), instead of being permanently
  dwarfed by it (previously 3.4 vs 13.4).
- Final `w_char` = `[8.53, 5.97, 6.24, 3.45, 3.46]` — pattern 0 retains a persistent edge
  (genuine character-layer memory of early experience) without the unbounded blowout
  (previously `[47.1, 3.55, 3.37, 2.53, 2.43]`).
- The `w_char` plot now shows a clean, graded picture: each pattern's character strength grows
  on a saturating curve during its own dominant phase, and the final ordering (0 > 2 > 1 > 3,4)
  blends primacy (pattern 0, first) and recency (pattern 2, most recently dominant) rather than
  one pattern eating all available strength.

**Remaining caveats — not fully clean yet:**

1. Phase 2 still shows 56.6% misretrieval to pattern 0 (vs. 13.6% in phase 3) — there's a
   transient right at the phase 1→2 boundary where `w_char[0]` is already near its phase-1 peak
   (close to the ceiling) while `w_char[1]` is still near baseline, so the imbalance briefly
   dominates until pattern 1 catches up. Visible in the `w_char` plot as the crossing point
   around step 550-600.
2. The "savings" check (steps to re-cross `w_fast>=1.5` on return to pattern 0) still reads
   0 steps, but this is not clean evidence of relearning savings: `w_fast[0]` was already at
   1.99 (above the 1.5 threshold) at the *start* of phase 4, because it never fully decayed
   below threshold during phases 2-3. Would need a stricter test (threshold set above the
   phase-2/3 trough, or measuring rate-of-increase rather than absolute-threshold-crossing) to
   actually demonstrate savings rather than residual elevation.

**Verdict:** the saturation fix (not decoupling) is the right shape of fix — two-layer strength
splitting can produce genuine bounded persistence + plasticity when the slow layer's growth
saturates. Not yet a fully clean result; the phase-2 transient and the contaminated savings
metric are real open issues, not polish.

---

## 2026-07-07 — Infini-attention (linear) vs Hopfield (iterative) retrieval on ambiguous/partial queries

**Notebook:** `notebooks/hopfield/infini_vs_hopfield_retrieval.ipynb`

**Question:** Does iterative energy-based Hopfield retrieval outperform Infini-attention's
single-shot linear associative read specifically on ambiguous/partial queries, and does it
scale better with capacity? Formulas used exactly as specified (Munkhdalai et al 2024 linear
rule with `sigma = elu(x)+1`; Ramsauer et al 2020 Hopfield energy/update rule, generalized to
hetero-associative K/V so both backends read the same stored pairs — iterate in key-space,
read out value only after convergence).

**Setup:** N random unit-norm keys in `dim_k=64`, one-hot values (label = argmax decode).
Three query types (clean / partial-noised / ambiguous-blend-of-two), plus a capacity sweep
N = 5..640 tracking mean separation Δᵢ alongside accuracy.

**Results:**

- Partial-query accuracy (N=5, 200 trials): linear 23%, Hopfield 81%.
- Ambiguous-query behavior (N=5, 200 trials): linear margin ~0.003 (fully flat, no signal),
  Hopfield margin ~0.69, energy ~-0.38 (vs ~-0.50 on clean queries — a real, measurable
  "less resolved" signal). Hopfield's top-2 weights landed on the true blended pair 79% of
  the time (linear: 17.5%).
- Capacity curve: Hopfield accuracy starts at 76% (N=5) and monotonically collapses to 0%
  by N≈160-320, tracking the shrinking mean separation Δᵢ (0.81 → 0.62 over that range).
  Linear accuracy is ~floor (chance-level, ~1/N) at every N tested, including N=5.

**Honest caveat — this is not a clean win for Hopfield:** the linear backend's near-chance
performance held even on *clean* queries, which shouldn't happen if the linear read were
doing anything query-dependent. Traced it to the `elu(x)+1` feature map applied directly to
raw random keys: `phi(Q)·phi(K_i) ≈ dim_k + sum(Q) + sum(K_i) + Q·K_i`. The constant `dim_k`
offset (64) and each key's own feature-sum (`sum(K_i)`, O(8)) dwarf the actual similarity term
`Q·K_i` (O(0.1) for random unit vectors in 64-d). So the linear rule, run on *untrained* raw
keys, is structurally near-blind to the query — it's not that it loses to Hopfield on
ambiguity specifically, it barely reads the query at all in this setup. Real Infini-attention
depends on learned Q/K projections to shape representations so this additive term isn't
dominant; this synthetic test has no learned projections, so the comparison isn't fully
apples-to-apples for the linear side. This is a property of the formula on raw features, not
an implementation bug (verified: consistently landing on whichever pattern has the largest
`sum(K_i)` regardless of query).

**Verdict:**
1. On 3c (ambiguous queries) and at small-N capacity, Hopfield clearly outperforms and gives
   a genuinely distinguishable "unresolved" signal (lower margin, higher energy) rather than
   confidently guessing wrong — this part supports the hypothesis, cleanly.
2. On the capacity question (test 4), Hopfield does **not** show durable better scaling — it
   converges to the same 0%-accuracy failure mode as linear once N grows enough to erode
   separation, exactly the metastable-averaging failure predicted by Ramsauer's own theorem.
   No free lunch at high N.
3. The magnitude of Hopfield's win at low N is likely overstated by the linear backend's
   near-chance floor being partly an artifact of no learned projections, not solely a
   Hopfield-vs-linear architectural difference. A fairer comparison would need learned (even
   lightly trained) Q/K projections for the linear side before concluding Hopfield's iterative
   read is intrinsically superior on ambiguity — right now the experiment mostly shows
   "single-shot linear read on raw features doesn't address by content at all," which is a
   weaker and different claim than "Hopfield resolves ambiguity better than a working linear
   reader would."

**Next steps (not started, flagging for discussion):** to isolate the ambiguity-resolution
question cleanly, would want a version where the linear backend gets *some* minimal learned
or at least similarity-preserving projection so it actually addresses by content before
re-running 3c/4 — otherwise the current result mostly demonstrates a degenerate linear
baseline rather than a fair head-to-head.

---

## 2026-07-07 (correction) — confound fix + fair rerun of 3c/capacity

**Notebook:** `notebooks/hopfield/infini_vs_hopfield_retrieval.ipynb` (same notebook, linear backend patched)

**What changed:** applied the minimal fix flagged above — a **fixed (not learned)** centering
projection `P = I - (1/dim_k) 1 1^T` applied to both Q and K before the `elu(x)+1` feature map,
i.e. mean-subtract each vector so `sum(Q)=0` and `sum(K_i)=0`. This directly zeroes the additive
bias terms identified in the original confound, leaving `Q·K_i` as the only thing that varies
across stored patterns. No learned parameters, no change to the Hopfield backend, no new query
types.

**Verification (done before rerunning 3c/4, per the honesty gate):** clean-query accuracy across
1413 trials spanning N=3..50 (dim_k=64): **100%**. Fix confirmed before touching anything else.

**Corrected results:**

- Partial-query accuracy (N=5, 200 trials): linear **79%**, Hopfield 81% — now roughly tied,
  not a 23-vs-81 blowout.
- Ambiguous-query behavior (N=5, 200 trials): linear margin now ~0.0001 (still near-zero, but
  for a different reason than before — it now genuinely reads the query and reports a flat,
  symmetric split when the query truly is ambiguous). Linear's top-2 choices matched the true
  blended pair **100%** of the time (up from 17.5%). Hopfield: margin 0.69, top-2-is-true-pair
  79% of the time — meaning 21% of the time Hopfield *confidently* (high margin) converges to a
  pattern that isn't even one of the two ambiguous candidates.
- Capacity curve (dim_k=64, partial-query accuracy, 150 trials/point):

  | N | Δᵢ | linear acc | hopfield acc |
  |---|---|---|---|
  | 5   | 0.81 | 0.74 | 0.76 |
  | 10  | 0.78 | 0.62 | 0.63 |
  | 20  | 0.77 | 0.53 | 0.37 |
  | 40  | 0.74 | 0.45 | 0.12 |
  | 80  | 0.70 | 0.33 | 0.02 |
  | 160 | 0.67 | 0.22 | 0.00 |
  | 320 | 0.64 | 0.18 | 0.00 |
  | 640 | 0.62 | 0.13 | 0.00 |

  Linear now degrades gracefully and clearly **outperforms** Hopfield from N=20 upward; Hopfield
  hits a 0% floor by N=160 while linear still retrieves correctly ~13-22% of the time even at
  N=320-640.

**Verdict — this reverses the original conclusion, not confirms it:**

1. The original "Hopfield wins on ambiguity" result was **entirely an artifact of the broken
   linear baseline**. With a fair (content-sensitive) linear read, the two backends are
   statistically tied on partial-query accuracy at low N (79% vs 81%), and linear's behavior on
   the truly-ambiguous query (3c) is arguably *better calibrated* than Hopfield's — it reports a
   flat, symmetric, low-confidence split 100% of the time on the correct pair, whereas Hopfield's
   sharper softmax dynamics produce higher-margin (falsely confident) answers that miss both true
   candidates 21% of the time.
2. On capacity scaling (test 4), the fixed linear backend **clearly beats** Hopfield, not the
   other way around. Hopfield's iterative read is more brittle under crowding, not more robust:
   once separation degrades, its softmax sharpening accelerates collapse into a single
   (frequently wrong) metastable state, while linear's unnormalized additive read degrades
   smoothly.
3. **The original wedge idea — that iterative Hopfield-style retrieval outperforms single-shot
   linear retrieval on ambiguous/partial queries and scales better — is not supported by this
   corrected, fair comparison.** If anything, the evidence now points the other way on capacity.
   This is a clean negative for the hypothesis as originally framed, not a tuning artifact — it
   should be treated as a real result, not spun as validating the original wedge.

**Caveats on this corrected result itself:** the centering fix is a generic, un-learned patch
targeted at removing one specific bias artifact discovered in dim=64, N-vs-random-key space; it
is not a claim that linear associative reads are *in general* superior to modern Hopfield
retrieval — real Infini-attention's linear read lives inside a trained transformer with learned
Q/K/V projections and a compressive-memory-with-delta-rule update this test didn't use, and real
deployed Hopfield/attention layers aren't run at fixed beta=8 with no learned temperature. Treat
this as: within this narrow synthetic KV setup, with both backends given a fair (content-sensitive)
read, the iterative Hopfield advantage hypothesized at the outset did not materialize.
