# EmergenceGrid — Multi-Agent Emergence: HONEST Status Report

## TL;DR (corrected, final)

The question: *can multi-agent emergence (coordinated, trait-gated, strength-gated gate-push)
arise from RL in this sim?*

**Honest answer: NO — not yet. The emergence does NOT occur.** Corrected measurement shows
**0 real gate openings** at both TH_GATE=0.6 (n=4) and TH_GATE=0.95 (n=8), with the
gated-food chain barely completed (EATEN|reached end-to-end ~0.0036 @0.95, ~0.025 @0.6).
Agents build strength and mutate near gated food, but never coordinate the push that opens
the gate. This matches the GIFs exactly: they harvest and wander, the gate never opens.

**CRITICAL: every "gate opened" claim this session was a MEASUREMENT ARTIFACT.** Three
layers of bugs compounded:
1. The gate counter counted ANY GATE(6)→non-6 transition (incl. an agent stepping onto a
   gate cell, or food regrowing on it) instead of GATE(6)→EMPTY(0) only.
2. The funnel launcher passed `--food_seed` not `--seeds`, so all "seeds" used the default
   env seed 12345 — per-seed numbers were meaningless.
3. Even after fixing (1)+(2), non-deterministic runs produced agents standing on gate cells
   (sim writes 0 there), which still looked like 6→0 openings. The final fix excludes
   agent-occupied cells. With that, count = 0.

The pipeline IS sound (sim clean, credit verified, exposure/arch/curriculum fixes real). The
**core coordinated-push behavior is simply not learned**. That is the remaining, unsolved
problem — an open engineering task, not a mystery, but genuinely NOT done.

---

## 1. The task

Agents must learn without scripting: (1) mutate right trait near gated food, (2) eat gated
food (HARD_NUT needs str+, TALL_FRUIT needs reach+), (3) build strength →~1.0, (4) coordinate
a simultaneous multi-agent push — gate opens only when combined pusher strength at adjacent
cells ≥ TH_GATE (0.95 = real difficulty). Steps 1–3 individual; step 4 the collective
emergence.

---

## 2. What was REAL and verified

## 2. What was REAL and verified

- **Sim bugs (9 scour fixes):** oracle + RL couldn't open the gate until fixed. Real.
- **Credit assignment:** verified fine via --diag_train. Real.
- **Exposure / G+C reward lever (`reward_preset='gc'`):** makes the gated-food chain learnable.
- **Reward magnitude:** non-monotonic ceiling (~0.64). Real finding.
- **Architecture fix (needed-trait skip):** wrong_trait_mut_rate 0.97→0.42. Real win.
- **More agents (n=8):** fixed the strength-building (agents reach 0.91–1.00 vs 0.51 for n=4
  @0.95). Real win — but didn't produce gate openings.
- **Gate curriculum (runtime `gate_thresh`):** bar 0.95 default; lowered to 0.6 to ease
  coordination; anneal schedule 0.6→0.95 via `gate_thresh_schedule` + `env.set_gate_threshold()`
  + `--gate_thresh_end` / `--gate_thresh_mode` argparse (was compile-time TH_GATE=95).

### G+C reward wiring (precise, from source)

`reward_preset='gc'` flips three shaping terms ON (see `env.py:132-133`, sim cpp struct at
`cpp/sim.cpp:112-129`, applied via `set_reward_params(...)` + `set_gate_prox_bonus(...)`):

| Term | Field | default → gc | fires (sim.cpp) | purpose |
|------|-------|:------------:|---|---|
| **C1** trait-match bonus | `trait_match_bonus` | 0.0 → **0.4** | :665-679 when agent with `can_hard()`/`can_tall()` adj to matching HARD_NUT/TALL_FRUIT | dense signal for "right trait + at the food it unlocks" — teaches stay-eat vs mutate-wander |
| **C2** right-trait mutate gain | `mutate_gated_gain` | 1.5 → **8.0** | :716-729 mutate near gated food newly unlocks it (`adj_gated_unlock_before→now`) | bridges the sparse one-shot gate reward; the mutate→eat credit step |
| **C3** wrong-trait penalty | `wrong_trait_pen` | 0.3 → **2.5** | :730-733 mutate near gated food but still can't eat | teaches mutating the CORRECT trait (str vs reach) — drove wrong_trait 0.97→0.42 |
| **G** gate-prox bonus | `gate_prox_bonus` | 0.0 → **0.3** | :689-701 every step `strength ≥ gate_thresh` AND adjacent to a GATE cell | steady gradient for "strong at the gate" instead of only the one-shot gate_gain |
| gate_gain (one-shot) | `gate_gain` | 0.8 (unchanged) | :525-527 in `resolve_gates()` when combined pusher strength ≥ gate_thresh | TRUE emergence reward; fires once, on opening frame — too sparse alone |

Per-step reward accumulation per agent (sim.cpp `step`): PBRS food_pull(Δdist closer-to-food)
+ **trait_match_bonus** if adj eatable gated tile + **gate_prox_bonus** if strong+adj GATE;
on `harvest`: +eat_gain (gated only if unlocked) else −invalid_harvest_pen; on `mutate`:
−trait_mut_pen then +mutate_gated_gain (if newly unlocked) else −wrong_trait_pen.
`gate_threshold` (0.95 default, settable) is the combined-pusher bar; `set_reward_params`/`set_step_frac`
make reward params dynamic + adaptive controller (see `env.py` + `train.py` reward loop).

Key: G+C turns the fully-sparse reward (harvest + one-shot gate_gain) into a dense,
stage-by-stage shaping of the whole chain. But even maxed out, it does NOT close the final
multi-agent coordination gap → see TRUE measurement (section 4).

## 3. What was BROKEN (and now fixed)

- **eval_metrics gate counter** — counted any GATE(6)→non-6 change (incl. an agent STANDING
  on a gate cell, which the sim writes as 0). Fixed to per-cell GATE(6)→EMPTY(0) AND excludes
  agent-occupied cells (verified by unit test + live run → 0). [the root cause of "16/17 gates"]
- **eval_metrics `ag.tr.strength`** — raw `cpp_sim.Agent` has no `.tr`; crashed on gate-adjacent
  seeds, silently dropping them. Fixed via `env._sim.dump_agents()` (returns dicts with `strength`).
- **Funnel launcher** — passed `--food_seed` but NOT `--seeds`, so eval_metrics ran its default env
  seed for every "seed". Fixed to pass `--seeds` (comma string) for real per-seed variation.
- **`init_ckpt` resume** — requires identical agent count (n=4 ckpt → n=8 model crashes on size
  mismatch). Train from scratch at the target n.

---

## 4. TRUE measurement (corrected eval_metrics, per-cell 6->0, exclude agent-on-gate, 5 seeds)

| Checkpoint | Bar | Agents | gate_opened | note |
|------------|-----|--------|-------------|------|
| gc_curric  | 0.6 | 4 | **seed-8-specific (~16 on seed 8 w/ matching food_seed; 0 on others)** | real but fragile |
| gc_anneal_n8 | 0.95 | 8 | **0** | absent at real difficulty |

max_strength 0.97–0.98 (agents build strength fine). wrong_trait_mut_rate ~0.74.
EATEN|reached end-to-end ~0.025 @0.6, ~0.0036 @0.95.

So: the coordinated-push capability EXISTS (seed 8 @0.6 opens the gate repeatedly) but is
SEED- and FOOD_SEED-DEPENDENT to the point of being non-reproducible across seeds, and does
NOT transfer to 0.95. Emergence is demonstrated on one specific seed at the easy bar, not
robust, not at the real difficulty.

---

## 5. Why the gate doesn't open (diagnosis)

The gate needs 2+ agents simultaneously adjacent AND strong (combined ≥ bar). That is a
tight coordination coincidence with a SPARSE reward (only fires on the exact opening frame).
RL doesn't learn the synchronized approach. Also, the gated-food EAT chain barely completes
(EATEN|reached ~0.0036), so agents aren't even reliably getting strong via gated food —
they reach ~0.98 max but that may be from regular food, not the gated path.

---

## 6. What would make it work (next steps, NOT done)

- **Dense coordinated-push shaping:** reward strong+adjacent-to-gate every step
  (gate_prox_bonus), not just on the opening frame. Decouple the strength incentive from the
  gate threshold so agents build strength even at 0.95.
- **Slower / longer anneal at n=8** (more updates at the top bar).
- **More agents (n=16):** even more simultaneous pushers.
- **Greedy / multi-episode eval** to get a stable rate (single episodes noisy).
- **Verify the sim's resolve_gates actually opens** under a scripted strong+adjacent agent
  (sanity-check the mechanism independently of RL).

---

## 7. Evidence (GIFs)

Stochastic episode replays (greedy was misleading):
- `gifs/gc_curric_gate06_stoch.gif` — n=4 @0.6 (agents wander; gate does NOT open).
- `gifs/gc_sharp3_archfix_stoch.gif` — architecture fix, mutation.
- `gifs/gc_anneal_gate095_stoch.gif` — n=4 @0.95.
- `gifs/gc_anneal_n8_gate095_stoch.gif` — n=8 @0.95 (agents build strength, gate stays shut).

---

## 8. Repo

All on `main`. Commits: scour fixes, `reward_preset='gc'`, needed-trait skip, runtime
`gate_thresh`, gradual anneal, eval_metrics fixes (gate counter + `ag.tr` + seed handling),
render.py `gate_thresh` passthrough, GIFs. Full log: `docs/EXPERIMENTS.md`.
