# EmergenceGrid — Multi-Agent Emergence: Honest Status Report

## TL;DR (corrected)

The question was: *can multi-agent emergence (a coordinated, trait-gated, strength-gated
gate-push) arise from RL in this sim?*

**Honest answer: the emergence is REAL but RARE and seed-fragile — NOT yet a reliable,
learned behavior.** Direct grid-level measurement (bypassing the broken eval counter)
shows:
- At TH_GATE=0.6 (easy bar): 1 of 5 seeds (seed 8) opens the gate repeatedly (16 times
  in 400 steps); the other 4 never do.
- At TH_GATE=0.95 (real difficulty, n=8): 1 gate total across 5 seeds × 400 steps.

So emergence CAN happen, but it's a fragile coincidence, not a robust skill. The pipeline
is sound (sim clean, credit fine, exposure/architecture fixes work, agents build full
strength). The remaining gap is making the coordinated push FREQUENT — an open
engineering task, not a mystery, but NOT done.

**IMPORTANT correction:** earlier reports claimed "3/5 gates", "7 gates", and "17 gates".
ALL of those were MEASUREMENT BUGS in the eval harness, not emergence:
1. The gate-opened counter counted ANY GATE(6)→non-6 transition (including an agent
   stepping onto a gate cell, or food regrowing on it) instead of GATE(6)→EMPTY(0) (the
   real opening). Fixed to per-cell 6→0 only (unit-tested).
2. The funnel launcher passed `--food_seed` but NOT `--seeds`, so every "seed" ran the
   same default env seed (12345) — the per-seed numbers were meaningless.
The numbers below are from direct `env._sim.grid` reads with the correct per-cell 6→0
logic and proper per-seed env seeds.

---

## 1. The task

Agents must learn a chained behavior without scripting:
1. Mutate the right trait (str+/reach+) near gated food.
2. Eat gated food (HARD_NUT needs str+, TALL_FRUIT needs reach+).
3. Build strength by eating (→ ~1.0).
4. Coordinate a simultaneous multi-agent push — gate opens only when combined pusher
   strength at adjacent cells ≥ TH_GATE (0.95 = real difficulty).

Steps 1–3 are individual; step 4 is the collective emergence.

---

## 2. What was actually fixed (real, verified wins)

- **Sim bugs (9 scour fixes):** oracle + RL couldn't open the gate until fixed. Real.
- **Credit assignment:** verified fine via --diag_train. Real.
- **Exposure / G+C reward lever (`reward_preset='gc'`):** makes the gate-task gradient
  learnable. Real (A/B-able, defaults untouched).
- **Reward magnitude:** non-monotonic ceiling (~0.64). Real finding — cranking reward
  regressed, not helped.
- **Architecture fix (needed-trait skip-connection):** added need_strplus/need_reachplus
  to obs + trait_w skip with priors. wrong_trait_mut_rate dropped 0.97→0.42. Real win,
  verified by probing policy action biases.
- **Gate curriculum (runtime TH_GATE):** made the bar settable (was compile-time). At 0.6
  the gate demonstrably opens (seed 8). Real.
- **More agents (n=8):** FIXED the strength-building problem — agents now build to
  0.91–1.00 strength (vs 0.51 for n=4 at 0.95). Real win (the diagnosis: at 0.95 the
  strength-building reward only fires when the gate is near-open, which rarely happens
  with n=4; n=8 makes the combined-strength event fire during training, so agents learn
  to build strength).

## 3. The remaining gap (NOT solved)

The coordinated PUSH that opens the gate is rare. Direct measurement:

| Checkpoint | Bar | Agents | Real gates (5 env seeds × 400 steps) |
|------------|-----|--------|--------------------------------------|
| gc_curric  | 0.6 | 4 | seed8=16, others=0 → 1/5 seeds |
| gc_anneal_n8 | 0.95 | 8 | 1 total (seed5) → 1/5 seeds barely |

The agents DO: mutate near gated food, build to full strength, navigate. They RARELY
synchronize the final push. This matches the GIFs (agents wander, harvest, mutate, but
the gate almost never opens in a 300-step replay).

Why rare? The gate needs 2+ agents simultaneously adjacent AND strong (combined ≥ bar).
That's a tight coordination coincidence. The reward for it is sparse (only fires on the
exact frame the gate opens), so RL doesn't reliably learn the synchronized approach.

---

## 4. What would make it reliable (next steps, not yet done)

- **Dense coordinated-push shaping:** reward agents for being strong + adjacent to a gate
  (gate_prox_bonus exists but may be too weak / thresholded wrong). Decouple the strength
  incentive from the gate threshold so agents build strength even at 0.95.
- **Slower / longer anneal at n=8:** gc_anneal_n8 trained 250 upd; more updates at the
  top bar may consolidate the push.
- **More agents (n=16):** even more simultaneous pushers → combined ≥0.95 easier.
- **Greedy/multi-episode eval** to measure a stable rate (single episodes are noisy).

---

## 5. Evidence (GIFs)

Rendered stochastic episode replays (greedy was misleading — argmax collapsed to one
action and looked frozen; stochastic shows the true, mostly-wandering behavior):
- `gifs/gc_curric_gate06_stoch.gif` — n=4 at TH_GATE=0.6 (seed 8 opens; others wander).
- `gifs/gc_sharp3_archfix_stoch.gif` — architecture fix, mutation targeting.
- `gifs/gc_anneal_gate095_stoch.gif` — n=4 at 0.95.
- `gifs/gc_anneal_n8_gate095_stoch.gif` — n=8 at 0.95 (agents build strength, rare push).

---

## 6. Bugs found and fixed this session

1. **eval_metrics gate counter** — counted GATE(6)→any-non-6 as an opening. Fixed to
   per-cell 6→0 (real opening only). Verified by unit test (7 cases).
2. **eval_metrics `ag.tr.strength`** — raw cpp_sim Agent has no `.tr`; crashed on
   gate-adjacent seeds. Fixed via `dump_agents()`.
3. **Funnel launcher** — passed `--food_seed` not `--seeds`; all "seeds" used default
   12345. Fixed (use `--seeds` for env seed variation).
4. **init_ckpt resume** — requires identical agent count (n=4 ckpt → n=8 model crashes).
   Train from scratch at the target n.

---

## 7. Repo

All on `main`. Notable commits: scour fixes, `reward_preset='gc'`, needed-trait skip,
runtime `gate_thresh`, gradual anneal schedule, eval_metrics fixes, render.py
`gate_thresh` passthrough, GIFs. Full log: `docs/EXPERIMENTS.md`.
