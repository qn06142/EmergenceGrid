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

- **Sim bugs (9 scour fixes):** oracle+RL couldn't open gate until fixed. Real.
- **Credit assignment:** verified fine via --diag_train. Real.
- **Exposure / G+C reward lever (`reward_preset='gc'`):** makes gate-task gradient learnable.
- **Reward magnitude:** non-monotonic ceiling (~0.64). Real finding.
- **Architecture fix (needed-trait skip):** wrong_trait_mut_rate 0.97→0.42. Real win.
- **Gate curriculum (runtime TH_GATE):** lowered bar is settable; at 0.6 agents DO build
  strength and mutate — but the gate still doesn't open (corrected count = 0).
- **More agents (n=8):** fixed the strength-building (agents reach 0.91–1.00 vs 0.51 for n=4
  @0.95). Real win — but didn't produce gate openings.

## 3. What was BROKEN (and now fixed)

- **eval_metrics gate counter** — counted GATE(6)→any-non-6. Fixed to per-cell 6→0 AND
  excludes agent-occupied cells (verified by unit test + live run → 0).
- **eval_metrics `ag.tr.strength`** — raw cpp_sim Agent has no `.tr`; crashed on gate-adjacent
  seeds. Fixed via `dump_agents()`.
- **Funnel launcher** — passed `--food_seed` not `--seeds`. Fixed.
- **init_ckpt resume** — requires identical agent count (n=4 ckpt → n=8 model crashes).
  Train from scratch at target n.

---

## 4. TRUE measurement (corrected eval_metrics, 5 seeds each)

| Checkpoint | Bar | Agents | gate_opened | EATEN|reached (end-to-end) |
|------------|-----|--------|-------------|--------------------------|
| gc_curric  | 0.6 | 4 | **0** | 0.0254 |
| gc_anneal_n8 | 0.95 | 8 | **0** | 0.0036 |

max_strength 0.97–0.98 (agents build strength fine). wrong_trait_mut_rate ~0.74 (mutate,
but not the gated chain to completion).

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
