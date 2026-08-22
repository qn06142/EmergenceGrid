# Experiments & Results

Honest account of what emerged, what didn't, and why. Metrics are **real food
collected** (`harv/step`, inventory increments), never the harvest-button rate.

## Curriculum progression

| Level | World | Targeted emergence | Result |
|-------|-------|-------------------|--------|
| 0 | open grid, free food | navigate + harvest | ✅ reliable collecting |
| 1 | + walls | path-find | ✅ reliable, symmetric up/down |
| 2 | + gated food (needs traits) | **mutate own traits → eat** | ⚠️ **partial: ~8–10% real collection** |
| 3 | + oases behind gates | mutate strength → open gate → safe food | ❌ gate-opening did not emerge in budget |
| 4–5 | + hazards / sparse | robustness | not reached |

## Headline result — Curriculum 2 trait emergence

The agent, starting with traits that cannot eat HARD_NUT/TALL_FRUIT, learns to
**mutate its own traits** and then consume gated food it could not eat before.
Across seeds (probe counting real `inv++` over 600 steps, big model):

- mutation activity: **~25–30%** of steps
- gated food actually eaten: **~8–10%** of steps (best checkpoint `ckpts/_L2b/L2b_policy_step600.pt`)
- reproducible across seeds 12345 / 777 / 2024

This is genuine emergence: the trait→food mapping is not in the loss, only in the
world structure + local reward. Example trajectory: `gifs/L2_emergence.gif`
(seed 99, ~24 eats / 300 steps).

Why ~10% and not higher: the behavior is a 3-step chain — navigate to a gated
tile, mutate the *correct* trait (50% chance of the wrong one), harvest while
adjacent. Movement + wrong-trait mutations + approach necessarily consume most of
the 64-step episode, so the eatable-step fraction is bounded well below 100%.
We treat ~10% as the practical ceiling for dense mixed gated food, not a bug.

## What we tried that did NOT help (and why)

Exhaustive variants, all measured by probe; every reward/pressure tweak **regressed**
vs the plain dense baseline:

| variant | gated eating | note |
|---|---|---|
| plain dense, 600 upd | **7.8–9.7%** | ✅ BEST |
| sparser food (density_div 100) | 0.3–0.5% | starved → harvest-spam returned |
| longer training (1500 upd) | 2.7–3.2% | over-trained, unlearned |
| trait-match shaping (+adjacent-gated bonus) | ~0% | agent wandered |
| passive hunger, soft (decay 0.05) | 2.2–4.8% | survivable without eating |
| passive hunger, strong (decay 0.13) | ~0% / 3% | collapsed (died) or worse |

Lesson: the bottleneck is **credit assignment** on the 3-step (reach → mutate →
eat) chain, not motivation. Hunger/pressure doesn't fix it — it either starves
the agent before it can navigate to far gated food, or perturbs the working
policy. We reverted the hunger/shaping experiments (constants back to 0) so the
shipped code matches the best baseline.

## Curriculum 3 — gate-opening (not yet emerged)

Setup: oases (walled safe-food pockets) with GATE entrances; a gate opens when
cumulative adjacent strength ≥ 1.10. Agent starts strength 0.35–0.5, so opening
needs ~6 strength-mutations (each +0.12, 15-step cooldown) ≈ 90 steps before any
gate reward. The chain is longer and more delayed than curriculum 2, and the L3
run (init from the L2 checkpoint, 800 updates) showed `gateopen = 0` throughout —
no gate ever opened.

Promising next steps (untested):
- **Lower `TH_GATE`** (1.10 → ~0.7) so fewer mutations are needed — shorter chain.
- **Gate-adjacency shaping**: small reward for being next to a gate with enough
  strength, bridging the credit gap (the analogous trait-match bonus failed for
  food, but gates are rarer/salient, so it may differ).
- **One gated type first**: train only HARD_NUT → mutate-strength → eat before
  adding TALL_FRUIT, halving the trait-mapping burden.

## Known gaps

- **Predators never spawn**: `resolve_predators` runs but the `predators` list is
  never populated, so curriculum 3+ has no predator pressure (only oases/gates).
- **PBRS was zeroed for curriculum ≥ 3** in both `sim.cpp` and `env.py` originally;
  fixed so PBRS fires at all curricula (annealed). Verify before relying on L4/L5.

## Metric discipline

- Report `harv/step` (real collection), never `topact[5:100%]` (harvest-button
  mash, which can be 100% while eating 0 food).
- `dist` in logs is `obs[-1]*40` (normalized); 2.0 = far, 0.5 = on food.
- Entropy (`ent`) must stay >~0.5; collapse to ~0 means single-action lock-up.

## Closed-loop adaptive rewards (experiment, 2026-08-21)

Hypothesis: static reward constants cap learning; instead, *react* to the
agent's actual failure mode each episode (closed-loop), not a blind time
schedule. Implemented in `cpp/sim.cpp` (9-counter `Diag` accumulator) +
`src/train.py` (`adaptive_reward_params()`, `--adaptive` flag) +
`src/eval_metrics.py` (rigorous multi-seed harness).

Failure-mode -> reward lever map (each nudged + clamped per update):
  1. harvest-spam (invalid/(invalid+valid) high) -> RAISE invalid_harvest_pen
  2. moves-away (move_away > move_closer)        -> RAISE food_pull + nav_alpha
  3. mutate-but-no-eat (mutate high, valid low)  -> LOWER trait_mut_pen
  4. gate-adjacent-but-weak (gate_adj high)      -> RAISE trait_match_bonus
  5. dying (dead>0)                               -> RAISE eat_gain

Results (multi-seed harness, mean +/- std over 5 seeds x 400 steps):

| config | curriculum | collect_rate | invalid_harvest_rate | mutate->eat |
|--------|-----------|--------------|---------------------|-------------|
| static | 1 | 0.045 +/- 0.030 | 0.871 +/- 0.072 | 0.297 +/- 0.167 |
| schedule(linear) | 1 | 0.071 +/- 0.018 | 0.796 +/- 0.047 | 0.797 +/- 0.275 |
| adaptive | 1 | 0.064 +/- 0.029 | 0.546 +/- 0.194 | 1.165 +/- 0.598 |
| static | 2 | 0.076 +/- 0.055 | 0.685 +/- 0.209 | 0.825 +/- 0.511 |
| adaptive | 2 | 0.071 +/- 0.034 | 0.421 +/- 0.243 | 0.740 +/- 0.327 |
| adaptive | 3 | 0.076 +/- 0.031 | 0.715 +/- 0.112 | 0.719 +/- 0.339 |

Honest conclusions (cont.):
- **L2 funnel (full instrumentation) found the real bottleneck**: the agent
  REACHES gated food (~0.20/step) and HAS traits, but barely mutates there
  (mut_near|reached = 0.029 static / 0.098 adaptive) because `harvest=+379`
  reward dominates net-negative mutate actions. The ~10% ceiling is
  **reward dominance / credit-assignment**, not navigation or trait access.
- **D reshaping (free + shaped mutation near gated) FAILED**: EATEN|reached
  0.098 (D) < 0.130 (static). Making mutation free near gated food wasn't
  enough — the 15-step cooldown + need to stay-adjacent-and-harvest is a
  behavior the policy can't sustain, and dense regular-food harvest outcompetes.

## nstep=128 horizon hypothesis -- FALSIFIED

We raised nstep 64->128 (static reward) on L2 and L3 to test whether the
episode horizon was the blocker. Funnel results (5 seeds x 400 steps):

| config | curriculum | mut_near\|reached | EATEN\|reached | wrong_trait_mut | max_strength |
|--------|-----------|-------------------|----------------|-----------------|--------------|
| static (nstep64) | 2 | 0.029 | 0.130 | 0.39 | - |
| **nstep128** | 2 | **0.135** (4.7x) | 0.139 (flat) | 0.63 (worse) | - |
| adaptive (nstep64) | 3 | 0.020 | 0.159 | 0.16 | 0.51 |
| **nstep128** | 3 | 0.020 | 0.159 | 0.16 | **0.57** (still <1.10) |

- L2: longer episodes let the agent linger+mutate at gated food (mut_near 4.7x),
  but the chain is STILL not completed (EATEN|reached flat) and wrong-trait
  flailing got worse (0.63). More attempts, no completion.
- L3: even with 128 steps, `max_strength` only reached 0.57 (< 1.10 threshold),
  `gate_chain_possible=False`, `gate_reachable 0/5`. The agent never accumulates
  strength toward the gate — it optimizes harvest-spam instead.

**Correction to the earlier "L3 is a horizon artifact" note**: the horizon was a
*necessary* condition we removed, but it was NOT *sufficient*. The actual blocker
is that the agent rationally takes the easy +15 harvest payoff over the long,
cooldown-gated, fragile mutate->eat loop. This is a **temporal-credit /
exploration** problem, not an episode-length problem. Raising nstep alone does
not fix it.

## What has been ruled out (so we stop re-hypothesizing blindly)

1. Static vs dynamic/adaptive rewards -- no collection gain (mechanically works,
   doesn't break ceiling).
2. Reward reshaping at the gated tile (D: free + shaped mutation) -- made the
   chain worse.
3. Episode horizon (nstep 64->128) -- more mutate attempts, no completion.

## Remaining honest levers (untested)

- **E**: make gated-food eating worth FAR more than regular (eat_gain gated x5-10)
  so one completed chain beats a harvest-spam session.
- **F**: remove/shorten the 15-step mutate cooldown near gated food, or add dense
  "hold-adjacent-and-harvest" shaping so the policy can sustain the 3-step chain.
- **G**: lower TH_GATE (1.10) / trait cooldown (15) so the chain is shorter.
