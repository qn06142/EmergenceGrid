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

Honest conclusions:
- The controller **mechanically works**: on L1 it cut invalid-harvest rate
  0.87 -> 0.55 and on L2 0.69 -> 0.42 (lever #1 fired correctly), and it was
  more *consistent* (lower std) than static.
- But it did **NOT break the collection ceiling** — collect_rate is flat vs
  static within noise on both L1 and L2. So dynamic/adaptive rewards are not
  the crux of the ~10% ceiling.
- The harness itself **exposed the prior metric flaw**: std is +/- 0.03-0.05,
  so the earlier "0.112 vs 0.107 vs 0.099" single-seed comparisons were
  NOISE, not signal. This validates the "our metrics were flawed" concern.

## L3 gate-opening is a HORIZON artifact (not a reward problem)

`gateopen=0` on L3 is NOT a learning failure. Opening a gate needs cumulative
strength >= 1.10; starting at 0.35 + 0.12/mutate with a 15-step trait cooldown
requires ~7 mutations x 15 = **~105 steps**, but `nstep=64`. So gate-opening
is structurally impossible within an episode. The harness reports
`gate_reachable_seeds = 0/5` (agent never gets adjacent-with-strength),
confirming the gate lever (#4) never even had a chance to fire. Fixing L3
means raising `nstep` (or lowering `TH_GATE` / cooldown), NOT reshaping rewards.
