# Architecture

How the three layers fit together and the exact shapes/spaces the policy sees.

## 1. System overview

```
  cpp/sim.cpp  (C++ gridworld, fast, deterministic)  --compiled via pybind11-->  cpp_sim.pyd
        │  cpp_sim.Sim(W,H,n,seed,curriculum,respawn,food_seed,food_seed_dist,food_density_div)
        ▼
  src/env.py  (EmergenceGrid: Gym-style wrapper; obs assembly + PBRS reward shaping)
        │  step(actions) -> (obs, rewards, dones, info)   obs: (N, OBS_DIM) float32
        ▼
  src/train.py  (PPO loop: collect rollouts, GAE, ppo_update_agent)
        │
        ▼
  src/model.py  (AgentPolicyBatch: CNN spatial encoder + GRU + policy/value heads)
  src/ppo.py    (RolloutBuffer, RewardNormalizer, PPOTrainer)
```

The C++ sim is the source of truth for world dynamics and reward. Python only
wraps it, adds a Potential-Based Reward Shaping (PBRS) navigation bonus on top,
and runs the PyTorch policy. The compiled `.pyd` MUST be rebuilt after any
`sim.cpp` change (`python build_cpp.py`) — editing `sim.cpp` alone has no effect
until recompiled.

## 2. World

- W×H grid (64×64 default), walled border (not a torus).
- Discrete time, simultaneous actions.
- Tile types (env.py constants): EMPTY, FOOD, HARD_NUT, TALL_FRUIT, GAP, WALL,
  GATE, HAZARD, PREDATOR, OASIS, MARKER.

## 3. Observation (`OBS_DIM`, per agent)

For a 64×64 grid, `OBS_DIM = 50621`:

| block | size | contents |
|-------|------|----------|
| global patch | 64·64·12 = 49152 | full-map tile encoding, 12 channels (one per tile type) |
| own-state | 14 | agent traits (strength, reach, speed, perception, social, small, can_hard, can_tall, metabolism, …) |
| 11×11 local patch | 121·12 = 1452 | local tile view around the agent (used by the CNN path in model.py) |
| food vector | 3 | (food_dx, food_dy, food_dist) — unit vector + normalized distance to nearest food (incl. gated) |

The food vector is what enables navigation: the policy is pushed toward the
nearest food (PBRS + this signal), and the C++ food index includes HARD_NUT/
TALL_FRUIT so gated food is a valid navigation target.

> Note: `train.py` logs `dist` as `obs[-1]*40` (the normalized food distance ×40).
> `dist 2.0` in a log means normalized 0.05 (far), NOT adjacent — do not read raw
> `dist` as "close".

## 4. Action space (13 discrete)

| id | action | id | action |
|----|--------|----|--------|
| 0 | idle | 7 | signal |
| 1 | UP (dy=-1) | 8 | strength+ |
| 2 | RIGHT (dx=+1) | 9 | strength- |
| 3 | DOWN (dy=+1) | 10 | reach+ |
| 4 | LEFT (dx=-1) | 11 | reach- |
| 5 | harvest | 12 | speed+ |
| 6 | share | | |

Gated food requires a trait: **HARD_NUT → strength ≥ 0.6**, **TALL_FRUIT →
reach ≥ 0.6**. Gates open when cumulative adjacent agent strength ≥ 1.10.

## 5. Policy network (`model.py` — `AgentPolicyBatch`)

Three stages (see the class docstring for detail):

1. **Vision backbone** — no-pool dilated CNN (MiniGrid-style) over the 11×11
   local patch, 12 channels, 4 layers with dilations 1/2/4/8 → ~31-tile receptive
   field. Stride-1 + pad=dilation keeps gates/walls crisp (the old 1024-token ViT
   was "wall-blind").
2. **Spatial pool** — feature map pooled to 8×8 (not global-average, which erased
   *where* walls were and broke navigation).
3. **Recurrent core** — GRU over pooled features + own-state vector; a 2-layer
   reasoning-head MLP is applied to the GRU output before the action/value heads
   (extra capacity for trait-matching).

A directional prior `food_w` is added to action logits (toward food-direction,
small harvest bias when adjacent) — a *prior*, not a reward; the policy can
override it (e.g. mutate instead of harvest when it lacks the trait).

Bigger-model runs use `d_model=gru_hidden=head_dim=256` (~14.6M params at N=1).

## 6. Reward (`sim.cpp` constants)

| term | constant | value | effect |
|------|----------|-------|--------|
| PBRS nav bonus | `nav_alpha` | 0.25/0.15/0.10 | potential; rewards closing distance to food (not camping) |
| FOOD_PULL | `FOOD_PULL` | 1.0 | potential; paid only on steps that move closer to food |
| eat | `EAT_GAIN` | 15.0 | +reward when harvesting adjacent food |
| invalid harvest | `INVALID_HARVEST_PEN` | 0.5 | −reward for harvesting with no food adjacent (kills spam) |
| mutate | `TRAIT_MUT_PEN` | 1.0 | −per mutate, but +1.0 when gaining a trait it lacked (net ~0 unless it unlocks food) |
| gate open | `GATE_GAIN` | 0.8 | +to agents opening a gate |
| death | `DEATH_PEN` | 2.0 | −when energy hits 0 |
| passive hunger | `STARVE_DECAY`/`STARVE_PEN` | 0.0 (disabled) | experiment; reverting to 0 after it did not improve gated eating |

`harv/step` in training logs = mean inventory increments / nstep = **real** food
collected (not the harvest-button press rate). Always report `harv/step`.

## 7. PPO training (`train.py` + `ppo.py`)

- Clipped surrogate + GAE (γ=0.99, λ=0.95) + value loss + entropy.
- **RewardNormalizer** (Welford) per-agent before buffering — keeps vf loss stable
  (raw returns span ~[−30,+600], which exploded the critic to vf loss 300–1600).
- **Entropy floor** (`ent_floor=0.5`) via `weight·clamp(ent_floor−ent,0)` added to
  loss — prevents the policy collapsing to a single constant action.
- Per-episode world reset at each update so finite food is replenished (no
  pocket-feeding from in-place regen).
- Resume: `--resume --exp NAME` continues from the latest `*_stepN.pt` (policy+opt).
