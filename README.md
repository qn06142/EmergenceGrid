# EmergenceGrid — Multi-Agent RL Trait Emergence

A from-scratch multi-agent reinforcement learning simulation where agents must
**discover trait mutations** (strength / reach / speed) to overcome gated food
and physical obstacles. The project pairs a hand-written C++ gridworld simulator
(`cpp/sim.cpp`, exposed to Python via pybind11) with a PyTorch PPO policy
(`src/`).

The research question: *can useful traits (and the behaviors that require them)
emerge purely from reward, without being hard-coded?*

## Curriculum

| Level | World | Emergent behavior targeted |
|-------|-------|----------------------------|
| 0 | open grid, free food | navigate + harvest |
| 1 | + walls | path-find around obstacles |
| 2 | + **gated food** (HARD_NUT needs strength, TALL_FRUIT needs reach) | **mutate own traits** to eat |
| 3 | + oases behind **gates** (open by cumulative strength) | mutate strength → open gate → safe food |
| 4–5 | + hazards / sparse | robustness |

## Results

- **Curriculum 0–1:** agents reliably learn to navigate, path around walls, and
  harvest. (`gifs/L1rand_collect.gif`)
- **Curriculum 2 (trait emergence):** the agent learns to **mutate its own
  traits** and eat gated food it could not eat beforehand. Measured real
  collection (food actually eaten, not just the harvest button pressed) reaches
  **~8–10% of steps** with ~25–30% mutation activity — reproducible across
  seeds. This is the headline emergence result.
  (`gifs/L2_emergence.gif`, best checkpoint `ckpts/_L2b/L2b_policy_step600.pt`)
- **Curriculum 3 (gate-opening):** investigated; the longer credit-assignment
  chain (mutate strength ~6× to reach the gate threshold, then the gate opens)
  did not reliably emerge within the training budget. Left as the next step —
  see *Limitations* below.

> Honest metric note: we report *real* food collection (`harv/step`, counted
> from inventory increments in the sim), **not** "top action = harvest 100%",
> which only indicates the agent is mashing the harvest button regardless of
> whether food is present.

## Build & Run

Requires **Windows + VS2022 BuildTools** (for the C++ sim) and Python 3.11+.

```bash
# 1. create env + deps
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. compile the C++ simulator into cpp_sim.pyd (needs MSVC in PATH)
python build_cpp.py

# 3. train (single agent, curriculum 2, big model)
python src/train.py --n 1 --grid 64 --k 2 --nstep 64 --nupd 600 \
    --curriculum 2 --gated_food 2 --food_seed_dist 8 --food_density_div 50 \
    --food_regen_mode 2 --ent_coef 0.05 --ent_floor 0.5

# 4. render a gif from a checkpoint
python src/render.py --ckpt ckpts/_L2b/L2b_policy_step600.pt --n 1 --grid 64 \
    --steps 500 --seed 99 --gated_food 2 --food_regen_mode 2 \
    --d_model 256 --gru_hidden 256 --head_dim 256 --out gifs/L2_emergence.gif

# 5. sanity checks
python tests/test_sanity.py
```

### Key training flags
- `--curriculum N` — world complexity (0–5, see table).
- `--gated_food 2` — gated-dominant food (agent must mutate to eat).
- `--food_seed_dist 8` — food spawns at a distance, forcing navigation.
- `--food_density_div 50` — food density divisor (higher = sparser).
- `--food_regen_mode 2` — food respawns at random empty cells (no pocket-feeding).
- `--ent_coef 0.05 --ent_floor 0.5` — entropy regularization that prevents the
  policy collapsing to a single constant action.
- `--d_model 256 --gru_hidden 256 --head_dim 256` — larger model used for the
  trait-emergence runs (cognitive load of trait-matching).

## Documentation

- `docs/ARCHITECTURE.md` — sim ↔ Python ↔ PPO data flow, full obs layout,
  action space, reward breakdown, policy network design.
- `docs/EXPERIMENTS.md` — honest results per curriculum (L2 trait emergence at
  ~8–10% real collection; L3 gate-opening not yet emerged), what was tried and
  why it didn't help, known gaps, and metric discipline.
- In-code docstrings in `src/model.py`, `src/env.py`, `src/ppo.py` explain the
  architecture and the stability fixes (RewardNormalizer, entropy floor).

> The two root-level `design.md` and `EMERGENCEGRID_DUMP.md` are **earlier draft /
> postmortem notes** from the project's original "emergent cooperation" direction
> (2026-08-14) and predate the trait-emergence results. They are kept for history
> but `docs/` is the current, accurate writeup.

```
cpp/sim.cpp            C++ gridworld simulator (pybind11 module cpp_sim)
build_cpp.py           MSVC build script for cpp_sim.pyd (portable paths)
src/train.py           PPO training loop
src/env.py             Gym-style wrapper around cpp_sim
src/model.py           AgentPolicyBatch (CNN spatial encoder + GRU + policy head)
src/ppo.py             Rollout buffer, GAE, RewardNormalizer
src/render.py          render a trajectory to a GIF
tests/test_sanity.py   determinism / obs-shape / harvest sanity checks
gifs/                  example trajectories
```

## Limitations / next steps
- **Credit-assignment ceiling at curriculum 2:** the 3-step chain
  (navigate → mutate correct trait → harvest) plateaus near ~10% real
  collection; movement overhead naturally bounds the eatable-step fraction.
- **Curriculum 3 gate-opening** did not emerge within budget — the mutation
  chain to open a gate is long (~6 strength mutations) with delayed reward.
  Promising next steps: lower the gate-strength threshold, or add a small
  shaping reward for being adjacent to a gate with sufficient strength.
- No predators actually spawn in the current build (`resolve_predators` is a
  no-op because the predator list is never populated) — a known gap vs the
  designed curriculum.

## Credits
Built by Wheatley. C++ simulator + PyTorch PPO policy, trained on a single
Windows GPU.
