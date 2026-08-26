"""
Python wrapper around the C++ sim (cpp_sim.Sim).

Observation layout (per agent), total OBS_DIM = 50621 for a 64x64 grid:
  - Global patch: GRID_H*GRID_W*SPAT_C = 64*64*12 = 49152 floats. A full-map
    one-hot-ish encoding of the world (tile type per cell x 12 channels), so the
    agent sees walls/food/gates/agents everywhere. (Used by the ViT path; the CNN
    path additionally takes an 11x11 local patch — see model.py.)
  - Own-state vector: 14 traits + 11x11 patch(121*12) + (food_dx, food_dy, food_dist)
    = 14 + 1452 + 3 = 1469 floats describing the agent itself and its relation to
    the nearest food (direction unit-vector + normalized distance). The food-direction
    signal is what lets the policy navigate to food (including gated food — the C++
    food index includes HARD_NUT/TALL_FRUIT).  (49152 + 1469 = 50621 total.)

Action space (13 discrete actions): see ACTION_MAP in the sim. 0 idle, 1-4 move
(UP/RIGHT/DOWN/LEFT), 5 harvest, 6 share, 7 signal, 8-12 mutate traits
(str+-/reach+-/speed+). Gated food requires a trait: HARD_NUT -> strength>=0.6,
TALL_FRUIT -> reach>=0.6.

Reward (computed in C++, see sim.cpp constants):
  - PBRS navigation bonus (potential-based, annealed by curriculum) rewarding steps
    that decrease distance to nearest food (NOT camping — skipped when adjacent).
  - FOOD_PULL potential on the step food gets closer (was a state reward; changed to
    potential so the agent must EAT rather than orbit food for the pull).
  - harvest: +EAT_GAIN (15) when adjacent to harvestable food, -INVALID_HARVEST_PEN
    (0.5) otherwise (kills free harvest-spam).
  - mutate: -TRAIT_MUT_PEN (1.0) per mutate, but +1.0 when it *gains* a trait it
    lacked (can_hard/can_tall) — net ~0 unless the mutation unlocks gated food.
  - gate: +GATE_GAIN (0.8) to agents who open a gate (cumulative strength >= threshold).
  - death: -DEATH_PEN (2.0) when energy hits 0.

env.step(actions) returns (obs, rewards, dones, info). obs is (N, OBS_DIM) float32.
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import cpp_sim

# Tile Type Constants
EMPTY = 0
FOOD = 1
HARD_NUT = 2
TALL_FRUIT = 3
GAP = 4
WALL = 5
GATE = 6
HAZARD = 7
PREDATOR = 8
OASIS = 9
MARKER = 10


@dataclass(slots=True)
class Traits:
    strength: float
    reach: float
    speed: float
    perception: int
    metabolism: float
    social: float
    size_small: int
    _can_hard: bool
    _can_tall: bool
    _can_small: bool

    @property
    def can_hard(self) -> bool:
        return self._can_hard

    @property
    def can_tall(self) -> bool:
        return self._can_tall

    @property
    def can_small(self) -> bool:
        return self._can_small


@dataclass(slots=True)
class Agent:
    idx: int
    x: int
    y: int
    energy: float
    inv: int
    alive: bool
    last_action: int
    cooldown: int
    traits: Traits


class EmergenceGrid:
    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        n_agents: int = 16,  # Defaulted to 16 based on the architecture specs
        seed: int = 0,
        fallacies: bool = False,
        predator_every: int = 0,
        respawn: bool = True,
        curriculum: int = 5,
        food_seed: int = 0,
        food_seed_dist: int = 1,
        food_density_div: int = 50,
        food_regen_mode: int = 2,  # 0=none, 1=in-place (pocket-feeds), 2=random empty cell
        gated_food: int = 1,  # 0=none, 1=regular+trickle gated, 2=gated-dominant (mutate to eat)
        reward_preset: str = 'default',  # 'default' | 'gc' (G+C reward-density lever)
    ):
        self._sim = cpp_sim.Sim(width, height, n_agents, seed, curriculum, respawn, food_seed, food_seed_dist, food_density_div)
        self._sim.set_food_regen_mode(food_regen_mode)
        self._sim.set_gated_food(gated_food)
        # Default reward params (mirror sim.cpp RewardParams defaults). Override per
        # update via set_reward_params() to anneal during training.
        # reward_preset='gc' applies the G+C reward-DENSITY lever
        # (diagnosed via --diag_train: the only positive gradient in the gate task
        # is too sparse to learn). It raises trait_match_bonus (bridge mutate->eat),
        # mutate_gated_gain, sharpens wrong_trait_pen, and adds a dense
        # gate_prox_bonus for being strong & adjacent to a gate.
        _PRESETS = {
            'default': {},
            'gc': dict(trait_match_bonus=0.4, mutate_gated_gain=5.0,
                       wrong_trait_pen=1.5, gate_prox_bonus=0.3),
        }
        if reward_preset not in _PRESETS:
            raise ValueError(f"unknown reward_preset={reward_preset!r}")
        self.reward_params = dict(
            food_pull=1.0, nav_alpha=0.15, eat_gain=15.0, eat_gain_regular=15.0,
            invalid_harvest_pen=0.5, trait_mut_pen=1.0,
            trait_mut_pen_gated=0.0,   # A: mutation is FREE when adjacent to gated food
            gate_gain=0.8, trait_match_bonus=0.0,
            mutate_gated_gain=1.5,     # C: +reward for mutating the RIGHT trait near gated
            wrong_trait_pen=0.3,       # C: -reward for mutating WRONG trait near gated
            gate_prox_bonus=0.0)       # G: dense + for strong(>=gate thr) & adjacent to gate
        self.reward_params.update(_PRESETS[reward_preset])
        self._apply_reward_params()

        self.W = width
        self.H = height
        self.n_agents = n_agents
        self.respawn = respawn
        self.curriculum = curriculum
        self.food_seed = food_seed
        self.food_seed_dist = food_seed_dist
        self.food_density_div = food_density_div
        self.food_regen_mode = food_regen_mode
        self.gated_food = gated_food
        self.step_count = 0

        self.grid: List[int] = []
        self.gate_cells: List[Tuple[int, int]] = []
        self.oasis_cells: List[Tuple[int, int]] = []
        self.agents: List[Agent] = []
        self._food_dist: Optional[np.ndarray] = None

        self._refresh()

    def _apply_reward_params(self):
        rp = self.reward_params
        self._sim.set_reward_params(
            rp['food_pull'], rp['nav_alpha'], rp['eat_gain'], rp['eat_gain_regular'],
            rp['invalid_harvest_pen'], rp['trait_mut_pen'],
            rp['trait_mut_pen_gated'], rp['gate_gain'], rp['trait_match_bonus'],
            rp['mutate_gated_gain'], rp['wrong_trait_pen'])
        self._sim.set_gate_prox_bonus(rp['gate_prox_bonus'])
        self._sim.set_step_frac(getattr(self, 'step_frac', 0.0))

    def set_reward_params(self, **kw):
        """Override reward parameters (e.g. to anneal over training).
        Supported keys: food_pull, nav_alpha, eat_gain, eat_gain_regular,
        invalid_harvest_pen, trait_mut_pen, trait_mut_pen_gated, gate_gain,
        trait_match_bonus, mutate_gated_gain, wrong_trait_pen, gate_prox_bonus."""
        self.reward_params.update(kw)
        self._apply_reward_params()

    def get_diag(self):
        """Return the C++ sim's closed-loop diagnostics tuple (9 fields):
        (steps, harvest_invalid, harvest_valid, move_away, move_closer,
         mutate_steps, gate_adj, gate_adj_strong, dead)."""
        return self._sim.get_diag()

    def get_diag_full(self):
        """Return the COMPLETE instrumentation dict (pipeline funnel, trait
        dynamics, ground-truth distances, gate progress, reward probe)."""
        return self._sim.get_diag_full()

    @property
    def gate_threshold(self):
        """Sim's actual gate-opening threshold (TH_GATE/100). Exposed so the
        funnel/eval uses the REAL value instead of a hardcoded 1.10 that drifts."""
        return float(self._sim.gate_threshold())

    def set_step_frac(self, f: float):
        self.step_frac = float(f)
        self._sim.set_step_frac(float(f))

    def adjacent_harvestable(self, a: Agent) -> bool:
        return self._sim.adjacent_harvestable(self._sim.agents[a.idx])

    def _refresh(self) -> None:
        """Syncs Python state with the underlying C++ engine."""
        self.grid = self._sim.grid
        self.gate_cells = [(g[0], g[1]) for g in self._sim.gate_cells]
        self.oasis_cells = [(o[0], o[1]) for o in self._sim.oasis_cells]

        self.agents = [
            Agent(
                idx=d["idx"], x=d["x"], y=d["y"], energy=d["energy"], inv=d.get("inv", 0),
                alive=d["alive"], last_action=d["last_action"], cooldown=d["cooldown"],
                traits=Traits(
                    strength=d["strength"], reach=d["reach"], speed=d["speed"],
                    perception=d["perception"], metabolism=d["metabolism"],
                    social=d["social"], size_small=d["size_small"],
                    _can_hard=d["can_hard"], _can_tall=d["can_tall"], _can_small=d["can_small"]
                )
            ) for d in self._sim.dump_agents()
        ]
        self._food_dist = None

    @property
    def food_dist(self) -> np.ndarray:
        if self._food_dist is None:
            self._food_dist = self._compute_food_dist_for_probe()
        return self._food_dist

    def _compute_food_dist_for_probe(self) -> np.ndarray:
        """Computes nearest food Manhattan distance. Replaces 1e9 with max possible distance."""
        H, W = self.H, self.W
        max_dist = H + W  # Cap distance to prevent gradient explosions
        d = np.full((H, W), max_dist, dtype=np.int32)

        foods = [
            (x, y) for y in range(H) for x in range(W)
            if self.grid[y * W + x] in (FOOD, OASIS, HARD_NUT, TALL_FRUIT)
        ]

        for (fx, fy) in foods:
            for y in range(H):
                for x in range(W):
                    dd = abs(x - fx) + abs(y - fy)
                    if dd < d[y, x]:
                        d[y, x] = dd
        return d

    def reset(self) -> np.ndarray:
        obs = self._sim.reset()
        self.step_count = 0
        self._refresh()
        obs_arr = np.array(obs, dtype=np.float32)
        # Track distance to nearest food for PBRS navigation reward
        self._prev_dist = [obs_arr[i, -1] * 40.0 for i in range(self.n_agents)]
        return obs_arr

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], List[bool], Dict[str, Any]]:
        self.step_count += 1
        obs, rew, done = self._sim.step([int(a) for a in actions])
        self._refresh()
        obs_arr = np.array(obs, dtype=np.float32)
        rew_list = list(rew)

        # Potential-Based Reward Shaping (PBRS) for navigation (Ng et al. 1999).
        # NOTE: the C++ sim applies its OWN rp.nav_alpha UNCONDITIONALLY (no
        # curriculum gate) toward nearest_food_dist when eat_gain_regular>0, and
        # toward gates/gated food when eat_gain_regular==0. To avoid a double nav
        # signal in the NORMAL env, this Python food-PBRS is only applied in
        # Phase-2 (eat_gain_regular==0), where the sim targets GATES and the agent
        # would otherwise have NO pull toward regular food for survival.
        # Alpha decays with curriculum as complexity grows; do NOT reward camping
        # adjacent (dist <= 1) to prevent the "harvest spam when adjacent" attractor.
        _pbrs_alpha = {0: 0.25, 1: 0.15, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
        nav_alpha = 0.0
        if self.reward_params.get('eat_gain_regular', 15.0) == 0.0:
            nav_alpha = _pbrs_alpha.get(self.curriculum, 0.10)
        if hasattr(self, '_prev_dist'):
            for i in range(self.n_agents):
                if self.agents[i].alive:
                    curr_dist = obs_arr[i, -1] * 40.0
                    delta = self._prev_dist[i] - curr_dist
                    # Only reward closing distance when not already adjacent.
                    # Skip when prev_dist <= 1: agent is camping next to food.
                    # Clamp teleport spikes (food respawn changes nearest-food pointer).
                    if self._prev_dist[i] > 1.0 and abs(delta) <= 3.0:
                        rew_list[i] += nav_alpha * delta
                    self._prev_dist[i] = curr_dist
                else:
                    self._prev_dist[i] = obs_arr[i, -1] * 40.0
        else:
            self._prev_dist = [obs_arr[i, -1] * 40.0 for i in range(self.n_agents)]

        info = {
            'step': self.step_count,
            'alive': [a.alive for a in self.agents],
            'energy': [round(a.energy, 2) for a in self.agents]
        }
        return obs_arr, rew_list, list(done), info

    def obs_dim(self, a: Optional[Any] = None) -> int:
        return self._sim.obs_dim()
