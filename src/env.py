"""
Python wrapper around the C++ sim (cpp_sim.Sim).
Observation shape: (N, 49166) float32.
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
    ):
        self._sim = cpp_sim.Sim(width, height, n_agents, seed, curriculum, respawn, food_seed, food_seed_dist, food_density_div)
        self._sim.set_food_regen_mode(food_regen_mode)
        self._sim.set_gated_food(gated_food)
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

    def adjacent_harvestable(self, a: Agent) -> bool:
        """Check if an agent is adjacent to harvestable food."""
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
        # The C++ sim disables its own nav_alpha for curriculum 0 and >= 3.
        # We fill that gap here so there's always a gradient to move toward food.
        # Alpha decays with curriculum as complexity grows; do NOT reward camping
        # adjacent (dist <= 1) to prevent the "harvest spam when adjacent" attractor.
        _pbrs_alpha = {0: 0.25, 1: 0.15, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
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
