# EmergenceGrid — Full Code Dump & Postmortem (for redesign by Gemini Pro)

This is the complete, working-as-far-as-it-goes state of a multi-agent RL project
attempting to induce **Emergent Cooperation** (not hand-coded; emergent from local
reward + world structure) in a 2D grid world. The author (Wheatley) hit a wall:
training is too slow AND the policy does not actually learn to forage. The C++ sim
engine is done and fast; the PPO training loop and the reward/learning design are
the problems. This doc is for a fresh design pass.

---

## 1. GOAL (what we're actually trying to build)

Induce emergent cooperation among 16 independent agents in a 64×64 grid world via
reinforcement learning (no global bonus). Agents should
spontaneously specialize (forager / striker / connector / scout) and coordinate
(e.g. push gates open together, share food) because the WORLD + REWARD structure
makes those behaviors locally beneficial — NOT because the reward tells them to.

Key constraints from the author:
- "i would detest against actually changing our environment to google deepmind's,
  we just need to shape the rewards and world to force specialization."
- Agents should see the FULL grid (not a local view).
- No CNN max/avg-pool blur — fine tile/gate position must stay resolvable → use
  attention (ViT-style) encoder.
- Agents must know their OWN (x,y) position to navigate.
- Reward shaping is the lever, not copying an existing env.

### Observed failure (last run, L0 food-only, 5090)
```
upd    0 | ... harv/step 0.079   (random baseline)
upd   50 | ... harv/step 0.180
upd  100 | ... harv/step 0.195
upd  150 | ... harv/step 0.233
upd  200 | ... harv/step 0.150   <- COLLAPSED (regressed)
```
harv/step barely rose above random (0.079) and then went backwards. The policy
never learned robust foraging. Also the run took ~76 min/update → 700 updates
would be ~9 days. Two separate failures: (a) slow pace, (b) no learning.

---

## 2. ARCHITECTURE

- **16 independent PPO agents** (D1: independent policy nets), each with its OWN
  GRU memory + actor/critic heads, but sharing ONE visual encoder (cheap — the
  grid view is identical across agents in an env, so the encoder runs K times, not
  N*K). The visual encoder is NOT trained per-agent (shared), but the GRU/heads ARE
  per-agent. This means agents CAN diverge in behavior (different GRU/head weights)
  even with a shared vision backbone.
- **ViT-style grid encoder**: (64×64×12) grid → 4096 tokens (one per tile) → linear
  embed + 2D sinusoidal positional encoding → TransformerEncoder(2 layers, 4 heads,
  d_model=128) → CLS token output = grid embedding (128-d).
- **Obs dim = 49166** = 64×64×12 (grid channels) + 14 (own state).
- **PPO**: clipped surrogate + GAE(λ=0.95, γ=0.99) + value + entropy. Adam lr 2.5e-4.
- **Curriculum** flag (0..5): 0=food-only → 1=+walls → 2=+trait-gated food
  (hard_nut/tall_fruit/gap) → 3=+oasis/gates → 4=+hazards → 5=full (+predators).
  L0 (food-only) was being trained when it stalled.

### Why this obs dim / encoder
The author insisted on full-grid vision (no local view) and no CNN blur (attention
preserves exact tile positions). The 49166-dim obs is the full grid one-hot + own
state. This is expensive (ViT over 4096 tokens) but that's a deliberate choice.

---

## 3. THE WORLD (mechanics — faithful spec)

Grid 64×64, border walls. 16 agents. Tile types:
```
EMPTY=0 FOOD=1 HARD_NUT=2 TALL_FRUIT=3 GAP=4 WALL=5
GATE=6  HAZARD=7 PREDATOR=8 OASIS=9 MARKER=10
```
Actions (13): 0=noop, 1-4=move U/D/L/R, 5=harvest (adjacent 3×3), 6=share,
7=signal (drop marker), 8-12=trait mutate (strength±, reach±, speed+).

Agent traits (continuous):
- strength, reach, speed ∈ [0.05,1.0]
- perception ∈ {1,2,3,4}
- metabolism ∈ [0.5,1.5]
- social ∈ [0,1]
- size_small (0/1)
- Derived gates: `can_hard = strength>=0.6`, `can_tall = reach>=0.6`,
  `can_small = size_small`.
- Constraint: `strength+speed <= 1.3` (enforced by rescaling).
- Metabolism cap linked to social (high social caps the physical traits).

Spawn archetypes (biased trait samples): striker (high str), scout (high speed+perc,
small), runner (high reach+speed, small), connector (high social, small), generalist.

### Rewards (per step, per agent)
```
FOOD_PULL        = 0.02/(1+dist_to_nearest_food)   # weak, flat
NAV_ALPHA        = 0.05  # potential-based: +NAV_ALPHA*(dist_before - dist_after)
PROX_FOOD_BONUS  = 0.15  # if adjacent to harvestable food
EAT_GAIN         = 1.0 * (1+OASIS_BONUS=2.5 / HARD_BONUS=1.5 / TALL_BONUS=1.2)
STEP_PEN         = 0.01 * metabolism
STAGNATION_PEN   = 0.03  # if didn't move
DEATH_PEN        = 1.0
GATE_GAIN        = 0.8   # when gate opens (pushers)
PRED_GAIN        = 1.0   # when predator defeated (by combined strength >= TH_PRED=1.3)
SHARE_GAIN       = 0.4
SIGNAL_ENERGY    = 0.05  # cost
TRAIT_MUT_PEN    = 1.0   # cost of mutating
  + 0.4 bonus when a mutation CROSSes can_hard/can_tall threshold
ACT_COST_SHARE/SIGNAL/MUT = 0.05 each
HAZ_PEN          = 0.5   # standing on hazard
GATE_PROX        = 0.02  # adjacent to a gate
```
Energy: E_MAX=10. Death when energy<=0. **Respawn ON but at 0.3×E_MAX** (low
energy) so the population doesn't die out but isn't artificially inflated.

Mechanics:
- **Harvest**: adjacent (3×3) food/hard_nut/tall_fruit → energy + inventory; tile
  regrows after FOOD_REGEN=40 steps (oasis after 25).
- **Share**: give energy to a weaker orthogonally-adjacent agent.
- **Signal**: drop a MARKER on a nearby empty/food cell.
- **Mutate**: shift a trait (with cooldown 15); threshold-crossing gives a bonus.
- **Gates**: 2-wide×2-deep gate cells at 4 oases; open when total adjacent-agent
  strength >= TH_GATE=1.1. Pushers gain GATE_GAIN.
- **Predators**: roam; defeated when total adjacent strength >= TH_PRED=1.3.
- **Oasis**: rich food pocket behind gates.
- **Hazards**: drain energy.

---

## 4. THE DATA-STRUCTURE OPTIMIZATION (done, working)

The author's explicit instruction: "you shouldn't need a bfs for each food spread."
The original Python env recomputed a full-grid Manhattan distance transform
(grid-wide BFS-like 2-pass Chamfer) every step just to answer `food_dist` at the
~16 agent positions. Replaced in C++ with:

1. **Food list + coarse bucket grid** (8×8). `nearest_food_dist(x,y)` does an
   expanding-ring search over the bucket grid → O(local) per agent, NO grid-wide
   spread. `food_dist` is only queried at agent positions (for the NAV/FOOD_PULL
   reward), never as a full field.
2. **Occupancy grid** `occ[W*H]` (agent index+1) → O(1) collision check, replaces
   the O(N²) `any(agent at nx,ny)` linear scan.
3. Everything else ported 1:1 from the Python.

**Measured**: C++ sim = 0.28 ms/step (16 agents) vs Python 2.47 ms/step (optimized)
vs 51 ms/step (original). ~25× faster than the optimized Python, ~180× vs original.
The C++ `.so` lives on the 5090 box at `~/rl_emergence/cpp_sim.so` (and `src/cpp_sim.so`).
It does NOT exist on the laptop — so training MUST run on the 5090.

### C++ build (on 5090 box, venv `/home/qn06142/venv`)
```
g++ -O3 -shared -std=c++17 -fPIC cpp_sim.cpp \
    -I$(python -c 'import sysconfig;print(sysconfig.get_path("include"))') \
    -I$(python -c 'import pybind11;print(pybind11.get_include())') \
    -o cpp_sim.so
```
pybind11 is installed in the venv (`pip install pybind11`). Do NOT pip-install
anything else / modify the system on that box (commensal machine).

---

## 5. THE PACE PROBLEM (why training is ~76 min/update)

The C++ env is 0.28 ms/step → 256 env steps/update (nstep=64 × k=4) = 0.07s. So
the ENV is NOT the bottleneck anymore. The bottleneck is **train.py's Python
rollout loop** (lines ~103-166): per step it does
- `np.concatenate(obs_now)` (Python/numpy alloc),
- a 128-iteration double `for e in range(k): for i in range(n)` to build
  `hid_batched`,
- per-step `.detach()` buffer appends (256 tiny GPU tensors),
- a per-env Python `step()` call (4 of them) with Python-side gate-open detection
  loops.

On the 5090 box the CPU is CONTENDED (another tenant runs ffmpeg + a python job),
so this Python loop crawls. The GPU sits at 100% but is starved by the slow CPU
rollout. ~76 min/update → unusable.

**What a redesign should fix**: vectorize the rollout (batched obs concat once,
no per-step Python double-loop), and/or move the whole rollout into C++ (return
a batch of obs/rews/dones without Python per-step calls). The env C++ already
returns exactly what's needed; the glue in train.py is the slow part.

---

## 6. THE LEARNING PROBLEM (why harv/step stays near random)

Even in the simplest L0 (food-only) curriculum, with the NAV_ALPHA potential-based
reward added to bootstrap navigation, harv/step only reached 0.233 and collapsed.
Hypotheses (NOT resolved — needs fresh design):

- **FOOD_PULL + NAV reward too weak / mis-scaled** relative to metabolic cost and
  the credit-assignment length. A 0.05 reward for getting 1 tile closer to food is
  tiny vs the exploration noise. The agent may not reliably learn "move toward food
  → harvest" because the gradient is too flat and the action space (13 acts) is too
  large for sparse-ish signal.
- **13 discrete actions with 1 harvest among move/noop/share/signal/mutate** dilutes
  the probability mass on harvest; random harvest rate is low so the policy needs
  strong shaping to discover it.
- **GRU memory + full-grid attention may be overkill / hard to train** for what is
  essentially "go to food". Maybe a simpler policy (or curriculum that makes food
  trivially locatable) is needed first.
- **The author's own diagnosis**: "the agents respawn with high energy, therefore
  causing avge to be artificially inflated" (fixed: respawn at 0.3×E_MAX) and
  "i don't see any open gates, nor do i see any agents actually doing anything"
  (gates never opened; agents in action-loops). The metrics were flawed (avgE
  inflated by respawn). Replaced avgE with deaths/step, harv/step, gateopen.
- A measured probe showed **toward-food move rate = 0.046** (WORSE than random
  ~0.25) at one checkpoint — the policy was essentially random despite 60 updates.

**What Gemini should reconsider**: the reward curve, the action-space granularity,
whether full-grid ViT is necessary for L0 (maybe a much smaller encoder for early
curriculum), and whether the curriculum ordering actually bootstraps (food-only
first, then walls, then gates, etc.). The author explicitly wants world/reward
shaping to FORCE the target behavior, not a different published env.

---

## 7. FULL SOURCE

### 7.1 `cpp/sim.cpp` (the C++ sim engine — DONE, FAST, WORKING)
Compiled to `cpp_sim.so`. 484 lines. Key structures: `Sim` with `grid`, `agents`,
`foods` (list) + `bucket` (8×8 coarse grid), `occ` (occupancy). `step(actions)`
returns `(obs ndarray (N,49166) float32, rewards list, dones list)`. `reset()`
returns obs. `dump_agents()` returns agent state for the Python wrapper.
`nearest_food_dist` = expanding-ring bucket search (the anti-BFS). `obs_dim()`=49166.

Full file content below (verbatim from `/d/rl-emergence/cpp/sim.cpp`):

```cpp
// EmergenceGrid C++ core (pybind11 module: cpp_sim).
// Drop-in replacement for the Python env. Same mechanics, but:
//   * food distance is NOT a grid-wide BFS/DT -- we keep a food list + coarse
//     bucket grid and answer nearest-food queries in O(local) per agent.
//   * occupancy grid (occ[W*H]) replaces the O(N^2) agent-collision scan.
// Obs is (N, 49166) float32 -- identical layout to the Python version.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

enum { EMPTY=0, FOOD=1, HARD_NUT=2, TALL_FRUIT=3, GAP=4, WALL=5,
       GATE=6, HAZARD=7, PREDATOR=8, OASIS=9, MARKER=10 };

static const int TH_STR=60, TH_REACH=60, TH_GATE=110, TH_PRED=130; // x100
static const int E_MAX=1000;
static const float E_MAX_F=10.0f;
static const float EAT_GAIN=1.0f, OASIS_BONUS=2.5f, HARD_BONUS=1.5f, TALL_BONUS=1.2f;
static const float GATE_GAIN=0.8f, PRED_GAIN=1.0f, HAZ_PEN=0.5f;
static const float SHARE_GAIN=0.4f, SIGNAL_ENERGY=0.05f;
static const float STEP_PEN=0.01f, DEATH_PEN=1.0f, STAGNATION_PEN=0.03f;
static const float PROX_FOOD_BONUS=0.15f, FOOD_PULL=0.02f, NAV_ALPHA=0.05f, GATE_PROX=0.02f;
static const float TRAIT_MUT=0.12f, TRAIT_MUT_PEN=1.0f;
static const int TRAIT_COOLDOWN=15;
static const float ACT_COST_SHARE=0.05f, ACT_COST_SIGNAL=0.05f, ACT_COST_MUT=0.05f;
static const int FOOD_REGEN=40, OASIS_REGEN=25;
static const int CHAN=12;
static const int OWN_DIM=14;

struct Traits {
    float strength, reach, speed; int perception; float metabolism;
    int size_small; float social;
    bool can_hard() const { return strength >= 0.6f; }
    bool can_tall() const { return reach >= 0.6f; }
    bool can_small() const { return size_small==1; }
};

struct Agent { int idx, x, y; Traits tr; float energy; int inv;
               bool alive; int last_action; int cooldown; };

struct Sim {
    int W, H, n_agents, curriculum; bool respawn; uint32_t seed;
    std::mt19937 rng;
    std::vector<int> grid, regen_t, regen_type, occ;
    std::vector<Agent> agents;
    std::vector<std::array<int,2>> predators, oasis_cells, gate_cells;
    std::vector<std::array<int,2>> foods;
    int B=8; std::vector<std::vector<int>> bucket;
    int step_count=0;
    // ... (constructor, build_grid, place_oasis, sample_traits, spawn_agents,
    //      rebuild_food_index, nearest_food_dist, tile_passable,
    //      adjacent_harvestable, adjacent_gate, agent_at, harvest, share,
    //      signal, mutate, resolve_hazards, resolve_predators, resolve_gates,
    //      regen_tiles, respawn_dead, step, reset, obs_dim, dump_agents)
    // All logic is in the verbatim file at /d/rl-emergence/cpp/sim.cpp.
};

PYBIND11_MODULE(cpp_sim, m) {
    py::class_<Sim>(m,"Sim")
        .def(py::init<int,int,int,uint32_t,int,bool>())
        .def("step", &Sim::step)
        .def("reset", &Sim::reset)
        .def("obs_dim", &Sim::obs_dim)
        .def("dump_agents", &Sim::dump_agents)
        .def_readonly("W",&Sim::W).def_readonly("H",&Sim::H)
        .def_readonly("grid",&Sim::grid)
        .def_readonly("oasis_cells",&Sim::oasis_cells)
        .def_readonly("gate_cells",&Sim::gate_cells)
        .def_readonly("agents",&Sim::agents);
}
```
(For the complete, compile-ready implementation, read `/d/rl-emergence/cpp/sim.cpp`
— every method body is there. The `.so` is already built on the 5090.)

### 7.2 `src/env.py` (thin Python wrapper around cpp_sim — DONE)
```python
"""Thin wrapper around the C++ sim (cpp_sim.Sim). Same API as the old Python env
so train.py / render.py / model.py are untouched. Obs = (N, 49166) float32."""
import numpy as np, os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path: sys.path.insert(0, _PROJ_ROOT)
import cpp_sim

EMPTY=0; FOOD=1; HARD_NUT=2; TALL_FRUIT=3; GAP=4; WALL=5
GATE=6; HAZARD=7; PREDATOR=8; OASIS=9; MARKER=10

class Traits:
    def __init__(self, d):
        self.strength=d["strength"]; self.reach=d["reach"]; self.speed=d["speed"]
        self.perception=d["perception"]; self.metabolism=d["metabolism"]
        self.social=d["social"]; self.size_small=d["size_small"]
        self._can_hard=d["can_hard"]; self._can_tall=d["can_tall"]; self._can_small=d["can_small"]
    @property
    def can_hard(self): return self._can_hard
    @property
    def can_tall(self): return self._can_tall
    @property
    def can_small(self): return self._can_small

class Agent:
    def __init__(self, d):
        self.idx=d["idx"]; self.x=d["x"]; self.y=d["y"]; self.energy=d["energy"]
        self.alive=d["alive"]; self.last_action=d["last_action"]; self.cooldown=d["cooldown"]
        self.traits=Traits(d)

class EmergenceGrid:
    def __init__(self, width=64, height=64, n_agents=6, seed=0,
                 fallacies=False, predator_every=0, respawn=True, curriculum=5):
        self._sim = cpp_sim.Sim(width, height, n_agents, seed, curriculum, respawn)
        self.W=width; self.H=height; self.n_agents=n_agents
        self.respawn=respawn; self.curriculum=curriculum
        self.step_count=0; self._refresh()
    def _refresh(self):
        self.grid=self._sim.grid
        self.gate_cells=[(g[0],g[1]) for g in self._sim.gate_cells]
        self.oasis_cells=[(o[0],o[1]) for o in self._sim.oasis_cells]
        self.agents=[Agent(d) for d in self._sim.dump_agents()]
        self._food_dist=None
    @property
    def food_dist(self):
        if self._food_dist is None: self._food_dist=self._compute_food_dist_for_probe()
        return self._food_dist
    def _compute_food_dist_for_probe(self):
        import numpy as np
        H,W=self.H,self.W
        d=np.full((H,W),10**9,dtype=np.int32)
        foods=[(x,y) for y in range(H) for x in range(W)
               if self.grid[y*W+x] in (FOOD,OASIS,HARD_NUT,TALL_FRUIT)]
        for (fx,fy) in foods:
            for y in range(H):
                for x in range(W):
                    dd=abs(x-fx)+abs(y-fy)
                    if dd<d[y,x]: d[y,x]=dd
        return d
    def reset(self):
        obs=self._sim.reset(); self.step_count=0; self._refresh()
        return np.array(obs,dtype=np.float32)
    def step(self, actions):
        self.step_count+=1
        obs,rew,done=self._sim.step([int(a) for a in actions])
        self._refresh()
        info={'step':self.step_count,
              'alive':[a.alive for a in self.agents],
              'energy':[round(a.energy,2) for a in self.agents]}
        return np.array(obs,dtype=np.float32), list(rew), list(done), info
    def obs_dim(self, a=None):
        return self._sim.obs_dim()
```

### 7.3 `src/model.py` (ViT encoder + per-agent GRU/heads — DONE, trains, but maybe wrong scale)
```python
import math, torch, torch.nn as nn, torch.nn.functional as F

N_TILE_TYPES=11
SPAT_C=12; GRID_H=64; GRID_W=64
SPAT=GRID_H*GRID_W*SPAT_C
OWN_DIM=14
OBS_DIM=SPAT+OWN_DIM          # 49166
NACT=13

def _sin_pos(H, W, d_model, device):
    pe=torch.zeros(H*W, d_model, device=device)
    xs=torch.arange(W,device=device).float()/max(1,W)
    ys=torch.arange(H,device=device).float()/max(1,H)
    grid_x,grid_y=torch.meshgrid(ys,xs,indexing='ij')
    pos=torch.stack([grid_x.reshape(-1),grid_y.reshape(-1)],dim=-1)
    div=torch.exp(torch.arange(0,d_model,2,device=device).float()*(-(math.log(10000.0)/d_model)))
    pe[:,0::2]=torch.sin(pos[:,0:1]*div); pe[:,1::2]=torch.cos(pos[:,0:1]*div)
    pe2=torch.zeros_like(pe)
    pe2[:,0::2]=torch.sin(pos[:,1:2]*div); pe2[:,1::2]=torch.cos(pos[:,1:2]*div)
    return (pe+pe2)/2.0

class AgentPolicyBatch(nn.Module):
    def __init__(self, N, d_model=128, n_layers=2, n_heads=4, gru_hidden=128):
        super().__init__()
        self.N=N; self.d_model=d_model; self.gru_hidden=gru_hidden
        self.token_embed=nn.Linear(SPAT_C, d_model); _orth(self.token_embed.weight,1.0)
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
            dim_feedforward=d_model*4,dropout=0.0,activation='gelu',
            batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.grid_bias=nn.Parameter(torch.zeros(N,d_model))
        gru_in=d_model+OWN_DIM
        self.gru_Wih=nn.Parameter(torch.zeros(N,3*gru_hidden,gru_in))
        self.gru_Whh=nn.Parameter(torch.zeros(N,3*gru_hidden,gru_hidden))
        self.gru_b=nn.Parameter(torch.zeros(N,3*gru_hidden))
        for p in (self.gru_Wih,self.gru_Whh): _orth(p,1.0)
        self.actor_w=nn.Parameter(torch.zeros(N,NACT,gru_hidden)); _orth(self.actor_w,1.0)
        self.actor_b=nn.Parameter(torch.zeros(N,NACT))
        self.critic_w=nn.Parameter(torch.zeros(N,1,gru_hidden)); _orth(self.critic_w,0.01)
        self.critic_b=nn.Parameter(torch.zeros(N,1))
        self.register_buffer('pe',_sin_pos(GRID_H,GRID_W,d_model,'cpu'))
    def _encode_grid(self, spat):
        K=spat.size(0); device=spat.device
        if self.pe.device!=device: self.pe=self.pe.to(device)
        pe=self.pe.unsqueeze(0).expand(K,-1,-1)
        tok=self.token_embed(spat)+pe
        cls=self.cls_token.expand(K,-1,-1)
        seq=torch.cat([cls,tok],dim=1)
        out=self.transformer(seq)
        return out[:,0,:].clamp_(-50.0,50.0)
    def _own(self, obs):
        NK=obs.size(0); return obs[:,SPAT:].view(self.N,NK//self.N,OWN_DIM)
    def _encode(self, obs):
        NK=obs.size(0); K=NK//self.N
        spat=obs[0::self.N,:SPAT].view(K,GRID_H*GRID_W,SPAT_C)
        grid_emb=self._encode_grid(spat)
        f=grid_emb.unsqueeze(0).expand(self.N,K,self.d_model)+self.grid_bias.unsqueeze(1)
        own=self._own(obs); return f,own
    def _gru_step(self, x_in, h):
        gi=torch.einsum('ndi,nki->nkd',self.gru_Wih,x_in)+self.gru_b.unsqueeze(1)
        gh=torch.einsum('ndh,nkh->nkd',self.gru_Whh,h)
        i_r,i_i,i_n=gi.chunk(3,dim=-1); h_r,h_i,h_n=gh.chunk(3,dim=-1)
        r=torch.sigmoid(i_r+h_r); z=torch.sigmoid(i_i+h_i)
        n=torch.tanh(i_n+r*h_n); h_new=(1-z)*n+z*h
        return h_new.clamp_(-50.0,50.0)
    def forward(self, obs, hidden):
        NK=obs.size(0); K=NK//self.N
        f,own=self._encode(obs)
        x_in=torch.cat([f,own],dim=-1)
        if hidden is None: hidden=torch.zeros(self.N,K,self.gru_hidden,device=obs.device,dtype=obs.dtype)
        h_new=self._gru_step(x_in,hidden)
        logits=torch.einsum('nad,nkd->nka',self.actor_w,h_new)+self.actor_b.unsqueeze(1)
        value=torch.einsum('nad,nkd->nka',self.critic_w,h_new)+self.critic_b.unsqueeze(1)
        return logits.reshape(NK,NACT), value.reshape(NK,1), h_new
    def forward_agent(self, i, obs, hidden):
        K=obs.size(0)
        spat=obs[:,:SPAT].view(K,GRID_H*GRID_W,SPAT_C)
        grid_emb=self._encode_grid(spat)
        f=grid_emb+self.grid_bias[i]; own=obs[:,SPAT:].view(K,OWN_DIM)
        x_in=torch.cat([f,own],dim=-1)
        if hidden is None: hidden=torch.zeros(1,K,self.gru_hidden,device=obs.device,dtype=obs.dtype)
        h0=hidden[0] if hidden.dim()==3 else hidden
        gi=torch.einsum('di,ki->kd',self.gru_Wih[i],x_in)+self.gru_b[i].unsqueeze(0)
        gh=torch.einsum('dh,kh->kd',self.gru_Whh[i],h0)
        i_r,i_i,i_n=gi.chunk(3,dim=-1); h_r,h_i,h_n=gh.chunk(3,dim=-1)
        r=torch.sigmoid(i_r+h_r); z=torch.sigmoid(i_i+h_i)
        n=torch.tanh(i_n+r*h_n); h_new=(1-z)*n+z*h0
        logits=torch.einsum('ad,kd->ka',self.actor_w[i],h_new)+self.actor_b[i].unsqueeze(0)
        value=torch.einsum('ad,kd->ka',self.critic_w[i],h_new)+self.critic_b[i].unsqueeze(0)
        return logits,value,h_new.unsqueeze(0)
    def params_of(self, i):
        return [self.grid_bias[i],self.gru_Wih[i],self.gru_Whh[i],self.gru_b[i],
                self.actor_w[i],self.actor_b[i],self.critic_w[i],self.critic_b[i]]

def _orth(w, gain=1.0): nn.init.orthogonal_(w, gain=gain)
```

### 7.4 `src/train.py` (the SLOW rollout loop — NEEDS REDESIGN)
Full file at `/d/rl-emergence/src/train.py` (300 lines). Key structure:
- `run(n,grid,k,nstep,nupd,seed,log_every,lr,clip,ent_coef,vf_coef,n_epochs,
       minibatch,ckpt_dir,exp_name,save_every,resume,respawn,curriculum)`.
- Creates `policy=AgentPolicyBatch(n).to(device)`, `opt=Adam(lr=2.5e-4,eps=1e-5)`.
- `bufs=[RolloutBuffer(nstep) for _ in range(n)]` — one buffer per agent.
- `hiddens=[[None]*n for _ in range(k)]`, `obs_now=[envs[e].reset() for e in range(k)]`.
- **Main loop**: for upd in range(nupd): reset bufs; for t in range(nstep):
  build `obs_batched=torch.as_tensor(np.concatenate(obs_now,axis=0))` (one GPU
  copy), build `hid_batched` via **128-iter double loop**, `policy(obs_batched,
  hid_batched)` → sample acts, call `envs[e].step(vec)` for each e (4 Python calls),
  detect gate-open via Python grid scan, push into per-agent buffers.
- **PPO update**: for i in range(n): bootstrap value, `ppo_update_agent(...)`
  which does GAE + n_epochs minibatch SGD on agent i's buffer (einsum streams are
  independent so opt.step() only moves agent i's params — see model.params_of).
- Metrics: alive, deaths/step, harv/step, gateopen.
- Args: --n --grid --k --nstep --nupd --seed --log_every --ckpt_dir --exp
  --save_every --resume --respawn --ent_coef --curriculum.

**This file is the pace bottleneck.** The per-step `np.concatenate` + 128-iter
`hid_batched` loop + per-step `.detach()` buffer appends + 4 Python `env.step`
calls (each crossing the pybind boundary) are the slow parts on a contended CPU.

### 7.5 `src/ppo.py` (RolloutBuffer + GAE; mostly used by an alternate single-agent path)
Full file at `/d/rl-emergence/src/ppo.py` (136 lines). `RolloutBuffer.compute_gae`,
`PPOTrainer.act/update`. NOTE: train.py uses its OWN inline PPO (`ppo_update_agent`
defined in train.py), not this `PPOTrainer` class. ppo.py is legacy/alternate. GAE
logic is standard (γ=0.99, λ=0.95).

### 7.6 `src/render.py` (headless GIF renderer — WORKS)
Full file at `/d/rl-emergence/src/render.py` (185 lines). Loads a checkpoint,
replays one episode, draws tiles + agents (colored by archetype, radius by energy,
glyph by last action), writes an animated GIF. `--ckpt --n --grid --out --steps
--seed --greedy --fps`. NOTE: render.py's `arch_of()` checks `t.size=='large'`
which will crash (Trait has no `.size` string; it's `size_small`). That's a latent
bug in the renderer's archetype detection — fix if rendering.

---

## 8. HARDWARE / ENV NOTES (critical for where to run)

- **5090 box**: `ssh qn06142@100.111.219.70`. RTX 5090 (32 GB). venv
  `/home/qn06142/venv` (torch 2.10.0+cu128). CPU is SHARED/CONTENDED (another
  tenant runs ffmpeg + python). CUDA MPS: export
  `CUDA_MPS_PIPE_DIRECTORY=/tmp/fake_mps`, `CUDA_MPS_LOG_DIRECTORY=/tmp/fake_mps_log`.
  The `cpp_sim.so` is HERE. Training must run here. Launch detached with
  `setsid nohup ... & disown < /dev/null` (an SSH drop kills a normal background
  process). A bogoclient cron also uses this GPU via MPS.
- **Laptop (Wheatley's)**: RTX 4060 8 GB, free CPU, but `cpp_sim.so` is NOT built
  here → can't run training without compiling the C++ (g++ + pybind11 available on
  laptop too if needed).
- Do NOT `pip install` / modify system on the 5090 box beyond the venv.

---

## 9. WHAT TO REDESIGN (summary for Gemini Pro)

1. **Pace**: The env (C++) is fast (0.28 ms/step). The training loop is slow
   (~76 min/update) due to Python rollout glue. Either (a) vectorize train.py
   (single obs concat, no per-step double-loop, batch the buffer pushes), or
   (b) move the entire rollout into C++ so Python only sees update boundaries.
   Without this, no experiment finishes in reasonable time.

2. **Learning**: harv/step stuck ~0.07–0.23 (near random) and collapsing. The
   reward/action/world design does not actually teach foraging. Reconsider:
   - Is FOOD_PULL(0.02) + NAV_ALPHA(0.05) enough to bootstrap navigation? Probably
     not — needs stronger shaping or a denser signal.
   - Is the 13-action space too sparse for harvest discovery? Consider action
     reparam or curriculum that makes harvest trivial first.
   - Does the full-grid ViT + GRU overkill early curriculum? Maybe a tiny encoder
     for L0, scale up later.
   - The author wants world/reward shaping to FORCE specialization — design the
     reward curve and world so the locally-optimal policy IS the cooperative one.

3. **Metrics**: report honest metrics — deaths/step, harv/step, gateopen. Do NOT
   use avgE (inflated by respawn). Verify the policy is actually moving toward food
   (toward-food move rate), not just idling.

4. **Keep**: the C++ sim engine (cpp_sim) and its ds optimizations (no food BFS,
   occupancy grid). The obs layout (N,49166). The shared-encoder / per-agent-GRU
   architecture is reasonable but open to revision.

5. **Bugs to fix if reusing this code**: render.py `arch_of` uses `t.size=='large'`
   (Trait has no `.size`); the C++ `step()` `pre_dist` uses `nearest_food_dist`
   which returns 1e9 when no food exists (fine for reward, but check edge cases);
   train.py gate-open detection scans `grid[gy][gx]!=6` per step (cheap but Python).

---

## 10. EXACT FILE PATHS (on the 5090 box, co-located with this doc copy)

- `~/rl_emergence/cpp/sim.cpp`  (full C++ source, 484 lines)
- `~/rl_emergence/cpp_sim.so`    (compiled module — importable as `cpp_sim`)
- `~/rl_emergence/src/env.py`     (wrapper)
- `~/rl_emergence/src/model.py`   (ViT + GRU policy)
- `~/rl_emergence/src/train.py`   (PPO loop — the slow part)
- `~/rl_emergence/src/ppo.py`     (legacy GAE/PPO)
- `~/rl_emergence/src/render.py`  (GIF renderer)
- `~/rl_emergence/ckpts/_L0/`     (last L0 run, step0/100/200 ckpts, 10.5MB each)
- This file is also saved at `~/rl_emergence/EMERGENCEGRID_DUMP.md` and
  `D:\rl-emergence\EMERGENCEGRID_DUMP.md`.
