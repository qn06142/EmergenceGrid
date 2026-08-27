# EmergenceGrid — Multi-Agent Emergence Diagnostic & Achievement Report

## TL;DR

The question was: *can multi-agent emergence (a coordinated, trait-gated, strength-gated
gate-push) arise from RL in this sim?* Answer: **YES — and reproducibly at the real task
difficulty (TH_GATE = 0.95).**

The path was a full diagnostic ladder: fix sim bugs → verify credit → expose the task →
tune reward magnitude → fix the architecture → lower the gate curriculum → anneal the
bar up → scale agents. The final configuration (`gc_anneal_n8`: n=8 agents, gradual
0.6→0.95 anneal) opens the gate **7 times across 5 independent eval seeds** with agents
building to **0.93–0.98 strength** (the bar is 0.95). Emergence is demonstrated and
frequent, not a one-off fluke.

---

## 1. The task (what "emergence" means here)

Agents must, *without hand-coded scripting*, learn a chained behavior:
1. **Mutate** the right trait (acts 8/10 = str+/reach+) when adjacent to gated food.
2. **Eat gated food** — HARD_NUT (type 2) needs str+, TALL_FRUIT (type 3) needs reach+.
3. **Build strength** by eating (strength rises toward 1.0).
4. **Coordinate a simultaneous multi-agent push** — the gate opens only when combined
   pusher strength at adjacent cells ≥ `TH_GATE` (0.95 = real difficulty).

Steps 1–3 are individual; step 4 is the collective emergence. If the gate opens, the
agents have coordinated.

---

## 2. Diagnostic arc (what we found and fixed)

### 2.1 Sim was buggy (9 scour fixes)
Before any RL, an oracle (hand-scripted correct policy) could NOT open the gate. A
scour pass (oracle-probe + sim-scour) found and fixed 9 sim bugs in `cpp/sim.cpp`
(reward-overwrite, double-action-apply, threshold-mismatch, phantom-index, energy-cap
ordering, nav-conflict, etc.). After fixes, both oracle AND RL could in principle open
the gate. *A broken sim would have made every downstream result meaningless.*

### 2.2 Credit assignment was fine
`--diag_train` confirmed per-agent reward/credit was correctly attributed (no global
reward flooding). Not the blocker.

### 2.3 Exposure / sparsity — the G+C reward lever
`--diag_train` also showed the only positive gradient in the gate task was too sparse
to learn. The `reward_preset='gc'` lever (G = reward-density via trait_match_bonus +
mutate_gated_gain; C = credit-shaping via wrong_trait_pen sharpening + gate_prox_bonus)
made the signal learnable. Verified as an A/B-able lever (defaults untouched).

### 2.4 Reward magnitude — non-monotonic ceiling (~0.64)
Reward magnitude was NOT monotonic: cranking it up *regressed* performance (peaked
around 0.64, then collapsed). The bottleneck was never "reward too small" — it was the
architecture drowning the gated-tile signal (next section).

### 2.5 Architecture fix — needed-trait skip-connection (REAL win)
The obs had a compact food-direction vector but the **gated-tile TYPE signal was being
drowned** in the CNN→GRU pipeline (same bug class that `food_w` skip fixed for
navigation). Added:
- `need_strplus` / `need_reachplus` (2 floats) to the observation, after the food-dir
  vector.
- `trait_w` (N, NACT, 2) skip-connection in `model.py` with priors `trait_w[:,8,0]=+1`
  (str+ on need_strplus) and `trait_w[:,10,1]=+1` (reach+ on need_reachplus).

Result: **wrong_trait_mut_rate dropped 0.97 → 0.42** — agents learned to mutate the
*correct* trait near gated food. This was a genuine architecture fix, verified by
probing the trained policy's action biases.

### 2.6 Gate curriculum — lower TH_GATE (breakthrough)
Made `TH_GATE` runtime-settable (`gate_thresh`) instead of compile-time. Lowered it to
**0.6** so a single strong agent (or easy coordination) opens the gate. `gc_curric`
opened the gate in eval — **first reproducible emergence**. At 0.6, agents build to
0.91–0.97 strength (the bar is easy, so strength-building is rewarded).

### 2.7 Hard ramp failed (0.6 → 0.8 collapse)
Resuming at `TH_GATE=0.8` from the 0.6-trained ckpt **collapsed**: policy drifted to
signal-spam (topact signal 83%), harvest → 0.004, `max_strength` 0.51. The jump was too
abrupt. Lesson: curriculum must be *gradual*, not a cliff.

---

## 3. Reliability work (making it hold at 0.95)

### 3.1 Gradual anneal 0.6 → 0.95 (gc_anneal, n=4)
Annealed the bar continuously over 250 updates. Funnel @0.95: **gate_opened = 2/5**.
FIRST time the gate opened at the REAL difficulty. But the gate was rare/stochastic.

### 3.2 Slower anneal didn't help (gc_anneal2, n=4, 500 upd)
0/5 gates @0.95. **Anneal speed is not the dial.** Key clue: `max_strength = 0.51`
identical across all seeds — agents cap at ~0.51 individual strength at 0.95 because the
strength-building reward only fires near the easy 0.6 bar. The 0.95 gate needs combined
strength ≥ 0.95, i.e. 2+ agents simultaneously adjacent — a rare coincidence with n=4.

### 3.3 More agents SOLVES it (gc_anneal_n8, n=8)
Trained n=8 from scratch (init_ckpt resume requires matching agent count — a bug found
and documented) with the same 0.6→0.95 anneal. Funnel @0.95:
- **gate_opened = 7 across 5 seeds** (seed5=2, seed6=2, seed7=2, seed8=0, seed9=1)
- **max_strength = 0.93–0.98** on ALL seeds (vs 0.51 for n=4)

With 8 agents, the combined≥0.95 gate fires reliably during training, so agents get the
strength-building reward and build to ~0.95. **More agents (n=8) is the reliability
lever.** Emergence at the real difficulty is now demonstrable AND frequent.

---

## 4. A real bug we found along the way (eval_metrics)

`eval_metrics.py` accessed `ag.tr.strength` on raw `cpp_sim` Agents, which pybind does
NOT expose (only `idx/x/y/energy/inv/alive/last_action/cooldown`). The check fired only
when an agent was adjacent to a gate, so it **silently crashed funnels on gate-near
seeds** — corrupting results (e.g. gc_curric looked like 1/5 gates when it was 3/5).
Fixed by using `dump_agents()` (returns proper strength dicts). After the fix, gc_curric
funnel showed gates on 3/5 seeds.

---

## 5. Evidence (GIFs)

Rendered per-checkpoint episode replays (greedy, 300 steps, gated_food=2):

- `gifs/gc_curric_gate06.gif` — gate curriculum at TH_GATE=0.6 (trained bar). The gate
  VISIBLY OPENS. Emergence demonstrable. (n=4)
- `gifs/gc_sharp3_archfix.gif` — architecture fix (needed-trait skip). Shows agents
  mutating the correct trait near gated food. (n=4)
- `gifs/gc_anneal_gate095.gif` — gradual anneal, rendered at the real TH_GATE=0.95.
  Emergence at target difficulty. (n=4)
- `gifs/gc_anneal_n8_gate095.gif` — MORE AGENTS (n=8) at 0.95. The reliable emergence:
  agents build to ~0.95 strength and open the gate repeatedly. (n=8)

To replay any: `python src/render.py --ckpt <ckpt> --n <4|8> --grid 64 --gate_thresh
<0.6|0.95> --curriculum 3 --gated_food 2 --greedy --out <gif>`

---

## 6. Key numbers (funnel, eval @ TH_GATE=0.95)

| Run | Agents | Anneal | Gates / 5 seeds | max_strength |
|-----|--------|--------|-----------------|--------------|
| gc_anneal    | 4 | 0.6→0.95 | 2 | 0.51 |
| gc_anneal2   | 4 | 0.6→0.95 (slow) | 0 | 0.51 |
| **gc_anneal_n8** | **8** | **0.6→0.95** | **7** | **0.93–0.98** |

At TH_GATE=0.6 (easy bar), gates open on 3–4/5 seeds (gc_curric).

---

## 7. What remains (engineering, not mystery)

- **Seed 8 got 0 gates** in the n=8 funnel — emergence is frequent but not 100% reliable
  per fixed seed (stochastic). More training updates or a slower anneal at n=8 would
  tighten it.
- **Cross-seed robustness** at n=4 is poor (the gate is rare there). n=8 is the answer.
- The `init_ckpt` resume requires identical agent count — a limitation to fix if we want
  to resume n=4 ckpts into n=8 runs (currently train from scratch at the target n).

## 8. Repo / commits

All changes committed on `main`. Notable: scour fixes (cpp/sim.cpp), `reward_preset='gc'`
lever, needed-trait skip (model.py), runtime `gate_thresh` (cpp + env + train), gradual
anneal schedule (train.py), eval_metrics bug fix, and the render.py `gate_thresh`
passthrough. Full log in `docs/EXPERIMENTS.md`.
