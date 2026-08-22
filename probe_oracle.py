"""Oracle / teacher-forcing probe: a SCRIPTED agent that always takes the
correct action (mutate the right trait when adjacent to gated food, navigate to
gated food / gates, hold adjacent to a gate once strong enough). If this oracle
can open gates / eat gated food, the TASK IS LEARNABLE and the bottleneck is the
LEARNER (PPO / architecture) -- not reward, horizon, or curriculum. If even the
oracle fails, the SIM MECHANICS (cooldown / threshold / gate logic) are broken.

Usage: python probe_oracle.py [--steps 400] [--seed 12345]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
from env import EmergenceGrid

# tile types (mirror sim.cpp)
EMPTY, FOOD, HARD_NUT, TALL_FRUIT, GAP, WALL, GATE = 0, 1, 2, 3, 4, 5, 6
HAZARD, PREDATOR, OASIS, MARKER = 7, 8, 9, 10
# actions
IDLE, UP, RIGHT, DOWN, LEFT, HARVEST = 0, 1, 2, 3, 4, 5
STR_P, STR_M, REACH_P, REACH_M, SPD_P = 8, 9, 10, 11, 12

ACT_NAMES = {0:'idle',1:'up',2:'right',3:'down',4:'left',5:'harvest',
             8:'str+',9:'str-',10:'reach+',11:'reach-',12:'speed+'}

def scan_nearest_target(env, x, y, targets=(HARD_NUT, TALL_FRUIT, GATE)):
    """Brute-force nearest tile of given types (oracle only; one-shot probe)."""
    best_d, best = 1e9, None
    for yy in range(env.H):
        for xx in range(env.W):
            t = env._sim.get_tile(xx, yy)
            if t in targets:
                d = abs(xx - x) + abs(yy - y)
                if d < best_d:
                    best_d, best = d, (xx, yy)
    return best

def nav_action(env, x, y, tx, ty):
    """Wall-aware greedy step toward (tx,ty): prefer larger axis, but if that cell
    is a WALL, fall back to the other axis; if both blocked, any free neighbor."""
    dx, dy = tx - x, ty - y
    WALL = 5
    def free(nx, ny):
        return 0 <= nx < env.W and 0 <= ny < env.H and env._sim.get_tile(nx, ny) != WALL
    # primary axis = larger abs distance
    if abs(dx) >= abs(dy) and dx != 0:
        if free(x + (1 if dx > 0 else -1), y):
            return RIGHT if dx > 0 else LEFT
        if dy != 0 and free(x, y + (1 if dy > 0 else -1)):
            return DOWN if dy > 0 else UP
    elif dy != 0:
        if free(x, y + (1 if dy > 0 else -1)):
            return DOWN if dy > 0 else UP
        if dx != 0 and free(x + (1 if dx > 0 else -1), y):
            return RIGHT if dx > 0 else LEFT
    # degenerate (on target or fully boxed): idle
    return IDLE

def bfs_step(env, x, y, tx, ty):
    """BFS from (x,y) to (tx,ty) avoiding WALL; return first-step action (or IDLE)."""
    from collections import deque
    W, H = env.W, env.H
    if (x, y) == (tx, ty):
        return IDLE
    q = deque([(x, y)])
    prev = {(x, y): None}
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == (tx, ty):
            break
        for dxx, dyy, a in ((1,0,RIGHT),(-1,0,LEFT),(0,1,DOWN),(0,-1,UP)):
            nx, ny = cx+dxx, cy+dyy
            if 0<=nx<W and 0<=ny<H and (nx,ny) not in prev:
                if env._sim.get_tile(nx, ny) == WALL:
                    continue
                prev[(nx, ny)] = (cx, cy, a)
                q.append((nx, ny))
    if (tx, ty) not in prev:
        return IDLE
    # walk back from target to find first step from (x,y)
    cx, cy = tx, ty
    while prev[(cx, cy)] is not None:
        pcx, pcy, pa = prev[(cx, cy)]
        if (pcx, pcy) == (x, y):
            return pa
        cx, cy = pcx, pcy
    return IDLE

def adjacent_spot(env, fx, fy):
    """Return a free (non-WALL, in-bounds) 4-neighbor cell of (fx,fy), or None."""
    for dxx, dyy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = fx+dxx, fy+dyy
        if 0<=nx<env.W and 0<=ny<env.H and env._sim.get_tile(nx, ny) != WALL:
            return (nx, ny)
    return None

def oracle_action(env, ag):
    x, y = ag['x'], ag['y']
    # scan 4-neighbors for gated food / gate
    adj_hard = adj_tall = adj_gate = adj_food = False
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = x+dx, y+dy
        if 0<=nx<env.W and 0<=ny<env.H:
            t = env._sim.get_tile(nx, ny)
            if t == HARD_NUT: adj_hard = True
            elif t == TALL_FRUIT: adj_tall = True
            elif t == GATE: adj_gate = True
            elif t == FOOD or t == OASIS: adj_food = True
    TH = 0.95   # matches sim TH_GATE=95 (x100). Strength caps at 1.0, so 0.95 opens gates.
    # 0) SURVIVE: if energy is low, go eat regular food (stand adjacent, then harvest).
    if ag['energy'] < 25:
        fpos = scan_nearest_target(env, x, y, targets=(FOOD, OASIS))
        if fpos is not None:
            spot = adjacent_spot(env, fpos[0], fpos[1])
            if spot is None:
                return IDLE
            if spot == (x, y):
                return HARVEST   # already adjacent to food: eat it
            return bfs_step(env, x, y, spot[0], spot[1])
    # 1) BUILD STRENGTH to gate threshold (1.10) -- only strength opens gates.
    if ag['cooldown'] == 0 and ag['strength'] < TH and (adj_hard or adj_gate):
        return STR_P
    # 2) hold adjacent to a gate once strong enough (let resolve_gates fire)
    if adj_gate and ag['strength'] >= TH:
        return IDLE
    # 3) harvest regular food for ENERGY (survival) whenever adjacent
    if adj_food:
        return HARVEST
    # 4) if strong enough but not at a gate, navigate to be ADJACENT to the nearest gate
    if ag['strength'] >= TH and not adj_gate:
        gpos = scan_nearest_target(env, x, y, targets=(GATE,))
        if gpos is not None:
            gspot = adjacent_spot(env, gpos[0], gpos[1])
            if gspot is not None:
                if gspot == (x, y):
                    return IDLE   # already adjacent to gate: hold (rule #2 handles open)
                nav = bfs_step(env, x, y, gspot[0], gspot[1])
                print(f"  [dbg r4] str={ag['strength']:.2f} at ({x},{y}) gpos={gpos} gspot={gspot} nav={nav}")
                return nav
    # 5) STAY PUT during cooldown when adjacent to a gated target, so the next
    #    mutation (after cooldown) lands on the same target.
    if (adj_hard or adj_tall or adj_gate) and ag['cooldown'] > 0:
        return IDLE
    # 6) navigate to be ADJACENT to nearest HARD_NUT (build strength toward gate)
    tgt = scan_nearest_target(env, x, y, targets=(HARD_NUT,))
    if tgt is None:
        tgt = scan_nearest_target(env, x, y, targets=(GATE,))
    if tgt is None:
        return IDLE
    tspot = adjacent_spot(env, tgt[0], tgt[1])
    if tspot is None:
        return IDLE
    if tspot == (x, y):
        return IDLE   # already adjacent to HARD_NUT; rule #1 will mutate
    return bfs_step(env, x, y, tspot[0], tspot[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--curriculum', type=int, default=3)
    ap.add_argument('--eat_gain_regular', type=float, default=0.0)
    args = ap.parse_args()

    env = EmergenceGrid(width=args.grid, height=args.grid, n_agents=1, seed=args.seed,
                        curriculum=args.curriculum, food_seed=0, food_seed_dist=1,
                        respawn=True, food_density_div=50, food_regen_mode=2, gated_food=1)
    env.set_reward_params(eat_gain_regular=args.eat_gain_regular)
    obs = env.reset()
    # diagnostic: count tile types in the initial grid
    counts = {}
    for yy in range(env.H):
        for xx in range(env.W):
            t = env._sim.get_tile(xx, yy)
            counts[t] = counts.get(t, 0) + 1
    names = {0:'EMPTY',1:'FOOD',2:'HARD_NUT',3:'TALL_FRUIT',4:'GAP',5:'WALL',
             6:'GATE',7:'HAZARD',8:'PREDATOR',9:'OASIS',10:'MARKER'}
    print("[diag] tile counts:", {names.get(k,k): v for k, v in sorted(counts.items())})
    prev_inv = env.agents[0].inv
    total_harvest = 0
    gate_opened = 0
    gate_reachable = False
    max_strength_seen = 0.0
    for step in range(args.steps):
        ags = env._sim.dump_agents()
        ag = ags[0]
        max_strength_seen = max(max_strength_seen, ag['strength'])
        act = oracle_action(env, ag)
        gc = env.gate_cells
        before = sum(1 for (gx, gy) in gc if env.grid[gy*env.W+gx] != GATE) if gc else 0
        obs, r, d, info = env.step([act])
        after = sum(1 for (gx, gy) in gc if env.grid[gy*env.W+gx] != GATE) if gc else 0
        if after > before:
            gate_opened += 1
        if env.agents[0].inv > prev_inv:
            total_harvest += 1
        prev_inv = env.agents[0].inv
        if any(env._sim.get_tile(gx, gy) == GATE and ag['strength'] >= 1.10
               for (gx, gy) in (gc or [])):
            gate_reachable = True
        if all(d):
            break
        if step % 40 == 0:
            agd = env._sim.dump_agents()[0]
            print(f"  step {step}: str={agd['strength']:.2f} reach={agd['reach']:.2f} "
                  f"cd={agd['cooldown']} e={agd['energy']:.0f} alive={agd['alive']} act={act}")
    print(f"[oracle] curriculum={args.curriculum} eat_gain_regular={args.eat_gain_regular}")
    print(f"  steps run, total_harvest(gated+reg)={total_harvest}")
    print(f"  gate_opened = {gate_opened}")
    print(f"  max_strength_seen = {max_strength_seen:.3f}")
    print(f"  gate_reachable (adjacent+strong at some pt) = {gate_reachable}")
    if gate_opened > 0:
        print("  >>> ORACLE OPENED GATES: task IS learnable; bottleneck is the LEARNER (PPO/architecture).")
    else:
        print("  >>> ORACLE FAILED to open gates: SIM MECHANICS are broken (cooldown/threshold/gate logic).")

if __name__ == '__main__':
    main()
