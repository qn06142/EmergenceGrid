"""Rigorous evaluation harness for EmergenceGrid policies.

Fixes the three metric flaws we identified:
  1. Normalizes by OPPORTUNITY/seed, not single noisy snapshots -> reports
     mean +/- std across MULTIPLE seeds.
  2. gateopen is reported as a RATE over REACHABLE episodes (an episode is
     "reachable" if the agent was ever adjacent to a gate with enough strength
     to open it). gateopen=0 is then distinguishable from "never reachable".
  3. Reports PER-FAILURE-MODE RATES (from the sim's closed-loop diagnostics)
     so a config change can be attributed to a specific mechanism, not a
     single ambiguous harv/step number.

Usage:
  python src/eval_metrics.py --ckpt ckpts/L1ctrl/L1ctrl_policy.pt --n 1 --grid 64 \
      --curriculum 1 --seeds 5 --steps 400 --greedy
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
from model import AgentPolicyBatch
from env import EmergenceGrid


def evaluate(ckpt, n=1, grid=64, steps=400, seeds=(12345,), greedy=False,
             curriculum=1, food_seed=0, food_seed_dist=1, food_density_div=50,
             food_regen_mode=2, gated_food=1, d_model=256, gru_hidden=256,
             head_dim=256, harvest_bias=0.0, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    policy = AgentPolicyBatch(n, d_model=d_model, gru_hidden=gru_hidden,
                              head_dim=head_dim).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    per_seed = []
    for sd in seeds:
        env = EmergenceGrid(width=grid, height=grid, n_agents=n, seed=sd,
                            curriculum=curriculum, food_seed=food_seed,
                            food_seed_dist=food_seed_dist, respawn=True,
                            food_density_div=food_density_div,
                            food_regen_mode=food_regen_mode, gated_food=gated_food)
        obs = env.reset()
        hid = torch.zeros(n, 1, policy.gru_hidden, device=device)

        # gate reachability tracking (L3): was the agent ever adjacent to a gate
        # with enough cumulative strength to open it?
        gate_reachable = False
        gate_opened = 0
        prev_inv = [a.inv for a in env.agents]
        total_harvest = 0
        total_steps = 0

        for step in range(steps):
            obs_b = torch.as_tensor(obs[:n], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, vals, h_new = policy(obs_b, hid)
            if harvest_bias:
                logits = logits.clone()
                logits[:, 5] += harvest_bias
            dist = torch.distributions.Categorical(logits=logits)
            acts = torch.argmax(logits, dim=-1) if greedy else dist.sample()
            acts_l = acts.cpu().tolist()
            # gate state before step
            gc = env.gate_cells
            before = sum(1 for (gx, gy) in gc
                         if env.grid[gy * env.W + gx] != 6) if gc else 0
            o, r, d, info = env.step(acts_l)
            after = sum(1 for (gx, gy) in gc
                        if env.grid[gy * env.W + gx] != 6) if gc else 0
            if after > before:
                gate_opened += 1
            # gate reachability: adjacent to a gate with strength >= threshold
            for ag in env._sim.agents:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = ag.x + dx, ag.y + dy
                        if 0 <= nx < env.W and 0 <= ny < env.H:
                            if env.grid[ny * env.W + nx] == 6:  # GATE
                                # TH_GATE = 1.10 in sim.cpp
                                if ag.tr.strength >= 1.10:
                                    gate_reachable = True
            for i in range(n):
                if env.agents[i].inv > prev_inv[i]:
                    total_harvest += 1
                prev_inv[i] = env.agents[i].inv
            total_steps += n
            hid = h_new[:, :, :].detach()
            obs = o
            if all(d):
                break

        # aggregate diagnostics from the sim (per-failure-mode rates)
        diag = env.get_diag()  # (steps, hi, hv, maway, mclose, mut, gadj, gstr, dead)
        steps_d, hi, hv, maway, mclose, mut, gadj, gstr, dead = diag
        steps_d = max(1, steps_d)
        invalid_rate = hi / (hi + hv + 1)
        away_rate = maway / (maway + mclose + 1)
        # mutate->eat conversion: of steps spent mutating, how many adjacent-eats?
        mut_to_eat = hv / (mut + 1)
        collect_rate = total_harvest / max(1, total_steps)  # honest harv/step
        per_seed.append(dict(
            seed=sd, collect_rate=collect_rate, invalid_rate=invalid_rate,
            away_rate=away_rate, mut_to_eat=mut_to_eat,
            gate_opened=gate_opened, gate_reachable=gate_reachable,
            dead=int(dead), diag_steps=int(steps_d)))

    # aggregate with mean +/- std across seeds
    def agg(key, fn=float):
        vals = [p[key] for p in per_seed]
        return float(np.mean(vals)), float(np.std(vals))
    m_c, s_c = agg('collect_rate')
    m_i, s_i = agg('invalid_rate')
    m_a, s_a = agg('away_rate')
    m_m, s_m = agg('mut_to_eat')
    reachable = sum(1 for p in per_seed if p['gate_reachable'])
    opened = sum(p['gate_opened'] for p in per_seed)
    # gateopen RATE over REACHABLE episodes (0 if none reachable -> honest "n/a")
    gate_rate = (opened / reachable) if reachable > 0 else float('nan')
    return dict(
        n_seeds=len(seeds),
        collect_rate=(m_c, s_c),
        invalid_rate=(m_i, s_i),
        away_rate=(m_a, s_a),
        mut_to_eat=(m_m, s_m),
        gate_opened_total=opened,
        gate_reachable_seeds=reachable,
        gate_rate_over_reachable=gate_rate,
        dead_total=sum(p['dead'] for p in per_seed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--curriculum', type=int, default=1)
    ap.add_argument('--seeds', type=str, default='12345,777,2024,99,42',
                    help='comma-separated seeds for multi-seed evaluation')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--harvest_bias', type=float, default=0.0)
    ap.add_argument('--food_seed', type=int, default=0)
    ap.add_argument('--food_seed_dist', type=int, default=1)
    ap.add_argument('--food_density_div', type=int, default=50)
    ap.add_argument('--food_regen_mode', type=int, default=2)
    ap.add_argument('--gated_food', type=int, default=1)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--gru_hidden', type=int, default=256)
    ap.add_argument('--head_dim', type=int, default=256)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s != '']
    t0 = time.time()
    R = evaluate(args.ckpt, n=args.n, grid=args.grid, steps=args.steps,
                 seeds=seeds, greedy=args.greedy, curriculum=args.curriculum,
                 food_seed=args.food_seed, food_seed_dist=args.food_seed_dist,
                 food_density_div=args.food_density_div,
                 food_regen_mode=args.food_regen_mode, gated_food=args.gated_food,
                 d_model=args.d_model, gru_hidden=args.gru_hidden,
                 head_dim=args.head_dim, harvest_bias=args.harvest_bias)
    print(f"[eval] {args.ckpt}")
    print(f"  seeds={R['n_seeds']}  steps={args.steps}  greedy={args.greedy}")
    print(f"  collect_rate (harv/step)   = {R['collect_rate'][0]:.4f} +/- {R['collect_rate'][1]:.4f}")
    print(f"  invalid_harvest_rate       = {R['invalid_rate'][0]:.4f} +/- {R['invalid_rate'][1]:.4f}")
    print(f"  move_away_rate             = {R['away_rate'][0]:.4f} +/- {R['away_rate'][1]:.4f}")
    print(f"  mutate->eat conversion     = {R['mut_to_eat'][0]:.4f} +/- {R['mut_to_eat'][1]:.4f}")
    print(f"  gate_opened (total)        = {R['gate_opened_total']}")
    print(f"  gate_reachable_seeds       = {R['gate_reachable_seeds']}/{R['n_seeds']}")
    gr = R['gate_rate_over_reachable']
    print(f"  gateopen RATE (reachable)  = {('%.4f' % gr) if gr==gr else 'n/a (0 reachable)'}")
    print(f"  dead (total)               = {R['dead_total']}")
    print(f"  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
