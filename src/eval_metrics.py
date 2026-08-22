"""Rigorous, FULLY-INSTRUMENTED evaluation harness for EmergenceGrid policies.

Measures EVERYTHING so we stop hypothesizing blind:
  - 3-step pipeline funnel (gated food): reached -> mutated_near -> gained_right
    -> harvested, with per-stage conversion rates (where does the chain leak?)
  - trait dynamics: gain/loss events, mean strength/reach
  - ground-truth distances (actual Manhattan, not the obs proxy)
  - gate progress (L3): max strength, gate-chain-possible, adjacency
  - reward probe: total reward yielded per action type
  - multi-seed: mean +/- std so A/B isn't single-seed noise

Usage:
  python src/eval_metrics.py --ckpt ckpts/L2static/L2static_policy.pt --n 1 \
      --grid 64 --curriculum 2 --seeds 5 --steps 400
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
from model import AgentPolicyBatch
from env import EmergenceGrid

ACTION_NAMES = ["idle", "up", "right", "down", "left", "harvest", "share",
                "signal", "str+", "str-", "reach+", "reach-", "speed+"]


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
        prev_inv = [a.inv for a in env.agents]
        total_harvest = 0
        gate_reachable = False
        gate_opened = 0

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
            gc = env.gate_cells
            before = sum(1 for (gx, gy) in gc
                         if env.grid[gy * env.W + gx] != 6) if gc else 0
            o, r, d, info = env.step(acts_l)
            after = sum(1 for (gx, gy) in gc
                        if env.grid[gy * env.W + gx] != 6) if gc else 0
            if after > before:
                gate_opened += 1
            for ag in env._sim.agents:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = ag.x + dx, ag.y + dy
                        if 0 <= nx < env.W and 0 <= ny < env.H:
                            if env.grid[ny * env.W + nx] == 6:  # GATE
                                # use the SIM's actual gate threshold (TH_GATE/100),
                                # not a hardcoded 1.10 that drifted from the sim.
                                if ag.tr.strength >= env.gate_threshold:
                                    gate_reachable = True
            for i in range(n):
                if env.agents[i].inv > prev_inv[i]:
                    total_harvest += 1
                prev_inv[i] = env.agents[i].inv
            hid = h_new[:, :, :].detach()
            obs = o
            if all(d):
                break

        d = env.get_diag_full()
        steps_d = max(1, int(d["steps"]))
        # pipeline funnel conversion rates
        reached = int(d["reached_gated"])
        mut_near = int(d["mutated_near_gated"])
        gained = int(d["gained_right_trait"])
        eaten = int(d["harvested_gated"])
        collect_rate = total_harvest / max(1, steps_d * n)
        per_seed.append(dict(
            seed=sd, collect_rate=collect_rate,
            invalid_rate=int(d["harvest_invalid"]) / (int(d["harvest_invalid"]) + int(d["harvest_valid"]) + 1),
            away_rate=int(d["move_away"]) / (int(d["move_away"]) + int(d["move_closer"]) + 1),
            # funnel conversion rates (each stage as fraction of prior stage)
            reached_per_step=reached / steps_d,
            mut_near_per_reached=(mut_near / reached) if reached else 0.0,
            gained_per_mut_near=(gained / mut_near) if mut_near else 0.0,
            eaten_per_gained=(eaten / gained) if gained else 0.0,
            eaten_per_reached=(eaten / reached) if reached else 0.0,
            wrong_trait_mut_rate=int(d["wrong_trait_mut"]) / (mut_near + 1),
            trait_gain_events=int(d["trait_gain_events"]),
            trait_loss_events=int(d["trait_loss_events"]),
            mean_strength=float(d["mean_strength"]),
            mean_reach=float(d["mean_reach"]),
            mean_dist_food=float(d["mean_dist_food"]),
            mean_dist_gated=float(d["mean_dist_gated"]),
            gated_reach_rate=(int(d["moved_closer_gated"]) /
                             (int(d["moved_closer_gated"]) + int(d["moved_away_gated"]) + 1)
                             if (int(d["moved_closer_gated"]) + int(d["moved_away_gated"]) > 0) else 0.0),
            max_strength=float(d["max_strength"]),
            gate_chain_possible=int(d["gate_chain_possible"]),
            gate_opened=gate_opened, gate_reachable=gate_reachable,
            dead=int(d["dead"]),
            rew_by_action=[float(x) for x in d["rew_by_action"]],
        ))

    def agg(key):
        vals = [p[key] for p in per_seed]
        return float(np.mean(vals)), float(np.std(vals))

    m_c, s_c = agg('collect_rate')
    m_i, s_i = agg('invalid_rate')
    m_a, s_a = agg('away_rate')
    # funnel
    fr, fs = agg('reached_per_step')
    mnr, mns = agg('mut_near_per_reached')
    gmr, gms = agg('gained_per_mut_near')
    egr, egs = agg('eaten_per_gained')
    epg, eps = agg('eaten_per_reached')
    wtm, wtms = agg('wrong_trait_mut_rate')
    ms, mss = agg('mean_strength')
    mr, mrs = agg('mean_reach')
    mdf, mdfs = agg('mean_dist_food')
    mdg, mdgs = agg('mean_dist_gated')
    gr, grs = agg('gated_reach_rate')
    mxs, mxss = agg('max_strength')
    reachable = sum(1 for p in per_seed if p['gate_reachable'])
    opened = sum(p['gate_opened'] for p in per_seed)
    gate_rate = (opened / reachable) if reachable > 0 else float('nan')
    # reward probe averaged across seeds (per-action mean reward)
    rew_mean = [0.0] * 13
    for p in per_seed:
        for i in range(13):
            rew_mean[i] += p['rew_by_action'][i]
    rew_mean = [v / len(per_seed) for v in rew_mean]
    return dict(
        n_seeds=len(seeds),
        gate_threshold=float(env.gate_threshold),
        collect_rate=(m_c, s_c), invalid_rate=(m_i, s_i), away_rate=(m_a, s_a),
        reached_per_step=(fr, fs),
        mut_near_per_reached=(mnr, mns),
        gained_per_mut_near=(gmr, gms),
        eaten_per_gained=(egr, egs),
        eaten_per_reached=(epg, eps),
        wrong_trait_mut_rate=(wtm, wtms),
        mean_strength=(ms, mss), mean_reach=(mr, mrs),
        mean_dist_food=(mdf, mdfs), mean_dist_gated=(mdg, mdgs),
        gated_reach_rate=(gr, grs),
        max_strength=(mxs, mxss),
        gate_opened_total=opened, gate_reachable_seeds=reachable,
        gate_rate_over_reachable=gate_rate,
        dead_total=sum(p['dead'] for p in per_seed),
        rew_by_action=rew_mean,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--curriculum', type=int, default=1)
    ap.add_argument('--seeds', type=str, default='12345,777,2024,99,42')
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
    R = evaluate(args.ckpt, n=args.n, grid=args.grid, steps=args.steps,
                 seeds=seeds, greedy=args.greedy, curriculum=args.curriculum,
                 food_seed=args.food_seed, food_seed_dist=args.food_seed_dist,
                 food_density_div=args.food_density_div,
                 food_regen_mode=args.food_regen_mode, gated_food=args.gated_food,
                 d_model=args.d_model, gru_hidden=args.gru_hidden,
                 head_dim=args.head_dim, harvest_bias=args.harvest_bias)
    p = lambda k: f"{R[k][0]:.4f} +/- {R[k][1]:.4f}"
    print(f"[eval] {args.ckpt}  seeds={R['n_seeds']} steps={args.steps} greedy={args.greedy}")
    print(f"  collect_rate (harv/step)   = {p('collect_rate')}")
    print(f"  invalid_harvest_rate       = {p('invalid_rate')}")
    print(f"  move_away_rate             = {p('away_rate')}")
    print("  --- 3-step pipeline funnel (gated food) ---")
    print(f"  reached_gated /step        = {p('reached_per_step')}")
    print(f"  mut_near | reached         = {p('mut_near_per_reached')}")
    print(f"  gained_right | mut_near    = {p('gained_per_mut_near')}")
    print(f"  EATEN | gained_right       = {p('eaten_per_gained')}")
    print(f"  EATEN | reached (end-to-end) = {p('eaten_per_reached')}")
    print(f"  wrong_trait_mut_rate       = {p('wrong_trait_mut_rate')}")
    print(f"  --- trait dynamics ---")
    print(f"  mean_strength              = {p('mean_strength')}")
    print(f"  mean_reach                 = {p('mean_reach')}")
    print(f"  --- ground-truth distances ---")
    print(f"  mean_dist_food             = {p('mean_dist_food')}")
    print(f"  mean_dist_gated            = {p('mean_dist_gated')}")
    print(f"  gated_reach_rate (closer/total) = {p('gated_reach_rate')}")
    print(f"  --- gate progress (L3) ---")
    print(f"  max_strength               = {p('max_strength')}")
    print(f"  gate_chain_possible        = {R['gate_reachable_seeds']>0 or R['max_strength'][0]>=R['gate_threshold']}")
    print(f"  gate_opened (total)        = {R['gate_opened_total']}")
    print(f"  gate_reachable_seeds       = {R['gate_reachable_seeds']}/{R['n_seeds']}")
    gr = R['gate_rate_over_reachable']
    print(f"  gateopen RATE (reachable)  = {('%.4f' % gr) if gr==gr else 'n/a (0 reachable)'}")
    print(f"  dead (total)               = {R['dead_total']}")
    rb = R['rew_by_action']
    top = sorted(range(13), key=lambda i: rb[i], reverse=True)[:4]
    print(f"  --- reward probe (mean reward/action) ---")
    print("  " + "  ".join(f"{ACTION_NAMES[i]}:{rb[i]:.2f}" for i in top))
    print(f"  [all actions] " + " ".join(f"{ACTION_NAMES[i]}={rb[i]:.2f}" for i in range(13)))


if __name__ == '__main__':
    main()
