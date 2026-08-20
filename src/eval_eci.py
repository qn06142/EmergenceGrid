"""ECI evaluation harness (§10): defensible emergence metric.

Loads a trained GROUP policy (AgentPolicyBatch(N), N=16) and a trained SOLO
policy (AgentPolicyBatch(1)) on the SAME 128x128 grid. Runs episodes and reports:
  - group:  alive%, avg energy, total credit (sum of local rewards)
  - solo:   alive%, avg energy, total credit for ONE agent (x16 for fair compare)
  - raw ECI   = (group credit - 16 * solo credit) / group credit
  - ECI_eff   = (group credit / group energy) - (solo credit / solo energy)
                [efficiency-normalized: did the GROUP extract more value per
                 unit energy than 16 isolated agents?]

A positive ECI_eff that is NOT explainable by headcount (the gate/predator are
strength-SUM locked, not count) is the defensible emergence signal.

Run: python src/eval_eci.py --group_ckpt ckpts/eg16_run1/eg16_run1_policy.pt \
                            --solo_ckpt ckpts/solo128/solo128_policy.pt \
                            --grid 128 --episodes 20
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from env import EmergenceGrid
from model import AgentPolicyBatch, OBS_DIM

HIDDEN = 128


def run_episode(policy, n, grid, seed, device, hidden=None):
    """Run one episode. policy: AgentPolicyBatch(n). Returns (alive, avgE, credit,
    energy_sum) across the episode's final state + cumulative reward."""
    env = EmergenceGrid(width=grid, height=grid, n_agents=n, seed=seed)
    obs = env.reset()
    hid = [None] * n
    total_credit = 0.0
    steps = 0
    max_steps = 400
    H = policy.gru_hidden
    while steps < max_steps:
        obs_b = torch.as_tensor(obs[:n], dtype=torch.float32, device=device)
        hid_b = torch.zeros(n, 1, H, device=device) if hidden is None \
            else torch.stack([hid[i] for i in range(n)], dim=0).to(device)
        with torch.no_grad():
            logits, vals, h_new = policy(obs_b, hid_b)
        dist = torch.distributions.Categorical(logits=logits)
        acts = dist.sample()
        acts_l = acts.cpu().tolist()
        o, r, d, info = env.step(acts_l)
        total_credit += sum(r)
        for i in range(n):
            if d[i]:
                hid[i] = None
            else:
                hid[i] = h_new[i:i + 1, :, :].detach().to('cpu')
        obs = o
        steps += 1
        if all(d):
            break
    alive = sum(1 for a in env.agents if a.alive)
    avg_e = sum(a.energy for a in env.agents) / n
    energy_sum = sum(a.energy for a in env.agents)
    return alive, avg_e, total_credit, energy_sum


def behavior_profile(policy, n, grid, episodes, device, tag, seed0=7000):
    """Verifiable behavior-richness tracker (replaces eyeballing GIFs).
    Per episode, count: border-pinned agents, harvest/share/signal action rates,
    % steps an agent was food-adjacent, death rate, mean lifetime (steps alive).
    Emergence needs agents to ENGAGE the world; this says if they do."""
    harv = shr = sig = move = 0
    tot_acts = 0
    border_pins = []
    food_adj = []
    deaths = 0
    lifetimes = []
    H = policy.gru_hidden
    for ep in range(episodes):
        env = EmergenceGrid(width=grid, height=grid, n_agents=n, seed=seed0 + ep)
        obs = env.reset()
        hid = torch.zeros(n, 1, H, device=device)
        born = [env.step_count] * n
        for step in range(400):
            ob = torch.as_tensor(obs[:n], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _, h_new = policy(ob, hid)
            acts = torch.argmax(logits, dim=-1).cpu().tolist()
            for a in acts:
                tot_acts += 1
                if a in (1, 2, 3, 4):
                    move += 1
                elif a == 5:
                    harv += 1
                elif a == 6:
                    shr += 1
                elif a == 7:
                    sig += 1
            o, r, d, info = env.step(acts)
            for i in range(n):
                a = env.agents[i]
                if a.alive:
                    if a.x <= 2 or a.x >= grid - 3 or a.y <= 2 or a.y >= grid - 3:
                        border_pins.append(1)
                    else:
                        border_pins.append(0)
                    if env.adjacent_harvestable(a):
                        food_adj.append(1)
                    else:
                        food_adj.append(0)
                if d[i] and a.alive is False:
                    pass
            # track deaths + lifetime
            for i in range(n):
                if d[i]:
                    deaths += 1
                    lifetimes.append(step - born[i])
                    born[i] = step
            hid = h_new.detach()
            obs = o
            if all(d):
                break
    def pct(x):
        return 100.0 * sum(x) / max(1, len(x))
    print(f"[{tag}] BEHAVIOR over {episodes} ep ({tot_acts} actions):")
    print(f"  move {pct([move]) if tot_acts else 0:.0f}% | "
          f"harvest {100.0*harv/max(1,tot_acts):.1f}% | "
          f"share {100.0*shr/max(1,tot_acts):.1f}% | "
          f"signal {100.0*sig/max(1,tot_acts):.1f}%")
    print(f"  border-pinned {pct(border_pins):.0f}% | "
          f"food-adjacent {pct(food_adj):.0f}% | "
          f"death-rate {100.0*deaths/max(1,episodes*n):.0f}% | "
          f"mean-lifetime {sum(lifetimes)/max(1,len(lifetimes)):.0f} steps")
    return dict(harv=harv, shr=shr, sig=sig, move=move, tot=tot_acts,
                border=pct(border_pins), foodadj=pct(food_adj),
                death=100.0 * deaths / max(1, episodes * n),
                lifetime=sum(lifetimes) / max(1, len(lifetimes)))





def evaluate(policy, n, grid, episodes, device, tag):
    credits, alives, avges, energies = [], [], [], []
    for ep in range(episodes):
        a, ae, c, e = run_episode(policy, n, grid, seed=9000 + ep, device=device)
        alives.append(a); avges.append(ae); credits.append(c); energies.append(e)
    alive_pct = 100.0 * sum(alives) / (episodes * n)
    avg_credit = sum(credits) / episodes
    avg_e = sum(avges) / episodes
    avg_energy = sum(energies) / episodes
    print(f"[{tag}] n={n} episodes={episodes} | alive {alive_pct:.0f}% | "
          f"avg credit/ep {avg_credit:.1f} | avgE {avg_e:.2f} | "
          f"totE/ep {avg_energy:.1f}")
    return dict(alive_pct=alive_pct, avg_credit=avg_credit, avg_e=avg_e,
                avg_energy=avg_energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group_ckpt', type=str, required=True)
    ap.add_argument('--solo_ckpt', type=str, required=True)
    ap.add_argument('--grid', type=int, default=128)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'

    group = AgentPolicyBatch(args.n).to(dev)
    group.load_state_dict(torch.load(args.group_ckpt, map_location=dev))
    group.eval()

    solo = AgentPolicyBatch(1).to(dev)
    solo.load_state_dict(torch.load(args.solo_ckpt, map_location=dev))
    solo.eval()

    g = evaluate(group, args.n, args.grid, args.episodes, dev, 'GROUP')
    s = evaluate(solo, 1, args.grid, args.episodes, dev, 'SOLO')

    # raw ECI
    raw_eci = (g['avg_credit'] - args.n * s['avg_credit']) / (g['avg_credit'] + 1e-8)
    # efficiency-normalized: value per energy
    g_eff = g['avg_credit'] / (g['avg_energy'] + 1e-8)
    s_eff = s['avg_credit'] / (s['avg_energy'] + 1e-8)
    eci_eff = g_eff - s_eff
    print(f"\n=== ECI (§10) ===")
    print(f"raw ECI   = {raw_eci:+.3f}   (group credit vs 16x solo credit)")
    print(f"ECI_eff   = {eci_eff:+.3f}   (group value/energy - solo value/energy)")
    print(f"  group value/energy = {g_eff:.3f} | solo value/energy = {s_eff:.3f}")
    if eci_eff > 0.05:
        print(">> POSITIVE efficiency-normalized ECI: group extracts more value")
        print("   per unit energy than 16 isolated agents -> emergence signal.")
    else:
        print(">> ECI_eff ~ 0 or negative: no emergent efficiency gain detected.")


if __name__ == '__main__':
    main()
