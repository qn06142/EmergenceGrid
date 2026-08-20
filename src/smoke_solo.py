"""Solo PPO smoke test (build step 2): prove a single independent policy learns
to survive on a small grid. Validates PPO correctness before scaling to 16 nets.

Run: python src/smoke_solo.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
import numpy as np
from env import EmergenceGrid
from model import AgentPolicyBatch, OBS_DIM, NACT
from ppo import PPOTrainer

K_DEF = 4            # parallel env copies
NSTEP_DEF = 64       # rollout length per update
NUPD_DEF = 5        # updates
GRID_DEF = 64


def to_torch(obs_list, device='cuda'):
    return torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)


def run_smoke(K=K_DEF, NSTEP=NSTEP_DEF, NUPD=NUPD_DEF, GRID=GRID_DEF):
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    grids = [EmergenceGrid(width=GRID, height=GRID, n_agents=1, seed=100 + i, respawn=True, curriculum=0)
             for i in range(K)]
    net = AgentPolicyBatch(1).to(device)
    trainer = PPOTrainer(net, lr=2.5e-4, n_steps=NSTEP, minibatch=32, device=device)
    H = net.gru_hidden
    h_stack = torch.zeros(1, K, H, device=device)

    for upd in range(NUPD):
        obs_now = [grids[i].reset()[0] for i in range(K)]
        h_stack.zero_()
        ep_rew = [0.0] * K
        ep_len = [0] * K
        for t in range(NSTEP):
            obs_b = to_torch(obs_now, device=device)        # (K, OBS_DIM)
            with torch.no_grad():
                acts, logp, vals, h_new = trainer.act(obs_b, h_stack)
            acts_l = acts.cpu().tolist()
            rews = []
            next_obs = []
            dones = []
            for i in range(K):
                o, r, d, info = grids[i].step([acts_l[i]])
                rews.append(float(r[0]))
                ep_rew[i] += r[0]; ep_len[i] += 1
                dones.append(bool(d[0]))
                if d[0]:
                    h_new[:, i:i + 1, :].zero_()
                next_obs.append(o[0])
            # push step into buffer
            trainer.buf.obs.append(obs_b)
            trainer.buf.acts.append(acts)
            trainer.buf.logp.append(logp)
            trainer.buf.rew.append(torch.tensor(rews, device=device))
            trainer.buf.val.append(vals)
            trainer.buf.don.append(torch.tensor([1.0 if x else 0.0 for x in dones],
                                                device=device))
            trainer.buf.hid.append(h_stack.detach().cpu())
            h_stack = h_new
            obs_now = next_obs

        # bootstrap final value
        obs_b = to_torch(obs_now, device=device)
        with torch.no_grad():
            _, last_val, _ = net(obs_b, h_stack)
        last_val = last_val.squeeze(-1)                     # (B,)
        last_don = torch.tensor([1.0 if not a.alive else 0.0
                                 for a in [g.agents[0] for g in grids]],
                                device=device)
        pol_loss, vf_loss = trainer.update(last_val, last_don)

        if upd % 10 == 0 or upd == NUPD - 1:
            avg_rew = sum(ep_rew) / K
            print(f"upd {upd:3d} | pol_loss {pol_loss:.3f} vf_loss {vf_loss:.3f} "
                  f"| avg_ep_rew(last) {avg_rew:.2f} | {time.time()-t0:.1f}s", flush=True)

    torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__),
                                              'smoke_solo_net.pt'))
    print("saved smoke_solo_net.pt", flush=True)


if __name__ == '__main__':
    run_smoke()
