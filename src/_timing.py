import sys, time
sys.path.insert(0, 'src')
import torch
from env import EmergenceGrid
from model import AgentPolicy
from ppo import PPOTrainer
from train import make_hid_stack

n, g, k = 16, 128, 2
envs = [EmergenceGrid(width=g, height=g, n_agents=n, seed=1 + e * 1000)
        for e in range(k)]
nets = [AgentPolicy().cuda() for _ in range(n)]
trs = [PPOTrainer(nets[i], device='cuda', n_steps=64) for i in range(n)]
H = nets[0].gru.hidden_size
hid = [[None] * n for _ in range(k)]
obs_now = [envs[e].reset() for e in range(k)]

# time one rollout (64 steps) across k=2 envs
t0 = time.time()
for t in range(64):
    obs_agent = [torch.tensor([obs_now[e][i] for e in range(k)], device='cuda')
                 for i in range(n)]
    acts_agent = []
    for i in range(n):
        hs = make_hid_stack([hid[e][i] for e in range(k)], k, H, 'cuda')
        with torch.no_grad():
            a, lp, v, hn = trs[i].act(obs_agent[i], hs)
        acts_agent.append(a)
        for e in range(k):
            hid[e][i] = hn[:, e:e + 1, :]
    acts_l = [a.cpu().tolist() for a in acts_agent]
    next_obs = []
    for e in range(k):
        vec = [acts_l[i][e] for i in range(n)]
        o, r, d, info = envs[e].step(vec)
        next_obs.append(o)
        for i in range(n):
            if d[i]:
                hid[e][i] = None
    obs_now = next_obs
print(f"rollout 64 steps @128/16/k2: {time.time()-t0:.1f}s")

# fill buffers and time one update per net
t1 = time.time()
for i in range(n):
    # rebuild a small buffer (reuse last obs)
    ob = torch.randn(k, 1144, device='cuda')
    hs = torch.zeros(1, k, H, device='cuda')
    with torch.no_grad():
        a, lp, v, hn = trs[i].act(ob, hs)
    for _ in range(64):
        trs[i].buf.obs.append(ob)
        trs[i].buf.acts.append(a)
        trs[i].buf.logp.append(lp)
        trs[i].buf.rew.append(torch.randn(k, device='cuda'))
        trs[i].buf.val.append(v)
        trs[i].buf.don.append(torch.zeros(k, device='cuda'))
        trs[i].buf.hid.append(hn.detach().cpu())
    with torch.no_grad():
        _, lv, _ = nets[i](ob, hs)
    lv = lv.squeeze(-1)
    trs[i].update(lv, torch.zeros(k, device='cuda'))
print(f"16 net updates (nstep64,k2): {time.time()-t1:.1f}s")
