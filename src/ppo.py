"""Minimal PPO (clipped surrogate + GAE + value + entropy) for ONE independent
agent policy. 16 of these run in parallel (one per agent), each with its own
rollout buffer collected from K env copies. CPU-free: all tensors on CUDA.

Each AgentPolicy owns one PPOTrainer. The harness calls collect() (batched GPU
inference over K envs) and update() (a few epochs of minibatch SGD).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RolloutBuffer:
    def __init__(self, n_steps, gamma=0.99, lam=0.95):
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.obs = []
        self.acts = []
        self.logp = []
        self.rew = []
        self.val = []
        self.don = []
        self.hid = []   # GRU hidden at each step (for backprop-through-time trim)

    def push(self, obs, act, logp, rew, val, don, hid):
        self.obs.append(obs)
        self.acts.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.don.append(don)
        self.hid.append(hid)

    def compute_gae(self, last_val, last_don):
        # last_val: (B,) tensor, last_don: (B,) tensor (0=not done,1=done)
        # self.val/self.rew/self.don are lists of (B,) tensors over T steps.
        T = len(self.don)   # actual collected steps (may be < n_steps)
        B = last_val.size(0)
        dev = last_val.device
        last_val = last_val.reshape(B)                      # ensure (B,)
        vals = self.val + [last_val]                       # (T+1, B)
        dones = self.don + [last_don]                      # (T+1, B)
        adv = [torch.zeros(B, device=dev) for _ in range(T)]
        gae = torch.zeros(B, device=dev)
        for t in reversed(range(T)):
            next_nonterm = 1.0 - dones[t + 1]              # (B,)
            next_val = vals[t + 1] * next_nonterm          # (B,)
            delta = self.rew[t] + self.gamma * next_val - vals[t]
            gae = delta + self.gamma * self.lam * next_nonterm * gae
            adv[t] = gae
        returns = [adv[t] + self.val[t] for t in range(T)]
        # stack -> (T, B)
        return torch.stack(adv), torch.stack(returns)


class RewardNormalizer:
    """Online running mean/variance (Welford) for reward normalization.
    Divides rewards by running std so the VF fits normalized returns.
    This is critical when reward scale spans -0.5 (invalid harvest) to +15
    (harvest), causing unnormalized VF loss in the hundreds."""
    def __init__(self, clip=10.0):
        self.clip = clip
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def normalize(self, r: float) -> float:
        self.count += 1
        old_mean = self.mean
        self.mean += (r - self.mean) / self.count
        self.var += (r - old_mean) * (r - self.mean)
        std = max((self.var / max(self.count - 1, 1)) ** 0.5, 1e-8)
        return float(np.clip(r / std, -self.clip, self.clip))


class PPOTrainer:
    def __init__(self, policy, lr=2.5e-4, n_steps=256, gamma=0.99, lam=0.95,
                 clip=0.2, ent_coef=0.05, vf_coef=0.5, n_epochs=4, minibatch=64,
                 max_grad_norm=0.5, device='cuda'):
        self.policy = policy.to(device)
        self.device = device
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.buf = RolloutBuffer(n_steps, gamma, lam)
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.minibatch = minibatch
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs_b, hidden):
        # obs_b: (B, OBS_DIM) tensor on device. Returns act(B), logp(B), val(B), h
        logits, val, h = self.policy(obs_b, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), val.squeeze(-1), h

    def update(self, last_val, last_don):
        adv, ret = self.buf.compute_gae(last_val, last_don)
        # adv, ret: (T, B)
        obs = torch.stack(self.buf.obs).to(self.device)      # (T,B,OBS_DIM)
        act = torch.stack(self.buf.acts).to(self.device)     # (T,B)
        old_logp = torch.stack(self.buf.logp).to(self.device)  # (T,B)
        adv_t = adv.to(self.device).detach()      # advantages are constants (no bp)
        ret_t = ret.to(self.device).detach()      # targets are constants
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T, B, _ = obs.shape
        flat_obs = obs.reshape(T * B, -1)
        flat_act = act.reshape(-1)
        flat_old = old_logp.reshape(-1)
        flat_adv = adv_t.reshape(-1)
        flat_ret = ret_t.reshape(-1)

        pol_losses = []
        vf_losses = []
        # Flatten hid: (T, 1, B, H) -> (T*B, H)
        flat_hid = torch.stack(self.buf.hid, dim=0).squeeze(1).reshape(T * B, -1).to(self.device)
        for _ in range(self.n_epochs):
            perm = torch.randperm(T * B, device=self.device)
            for start in range(0, T * B, self.minibatch):
                mb = perm[start:start + self.minibatch]
                o_mb = flat_obs[mb]
                h0_mb = flat_hid[mb].unsqueeze(0)   # (1, mb_len, H)
                logits, val, _ = self.policy(o_mb, h0_mb)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(flat_act[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - flat_old[mb])
                s1 = ratio * flat_adv[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * flat_adv[mb]
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(val.squeeze(-1), flat_ret[mb])
                loss = pol_loss + self.vf_coef * val_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()
                pol_losses.append(pol_loss.detach().cpu())
                vf_losses.append(val_loss.detach().cpu())
        self.buf.reset()
        return float(torch.stack(pol_losses).mean()), float(torch.stack(vf_losses).mean())
