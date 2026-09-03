"""QMIX for EmergenceGrid.

Minimal QMIX (Rashid et al. 2018) on top of the existing AgentPolicyBatch encoder.
Key point: per-agent REWARDS stay local (env.py reward terms are unchanged -- the
EmergenceGrid "local-only reward" discipline holds). QMIX centralizes only the
VALUE via a state-conditioned monotonic mixer, so the group credit for the rare
simultaneous gate push propagates back to the per-agent policies. That is the
algorithmic mechanism, not a reward term.

Pieces:
- `QMIXMixer`: 2-layer hypernetwork. state -> (embed) -> per-agent mixing weights
  (monotonic: `w = softplus(hnet) * positive`) + bias; Q_tot = sum_i q_i * w_i(state) + b(state).
- `QMIXBuffer`: like RolloutBuffer but stores the centralized `state` per step and
  per-agent actions; computes the double-Q target.
- `qmix_update(...)`: the TD-loss update over the shared policy (N independent
  weight sets via einsum, exactly as IPPO), with the mixer providing the joint target.

Obs/state note: the centralized state `s` is built in env.global_state() from
dump_agents() (positions, strength, traits of all agents) -- this is INFERRED state,
not a hand-authored group reward, so D5 (no global reward term) is respected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMIXMixer(nn.Module):
    """State-conditioned monotonic mixing network.

    Q_tot = b(s) + sum_i q_i * w_i(s), where w_i > 0 (monotonic via softplus/abs),
    so an individual agent's Q always has monotone effect on the joint Q (QMIX's
    theoretical guarantee that per-agent greedy action selection stays consistent
    with joint greedy selection).

    Architecture (standard QMIX / QTRAN-style):
      - state -> hypernet hidden (embed_dim) -> 2 hyper-layers producing
        first-order mixing weights (n_agents) and a bias; a second set produces
        the bias term (single scalar). All conditioned on the centralized state.
    """
    def __init__(self, state_dim: int, n_agents: int, embed_dim: int = 256,
                 mixing_dim: int = 64, hypernet_embed: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        # Hypernetwork: state -> mixing params. Two-layer with a skip connection.
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.Tanh(),
            nn.Linear(hypernet_embed, hypernet_embed),
            nn.Tanh(),
            nn.Linear(hypernet_embed, embed_dim * n_agents),
        )
        self.hyper_w1b = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.Tanh(),
            nn.Linear(hypernet_embed, embed_dim),
        )
        # V(s): state -> scalar baseline (value of being in this state)
        self.V = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.Tanh(),
            nn.Linear(hypernet_embed, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        """agent_qs: (B, n_agents) ; states: (B, state_dim).
        Returns Q_tot: (B,)."""
        B = agent_qs.size(0)
        n = self.n_agents
        E = self.embed_dim
        # First-order mixing weights, monotonic (positive).
        w1 = F.softplus(self.hyper_w1(states))            # (B, E*n)
        b1 = self.hyper_w1b(states)                       # (B, E)
        w1 = w1.view(B, n, E)                             # (B, n_agents, embed)
        # agent_qs (B, n) -> weighted sum into embed space: (B, embed)
        q_mix = torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1   # (B, E)
        q_mix = F.gelu(q_mix)                            # (B, E)
        q_tot = self.V(states).squeeze(-1)               # (B,)  -- V(s) baseline
        q_tot = q_tot + q_mix.sum(dim=-1)                # (B,)
        return q_tot


class ReplayBuffer:
    """Experience replay for QMIX. Stores full transitions (s,a,r,s',done,state,state')
    so the RARE rewarding joint-gate transitions get reused many times during training
    -- without replay, online QMIX only trains the taken action's Q once and (with
    uniform-random exploration) never differentiates the Q-heads (every action gets the
    same ~0 target -> uniform policy -> ent pinned at max). Replay is what makes distal
    rewards actually train the heads.

    Stored per push: obs_t (N*K,OBS), acts_t (N,K), rews_t (N,K) [LOCAL, D5-compliant],
    states_t (K,S), dones_t (N,K), obs_t1 (N*K,OBS), states_t1 (K,S), dones_t1 (N,K).
    """
    def __init__(self, capacity=20000):
        self.cap = capacity
        self.obs = []; self.acts = []; self.rews = []
        self.state = []; self.don = []
        self.obs1 = []; self.state1 = []; self.don1 = []

    def add_rollout(self, buf, obs_next, state_next, don_next):
        """Append a whole rollout (dict with keys obs/acts/rews/state/don, each a list
        of per-step tensors) as transitions, with the supplied next obs/state/done for
        the final step."""
        T = len(buf['obs'])
        for t in range(T):
            self.obs.append(buf['obs'][t]); self.acts.append(buf['acts'][t])
            self.rews.append(buf['rews'][t]); self.state.append(buf['state'][t])
            self.don.append(buf['don'][t])
            if t + 1 < T:
                self.obs1.append(buf['obs'][t + 1]); self.state1.append(buf['state'][t + 1])
                self.don1.append(buf['don'][t + 1])
            else:
                self.obs1.append(obs_next); self.state1.append(state_next)
                self.don1.append(don_next)
        self._trim()

    def _trim(self):
        while len(self.obs) > self.cap:
            self.obs.pop(0); self.acts.pop(0); self.rews.pop(0)
            self.state.pop(0); self.don.pop(0)
            self.obs1.pop(0); self.state1.pop(0); self.don1.pop(0)

    def __len__(self):
        return len(self.obs)

    def sample(self, batch, device):
        import random
        idx = random.sample(range(len(self.obs)), min(batch, len(self.obs)))
        obs = torch.stack([self.obs[i] for i in idx]).to(device)
        acts = torch.stack([self.acts[i] for i in idx]).to(device)
        rews = torch.stack([self.rews[i] for i in idx]).to(device)
        state = torch.stack([self.state[i] for i in idx]).to(device)
        don = torch.stack([self.don[i] for i in idx]).to(device)
        obs1 = torch.stack([self.obs1[i] for i in idx]).to(device)
        state1 = torch.stack([self.state1[i] for i in idx]).to(device)
        don1 = torch.stack([self.don1[i] for i in idx]).to(device)
        return obs, acts, rews, state, don, obs1, state1, don1


def qmix_update(policy, mixer, target_mixer, target_policy, replay, opt, device,
                n, k, nact, gamma=0.99, batch=512, epochs=4,
                ent_coef=0.05, ent_floor=0.5, max_grad_norm=0.5):
    """Replay-based double-Q TD update. Samples minibatches from `replay`, recomputes
    the double-Q target each epoch (target policy+mixer are slow/polyak), so rare joint
    events are seen many times and the per-agent Q-heads differentiate.

    Returns (td_err_mean, ent_mean).
    """
    if len(replay) < batch:
        return 0.0, 2.56
    for _ in range(epochs):
        obs, acts, rews, state, don, obs1, state1, don1 = replay.sample(batch, device)
        B, n_, k_ = obs.shape[0], obs.shape[1], obs.shape[2]
        # Treat each (sampled transition, parallel env) as one mixer batch element.
        # obs (B,n,k,OBS) -> (B*k, n, OBS); forward_q wants (N*K', OBS) with hidden (N,K',H)
        obs_f = obs.reshape(B * k_, n_, -1)                       # (B*k, n, OBS)
        obs1_f = obs1.reshape(B * k_, n_, -1)
        q = policy.forward_q(obs_f.reshape(B * k_ * n_, -1),
                              torch.zeros(n, B * k_, policy.gru_hidden, device=device)
                              )[0].reshape(B * k_ * n_, nact)      # (B*k*n, NACT)
        q = q.reshape(B * k_, n_, nact)                           # (B*k, n, NACT)
        # taken-action Q per agent: acts (B,n,k) -> (B*k, n)
        acts_r = acts.permute(0, 2, 1).reshape(B * k_, n_)        # (B*k, n)
        qt = q.gather(2, acts_r.unsqueeze(-1)).squeeze(-1)        # (B*k, n)
        q_tot = mixer(qt, state.reshape(B * k_, -1))              # (B*k,)
        # double-Q target: greedy next actions from TARGET policy, mixed by TARGET mixer
        with torch.no_grad():
            qn = target_policy.forward_q(obs1_f.reshape(B * k_ * n_, -1),
                                torch.zeros(n, B * k_, policy.gru_hidden, device=device)
                                )[0].reshape(B * k_ * n_, nact).reshape(B * k_, n_, nact)
            nxt_a = qn.argmax(dim=-1)                             # (B*k, n)
            qn_greedy = qn.gather(2, nxt_a.unsqueeze(-1)).squeeze(-1)  # (B*k, n)
            q_tot_next = target_mixer(qn_greedy, state1.reshape(B * k_, -1))  # (B*k,)
            rew_e = rews.sum(dim=1).reshape(B * k_)               # (B*k,) local reward summed over agents
            done_e = don1.any(dim=1).reshape(B * k_).float()      # (B*k,)
            target = rew_e + gamma * q_tot_next * (1.0 - done_e)
        loss_td = F.mse_loss(q_tot, target.detach())
        # entropy (penalty: push Q-heads to peak once rewarding actions get positive Q)
        p = torch.softmax(q, dim=-1)
        ent_mean = -(p * p.log()).sum(dim=-1).mean()
        loss = loss_td + ent_coef * ent_mean
        if ent_floor and ent_floor > 0.0:
            loss = loss + 0.1 * torch.clamp(ent_floor - ent_mean, min=0.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(mixer.parameters()),
                                       max_grad_norm)
        opt.step()
    return float((q_tot - target.detach()).abs().mean().detach().cpu()), \
           float(ent_mean.detach().cpu())
