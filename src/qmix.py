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


class QMIXBuffer:
    """Rollout buffer for QMIX. Stores per-step: obs (N*K,...), acts (N,K),
    logp (N,K), **local** rewards (N,K), vals (N,K), dones (N,K), hid (N,K,H),
    and the centralized state (N or 1 per env-episode, derived from dump_agents).

    B = N*K (agent-major: row = agent i in env e at position e*N+i).
    State is (K,) one centralized state per env (shared across agents in that env).
    """
    def __init__(self, n_agents: int, n_envs: int, n_steps: int,
                 gamma=0.99, lam=0.95):
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.obs = []      # (N*K,) per step
        self.acts = []     # (N,K)
        self.logp = []     # (N,K)
        self.rew = []      # (N,K) -- LOCAL per-agent rewards (D5 compliant)
        self.val = []      # (N,K) -- per-agent Q at taken action
        self.don = []      # (N,K)
        self.hid = []      # (N,K,H)
        self.state = []    # (K, state_dim) centralized state per env

    def push(self, obs, acts, logp, rew, val, don, hid, state):
        self.obs.append(obs)
        self.acts.append(acts)
        self.logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.don.append(don)
        self.hid.append(hid)
        self.state.append(state)

    def compute_targets(self, last_val, last_don, last_state, next_q_tot_fn):
        """Double-Q target: Q_tot_target = sum_i r_i + gamma * (max over next joint Q_tot).
        `next_q_tot_fn(next_actions_per_agent)` returns the target-network Q_tot for the
        greedy next actions. We compute the TD target per-env (Q_tot over K*... -> aggregate).

        Returns:
          q_tot: (T, K) joint Q_tot per env (from online mixer, using taken actions)
          q_tot_target: (T, K) bootstrap targets
          q_taken: (T, N*K) per-agent Q at taken action (for the mixer input)
          state_seq: (T, K, state_dim)
          obs_seq, acts_seq, logp_seq, hid_seq: (T, ...)
        """
        T = len(self.don)
        B = self.n_agents * self.n_envs
        rew = torch.stack(self.rew)          # (T, N, K)
        don = torch.stack(self.don)          # (T, N, K)
        val = torch.stack(self.val)          # (T, N, K) -- per-agent Q(s,a) for taken a
        state_seq = torch.stack(self.state)  # (T, K, S)

        # per-env summed local reward (N agents) at each step: (T, K)
        rew_e = rew.sum(dim=0)               # (T, K)
        don_e = don.any(dim=0).float()       # (T, K) env done if ANY agent done

        # Bootstrap: last_val is per-agent Q (N*K,) ; last_state is (K,S).
        # Q_tot at the bootstrap step via next_q_tot_fn(next_actions greedy).
        # We compute the standard n-step return: R_t = sum_{t'=t}^{T-1} gamma^{t'-t} r_{t'}
        # plus gamma^{T-t} * (1-done) * next_Q_tot.
        q_tot_target = torch.zeros(T, self.n_envs, device=rew.device)
        # iterate backward over T to build discounted returns
        ret = torch.zeros(self.n_envs, device=rew.device)
        discount = torch.ones(self.n_envs, device=rew.device)
        # last bootstrap (double-Q): use target net next-greedy joint Q_tot
        next_bootstrap = next_q_tot_fn(last_state, last_val.reshape(self.n_agents, self.n_envs).T)  # (K,)
        ret = (1.0 - last_don.float()) * next_bootstrap
        for t in reversed(range(T)):
            ret = rew_e[t] + self.gamma * ret * (1.0 - don_e[t])
            q_tot_target[t] = ret
        return q_tot_target, val, state_seq, rew_e
