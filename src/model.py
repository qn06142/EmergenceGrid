import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Architecture Dimensions
N_TILE_TYPES = 11
SPAT_C = 12
GRID_H = 64
GRID_W = 64
SPAT = GRID_H * GRID_W * SPAT_C
OWN_DIM = 14 + 121 * 12 + 3  # 14 traits + 11x11 patch(121*12) + (food_dx, food_dy, food_dist)
OBS_DIM = SPAT + OWN_DIM  # now includes 5x5 local patch for direct food sight
NACT = 13
SPATIAL_POOL = 8  # coarse grid the CNN feature map is pooled to (preserves WHERE)


def _orth(w: torch.Tensor, gain: float = 1.0) -> None:
    """Applies orthogonal initialization to weights."""
    nn.init.orthogonal_(w, gain=gain)


class AgentPolicyBatch(nn.Module):
    """Policy network for N independent agents, trained with shared weights.

    Architecture (three stages):
      1. Vision backbone — a *no-pool dilated CNN* (MiniGrid-style) over the
         11x11 local patch (SPAT_C=12 channels: walls, food types, gates, agents...).
         Stride-1 with pad=dilation preserves spatial resolution so fine structures
         (gates, single walls) stay crisp. 4 layers with dilations 1/2/4/8 give an
         ~31-tile receptive field — the agent can see food across most of the map,
         unlike the earlier 1024-token ViT whose local attention spanned only ~8 tiles
         (that ViT was "wall-blind" and got 72% of moves blocked on curriculum 1).
      2. Spatial pool — the CNN feature map is pooled to an 8x8 coarse grid
         (SPATIAL_POOL=8) instead of a single global average. The 8x8 keeps *where*
         things are (a global average erased wall position and made navigation fail).
      3. Recurrent core — a GRU over the pooled features + the agent's own-state
         vector (traits, energy, inventory, food-direction), producing a hidden state
         that carries memory across the episode. A 2-layer reasoning-head MLP is
         applied to the GRU output before the action/value heads, adding capacity for
         the cognitive load of trait-matching (which trait do I need? do I have it?).

    A directional prior (`food_w`) is added to the action logits: it nudges the
    agent toward the food-direction vector in its observation (UP if food is north,
    etc.) and adds a small harvest bias when food is adjacent. This is a *prior*, not
    a reward — it speeds learning but the policy can override it (e.g. to mutate
    instead of harvest when it lacks the trait for gated food).

    forward_agent(i, ob, hid) runs a single agent (used for probing/rendering);
    forward(obs_b, hid_b) runs the full batch (used in training).
    """
    def __init__(
        self,
        N: int,
        d_model: int = 256,
        gru_hidden: int = 256,
        head_dim: int = 256,
        freeze_vision: bool = False
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.gru_hidden = gru_hidden
        self.head_dim = head_dim

        # Vision Backbone: no-pool dilated CNN (MiniGrid-style). Stride-1 / pad=dilation
        # keeps spatial resolution (no blur, fine gates crisp). 4 layers with dilations
        # 1,2,4,8 give an ~31-tile receptive field -> can see food across most of the
        # 64x64 map, unlike the old 1024-token ViT whose local attention was ~8 tiles.
        self.cnn = nn.Sequential(
            nn.Conv2d(SPAT_C, 64, 3, padding=1, dilation=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=2, dilation=2),    nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=4, dilation=4),   nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=8, dilation=8), nn.GELU(),
        )
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                _orth(m.weight, 1.0)

        self.grid_bias = nn.Parameter(torch.zeros(N, d_model * SPATIAL_POOL * SPATIAL_POOL))

        # Freeze vision backbone if requested
        if freeze_vision:
            for param in self.cnn.parameters():
                param.requires_grad_(False)
            self.grid_bias.requires_grad_(False)

        # Per-Agent GRU Heads
        gru_in = d_model * SPATIAL_POOL * SPATIAL_POOL + OWN_DIM
        self.gru_Wih = nn.Parameter(torch.zeros(N, 3 * gru_hidden, gru_in))
        self.gru_Whh = nn.Parameter(torch.zeros(N, 3 * gru_hidden, gru_hidden))
        self.gru_b = nn.Parameter(torch.zeros(N, 3 * gru_hidden))

        for p in (self.gru_Wih, self.gru_Whh):
            _orth(p, 1.0)

        # Reasoning head: 2-layer MLP between GRU memory and the action/value heads.
        # This gives the policy "thinking" capacity to combine (memory of where food
        # is) + (own traits: do I have can_hard/can_tall?) + (food type: is it
        # HARD_NUT/TALL_FRUIT?) -> the correct mutate action. The old single-linear
        # head couldn't represent that trait-matching computation, which is why
        # mutation emerged but the mutate->eat loop never closed.
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_dim), nn.GELU(),
            nn.Linear(head_dim, gru_hidden),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                _orth(m.weight, 1.0)

        # Actor/Critic Heads
        self.actor_w = nn.Parameter(torch.zeros(N, NACT, gru_hidden))
        _orth(self.actor_w, 1.0)
        self.actor_b = nn.Parameter(torch.zeros(N, NACT))

        self.critic_w = nn.Parameter(torch.zeros(N, 1, gru_hidden))
        _orth(self.critic_w, 1.0)
        self.critic_b = nn.Parameter(torch.zeros(N, 1))

        # Goal skip-connection: the compact food vector (food_dx, food_dy, food_dist)
        # lives at the tail of `own` but was being drowned in the 1469-dim GRU input
        # (supervised "move toward food" stalled at 0.17 despite a linear probe on the
        # 3-vector hitting 0.65). Project it DIRECTLY onto the action/value heads so the
        # policy can't ignore the goal signal. Goal-conditioned (standard for gridworlds).
        self.food_w = nn.Parameter(torch.zeros(N, NACT, 3))
        self.food_wc = nn.Parameter(torch.zeros(N, 1, 3))
        with torch.no_grad():
            # Directional navigation prior:
            # Action 1 = UP (dyn < 0)
            self.food_w[:, 1, 1] = -1.0
            # Action 2 = RIGHT (dxn > 0)
            self.food_w[:, 2, 0] = 1.0
            # Action 3 = DOWN (dyn > 0)
            self.food_w[:, 3, 1] = 1.0
            # Action 4 = LEFT (dxn < 0)
            self.food_w[:, 4, 0] = -1.0
            # Action 5 = HARVEST (distance near 0)
            self.food_w[:, 5, 2] = -1.0
            # Critic prior: closer to food -> higher expected return
            self.food_wc[:, 0, 2] = -1.0
            # Constant harvest bias: argmax must prefer HARVEST when on/adjacent to
            # food. The -dist prior alone vanishes at dist 0 (agent standing on food
            # gets 0 harvest signal -> greedy paces on top without eating). A flat
            # +1.0 bias makes harvest win at argmax when near food; the -dist term
            # still suppresses it when far. INVALID_HARVEST_PEN=0 so empty presses
            # are free.
            self.actor_b[:, 5] = 1.0
        # Multiplier on the food-skip so it provides a strong guidance signal
        # that RL refines as obstacle avoidance, gates, and traits enter.
        self.food_scale = 2.0

    def _encode_grid(self, spat_big: torch.Tensor) -> torch.Tensor:
        """(K, H*W, SPAT_C) -> (K, d_model*SPATIAL_POOL^2) via no-pool dilated CNN +
        coarse spatial pooling. Unlike the old global avg-pool (which collapsed all
        position and made the agent wall-blind), we pool to an 8x8 grid so the
        agent keeps a downsampled MAP of where walls/food are."""
        K = spat_big.size(0)
        g = spat_big.view(K, GRID_H, GRID_W, SPAT_C).permute(0, 3, 1, 2)  # (K, C, H, W)
        feat = self.cnn(g)                                       # (K, d_model, H, W)
        coarse = torch.nn.functional.adaptive_avg_pool2d(feat, (SPATIAL_POOL, SPATIAL_POOL))
        emb = coarse.flatten(1)                                  # (K, d_model*SPATIAL_POOL^2)
        return torch.clamp(emb, min=-10.0, max=10.0)

    def _own(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is agent-major: row (e*N + i) = agent i in env e. So the own block
        # of env e is the N rows [e*N : e*N+N]. Reshape as (K, N, OWN) then move N
        # to the front -> (N, K, OWN) where own[i, e] = agent i's features in env e.
        NK = obs.size(0)
        K = NK // self.N
        return obs[:, SPAT:].view(K, self.N, OWN_DIM).transpose(0, 1)

    def _encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        NK = obs.size(0)
        K = NK // self.N
        spat = obs[0::self.N, :SPAT].view(K, GRID_H * GRID_W, SPAT_C)
        grid_emb = self._encode_grid(spat)                       # (K, d_model*P^2) per-env map
        # All N agents in env e share env e's spatial map; grid_bias adds per-agent
        # specialization. (Previously grid_emb was (K,d_model) and broadcast to N,
        # but now it's spatial so we share it directly across agents in the env.)
        f = grid_emb.unsqueeze(0).expand(self.N, K, grid_emb.size(1)) + self.grid_bias.unsqueeze(1)
        own = self._own(obs)
        return f, own

    def _gru_step(self, x_in: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = torch.einsum('ndi,nki->nkd', self.gru_Wih, x_in) + self.gru_b.unsqueeze(1)
        gh = torch.einsum('ndh,nkh->nkd', self.gru_Whh, h)
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_i + h_i)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h
        return torch.clamp(h_new, min=-10.0, max=10.0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        NK = obs.size(0)
        K = NK // self.N
        f, own = self._encode(obs)
        x_in = torch.cat([f, own], dim=-1)
        if hidden is None:
            hidden = torch.zeros(self.N, K, self.gru_hidden, device=obs.device, dtype=obs.dtype)
        h_new = self._gru_step(x_in, hidden)
        h_head = self.head(h_new)
        logits = torch.einsum('nad,nkd->nka', self.actor_w, h_head) + self.actor_b.unsqueeze(1)
        value = torch.einsum('nad,nkd->nka', self.critic_w, h_head) + self.critic_b.unsqueeze(1)
        # Goal skip-connection: project food_dx/dy/dist directly onto the heads.
        # food is (N, K, 3) agent-major; food_w is (N, NACT, 3) -> (N, K, NACT).
        food = self._own(obs)[:, :, -3:]                       # (N, K, 3)
        flog = torch.einsum('iac,ikc->ika', self.food_w, food)  # (N, K, NACT)
        fval = torch.einsum('ic,ikc->ik', self.food_wc.squeeze(1), food)  # (N, K)
        logits = logits + self.food_scale * flog
        value = value + self.food_scale * fval.unsqueeze(-1)
        logits = logits.reshape(NK, NACT)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -float('inf'))
        return logits, value.reshape(NK, 1), h_new

    def forward_agent(
        self,
        i: int,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Agent-i slice of the same forward used by `forward` (so the PPO ratio
        # new_logp/old_logp is consistent with the rollout's `forward`).
        K = obs.size(0)
        spat = obs[:, :SPAT].view(K, GRID_H * GRID_W, SPAT_C)
        grid_emb = self._encode_grid(spat)
        f = grid_emb + self.grid_bias[i]                 # (K, d_model)
        own = obs[:, SPAT:].view(K, OWN_DIM)             # (K, OWN_DIM)
        x_in = torch.cat([f, own], dim=-1)               # (K, d_model+OWN_DIM)
        if hidden is None:
            hidden = torch.zeros(1, K, self.gru_hidden, device=obs.device, dtype=obs.dtype)
        # Replicate forward()'s _gru_step arithmetic exactly for agent i.
        Wih = self.gru_Wih[i]        # (3H, gru_in)
        Whh = self.gru_Whh[i]        # (3H, H)
        b   = self.gru_b[i]          # (3H,)
        h0  = hidden[0]              # (K, H)
        gi = x_in @ Wih.t() + b      # (K, 3H)
        gh = h0 @ Whh.t()            # (K, 3H)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_i + h_i)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h0  # (K, H)
        h_head = self.head(h_new)   # reasoning head (shared across agents)
        logits = torch.einsum('ad,kd->ka', self.actor_w[i], h_head) + self.actor_b[i].unsqueeze(0)
        value = torch.einsum('ad,kd->ka', self.critic_w[i], h_head) + self.critic_b[i].unsqueeze(0)
        # Goal skip-connection (consistent with forward()).
        food = own[:, -3:]                               # (K, 3)
        logits = logits + self.food_scale * torch.einsum('ac,kc->ka', self.food_w[i], food)
        value = value + self.food_scale * torch.einsum('c,kc->k', self.food_wc[i, 0], food).unsqueeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -float('inf'))
        return logits, value, h_new.unsqueeze(0)

    def params_of(self, i: int):
        return [
            self.grid_bias[i], self.gru_Wih[i], self.gru_Whh[i], self.gru_b[i],
            self.actor_w[i], self.actor_b[i], self.critic_w[i], self.critic_b[i],
            self.food_w[i], self.food_wc[i]
        ]
