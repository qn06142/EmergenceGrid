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

# Tokenization: stride the 64x64 grid to 32x32 = 1024 tokens (2x2 patch).
# Keeps obs dim identical (C++ untouched) but cuts ViT attention ~16x vs 4096.
PATCH = 2
TOK_H = GRID_H // PATCH
TOK_W = GRID_W // PATCH
N_TOK = TOK_H * TOK_W  # 1024


def _orth(w: torch.Tensor, gain: float = 1.0) -> None:
    """Applies orthogonal initialization to weights."""
    nn.init.orthogonal_(w, gain=gain)


def _sin_pos(H: int, W: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Generates 2D sinusoidal positional encodings (H*W tokens)."""
    pe = torch.zeros(H * W, d_model, device=device)
    xs = torch.arange(W, device=device).float() / max(1, W)
    ys = torch.arange(H, device=device).float() / max(1, H)

    grid_x, grid_y = torch.meshgrid(ys, xs, indexing='ij')
    pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(pos[:, 0:1] * div)
    pe[:, 1::2] = torch.cos(pos[:, 0:1] * div)

    pe2 = torch.zeros_like(pe)
    pe2[:, 0::2] = torch.sin(pos[:, 1:2] * div)
    pe2[:, 1::2] = torch.cos(pos[:, 1:2] * div)

    return (pe + pe2) / 2.0


class AgentPolicyBatch(nn.Module):
    def __init__(
        self,
        N: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        gru_hidden: int = 64,
        freeze_vision: bool = False
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.gru_hidden = gru_hidden

        # Vision Backbone (1024 tokens, shrunk for fast per-iteration training)
        self.token_embed = nn.Linear(SPAT_C, d_model)
        _orth(self.token_embed.weight, 1.0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.grid_bias = nn.Parameter(torch.zeros(N, d_model))

        # Positional Encoding Buffer (for the 1024-token grid)
        self.register_buffer('pe', _sin_pos(TOK_H, TOK_W, d_model, torch.device('cpu')))

        # Freeze vision backbone if requested
        if freeze_vision:
            for param in [self.token_embed.weight, self.token_embed.bias, self.cls_token, self.grid_bias]:
                if param is not None:
                    param.requires_grad_(False)
            for param in self.transformer.parameters():
                param.requires_grad_(False)

        # Per-Agent GRU Heads
        gru_in = d_model + OWN_DIM
        self.gru_Wih = nn.Parameter(torch.zeros(N, 3 * gru_hidden, gru_in))
        self.gru_Whh = nn.Parameter(torch.zeros(N, 3 * gru_hidden, gru_hidden))
        self.gru_b = nn.Parameter(torch.zeros(N, 3 * gru_hidden))

        for p in (self.gru_Wih, self.gru_Whh):
            _orth(p, 1.0)

        # Actor/Critic Heads
        self.actor_w = nn.Parameter(torch.zeros(N, NACT, gru_hidden))
        _orth(self.actor_w, 1.0)
        self.actor_b = nn.Parameter(torch.zeros(N, NACT))

        self.critic_w = nn.Parameter(torch.zeros(N, 1, gru_hidden))
        _orth(self.critic_w, 1.0)
        self.critic_b = nn.Parameter(torch.zeros(N, 1))

    def _tokens_from_spat(self, spat_big: torch.Tensor) -> torch.Tensor:
        """Downsample (K, H*W, SPAT_C) -> (K, N_TOK=1024, SPAT_C) by 2x2 stride."""
        K = spat_big.size(0)
        g = spat_big.view(K, GRID_H, GRID_W, SPAT_C)
        g = g[:, ::PATCH, ::PATCH, :]          # (K, TOK_H, TOK_W, SPAT_C)
        return g.reshape(K, N_TOK, SPAT_C)

    def _encode_grid(self, spat_big: torch.Tensor) -> torch.Tensor:
        K = spat_big.size(0)
        device = spat_big.device
        if self.pe.device != device:
            self.pe = self.pe.to(device)

        tok_in = self._tokens_from_spat(spat_big)          # (K, N_TOK, SPAT_C)
        pe = self.pe.unsqueeze(0).expand(K, -1, -1)        # (K, N_TOK, d)
        tok = self.token_embed(tok_in) + pe
        cls = self.cls_token.expand(K, -1, -1)
        seq = torch.cat([cls, tok], dim=1)                # (K, 1+N_TOK, d)
        out = self.transformer(seq)
        return torch.clamp(out[:, 0, :], min=-10.0, max=10.0)

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
        grid_emb = self._encode_grid(spat)
        f = grid_emb.unsqueeze(0).expand(self.N, K, self.d_model) + self.grid_bias.unsqueeze(1)
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
        logits = torch.einsum('nad,nkd->nka', self.actor_w, h_new) + self.actor_b.unsqueeze(1)
        value = torch.einsum('nad,nkd->nka', self.critic_w, h_new) + self.critic_b.unsqueeze(1)
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
        # new_logp/old_logp is consistent with the rollout's `forward`). The prior
        # manual GRU here didn't chain timesteps (used a fixed h0), desyncing the
        # update from the rollout and preventing any learning.
        K = obs.size(0)
        spat = obs[:, :SPAT].view(K, GRID_H * GRID_W, SPAT_C)
        grid_emb = self._encode_grid(spat)
        f = grid_emb + self.grid_bias[i]                 # (K, d_model)
        own = obs[:, SPAT:].view(K, OWN_DIM)             # (K, OWN_DIM)
        x_in = torch.cat([f, own], dim=-1)               # (K, d_model+OWN_DIM)
        if hidden is None:
            hidden = torch.zeros(1, K, self.gru_hidden, device=obs.device, dtype=obs.dtype)
        # Replicate forward()'s _gru_step arithmetic exactly for agent i, but with
        # indexed per-agent params (gru_Wih has an N=16 leading dim, so we can't feed
        # it a single-agent tensor without an n-dim mismatch). This matches forward()
        # (per-k, non-recurrent across time) so the PPO ratio new_logp/old_logp stays
        # consistent with the rollout's forward() call.
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
        logits = torch.einsum('ad,kd->ka', self.actor_w[i], h_new) + self.actor_b[i].unsqueeze(0)
        value = torch.einsum('ad,kd->ka', self.critic_w[i], h_new) + self.critic_b[i].unsqueeze(0)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -float('inf'))
        return logits, value, h_new.unsqueeze(0)

    def params_of(self, i: int):
        return [
            self.grid_bias[i], self.gru_Wih[i], self.gru_Whh[i], self.gru_b[i],
            self.actor_w[i], self.actor_b[i], self.critic_w[i], self.critic_b[i]
        ]
