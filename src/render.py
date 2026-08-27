"""Headless GIF renderer for EmergenceGrid (demo + emergence inspection).

Loads a trained AgentPolicyBatch checkpoint and replays ONE episode on a single
grid, drawing each frame:
  - tiles (food / hard_nut / tall_fruit / gap / wall / gate / oasis / hazard /
    marker) with design-doc colors
  - agents: circle colored by ARCHETYPE, radius scaled by energy, with a glyph
    for last_action (move dir / harvest / share / signal)
  - HUD: step, alive count, avg energy, cumulative group credit
Outputs an animated GIF. Group ckpt (n=16) renders all 16 agents on one grid;
solo ckpt (n=1) zooms the single agent.

Run:
  python src/render.py --ckpt ckpts/eg16_g64/eg16_g64_policy_step400.pt \
      --n 16 --grid 64 --out gifs/group_step400.gif --steps 300 --greedy
  python src/render.py --ckpt ckpts/solo_g64/solo_g64_policy_step800.pt \
      --n 1 --grid 64 --out gifs/solo.gif --steps 300
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch

# matplotlib without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import numpy as np
from PIL import Image

from env import (EmergenceGrid, EMPTY, FOOD, HARD_NUT, TALL_FRUIT, GAP, WALL,
                 GATE, HAZARD, PREDATOR, OASIS, MARKER)
from model import AgentPolicyBatch, OBS_DIM

HIDDEN = 128


def fixpath(p):
    if p is None:
        return None
    p = os.path.abspath(p)
    # MSYS form /d/... (forward slash)
    if p.startswith('/') and len(p) >= 3 and p[1].isalpha() and p[2:3] == '/':
        p = p[1].upper() + ':/' + p[3:]
    # double-prefix D:\d\...
    import re
    m = re.match(r'^([A-Za-z]):\\d\\(.*)$', p)
    if m:
        p = m.group(1) + ':/' + m.group(2)
    return p

# tile colors (RGB 0-255)
TILE_COLOR = {
    EMPTY: (18, 18, 24),
    FOOD: (40, 140, 55),
    HARD_NUT: (120, 110, 40),
    TALL_FRUIT: (30, 150, 150),
    GAP: (90, 90, 100),
    WALL: (60, 60, 70),
    GATE: (200, 120, 30),
    HAZARD: (170, 40, 40),
    PREDATOR: (180, 60, 180),
    OASIS: (60, 200, 90),
    MARKER: (230, 220, 60),
}

ARCH_COLORS = {
    'striker': (220, 70, 70),     # red
    'scout': (70, 170, 230),      # blue
    'runner': (120, 220, 120),    # green
    'connector': (230, 150, 90),  # orange
    'generalist': (200, 200, 200), # gray
}


def arch_of(agent):
    # agent.traits has no archetype string; recover from trait signature
    t = agent.traits
    if t.strength >= 0.6 and t.reach < 0.5 and t.size_small == 0:
        return 'striker'
    if t.speed >= 0.6 and t.perception >= 3:
        return 'scout'
    if t.speed >= 0.7 and t.reach >= 0.5:
        return 'runner'
    if t.social >= 0.6:
        return 'connector'
    return 'generalist'


def draw_frame(ax, env, harvests, mean_dist, step, cell=11):
    H, W = env.H, env.W
    # background image
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = TILE_COLOR.get(env.grid[y * W + x], (0, 0, 0))
    ax.imshow(img)
    # agents
    for a in env.agents:
        if not a.alive:
            continue
        col = np.array(ARCH_COLORS.get(arch_of(a), (255, 255, 255))) / 255.0
        en = a.energy / 10.0
        r = 0.18 + 0.22 * en  # radius scales with energy
        circ = Circle((a.x, a.y), r, color=col, zorder=5)
        ax.add_patch(circ)
        # action glyph
        act = a.last_action
        if act in (1, 2, 3, 4):
            dx, dy = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}[act]
            ax.add_patch(FancyArrow(a.x, a.y, dx * 0.35, dy * 0.35,
                                     width=0.04, head_width=0.18,
                                     color='white', zorder=6))
        elif act == 5:  # harvest: small inner dot
            ax.add_patch(Circle((a.x, a.y), 0.08, color='white', zorder=6))
        elif act == 6:  # share: ring
            ax.add_patch(Circle((a.x, a.y), 0.30, fill=False,
                                edgecolor='white', linewidth=1.2, zorder=6))
        elif act == 7:  # signal: cross
            ax.plot([a.x - 0.3, a.x + 0.3], [a.y, a.y], color='yellow', lw=1.2, zorder=6)
            ax.plot([a.x, a.x], [a.y - 0.3, a.y + 0.3], color='yellow', lw=1.2, zorder=6)
        elif act in (8, 9, 10, 11, 12):  # trait mutation: magenta diamond
            ax.add_patch(plt.Polygon(
                [(a.x, a.y - 0.28), (a.x + 0.28, a.y), (a.x, a.y + 0.28),
                 (a.x - 0.28, a.y)], closed=True, fill=True,
                facecolor='magenta', edgecolor='white', linewidth=0.6, zorder=6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {step} | alive {sum(1 for a in env.agents if a.alive)}/"
                 f"{env.n_agents} | avgE {np.mean([a.energy for a in env.agents]):.1f}"
                 f" | harvests {harvests} | meandist {mean_dist:.2f}", fontsize=10)


def render(ckpt, n, grid, steps, out, seed=7, greedy=False, cell=11, fps=8,
            food_seed=0, food_seed_dist=1, curriculum=0, food_density_div=50,
            harvest_bias=0.0, food_regen_mode=2, food_scale=None, gated_food=1,
            d_model=256, gru_hidden=256, head_dim=256, gate_thresh=0.95):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = AgentPolicyBatch(n, d_model=d_model, gru_hidden=gru_hidden, head_dim=head_dim).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    if food_scale is not None:
        policy.food_scale = float(food_scale)   # strengthen/weaken directional prior at eval

    env = EmergenceGrid(width=grid, height=grid, n_agents=n, seed=seed,
                        curriculum=curriculum, food_seed=food_seed,
                        food_seed_dist=food_seed_dist, respawn=True,
                        food_density_div=food_density_div,
                        food_regen_mode=food_regen_mode, gated_food=gated_food,
                        gate_thresh=gate_thresh)
    obs = env.reset()
    hid = torch.zeros(n, 1, policy.gru_hidden, device=device)
    frames = []
    harvests = 0
    prev_inv = [a.inv for a in env.agents]

    def mean_dist():
        g = env._sim.grid; W = env._sim.W; Hh = env._sim.H
        tot = 0; c = 0
        for ag in env._sim.agents:
            best = 999
            for yy in range(Hh):
                for xx in range(W):
                    if g[yy * W + xx] == 1:
                        d = abs(xx - ag.x) + abs(yy - ag.y)
                        if d < best:
                            best = d
            tot += best; c += 1
        return tot / max(c, 1)

    fig, ax = plt.subplots(figsize=(grid * cell / 72.0, grid * cell / 72.0),
                           dpi=72)
    for step in range(steps):
        obs_b = torch.as_tensor(obs[:n], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, vals, h_new = policy(obs_b, hid)
        if harvest_bias:
            logits = logits.clone()
            logits[:, 5] += harvest_bias   # constant harvest nudge (eval-only)
        dist = torch.distributions.Categorical(logits=logits)
        # Decode: greedy (argmax) for eval inspection, else stochastic (training).
        acts = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        acts_l = acts.cpu().tolist()
        o, r, d, info = env.step(acts_l)
        # truthful harvest count: inventory increased this step
        for i in range(n):
            if env.agents[i].inv > prev_inv[i]:
                harvests += 1
            prev_inv[i] = env.agents[i].inv
        hid = h_new[:, :, :].detach()
        draw_frame(ax, env, harvests, mean_dist(), step, cell)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[..., :3]))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(grid * cell / 72.0, grid * cell / 72.0),
                               dpi=72)
        obs = o
        if all(d):
            break
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"wrote {out} ({len(frames)} frames, {steps} steps, "
          f"harvests {harvests}, meandist {mean_dist():.2f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--out', default='gifs/episode.gif')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--food_seed', type=int, default=0,
                    help='drop food next to spawn (matches training)')
    ap.add_argument('--food_seed_dist', type=int, default=1,
                    help='Manhattan ring distance for seeded food')
    ap.add_argument('--curriculum', type=int, default=0)
    ap.add_argument('--food_density_div', type=int, default=50,
                   help='base food = W*H/food_density_div (higher = sparser)')
    ap.add_argument('--harvest_bias', type=float, default=0.0,
                   help='constant +bias added to action 5 (harvest) logits at eval '
                        '(makes greedy argmax prefer harvest when on food)')
    ap.add_argument('--food_regen_mode', type=int, default=2,
                   help='food regen after harvest: 0=none, 1=in-place (pocket-feeds), '
                        '2=random empty cell (default)')
    ap.add_argument('--gated_food', type=int, default=1,
                   help='gated food at curriculum>=2: 0=none, 1=regular+trickle gated, '
                        '2=gated-dominant (agent must mutate can_hard/can_tall to eat)')
    ap.add_argument('--gate_thresh', type=float, default=0.95,
                   help='gate opens at combined pusher strength >= this (must match the '
                        'bar the checkpoint was trained at: gc_curric=0.6, gc_anneal=0.95)')
    ap.add_argument('--food_scale', type=float, default=None,
                   help='override directional-prior strength at eval (model default 2.0; '
                        'init_ckpt sets 8.0). Higher = agent commits harder to moving toward food.')
    ap.add_argument('--d_model', type=int, default=256, help='CNN/feature width (must match ckpt)')
    ap.add_argument('--gru_hidden', type=int, default=256, help='GRU hidden + head width (must match ckpt)')
    ap.add_argument('--head_dim', type=int, default=256, help='reasoning-head MLP width (must match ckpt)')
    a = ap.parse_args()
    render(fixpath(a.ckpt), a.n, a.grid, a.steps, fixpath(a.out), seed=a.seed,
           greedy=a.greedy, fps=a.fps, food_seed=a.food_seed,
           food_seed_dist=a.food_seed_dist, curriculum=a.curriculum,
           food_density_div=a.food_density_div, harvest_bias=a.harvest_bias,
           food_regen_mode=a.food_regen_mode, food_scale=a.food_scale,
           gated_food=a.gated_food,
           d_model=a.d_model, gru_hidden=a.gru_hidden, head_dim=a.head_dim,
           gate_thresh=a.gate_thresh)
