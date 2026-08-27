"""16-net EmergenceGrid training harness (D1 independent, D3 128x128/N=16).

Single batched policy AgentPolicyBatch(16) holds 16 independent weight sets
(vectorized inference). ONE Adam optimizer over all params; each agent's PPO
backward only touches its own slice (einsum streams independent) so per-agent
optimizer.step() updates only that agent. Rollout: one batched forward over all
N*K obs -> 16x fewer GPU calls.

Run: python src/train.py [--n 16 --grid 128 --k 8 --nstep 64 --nupd 2000]
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
print("[train.py] module loaded", flush=True)
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', message='.*grad attribute of a Tensor that is not a leaf.*')


def fixpath(p):
    """Normalize MSYS-style paths (/d/rl-emergence/...) to native Windows
    (D:/rl-emergence/...) so torch.save doesn't land in a phantom D:\\d\\... dir."""
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

from model import AgentPolicyBatch, OBS_DIM
from ppo import RolloutBuffer, RewardNormalizer
from env import EmergenceGrid

def make_hid_stack_batched(hiddens, K, H, device):
    # hiddens: list of K entries, each (1,1,H) or None -> (1,K,H)
    parts = [h if h is not None else torch.zeros(1, 1, H, device=device)
             for h in hiddens]
    return torch.cat(parts, dim=1)


def reward_schedule(p, mode='none'):
    """Return annealed reward params for training progress p in [0,1].

    The hypothesis under test (user): static reward constants are the crux of the
    ~9.7% ceiling -- a fixed reward can't both DISCOVER a behavior early (needs
    forgiving, exploration-friendly signal) and SHARPEN it late (needs tight
    signal). So we anneal:
      - food_pull / nav_alpha: HIGH early (discover food-finding), decay late.
      - invalid_harvest_pen:   LOW early (don't punish fumbling), rise late.
      - eat_gain:              constant (the core collection reward).
      - trait_mut_pen:         constant (mutate cost).
    'none'   -> static defaults (control / A-B baseline).
    'linear' -> linear anneal from early->late over p in [0,1].
    """
    rp = dict(food_pull=1.0, nav_alpha=0.15, eat_gain=15.0, eat_gain_regular=15.0,
              invalid_harvest_pen=0.5, trait_mut_pen=1.0,
              trait_mut_pen_gated=0.0, gate_gain=0.8, trait_match_bonus=0.0,
              mutate_gated_gain=1.5, wrong_trait_pen=0.3)
    if mode == 'none' or p is None:
        return rp
    p = max(0.0, min(1.0, p))
    if mode == 'linear':
        rp['food_pull'] = 2.0 - 1.0 * p          # 2.0 -> 1.0
        rp['nav_alpha'] = 0.30 - 0.15 * p         # 0.30 -> 0.15
        rp['invalid_harvest_pen'] = 0.1 + 0.4 * p  # 0.1 -> 0.5
    return rp


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def adaptive_reward_params(diag_agg, rp, k=8, step=0.15):
    """CLOSED-LOOP reward adaptation: react to the agent's ACTUAL failure mode
    (measured from sim diagnostics this episode), not a blind time schedule.

    diag_agg = sum of per-env (steps, harvest_invalid, harvest_valid, move_away,
    move_closer, mutate_steps, gate_adj, gate_adj_strong, dead) across all K envs.

    Failure-mode -> lever mapping (each nudged by `step`, then clamped):
      1. harvest-spam (invalid/(invalid+valid) high)  -> RAISE invalid_harvest_pen
      2. moves-away (move_away > move_closer)          -> RAISE food_pull + nav_alpha
      3. mutate-but-no-eat (mutate_steps high, valid low) -> LOWER trait_mut_pen
      4. gate-adjacent-but-weak (gate_adj high)       -> RAISE trait_match_bonus
         (shaping for being next to gated things; bridges L3 credit cliff)
      5. dying (dead>0)                                -> RAISE eat_gain slightly
    """
    steps, hi, hv, maway, mclose, mut, gadj, gstr, dead = diag_agg
    steps = max(1, steps)
    invalid_rate = hi / (hi + hv + 1)
    # 1. harvest-spam -> penalize invalid harvest harder
    if invalid_rate > 0.15:
        rp['invalid_harvest_pen'] = clamp(rp['invalid_harvest_pen'] + step, 0.1, 3.0)
    elif invalid_rate < 0.03:
        rp['invalid_harvest_pen'] = clamp(rp['invalid_harvest_pen'] - step * 0.5, 0.1, 3.0)
    # 2. moving away from food more than toward -> boost navigation signal
    if maway > mclose:
        rp['food_pull'] = clamp(rp['food_pull'] + step * 0.5, 0.2, 3.0)
        rp['nav_alpha'] = clamp(rp['nav_alpha'] + step * 0.1, 0.02, 0.5)
    # 3. mutating a lot but not eating gated food -> make mutating cheaper
    if mut > 0.2 * steps and hv < 0.1 * steps:
        rp['trait_mut_pen'] = clamp(rp['trait_mut_pen'] - step * 0.3, 0.1, 2.0)
    # 4. hanging near gates but never opening -> shape gate adjacency
    if gadj > 0.1 * steps:
        rp['trait_match_bonus'] = clamp(rp['trait_match_bonus'] + step * 0.2, 0.0, 1.0)
    # 5. agents dying -> eating more valuable
    if dead > 0:
        rp['eat_gain'] = clamp(rp['eat_gain'] + step * 2.0, 5.0, 30.0)
    return rp


def run(n=16, grid=128, k=8, nstep=64, nupd=2000, seed=12345, log_every=50,
        lr=2.5e-4, clip=0.2, ent_coef=0.05, vf_coef=0.5, n_epochs=4,
        minibatch=64, ckpt_dir=None, exp_name='eg16', save_every=200,
        resume=False, respawn=False, curriculum=5, food_seed=0, food_seed_dist=1,
        food_density_div=50, init_ckpt=None, food_regen_mode=2, freeze_vision=False,
        gated_food=1, d_model=256, gru_hidden=256, head_dim=256, ent_floor=0.5,
        reward_schedule_mode='none', adaptive=False, eat_gain_regular=15.0,
        diag_train=False, reward_preset='default', gate_thresh=0.95):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = fixpath(ckpt_dir)
    print(f"[train] n={n} grid={grid} k={k} nstep={nstep} nupd={nupd} device={device}", flush=True)

    envs = [EmergenceGrid(width=grid, height=grid, n_agents=n, seed=seed + e * 1000,
                         respawn=respawn, curriculum=curriculum, food_seed=food_seed,
                         food_seed_dist=food_seed_dist, food_density_div=food_density_div,
                         food_regen_mode=food_regen_mode, gated_food=gated_food,
                         reward_preset=reward_preset, gate_thresh=gate_thresh)
            for e in range(k)]
    policy = AgentPolicyBatch(n, freeze_vision=freeze_vision,
                              d_model=d_model, gru_hidden=gru_hidden, head_dim=head_dim).to(device)
    H = policy.gru_hidden
    print(f"[model] d_model={d_model} gru_hidden={gru_hidden} head_dim={head_dim} "
          f"params={sum(p.numel() for p in policy.parameters()):,}", flush=True)
    if init_ckpt:
        sd = torch.load(fixpath(init_ckpt), map_location=device)
        policy.load_state_dict(sd)
        # food_scale is a plain float (not a Parameter) so it isn't in the
        # state_dict. The food pretrain trained at scale 8 -> restore it so the
        # goal signal keeps dominating the (random) GRU/CNN logits during RL.
        policy.food_scale = 8.0
        print(f"[init] loaded weights from {init_ckpt} (food_scale={policy.food_scale})", flush=True)
    import sys as _sys
    print('[DEBUG] H=%d policy.gru_hidden=%d' % (H, policy.gru_hidden), file=_sys.stderr, flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    start_upd = 0
    if resume and ckpt_dir:
        # find latest *_policy_stepN.pt
        import glob as _gl
        cands = _gl.glob(os.path.join(ckpt_dir, f"{exp_name}_policy_step*.pt"))
        if cands:
            def _num(p):
                import re
                m = re.search(r'_step(\d+)\.pt$', p)
                return int(m.group(1)) if m else -1
            latest = max(cands, key=_num)
            policy.load_state_dict(torch.load(latest, map_location=device))
            start_upd = _num(latest)            # resume from the policy's step
            ocands = _gl.glob(os.path.join(ckpt_dir, f"{exp_name}_opt_step*.pt"))
            if ocands:
                olatest = max(ocands, key=_num)
                sd = torch.load(olatest, map_location=device)
                opt.load_state_dict(sd['opt'])
                print(f"[resume] loaded {os.path.basename(latest)} "
                      f"(upd {start_upd}) + optimizer")
            else:
                print(f"[resume] loaded {os.path.basename(latest)} "
                      f"(upd {start_upd}) [no opt ckpt; fresh optimizer]")
        else:
            print(f"[resume] no checkpoint found in {ckpt_dir}; starting fresh")

    # per-agent rollout buffers (obs stored as (T,K,OBS_DIM))
    # Per-agent reward normalizers: persist across updates to accumulate stable
    # running std. Normalizing before GAE is critical because raw reward spans
    # [-0.5 (invalid harvest) to +15 (eat)] causing VF loss in the hundreds.
    bufs = [RolloutBuffer(nstep) for _ in range(n)]
    rew_norms = [RewardNormalizer(clip=10.0) for _ in range(n)]
    hid_batched = torch.zeros(n, k, H, device=device)
    obs_now = [envs[e].reset() for e in range(k)]
    ep_rew = [[0.0] * n for _ in range(k)]
    # closed-loop adaptive reward state (persisted across updates)
    rp_state = dict(food_pull=1.0, nav_alpha=0.15, eat_gain=15.0, eat_gain_regular=15.0,
                    invalid_harvest_pen=0.5, trait_mut_pen=1.0,
                    gate_gain=0.8, trait_match_bonus=0.0)
    t0 = time.time()

    for upd in range(start_upd, nupd):
        # Reset world + hidden at the start of each update's rollout so food is
        # replenished per-episode (finite food per episode, NOT in-place regen ->
        # no pocket-feeding). Without this, no-regen training depletes food on the
        # first rollout and trains on empty maps forever.
        obs_now = [envs[e].reset() for e in range(k)]
        # Apply reward parameters for this training step.
        # Priority: closed-loop adaptive > schedule > static defaults.
        if adaptive:
            sched = dict(rp_state)          # carried over from last iteration's adaptation
        else:
            p = upd / max(1, nupd - 1)
            sched = reward_schedule(p, reward_schedule_mode)
        # Phase-2 curriculum: regular-food reward can be zeroed (energy still granted
        # in sim.cpp harvest()) so harvest-spam gives no reward -- only gated/gate
        # food pays. Injected here so it overrides schedule/adaptive values.
        sched['eat_gain_regular'] = eat_gain_regular
        for e in range(k):
            envs[e].set_step_frac(upd / max(1, nupd - 1))
            envs[e].set_reward_params(**sched)
        hid_batched = torch.zeros(n, k, H, device=device)
        for b in bufs:
            b.reset()
        alive_count = [n] * k
        upd_deaths = 0
        upd_harvest = 0
        upd_gateopen = 0
        # diag_train: per-step gate-context capture to separate BEHAVIOR
        # (does the agent take MUTATE near an unlocked gated tile?) from CREDIT
        # (is mean GAE advantage positive there?). Filled only when --diag_train.
        rollout_ctx = [] if diag_train else None
        _ctx_bins = {} if diag_train else None  # (act,adj_gated,adj_gated_unlock)->[(adv,1)]

        for t in range(nstep):
            # build batched obs (N*K, OBS_DIM) for ONE forward -- a single GPU copy
            obs_batched = torch.as_tensor(np.concatenate(obs_now, axis=0),
                                         dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, vals, h_new = policy(obs_batched, hid_batched)
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()                      # (N*K,)
            logp = dist.log_prob(acts)
            # reshape to (N, K)
            acts_m = acts.view(n, k)
            logp_m = logp.view(n, k)
            vals_m = vals.view(n, k).squeeze(-1)      # (N,K)
            h_new_m = h_new                           # (N,K,H)

            rews = [[0.0] * n for _ in range(k)]
            dones = [[False] * n for _ in range(k)]
            next_obs = []
            for e in range(k):
                vec = [int(acts_m[i, e].item()) for i in range(n)]
                # gate state before step (for open detection)
                gc = envs[e].gate_cells
                before = sum(1 for (gx, gy) in gc if envs[e].grid[gy * envs[e].W + gx] != 6) \
                    if gc else 0
                inv_before = [envs[e]._sim.agents[i].inv for i in range(n)]
                o, r, d, info = envs[e].step(vec)
                after = sum(1 for (gx, gy) in gc if envs[e].grid[gy * envs[e].W + gx] != 6) \
                    if gc else 0
                if after > before:
                    upd_gateopen += 1
                for i in range(n):
                    if envs[e]._sim.agents[i].inv > inv_before[i]:
                        upd_harvest += 1
                    if d[i] and envs[e].agents[i].alive is False:
                        upd_deaths += 1
                next_obs.append(o)
                # ---- diag_train context capture (after step) ----
                if rollout_ctx is not None:
                    grid_e = envs[e].grid
                    W_e = envs[e].W
                    da = envs[e]._sim.dump_agents()
                    for i in range(n):
                        a = da[i]
                        ax, ay = a['x'], a['y']
                        adj_gated = False
                        adj_gated_unlock = False
                        dg = 999
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = ax + dx, ay + dy
                                if 0 <= nx < W_e and 0 <= ny < W_e:
                                    tt = grid_e[ny * W_e + nx]
                                    if tt == 2 or tt == 3:  # HARD_NUT / TALL_FRUIT
                                        adj_gated = True
                                        dg = min(dg, max(abs(dx), abs(dy)))
                                        if (tt == 2 and a['can_hard']) or (tt == 3 and a['can_tall']):
                                            adj_gated_unlock = True
                        rollout_ctx.append((int(acts_m[i, e].item()), adj_gated,
                                            adj_gated_unlock, float(a['strength']), dg,
                                            i, e, t))
                for i in range(n):
                    rews[e][i] = float(r[i])
                    dones[e][i] = bool(d[i])
                    ep_rew[e][i] += r[i]
                    if d[i]:
                        h_new_m[i, e, :].zero_()

            for i in range(n):
                bufs[i].obs.append(obs_batched[i::n].detach())
                bufs[i].acts.append(acts_m[i])
                bufs[i].logp.append(logp_m[i])
                # Normalize rewards before GAE: divides by running std so the
                # critic fits unit-scale returns regardless of harvest magnitude.
                norm_rew = torch.tensor(
                    [rew_norms[i].normalize(rews[e][i]) for e in range(k)],
                    device=device)
                bufs[i].rew.append(norm_rew)
                bufs[i].val.append(vals_m[i])
                bufs[i].don.append(torch.tensor([1.0 if dones[e][i] else 0.0
                                                  for e in range(k)], device=device))
                bufs[i].hid.append(hid_batched[i:i + 1, :, :].detach())   # (1,K,H)
            hid_batched = h_new_m
            obs_now = next_obs

        # ---- CLOSED-LOOP adaptation: read this update's diagnostics, react ----
        if adaptive:
            agg = [0] * 9
            for e in range(k):
                d = envs[e].get_diag()
                for j in range(9):
                    agg[j] += int(d[j])
            rp_state = adaptive_reward_params(tuple(agg), rp_state, k=k)
            if log_every and upd % log_every == 0:
                print(f"  [adapt] ihp={rp_state['invalid_harvest_pen']:.2f} "
                      f"fp={rp_state['food_pull']:.2f} nav={rp_state['nav_alpha']:.2f} "
                      f"tmp={rp_state['trait_mut_pen']:.2f} eg={rp_state['eat_gain']:.1f} "
                      f"tmb={rp_state['trait_match_bonus']:.2f} | diag hi={agg[1]} hv={agg[2]} "
                      f"away={agg[3]} close={agg[4]} mut={agg[5]} dead={agg[8]}", flush=True)

        # action histogram for agent 0 (before updates)
        import collections as _col
        ah = _col.Counter()
        for t_act in bufs[0].acts:
            for a_val in t_act.cpu().tolist():
                ah[int(a_val)] += 1
        top = ah.most_common(4)
        topstr = ",".join("%d:%.0f%%" % (a_val, 100.0 * c / max(sum(ah.values()), 1)) for a_val, c in top)

        # per-agent PPO update
        losses = []
        for i in range(n):
            # bootstrap last value from final obs of agent i
            obs_i = torch.stack([torch.tensor(obs_now[e][i], dtype=torch.float32,
                                                device=device) for e in range(k)])
            h_i = hid_batched[i:i + 1, :, :]
            with torch.no_grad():
                _, lv, _ = policy.forward_agent(i, obs_i, h_i)
            bufs[i].bootstrap_val = lv.squeeze(-1).detach().cpu()
            bufs[i].bootstrap_don = torch.tensor(
                [1.0 if not envs[e].agents[i].alive else 0.0 for e in range(k)],
                device=device).detach().cpu()
            pl, vl, ent, adv_i = ppo_update_agent(policy, opt, i, bufs[i], device, clip,
                                      ent_coef, vf_coef, n_epochs, minibatch, ent_floor)
            losses.append((pl, vl, ent))
            # diag_train: bin adv by (action x gate-context) for agent i
            if rollout_ctx is not None:
                adv_l = adv_i.reshape(-1)  # (T*B,)
                for (act_, adj, adju, str_, dg_, ci, ce, ct) in rollout_ctx:
                    if ci != i:
                        continue
                    # map flat (ct, ce) -> position in adv (T,B) with B ordered by e
                    idx_flat = ct * k + ce
                    if 0 <= idx_flat < adv_l.numel():
                        bin_key = (act_, adj, adju)
                        _ctx_bins.setdefault(bin_key, []).append((float(adv_l[idx_flat]), 1.0))

        alive = sum(1 for e in range(k) for i in range(n)
                    if envs[e].agents[i].alive)
        avg_en = sum(envs[e].agents[i].energy
                     for e in range(k) for i in range(n)) / (k * n)
        # diag_train: print advantage-by-(action x gate-context) table
        if _ctx_bins is not None and (upd % log_every == 0 or upd == nupd - 1):
            from collections import defaultdict as _df
            rows = []
            for (act_, adj, adju), lst in _ctx_bins.items():
                n_ = len(lst)
                mean_adv = sum(v for v, _ in lst) / n_
                rows.append((act_, adj, adju, n_, mean_adv))
            # sort by count desc, show every (act,context) seen
            rows.sort(key=lambda r: -r[3])
            print(f"  [diag] adv by (act, adj_gated, adj_unlock) -- "
                  f"BEHAVIOR: low count on (8-12,*,True)? CREDIT: mean_adv sign:", flush=True)
            for act_, adj, adju, n_, mean_adv in rows:
                print(f"    act={act_:2d} adj_gated={int(adj)} adj_unlock={int(adju)} "
                      f"n={n_:5d} mean_adv={mean_adv:+.4f}", flush=True)
            _ctx_bins.clear()
            rollout_ctx.clear()
        if ckpt_dir and save_every and (upd % save_every == 0):
            os.makedirs(ckpt_dir, exist_ok=True)
            ip = os.path.join(ckpt_dir, f"{exp_name}_policy_step{upd}.pt")
            torch.save(policy.state_dict(), ip)
            # optimizer + step sidecar so we can resume cleanly
            op = os.path.join(ckpt_dir, f"{exp_name}_opt_step{upd}.pt")
            torch.save({'opt': opt.state_dict(), 'upd': upd + 1}, op)
            print(f"  [ckpt] {os.path.abspath(ip)} ({os.path.getsize(ip)} bytes)", flush=True)
        # always checkpoint the FINAL update so a completed run never loses its
        # trained policy (the save_every gate above only fires on multiples).
        if ckpt_dir and upd == nupd - 1:
            os.makedirs(ckpt_dir, exist_ok=True)
            ip = os.path.join(ckpt_dir, f"{exp_name}_policy_final.pt")
            torch.save(policy.state_dict(), ip)
            op = os.path.join(ckpt_dir, f"{exp_name}_opt_final.pt")
            torch.save({'opt': opt.state_dict(), 'upd': upd + 1}, op)
            print(f"  [ckpt-final] {os.path.abspath(ip)} ({os.path.getsize(ip)} bytes)", flush=True)
        if upd % log_every == 0 or upd == nupd - 1:
            pl = sum(l[0] for l in losses) / n
            vl = sum(l[1] for l in losses) / n
            denom = k * n * nstep
            # mean manhattan distance to nearest food (FOOD=1) across all agents
            meandist = float(np.mean([obs_now[e][i][-1] * 40.0 for e in range(k) for i in range(n)]))
            ent = sum(l[2] for l in losses) / n
            print('upd %4d | pol %.3f vf %.3f ent %.2f | alive %d/%d deaths/step %.3f harv/step %.3f dist %.2f topact[%s] gateopen %d | %.0fs' % (
                upd, pl, vl, ent, alive, k * n, upd_deaths / denom, upd_harvest / denom, meandist, topstr, upd_gateopen, time.time() - t0), flush=True)

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{exp_name}_policy.pt")
        torch.save(policy.state_dict(), path)
        assert os.path.getsize(path) > 10000, f"save too small: {path}"
        op = os.path.join(ckpt_dir, f"{exp_name}_opt.pt")
        torch.save({'opt': opt.state_dict(), 'upd': nupd}, op)
        print(f"saved batched policy to {os.path.abspath(path)} "
              f"({os.path.getsize(path)} bytes)", flush=True)


def ppo_update_agent(policy, opt, i, buf, device, clip, ent_coef, vf_coef,
                     n_epochs, minibatch, ent_floor=0.0):
    # bootstrap values set on buf by caller
    adv, ret = buf.compute_gae(buf.bootstrap_val.to(device),
                               buf.bootstrap_don.to(device))
    T, B = adv.shape
    obs = torch.stack(buf.obs).to(device)             # (T,K,OBS_DIM)
    act = torch.stack(buf.acts).to(device)
    old_logp = torch.stack(buf.logp).to(device)
    adv_t = adv.to(device).detach()
    ret_t = ret.to(device).detach()
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    flat_obs = obs.reshape(T * B, -1)
    flat_act = act.reshape(-1)
    flat_old = old_logp.reshape(-1)
    flat_adv = adv_t.reshape(-1)
    flat_ret = ret_t.reshape(-1)
    # Stack hidden states: (T, 1, K, H) -> (T*K, H)
    flat_hid = torch.stack(buf.hid, dim=0).squeeze(1).reshape(T * B, -1).to(device)
    pol_losses = []
    vf_losses = []
    ent_sum = 0.0
    ent_n = 0
    for _ in range(n_epochs):
        perm = torch.randperm(T * B, device=device)
        for start in range(0, T * B, minibatch):
            mb = perm[start:start + minibatch]
            o_mb = flat_obs[mb]
            h0 = flat_hid[mb].unsqueeze(0)            # (1, minibatch, H)
            logits, value, _ = policy.forward_agent(i, o_mb, h0)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(flat_act[mb])
            entropy = dist.entropy().mean()
            ent_sum += float(entropy.detach().cpu())
            ent_n += 1
            ratio = torch.exp(new_logp - flat_old[mb])
            s1 = ratio * flat_adv[mb]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * flat_adv[mb]
            pol_loss = -torch.min(s1, s2).mean()
            val_loss = F.mse_loss(value.squeeze(-1), flat_ret[mb])
            loss = pol_loss + vf_coef * val_loss - ent_coef * entropy
            if ent_floor > 0.0:
                # Entropy FLOOR: penalize entropy BELOW the floor so the policy
                # can never collapse to a single constant action (the PPO
                # single-action attractor that produced topact[5:100%]). This is
                # not a move penalty -- it only stops degenerate constant-output
                # policies, leaving free movement intact.
                loss = loss + 0.1 * torch.clamp(ent_floor - entropy, min=0.0)
            opt.zero_grad()
            loss.backward()
            # einsum streams are independent: agent i's backward only populates
            # agent i's param grads; others stay None. So step() moves only i.
            torch.nn.utils.clip_grad_norm_(policy.params_of(i), 0.5)
            opt.step()
            pol_losses.append(pol_loss.detach().cpu())
            vf_losses.append(val_loss.detach().cpu())
    buf.reset()
    return (float(torch.stack(pol_losses).mean()), float(torch.stack(vf_losses).mean()),
            ent_sum / max(ent_n, 1), adv.detach().cpu())


if __name__ == '__main__':
    print("[train.py] ENTER MAIN", flush=True)
    import torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--grid', type=int, default=128)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--nstep', type=int, default=64)
    ap.add_argument('--nupd', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--ckpt_dir', type=str, default=None)
    ap.add_argument('--exp', type=str, default='eg16')
    ap.add_argument('--save_every', type=int, default=200)
    ap.add_argument('--resume', action='store_true',
                   help='resume from latest *_stepN.pt in ckpt_dir (policy+opt)')
    ap.add_argument('--respawn', action='store_true',
                   help='respawn dead agents at full energy (OFF for training: '
                        'real death pressure; keep ON only for GIF/episode viz)')
    ap.add_argument('--ent_coef', type=float, default=0.05,
                   help='entropy coefficient (higher = more exploration). 0.05 prevents the '
                        'single-action collapse that produced topact[5:100%] harvest-spam.')
    ap.add_argument('--lr', type=float, default=2.5e-4,
                   help='Adam learning rate')
    ap.add_argument('--curriculum', type=int, default=5,
                   help='world complexity: 0 food-only -> 5 full (train low->high)')
    ap.add_argument('--food_seed', type=int, default=0,
                   help='if >0, drop a food tile next to every spawn (fast L0 perception warmup)')
    ap.add_argument('--food_seed_dist', type=int, default=8,
                   help='Manhattan ring distance at which food_seed food is placed. MUST be >1: '
                        'dist=1 makes harvest-spam optimal (agent camps on adjacent food); '
                        '8+ forces real navigation.')
    ap.add_argument('--food_density_div', type=int, default=50,
                   help='base food = W*H/food_density_div. Higher = sparser world '
                        '(e.g. 400 -> ~10 food tiles on 64x64).')
    ap.add_argument('--init_ckpt', type=str, default=None,
                   help='path to a state_dict to LOAD AS INIT (e.g. a food_w pretrain). '
                        'Unlike --resume, this only seeds weights; training starts fresh at upd 0.')
    ap.add_argument('--food_regen_mode', type=int, default=2,
                   help='food regen after harvest: 0=none (finite food, agent must navigate '
                        'to scattered tiles), 1=in-place (regrows where eaten -> pocket-feeds '
                        'the agent), 2=random empty cell (default: food stays available but '
                        'never regrows on top of the agent)')
    ap.add_argument('--freeze_vision', action='store_true',
                   help='freeze the CNN+grid_bias backbone; only GRU/heads train. Use to '
                        'preserve a known-good perception (e.g. correct vertical nav) while '
                        'adapting the policy head to a new reward/world.')
    ap.add_argument('--gated_food', type=int, default=1,
                   help='gated food at curriculum>=2: 0=none, 1=regular+trickle gated, '
                        '2=gated-dominant (agent must mutate can_hard/can_tall to eat)')
    ap.add_argument('--d_model', type=int, default=256, help='CNN/feature width')
    ap.add_argument('--gru_hidden', type=int, default=256, help='GRU hidden + head width')
    ap.add_argument('--head_dim', type=int, default=256, help='reasoning-head MLP width')
    ap.add_argument('--ent_floor', type=float, default=0.5,
                   help='minimum entropy floor for the policy (prevents single-action '
                        'collapse / constant-output attractors). 0.5 is a sane default '
                        'for solo navigation.')
    ap.add_argument('--reward_schedule', type=str, default='none',
                   choices=['none', 'linear'],
                   help='anneal reward params over training (dynamic rewards). '
                        'none=static (control); linear=high pull/low pen early -> '
                        'low pull/high pen late. Tests whether static rewards cap learning.')
    ap.add_argument('--eat_gain_regular', type=float, default=15.0,
                   help='reward for eating REGULAR food (FOOD/OASIS). Phase-2 curriculum '
                        'sets this to 0 so harvest-spam gives energy (survival) but NO '
                        'reward -- only gated food + gates pay, forcing the agent off the '
                        'harvest-spam local optimum. Default 15.0 = same as gated eat_gain.')
    ap.add_argument('--adaptive', action='store_true',
                   help='CLOSED-LOOP reward adaptation: after each update, read sim '
                        'diagnostics (harvest-spam, move-away, mutate-no-eat, gate-adj, '
                        'death) and REACT by adjusting reward params. Overrides --reward_schedule.')
    ap.add_argument('--diag_train', action='store_true',
                   help='INSTRUMENTATION: capture per-step gate-context (adjacent to '
                        'gated tile? has matching trait? strength) and bin GAE advantage '
                        'by (action x context). Separates BEHAVIOR (does the policy emit '
                        'MUTATE near an unlocked gate?) from CREDIT (is mean advantage on '
                        'that action positive?).')
    ap.add_argument('--reward_preset', type=str, default='default',
                   choices=['default', 'gc'],
                   help="Reward-density lever. 'gc' = G+C preset: raises trait_match_bonus "
                        "(bridge mutate->eat), mutate_gated_gain, sharpens wrong_trait_pen, "
                        "and adds dense gate_prox_bonus for strong+adjacent-to-gate. "
                        "Diagnosed via --diag_train as the credit-sparse bottleneck.")
    ap.add_argument('--gate_thresh', type=float, default=0.95,
                   help="Gate opens at combined pusher strength >= this. Curriculum: lower (e.g. 0.6) to ease multi-agent coordination, then ramp up.")
    args = ap.parse_args()
    run(n=args.n, grid=args.grid, k=args.k, nstep=args.nstep, nupd=args.nupd,
        seed=args.seed, log_every=args.log_every, ckpt_dir=args.ckpt_dir,
        exp_name=args.exp, save_every=args.save_every, resume=args.resume,
        respawn=args.respawn, ent_coef=args.ent_coef, curriculum=args.curriculum,
        lr=args.lr, food_seed=args.food_seed, food_seed_dist=args.food_seed_dist,
        food_density_div=args.food_density_div, init_ckpt=args.init_ckpt,
        food_regen_mode=args.food_regen_mode, freeze_vision=args.freeze_vision,
        gated_food=args.gated_food,
        d_model=args.d_model, gru_hidden=args.gru_hidden, head_dim=args.head_dim,
        ent_floor=args.ent_floor, reward_schedule_mode=args.reward_schedule,
        adaptive=args.adaptive, eat_gain_regular=args.eat_gain_regular,
        diag_train=args.diag_train, reward_preset=args.reward_preset,
        gate_thresh=args.gate_thresh)
