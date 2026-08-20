# EmergenceGrid — Draft Design (2026-08-14)

## 0. Status
Draft. Goal already locked with owner: **demonstrate defensible emergent
cooperation in multi-agent RL** via a minimal grid world where agents have
traits + fallacies. Alone an agent merely survives; together the group exploits
high-value resources / repels threats it cannot alone. Minecraft RL is archived
(`D:\mc-rl_ARCHIVE_20260814.tar.gz`); this is the new direction.

## 1. Why this demonstrates emergence (the bar we must clear)
- Minimal env: no central controller, no "cooperate" reward term.
- Group strength is a function of *local interaction + trait complementarity*,
  not anything in the loss. If it appears, it emerged.
- Partial observability + fallacies guarantee the behavior is NOT an optimal
  solved policy, so we can't be accused of scripting it.
- Controlled ablations isolate the cause (see §10). That's what makes the claim
  defensible vs. "you shaped it."

## 2. Environment
- W×H grid (START 64×64 — "way bigger" so spatial structure + specialization
  can emerge; scales up), walled borders (no torus — keeps dynamics legible).
- Discrete time, simultaneous actions.
- Tile types:
  - `empty`
  - `food`        — cheap, low energy, ANY agent harvests
  - `hard_nut`    — high energy, harvestable only if Strength ≥ TH_STR
  - `tall_fruit`  — high energy, harvestable only if Reach ≥ TH_REACH
  - `gap`         — passable only by Size=small agents (others blocked)
  - `wall`        — blocks movement
  - `gate`        — a wall opening that opens only if Σ Strength of agents pushing
                    it the same step ≥ TH_GATE (cooperation-locked traversal, STRENGTH-SUM)
  - `hazard`      — static tile, damages on enter
  - `predator`    — mobile threat; beatable when Σ Strength of agents adjacent to
                    it ≥ TH_PRED. Defeat is a SUM-OF-TRAITS test, NOT an agent COUNT —
                    so complementarity (one strong agent + others, or several
                    moderate-strength agents) matters, not headcount.
  - `oasis`       — rich food cluster behind a `gate` (the prize)
- Resource regen: food regrows on a timer so survival is sustainable solo.

### 2.1 Step resolution (simultaneous actions, deterministic collisions)
All N agents declare intended actions for the step, then state changes resolve in
a FIXED deterministic order — NO unconditional simultaneous application (otherwise
two agents into one cell is undefined).
- Priority = Speed descending (fast agents win contested tiles — makes Speed
  mechanically deeper), tie-broken by a deterministic hash of the agent's (x,y)
  coordinates (reproducible, seed-stable).
- For each agent in priority order: if target cell is currently OCCUPIED (by an
  agent not yet resolved) or is wall/blocked/locked → move FAILS, agent stays.
  Else move succeeds and cell becomes occupied.
- Swaps resolve deterministically: if A,B swap, higher-priority A moves first into
  B's still-occupied cell (blocked); then B moves into A's vacated cell (succeeds).
  One-sided swaps create organic traffic — fine.
- Harvest/share/signal resolve in the same priority order (two agents harvesting
  the same tile → higher priority gets it; other no-ops).
- Predator/gate strength-sum checks use POST-move occupancy, so a contested tile
  can change who's adjacent to a predator this step.

## 3. Agents
- N agents (START 6).
- Per-agent state: energy (0..E_MAX, decays by metabolism/step; 0 = death),
  inventory (resource units), position, facing, alive.
- Partial observability: agent sees only a local window of radius R =
  PerceptionRadius + fallacies, never global state at inference.

## 4. Traits (fixed per agent for a run; randomized across agents)
Heterogeneity is what makes "together strong" possible and "alone" capped.
CRITICAL: traits are BOUNDED and CORRELATED with costs — every advantage is paid
for, so no agent can be strong+fast+perceptive+social simultaneously. This is
what FORCES division of labor: the strong agent is slow and hungry; the perceptive
agent can't fight; the social agent is individually weak. Alone, each is crippled;
together, the gaps cover each other. (Without binding tradeoffs, the emergent
claim collapses — a uniformly-good agent has no reason to need others.)

### 4.1 Trait list (each has a benefit AND a cost)
- Strength   0..1  — benefit: harvest hard_nut if ≥ TH_STR, push-power for gate,
                     predator-defeat Strength share.  COST: high Strength → higher
                     Metabolism (more food/step) + lower Speed (heavy).
- Reach      0..1  — benefit: harvest tall_fruit if ≥ TH_REACH.  COST: high Reach
                     → narrower Perception (all "eyes up", blind to ground detail)
                     + slightly higher Metabolism.
- Speed      0..1  — benefit: acts every step if high; low-speed skips steps.
                     COST: high Speed → higher Metabolism + lower Strength cap
                     (can't be both fast and strong; enforced in sampling).
- Perception 1..4  — benefit: local window radius (sees more, sees threats early).
                     COST: high Perception → lower Strength (fragile/small-frame) +
                     higher Metabolism (sensory cost).
- Metabolism 0.5..1.5 — energy decay multiplier.  NOT free: it is DERIVED from
                     the above (Strength/Speed/Perception load), NOT independently
                     sampled high. Low-metabolism agents are necessarily
                     low-capability on the paid traits.
- Size       small | large — small: passes gaps, low Metabolism, but LOW Strength
                     cap (can't push gate alone).  large: high Strength ceiling,
                     but blocked by gaps + higher Metabolism.
- SocialBias 0..1  — benefit: share(6)/signal(7) pay off (§8.4); coordinates.
                     COST: social agents are INDIVIDUALLY weak — sampling forces
                     their paid traits (Strength/Reach/Speed) BELOW average, and
                     they bleed energy on shares. A pure-social agent alone starves
                     faster and fights worse.

### 4.2 Binding sampler (enforces the tradeoffs — no free lunch)
- Draw a primary archetype (striker / scout / runner / connector / generalist-low).
- Paid traits are negatively correlated: e.g. Strength+Speed and Strength+Perception
  and Perception+Reach are anti-correlated (sample one high ⇒ others pulled low).
- Hard caps: Strength+Speed ≤ 1.3; Perception+Strength ≤ 1.3; SocialBias>0.6 ⇒
  mean(paid traits) ≤ 0.45 (social agents are individually weak by construction).
- Generalist-low archetype gets ALL traits mediocre (≈0.4) — the "jack of none"
  baseline that survives solo but contributes little to group strength, so the
  value of specialists is visible.
- Same sampler used for solo baseline (§10) so solo-vs-group gap is trait-fair.

### 4.3 Why this makes emergence real (not a feature buffet)
A strong+perceptive+fast agent is now IMPOSSIBLE by construction. Therefore:
- hard_nut needs Strength (slow/hungry), tall_fruit needs Reach (blind/low-metab),
- gaps need Size=small (weak), predator needs Σ Strength (so you need the striker),
- oasis gate needs Σ Strength (strikers, who are slow and can't see threats).
=> The group MUST combine a striker (fight/gate) + scout (see threats) + runner
  (reach/fetch) + connector (share/signal) to bank the high credits. None alone
  can. That compulsion is the emergent structure — and it comes from the trait
  tradeoffs, not from any reward cooperation term.

## 4.5 Model architecture (the agent's brain — must be NON-dumb)
A flat MLP over a flattened window can navigate but CANNOT do spatial
credit-assignment, hold temporal state, or model "the other agent has a blind
spot." Those capacities are exactly what emergence needs, so the net must earn
them. Design below. CPU constraint held (same stack as MC-RL): windows are small
(≤81 tokens) so a lightweight CNN + GRU is cheap even on CPU.

### 4.5.1 Perception encoder (spatial → relational)
Local window (2R+1)² cells; each cell = ONE TOKEN = position encoding + C grid
channels (tile one-hot, resource amount, agent presence, neighbor trait tags,
fallacy-warped hazard/predator). Feed the token set through SELF-ATTENTION
(2–3 layers, 2–4 heads) → relational features → flatten → GRU (§4.5.2).
- Self-attention is PRIMARY: it reasons relationally ("predator 3 cells N, my
  Strength low → defer"), which is what makes division-of-labor *representable*.
- CPU note: at a ≤81-token window this is trivial on CPU (O(n²)=6561/op/layer) —
  an earlier "attention CPU bottleneck" worry was UNFOUNDED at this scale. The
  earlier CNN-primary switch stands retracted; keep self-attention as the encoder.
- A LIGHTWEIGHT CNN (2 conv layers, 3×3) over the (C×(2R+1)×(2R+1)) slice is kept
  as an ALTERNATIVE/ablation encoder (local-conv baseline to prove attention isn't
  hiding needed capacity). Relational reasoning in core = attention + GRU +
  neighbor-embedding (§4.5.3).

### 4.5.2 Memory (recurrent — REQUIRED, not optional)
A GRU/LSTM cell wraps the fused features. Hidden state carried across steps
within an episode, RESET at episode boundary. This is non-negotiable: fallacies
(RecencyBias, FalseMemory), predator motion, and "agent was here, now gone"
demand temporal state. A memoryless agent is the "dumb agent" we're avoiding.
Standard PPO+LSTM recurrence; works on CPU.

### 4.5.3 Structured embeddings (self + others)
- Self/trait embedding: own trait vector + fallacy flags → MLP projection.
Conditions behavior on the agent's OWN nature (Strength/Reach/SocialBias/…).
- Neighbor embedding: visible neighbors' partial trait tags + last action → a
SET encoder (attention or mean-pool over the small visible set). This is the
SEED of opponent modeling — the agent can represent "the guy north is strong,
I should let him take the hard_nut." No explicit theory-of-mind module; it
emerges from the set encoding + shared training signal.

### 4.5.4 Fusion → heads
[spatial features] + [recurrent hidden] + [self-embedding] + [neighbor-embedding]
→ MLP trunk → two heads:
- Policy: categorical over 8 actions (0–7). Single head keeps emergence legible.
- Value: LOCAL critic (own return only) — proves coordination isn't a
value-sharing artifact. (Global critic = ablation only, see §10.)

### 4.5.5 Inductive biases we ALLOW vs BAKE IN
- ALLOWED (capacity, not a cooperation signal): spatial attention, recurrence,
self/neighbor embeddings. These let emergence be *representable*.
- BAKE-IN (forbidden in core): any weight-sharing that carries coordination
across agents (see D1), any global reward/bonus term (§8), any communication
channel that bypasses the world. Action 7 `signal` is WORLD-GROUNDED (a visible
marker tile), so it counts as capacity, not leakage.

## 5. Fallacies (systematic bounded-rationality — the engine of non-optimality)
Each agent gets 1–2 (randomly assigned). They perturb belief/decision, NOT the
true world — so agents are individually suboptimal and must cover each other's
blind spots. This is what forces emergence rather than optimal solo play.
- Overconfidence — believes hazard/predator is half its true distance
- RecencyBias    — expected resource location = last seen, ignores newer info
- Herding        — with prob p copies nearest agent's last action
- LossAversion   — freezes on a resource tile even under threat (won't flee)
- BlindSpot      — cannot perceive one compass direction (window column zeroed)
- FalseMemory    — maintains a phantom resource marker at a wrong cell

## 6. Action space (flat discrete, identical schema for every agent)
0 noop | 1 N | 2 E | 3 S | 4 W | 5 harvest (face tile) | 6 share (give 1 unit
to adjacent agent) | 7 signal (place a transient MARKER cell adjacent, see below).
Discrete + minimal keeps emergence legible (no continuous-control obfuscation).

### 6.1 `signal` = a cell that just exists (NO semantics)
- Action 7 writes a `marker` tile onto ONE adjacent cell (the 4 orthogonal
  neighbors + own cell — chosen by agent, but the cell carries NO meaning).
- The marker is a normal grid tile: any agent whose window covers it SEES "a
  marker cell exists here" in its observation (one extra channel), nothing more.
- Decays after 1 step (gone next step). Costs −0.05 energy (mechanical, §8.4).
- DELIBERATELY UNTYPED / UNSEMANTIC: there is no "danger" vs "rally" vs "come
  here" meaning. The agent either ignores it (it paid energy for nothing — no
  free lunch), or ascribes its OWN meaning to it empirically from correlation with
  outcomes. Any "communication" that arises is therefore ENTIRELY emergent, not
  designed. This is the strongest possible claim: we gave them a writable transient
  tile and a cost; we did not design a language.
- No other channel exists. share(6) and marker are the only world-grounded
  interactions; everything else is implicit via observation (neighbor trait tags +
  last action visible in the window, §4.5.3). No hidden/private/mailbox channel.

## 7. Observation (local, fixed schema — traits/fallacies vary BEHAVIOR not shape)
Flattened window (2R+1)²: tile one-hot, resource amount, adjacent-agent presence
+ their trait tags (partial), own energy/inv, own trait vector, last-action.
Fallacy effects applied *to this window* (e.g. BlindSpot zeroes a column;
Overconfidence shrinks hazard distance in the encoded tile).

## 8. Reward (LOCAL ONLY — this is the whole point)
Core rule: **every mechanic the observation exposes must be either CREDITED or
COSTED**, so there is no exploitably-unrestricted slack for the policy to find a
degenerate loophole in. The world is rich (traits, fallacies, locked tiles,
predators, gates, hazards); a too-sparse reward over a rich obs lets the agent
collapse to "camp one food tile" and ignore the whole emergent system. So reward
tracks the world's real structure. Crucially: all credits below are INDIVIDUAL
(local return). There is still NO group/cooperation bonus term — group strength
emerges from complementary-trait tiles/defense the solo agent simply cannot
access, not from a coordination reward.

### 8.1 Survival + metabolism
- +0.1 / step alive                 (survival pressure)
- −0.01 / step                      (discourage idling)
- −1.0 on death                     (terminal; no respawn in core)
- energy-decay already internal; no extra term (metabolism trait is the lever)

### 8.2 Harvest credits (tie reward to the trait-locked tiles)
- +EAT_GAIN on consuming food       (cheap; any agent)
- +EAT_GAIN·(1+HARD_BONUS) on consuming hard_nut   — CREDITS Strength-gated play
- +EAT_GAIN·(1+TALL_BONUS) on consuming tall_fruit  — CREDITS Reach-gated play
- consuming a tile you LACK the trait for = blocked at env level, so no credit
  (the lock is mechanical, not a reward choice — but the bonus ensures agents
  with the trait are *incentivized* to use it, not ignore it)

### 8.3 Cooperation-locked mechanics (STILL individual credit — no group bonus)
- On gate opening: each pushing agent that contributed Strength to Σ≥TH_GATE
  gets +GATE_GAIN. Credit is individual but the event REQUIRES complementarity.
- On predator defeat: each adjacent agent gets +PRED_GAIN ∝ its Strength share of
  the Σ that beat TH_PRED. A weak agent adjacent to a strong one still earns a
  small share → makes "stick near the strong one / cover his blind side" pay,
  WITHOUT a cooperation term. This is the emergence lever made incentive-consistent.

### 8.4 Anti-degeneracy / anti-exploit terms
- Hazard enter: −HAZ_PEN (so ignoring the fallacy-warped hazard costs)
- Flee pressure: if predator adjacent and Σ Strength < TH_PRED, −THREAT_PEN/step
  until out of range or defeated (credits NOT tanking into unwinnable fights)
- Hoarding cap: share(6) given to an agent in need yields +SHARE_GAIN to giver
  (incentivizes using SocialBias trait; credit is individual, event is social)
- Signal(7) costs a flat −0.05 energy MECHANICALLY (deducted in the env step, NOT
  a reward term). Communication is tied to survival metabolism, so agents only
  signal when expected coordination payoff (gate/oasis/predator) outweighs the
  metabolic cost. This HARD cost — not a tunable reward penalty — closes the
  degenerate signal/act loop: two agents can't farm variance by alternating
  signals, because each signal literally costs food they need to live.
- Oasis access: rich cluster behind gate; reaching+harvesting inside yields the
  same per-tile credits as 8.2 — the prize is just more high-bonus tiles behind
  a strength-sum gate, so it naturally concentrates group value there.

### 8.5 What stays OUT (by design — these are the emergence guarantees)
- NO global/team bonus, NO "cooperate" term, NO shaping toward a macro goal.
- All of 8.2–8.4 are local, per-agent, per-event. The ONLY way to bank the big
  credits (hard_nut, tall_fruit, oasis, predator-defeat share) is to hold/complement
  the right traits — so the agent is rewarded for BEING its trait and FINDING its
  complement, which is precisely the behavior we call emergent.

## 8.6 Observation↔reward completeness check (closes the exploit gap)
For each obs channel, a credited or costed event exists:
- tile types (food/hard_nut/tall_fruit/gap/gate/hazard/predator/oasis) → 8.2/8.3/8.4
- own trait vector → conditions which 8.2/8.3 credits are reachable (mechanical lock)
- neighbor trait tags + last action → 8.3 share/defense shares + set-embedding use
- energy/inv → 8.1/8.2 directly
- fallacy-warped channels → 8.4 hazard/threat penalties apply to the warped belief
  (so a fallacy that makes you walk into a hazard is its own cost — no free lunch)
No obs channel is "mostly unrestricted": every one maps to a real local payoff.

## 9. Learning
- DECISION LOCKED (D1=B): INDEPENDENT policies — one network per agent (6 nets,
  NO weight-sharing). Strongest emergence claim: no shared weights can carry
  coordination; it must arise from interaction in the shared env. Each net trains
  only on its OWN local return (LOCAL critic, §4.5.4). Opponent modeling emerges
  implicitly from simultaneous play; no theory-of-mind module.
- DECISION LOCKED (D2): traits FIXED-randomized per run (§4.2 sampler) for v1 —
  isolate behavioral emergence before any trait evolution.
- DECISION LOCKED (D4): fallacies added AFTER the clean baseline (build step 6).
- DECISION LOCKED (D5): NEVER a group/cooperate bonus (§8.5).

### 9.1 CPU-saturation rollout architecture (independent policies)
The env is ONE shared multi-agent sim (one grid, 6 agents). Rollout needs each
agent's own net to act. To keep all CPU cores busy under D1:
- SubprocVecEnv with K parallel env COPIES (each its own grid + 6 agents). Worker
  PROCESSES run the PURE-PYTHON grid step (no NN) and return obs/rewards — this
  RELEASES the GIL, so simulation scales across cores.
- MAIN process owns the 6 independent policy nets. For each rollout batch it
  gathers [K,6] obs, then per policy i runs a BATCHED forward pass over its K obs
  ([K, obs_dim]) → actions → shipped back to workers. Batching per-policy maximizes
  matmul utilization per net (BLAS multithreads the small nets).
- This DECOUPLES CPU-bound sim (workers) from NN inference (main) — producer/
  consumer, both stay saturated. K chosen so K × (threads/matmul) ≈ num_cores;
  set torch.set_num_threads to avoid oversubscription.
- GRU hidden states carried as [K, hidden] per policy; reset at episode boundary.
- Training: 6 independent PPO updates, one per policy buffer. NN-training ops
  release the GIL, so updates can interleave with the next rollout; if rollout is
  the bottleneck, train sequentially — either way cores stay busy.
- Reuse ThreadVecEnv/metrics plumbing from MC-RL (D:\mc-rl_ARCHIVE_20260814.tar.gz).

## 10. Metrics — how we PROVE emergence, not claim it
- Solo baseline: each agent alone in world → survival rate + energy (exp: survive, low).
- Group run: full N → survival + high-value harvest share (exp: strong).
- Emergent Cooperation Index = (group high-val harvest + group predator-defenses
  − Σ solo high-val − Σ solo predator-defenses) / group total.
  (predator defense is a strength-sum event, so credit goes to trait complementarity.)
- Division of Labor: entropy of trait→role assignment over time (low = specialization = emergence).
- Coordination onset: group-harvest vs training steps (look for phase transition).
- Ablation 1 homogeneous traits → does "together strong" vanish? (if yes → it's
  complementarity, not just "more bodies")
- Ablation 2 no fallacies → still emerge, or turn trivial/optimal?
- Ablation 3 local vs global critic → does value-sharing matter?
- Energy efficiency (ADD): cumulative energy spent per unit high-value credit,
  group vs solo. A group that survives longer but burns 5× the energy via
  inefficient pathing is brute-forcing the map, not cooperating. True emergence =
  MORE high-val credit PER ENERGY, not just more total. Report EnergyDelta =
  (group credit/energy) − mean(solo credit/energy). Refined Emergent Cooperation
  Index is efficiency-normalized: ECI = (group credit − Σ solo credit) / group
  energy — rewards efficient coordination, not map-brute-force.

## 11. Build order (each step verifiable before next)
1. Env skeleton: grid, tiles, move/harvest, energy/death, local obs (pure sim).
2. Single-agent sanity: random policy — survives? starves? tune decay/regen.
3. Traits in: prove hard_nut/tall_fruit/gap truly lock solo agents out.
4. Multi-agent + share/signal: confirm solo-survive vs group-strong gap exists.
5. PPO independent learners; train; log metrics (§10).
6. Fallacies layer; retrain; compare to no-fallacy baseline.
7. Ablations + render/visualization for the demo.

## 12. Decisions — LOCKED (2026-08-14 review)
- D1 = (B) INDEPENDENT policies (no shared weights).
- D2 = fixed-randomized traits per run (v1; evolve later, optionally).
- D3 = 128×128 grid, N=16 agents (OVERRIDE of original 64×64/N=6, per user
  2026-08-14 — "try 16 nets and a 128x128 grid"). Emergence at scale; needs more
  training steps to show structure. Solo baselines run per-archetype for ECI.
- D4 = fallacies added AFTER the clean baseline (build step 6).
- D5 = NEVER a group/cooperate bonus (§8.5).
Rationale: strongest defensible-emergence claim — no weight-sharing, no global
reward, traits locked so the observed cooperation can only come from local
interaction + trait complementarity.

## 13. Implementation status (as of 2026-08-14)
- Repo: D:\rl-emergence (venv with torch 2.11.0+cu128 CUDA=True, gymnasium 1.3,
  sb3 2.9, numpy 2.4, matplotlib). RTX 4060 (8GB) training GPU.
- src/env.py: EmergenceGrid sim implemented — grid+tiles, bounded-trait binding
  sampler (§4.2 verified: 0/500 super-agents), deterministic Speed-priority step
  resolution (§2.1), local obs window (fixed R=WINDOW=4, dim=1144, batches cleanly),
  trait-locked harvest, strength-sum gate/predator, unsemantic signal (mechanical
  −0.05 energy), reward completeness (§8), fixed obs shape.
  Verified at 128×128 / N=16: 2.3 ms/step (CPU sim), scales fine.
- src/model.py: AgentPolicy = spatial self-attention encoder (81 tokens) + GRU
  memory + actor/critic heads. ~180k params/net. Runs on CUDA.
- src/ppo.py: minimal PPO (clipped surrogate + GAE + value + entropy), per-net
  RolloutBuffer with GRU hidden gather for BPTT. Validated: solo agent learns
  (pol_loss→neg, vf_loss falls, ep_reward rises) on small grid.
- src/smoke_solo.py: solo PPO smoke — validates full pipeline before 16-net scale.
- tests/test_sanity.py: ALL PASS (energy law, trait locks, determinism, obs shape,
  binding sampler, signal cost).
- D3 OVERRIDDEN: 128×128 grid, N=16 agents (was 64×64/N=6).
- NOT YET: 16 independent PPO nets + batched GPU rollout (§9.1, SubprocVecEnv),
  predators in default run (optional), fallacies layer (D4), ablations, render,
  solo-baseline ECI (§10).
- Next: build the 16-net harness (one AgentPolicy+PPOTrainer per agent, batched
  GPU inference over K grid copies, SubprocVecEnv for CPU sim throughput),
  train solo baseline then group, log efficiency-normalized ECI.
