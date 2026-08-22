// EmergenceGrid C++ core (pybind11 module: cpp_sim).
// Drop-in replacement for the Python env. Same mechanics, but:
//   * food distance is NOT a grid-wide BFS/DT -- we keep a food list + coarse
//     bucket grid and answer nearest-food queries in O(local) per agent.
//   * occupancy grid (occ[W*H]) replaces the O(N^2) agent-collision scan.
//   * _agent_at / share / gate / predator use neighbor/bucket lookups.
// Obs is (N, 50621) float32 for a 64x64 grid (global 49152 + own 14 + 11x11
// patch 1452 + food-vector 3) -- identical layout to the Python version, so the
// PyTorch model + PPO in train.py are untouched.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <cmath>
#include <cstdint>

// MSVC has no ssize_t (used by pybind11 array API); define it for Windows builds.
#if defined(_MSC_VER) && !defined(ssize_t)
typedef long long ssize_t;
#endif

namespace py = pybind11;

// ---- tile types ----
enum {
    EMPTY=0, FOOD=1, HARD_NUT=2, TALL_FRUIT=3, GAP=4, WALL=5,
    GATE=6, HAZARD=7, PREDATOR=8, OASIS=9, MARKER=10
};

// ---- constants ----
static const int TH_STR=60, TH_REACH=60, TH_GATE=95, TH_PRED=130; // x100 (gate opens at strength>=0.95; below the 1.0 cap so float rounding reaches it)
static const int E_MAX=1000;       // energy in int x1000 internally? keep float
static const float E_MAX_F=50.0f;   // energy cap. MUST be > starting energy so that
                            // harvesting can actually REFUEL the agent; previously this
                            // was 10.0 and agents SPAWNED at the cap (a.energy=E_MAX_F),
                            // the agent could only decay and always starved. That made any
                            // long-horizon task (gate-opening needs ~105 steps) unwinnable.
static const float EAT_GAIN=15.0f, OASIS_BONUS=2.5f, HARD_BONUS=1.5f, TALL_BONUS=1.2f;
static const float GATE_GAIN=0.8f, PRED_GAIN=1.0f, HAZ_PEN=0.5f;
static const float SHARE_GAIN=0.4f, SIGNAL_ENERGY=0.05f;
static const float STEP_PEN=0.0f, DEATH_PEN=2.0f, STAGNATION_PEN=0.05f;
static const float STARVE_DECAY=0.0f, STARVE_PEN=0.0f;  // passive hunger experiment REVERTED (made gated eating worse). Kept as 0 = no passive hunger.
static const float PROX_FOOD_BONUS=0.30f, FOOD_PULL=1.0f, NAV_ALPHA=0.10f, GATE_PROX=0.02f;
static const float TRAIT_MUT=0.12f, TRAIT_MUT_PEN=1.0f;
static const float TRAIT_MATCH_BONUS=0.0f;  // shaping experiment REVERTED (made gated eating worse); 0 = disabled
static const float INVALID_HARVEST_PEN=0.5f, IDLE_FOOD_PEN=0.10f;
static const int TRAIT_COOLDOWN=15;
static const float ACT_COST_SHARE=0.05f, ACT_COST_SIGNAL=0.05f, ACT_COST_MUT=0.05f;
static const float BLOCKED_PEN=0.0f;  // attempted move that hit a wall/agent. Disabled: penalizing moves makes the agent "shy" (hangs back from food/walls). Collection goal takes priority.
static const int FOOD_REGEN=40, OASIS_REGEN=25;
static const int CHAN=12;
static const int OWN_DIM=14 + 121*12 + 3;  // 14 traits + 11x11 patch + (food_dir_x, food_dir_y, food_dist)

struct Traits {
    float strength, reach, speed;
    int perception;
    float metabolism;
    int size_small; // 1 if small
    float social;
    bool can_hard() const { return strength >= 0.6f; }
    bool can_tall() const { return reach >= 0.6f; }
    bool can_small() const { return size_small==1; }
};

struct Agent {
    int idx, x, y;
    Traits tr;
    float energy;
    int inv;
    bool alive;
    int last_action;
    int cooldown;
    int last_gated_dist = 1e9;   // for instrumentation: gated-food dist last step
};

struct Sim {
    int W, H, n_agents;
    int curriculum;
    bool respawn;
    uint32_t seed;
    int food_seed=0;
    int food_seed_dist=1;        // ring distance at which seeded food is placed
    int food_density_div=50;     // base food = W*H/food_density_div (higher = sparser)
    bool food_regen=true;        // harvested food regenerates after FOOD_REGEN steps
    int food_regen_mode=2;       // 0=no regen, 1=in-place (pocket-feeds), 2=random empty cell (default: no pocket-feeding)
    int gated_food=1;             // 0=no gated food, 1=regular+trickle of gated (default), 2=gated-dominant (agent must mutate to eat)
    std::mt19937 rng;
    std::vector<int> grid;       // H*W
    std::vector<Agent> agents;
    std::vector<std::array<int,2>> predators;
    std::vector<std::array<int,2>> oasis_cells, gate_cells;
    // regen: map index -> ticks
    std::vector<int> regen_t;     // per cell, 0 = none
    std::vector<int> regen_type;  // per cell, original tile (FOOD/OASIS/MARKER)
    // food structures (ds optimization): list of food positions + coarse bucket
    std::vector<std::array<int,2>> foods;
    int B=8;                      // bucket grid resolution
    std::vector<std::vector<int>> bucket; // B*B -> list of food indices
    std::vector<int> occ;         // occupancy grid (agent index+1, 0 empty)
    int step_count=0;
    // ---- dynamic reward parameters (settable at runtime; annealed by training progress) ----
    struct RewardParams {
        float food_pull = 1.0f;       // potential pull toward food
        float nav_alpha = 0.15f;      // PBRS navigation coefficient
        float eat_gain = 15.0f;       // reward for eating GATED food (HARD_NUT/TALL_FRUIT)
        float eat_gain_regular = 15.0f; // reward for eating REGULAR food (FOOD/OASIS).
                                         // Phase-2 env sets this to 0 so harvest-spam gives
                                         // energy (survival) but NO reward -> only gated/gate
                                         // food pays, forcing the agent off the spam optimum.
        float invalid_harvest_pen = 0.5f; // penalty for harvesting with no food adjacent
        float trait_mut_pen = 1.0f;   // penalty per mutate (when NOT adjacent to gated)
        float trait_mut_pen_gated = 0.0f; // mutate penalty when ADJACENT to gated (A: free)
        float gate_gain = 0.8f;       // reward for opening a gate
        float trait_match_bonus = 0.0f; // shaping bonus for adjacent eatable gated food
        float mutate_gated_gain = 1.5f;  // C: +reward for mutating the RIGHT trait when
                                         //    adjacent to gated food it then unlocks
        float wrong_trait_pen = 0.3f;    // C: -reward for mutating WRONG trait when
                                         //    adjacent to gated food (doesn't unlock it)
        float gate_prox_bonus = 0.0f;    // G: dense shaping for being STRONG (>=gate
                                         //    threshold) AND adjacent to a GATE, so the
                                         //    final "be strong at the gate" push is
                                         //    reinforced every step instead of only the
                                         //    one-shot gate_gain when it opens.
    };
    RewardParams rp;
    float step_frac = 0.0f;  // training progress in [0,1], set by Python each update

    // ---- closed-loop adaptation diagnostics ----
    // Accumulated over the episode (reset each reset()). Python reads these after
    // a rollout and adjusts rp.* to REACT to the agent's actual failure mode
    // (not on a blind time schedule). The 9 base fields feed the adaptive
    // controller via get_diag(); get_diag_full() returns the COMPLETE instrument
    // set for offline analysis (pipeline funnel, trait dynamics, distances, gate
    // progress, reward probe) so we stop hypothesizing blind.
    struct Diag {
        long steps = 0;          // agent-steps observed
        long harvest_invalid = 0;// act==5 while NOT adjacent to harvestable food
        long harvest_valid = 0;  // act==5 while adjacent (real eats + adjacent presses)
        long move_away = 0;      // a move action that INCREASED distance to nearest food
        long move_closer = 0;    // a move action that DECREASED distance
        long mutate_steps = 0;   // act in 8..12
        long gate_adj = 0;       // steps adjacent to a GATE
        long gate_adj_strong = 0;// steps adjacent to a GATE with enough strength to open
        long dead = 0;           // agent died this episode
        // ---- full instrumentation ----
        // 3-step pipeline funnel (gated food = HARD_NUT/TALL_FRUIT)
        long reached_gated = 0;      // steps adjacent to any gated tile
        long mutated_near_gated = 0; // mutate act while adjacent to gated
        long gained_right_trait = 0; // gained a trait that unlocks adjacent gated
        long harvested_gated = 0;    // ate gated food (inv++ on gated)
        long wrong_trait_mut = 0;    // mutated while adjacent to gated but it didn't unlock
        // trait dynamics
        long trait_gain_events = 0;  // a mutation that granted can_hard/can_tall
        long trait_loss_events = 0;  // a mutation that removed can_hard/can_tall
        double sum_strength = 0.0;   // running sum for mean strength (per agent-step)
        double sum_reach = 0.0;      // running sum for mean reach
        long trait_samples = 0;      // number of (agent,step) samples for the means
        // ground-truth distances (not the obs proxy)
        double dist_food_sum = 0.0;  // sum of nearest_food_dist per agent-step
        long dist_food_samples = 0;
        double dist_gated_sum = 0.0; // sum of nearest gated-food dist per agent-step
        long dist_gated_samples = 0;
        long moved_closer_gated = 0; // move action that decreased dist to nearest gated
        long moved_away_gated = 0;    // move action that increased dist to nearest gated
        // gate progress (L3)
        float max_strength = 0.0f;   // max strength any alive agent achieved
        long gate_chain_possible = 0;// 1 if max_strength >= TH_GATE at some point
        // reward probe: total reward yielded per action type (13 actions)
        double rew_by_action[13] = {0.0};
    };
    Diag diag;
    void reset_diag() { diag = Diag(); }

    Sim(int width, int height, int n, uint32_t sd, int curric, bool resp,
        int food_seed=0, int food_seed_dist=1, int food_density_div=50)
        : W(width), H(height), n_agents(n), curriculum(curric), respawn(resp),
          seed(sd), rng(sd), food_seed(food_seed), food_seed_dist(food_seed_dist),
          food_density_div(food_density_div) {
        grid.resize(W*H, EMPTY);
        regen_t.assign(W*H, 0);
        regen_type.assign(W*H, 0);
        occ.assign(W*H, 0);
        build_grid();
        spawn_agents();
        bucket.assign(B*B, {});
        rebuild_food_index();
        // Curriculum annealing of PBRS navigation coefficient (Ng et al. shaping):
        // dense nav reward is now a GENTLE nudge (alpha<=0.03): harvest (+5) dominates.
        // anneal to 0 so sparse gate/predator/hazard rewards take over.
        // L0 food-only -> 0.03 ; L1/L2 walls+gated food -> 0.02 ; L3+ -> 0.0
        // Anneal PBRS by curriculum, but KEEP a navigation signal at all levels
        // (curriculum 5 was wrongly zeroed -> no nav signal -> agent camps/spams).
        // Disable the PBRS bonus only when already adjacent to food (prev_dist<=1)
        // so camping on food isn't rewarded; the move-toward signal stays alive.
        // Curriculum baseline for PBRS navigation coefficient (Ng et al. shaping).
        // This is the DEFAULT; Python may override rp.* each update to anneal rewards.
        if      (curric >= 3) rp.nav_alpha = 0.10f;
        else if (curric >= 1) rp.nav_alpha = 0.15f;
        else                  rp.nav_alpha = 0.25f;
    }

    // Set reward parameters from Python (called each training update to anneal).
    void set_reward_params(const RewardParams &p) { rp = p; }
    // Set training progress fraction [0,1] (for any progress-dependent shaping).
    void set_step_frac(float f) { step_frac = f; }

    inline int idx(int x,int y) const { return y*W+x; }
    inline int bidx(int x,int y) const { return (y*B/W)*B + (x*B/W); }

    // Generate contiguous wall barriers (long runs) so navigation must route
    // AROUND them, instead of scattered singles that barely block. Each run is a
    // straight H or V segment with one random gap, so it never fully seals the
    // map (always routeable). count = number of barriers.
    void add_wall_runs(int count) {
        auto rint=[&](int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng); };
        for (int k=0;k<count;k++){
            bool horiz = rint(0,1)==0;
            int len = rint(8, std::max(8, (W+H)/5));
            int gap = rint(1, len-1);   // one gap so the barrier is never sealing
            if (horiz){
                int y=rint(4,H-5), x0=rint(2,W-3-len);
                for (int i=0;i<len;i++){ int x=x0+i; if (i==gap) continue;
                    if (0<x&&x<W-1&&0<y&&y<H-1) grid[idx(x,y)]=WALL; }
            } else {
                int x=rint(4,W-5), y0=rint(2,H-3-len);
                for (int i=0;i<len;i++){ int y=y0+i; if (i==gap) continue;
                    if (0<x&&x<W-1&&0<y&&y<H-1) grid[idx(x,y)]=WALL; }
            }
        }
    }

    void build_grid() {
        for (int x=0;x<W;x++){ grid[idx(x,0)]=WALL; grid[idx(x,H-1)]=WALL; }
        for (int y=0;y<H;y++){ grid[idx(0,y)]=WALL; grid[idx(W-1,y)]=WALL; }
        int L=curriculum;
        auto rint=[&](int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng); };
        auto rflt=[&](float a,float b){ std::uniform_real_distribution<float> d(a,b); return d(rng); };
        if (L>=2 && gated_food>=2){
            // gated-dominant: NO regular food. Agent must mutate traits
            // (can_hard/can_tall) to eat anything. Forces trait emergence.
            // Gated density follows food_density_div so SPARSE worlds (high
            // divisor) actually force navigation between scattered gated tiles
            // instead of camping on a dense pile.
            for (int i=0;i<W*H/food_density_div;i++){ int x=rint(2,W-3),y=rint(2,H-3); grid[idx(x,y)]=(rflt(0,1)<0.5f)?HARD_NUT:TALL_FRUIT; }
            for (int i=0;i<W*H/300;i++){ int x=rint(2,W-3),y=rint(2,H-3); grid[idx(x,y)]=GAP; }
        } else {
            for (int i=0;i<W*H/food_density_div;i++){ int x=rint(2,W-3),y=rint(2,H-3); if (grid[idx(x,y)]==EMPTY) grid[idx(x,y)]=FOOD; }
            if (L>=1) add_wall_runs((L>=2)?10:7);
            if (L>=2 && gated_food>=1){
                // regular + trickle of gated (default L2): regular dense, gated sparse.
                int reg_div = food_density_div;
                int gated_div = 90;
                for (int i=0;i<W*H/reg_div;i++){ int x=rint(2,W-3),y=rint(2,H-3); if (grid[idx(x,y)]==EMPTY) grid[idx(x,y)]=FOOD; }
                for (int i=0;i<W*H/gated_div;i++){ int x=rint(2,W-3),y=rint(2,H-3); grid[idx(x,y)]=(rflt(0,1)<0.5f)?HARD_NUT:TALL_FRUIT; }
                for (int i=0;i<W*H/300;i++){ int x=rint(2,W-3),y=rint(2,H-3); grid[idx(x,y)]=GAP; }
            }
        }
        if (L>=4) for (int i=0;i<W*H/400;i++){ int x=rint(2,W-3),y=rint(2,H-3); grid[idx(x,y)]=HAZARD; }
        if (L>=3) place_oasis(); else { oasis_cells.clear(); gate_cells.clear(); }
    }

    void place_oasis() {
        oasis_cells.clear(); gate_cells.clear();
        std::array<std::array<int,2>,4> cells={{{{W/4,H/4}},{{3*W/4,H/4}},{{W/4,3*H/4}},{{3*W/4,3*H/4}}}};
        for (auto &c : cells) {
            int gx=c[0], gy=c[1];
            for (int dx=-2;dx<=2;dx++) for (int dy=-2;dy<=2;dy++){
                int nx=gx+dx, ny=gy+dy;
                if (!(0<nx && nx<W-1 && 0<ny && ny<H-1)) continue;
                if (abs(dx)<=1 && abs(dy)<=1){ grid[idx(nx,ny)]=OASIS; oasis_cells.push_back({nx,ny}); }
                else grid[idx(nx,ny)]=WALL;
            }
            for (int col : {gx,gx+1}) for (int row : {gy+2,gy+3}){
                if (0<col && col<W-1 && 0<row && row<H-1){ grid[idx(col,row)]=GATE; gate_cells.push_back({col,row}); }
            }
        }
    }

    Traits sample_traits(const std::string &arch) {
        auto rflt=[&](float a,float b){ std::uniform_real_distribution<float> d(a,b); return d(rng); };
        auto rint=[&](int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng); };
        float strength,reach,speed; int perception; float social; int small;
        if (arch=="striker"){ strength=rflt(0.7f,1.0f); reach=rflt(0.1f,0.45f); speed=rflt(0.1f,0.4f); perception=rint(1,2); social=rflt(0,0.3f); small=0; }
        else if (arch=="scout"){ strength=rflt(0.1f,0.4f); reach=rflt(0.1f,0.5f); speed=rflt(0.4f,0.8f); perception=rint(3,4); social=rflt(0,0.3f); small=1; }
        else if (arch=="runner"){ strength=rflt(0.1f,0.4f); reach=rflt(0.5f,0.9f); speed=rflt(0.7f,1.0f); perception=rint(2,3); social=rflt(0,0.3f); small=1; }
        else if (arch=="connector"){ strength=rflt(0.1f,0.4f); reach=rflt(0.1f,0.4f); speed=rflt(0.3f,0.7f); perception=rint(2,3); social=rflt(0.6f,1.0f); small=1; }
        else { strength=rflt(0.35f,0.5f); reach=rflt(0.35f,0.5f); speed=rflt(0.35f,0.5f); perception=rint(2,3); social=rflt(0.2f,0.5f); small=rint(0,1); }
        if (strength+speed>1.3f){ float s=1.3f/(strength+speed); strength*=s; speed*=s; }
        float eff_per=perception/4.0f; float cap=1.3f-eff_per;
        if (strength>cap) strength=std::max(0.1f,cap);
        if (social>0.6f){ float mean=(strength+reach+speed)/3.0f; if (mean>0.45f){ float s=0.45f/mean; strength*=s; reach*=s; speed*=s; } }
        float metabolism=0.5f+0.5f*(strength+speed+perception/4.0f)/2.0f;
        metabolism=std::min(1.5f,std::max(0.5f,metabolism));
        Traits t; t.strength=std::round(strength*1000)/1000.0f; t.reach=std::round(reach*1000)/1000.0f;
        t.speed=std::round(speed*1000)/1000.0f; t.perception=perception; t.metabolism=std::round(metabolism*1000)/1000.0f;
        t.size_small=small; t.social=std::round(social*1000)/1000.0f; return t;
    }

    void spawn_agents() {
        std::vector<std::string> archs={"striker","scout","runner","connector","generalist","generalist"};
        std::shuffle(archs.begin(),archs.end(),rng);
        agents.clear();
        for (int i=0;i<n_agents;i++){
            Agent a; a.idx=i; a.tr=sample_traits(archs[i%archs.size()]);
            int tries=0;
            while(tries++<10000){ int x=rng()%(W-4)+2, y=rng()%(W-4)+2; if (grid[idx(x,y)]==EMPTY){ a.x=x; a.y=y; break; } }
            a.energy=0.6f*E_MAX_F; a.inv=0; a.alive=true; a.last_action=0; a.cooldown=0;
            agents.push_back(a);
        }
        for (auto &a:agents) occ[idx(a.x,a.y)]=a.idx+1;
        // food_seed>0: drop a food tile a few tiles from each spawn so the policy
        // must learn to PATH to it (curriculum). food_seed_dist = Manhattan ring.
        if (food_seed>0){
            for (auto &a:agents) seed_food_ring(a, food_seed_dist);
            rebuild_food_index();
        }
    }

    void rebuild_food_index() {
        foods.clear(); bucket.assign(B*B,{});
        for (int y=0;y<H;y++) for (int x=0;x<W;x++){
            int t=grid[idx(x,y)];
            if (t==FOOD||t==OASIS||t==HARD_NUT||t==TALL_FRUIT){
                foods.push_back({x,y});
                bucket[bidx(x,y)].push_back((int)foods.size()-1);
            }
        }
    }

    // nearest food Manhattan dist via expanding ring over bucket grid (no full DT)
    int nearest_food_dist(int x, int y) const {
        int best=1e9;
        int cx=x*B/W, cy=y*B/H;
        // search rings of buckets
        for (int r=0;r<=B;r++){
            bool found=false;
            for (int dx=-r;dx<=r;dx++) for (int dy=-r;dy<=r;dy++){
                if (abs(dx)!=r && abs(dy)!=r) continue; // ring only
                int bx=cx+dx, by=cy+dy;
                if (bx<0||bx>=B||by<0||by>=B) continue;
                for (int fi:bucket[by*B+bx]){
                    int fx=foods[fi][0], fy=foods[fi][1];
                    // guard: harvest() re-pushes the EMPTYed cell into the food
                    // index (and regen adds a NEW cell without removing the old),
                    // so the index can contain phantom EMPTY tiles. Skip them so
                    // nav never targets a dead cell.
                    int ft=grid[idx(fx,fy)];
                    if (ft!=FOOD && ft!=OASIS) continue;
                    int d=abs(fx-x)+abs(fy-y);
                    if (d<best){ best=d; if (d<=r) {found=true;} }
                }
            }
            if (found && best<=r) break; // closest food in this ring is exact
        }
        return best;
    }

    // ground-truth nearest GATED-food (HARD_NUT/TALL_FRUIT) Manhattan distance.
    // Full-grid scan (gated tiles are sparse); used only for instrumentation.
    int nearest_gated_dist(int x, int y) const {
        int best = 1e9;
        for (int yy=0; yy<H; yy++) for (int xx=0; xx<W; xx++) {
            int t = grid[idx(xx,yy)];
            if (t==HARD_NUT || t==TALL_FRUIT) {
                int d = abs(xx-x) + abs(yy-y);
                if (d < best) best = d;
            }
        }
        return best;
    }

    // Navigation TARGET distance: the food/gate the nav potential should pull
    // toward. In the normal env all food is reward-bearing -> use nearest_food_dist
    // (which includes gated). In Phase-2 (eat_gain_regular==0) regular food pays
    // NOTHING, so pulling toward it is a dead compass -> target only the
    // reward-bearing tiles (gated food + gates). This is what makes the Phase-2
    // curriculum actually "encourage the gate system" instead of parking on the
    // (now unrewarding) dense regular food.
    int nearest_goal_dist(int x, int y) const {
        if (rp.eat_gain_regular > 0.0f) return nearest_food_dist(x, y);
        int best = 1e9;
        for (int yy=0; yy<H; yy++) for (int xx=0; xx<W; xx++) {
            int t = grid[idx(xx,yy)];
            if (t==HARD_NUT || t==TALL_FRUIT || t==GATE) {
                int d = abs(xx-x) + abs(yy-y);
                if (d < best) best = d;
            }
        }
        return best;
    }

    bool tile_passable(const Agent &a, int t) const {
        if (t==WALL) return false;
        if (t==GAP && !a.tr.can_small()) return false;
        return true;
    }

    bool adjacent_harvestable(const Agent &a) const {
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
            int t=grid[idx(nx,ny)];
            if (t==FOOD||t==OASIS) return true;
            if (t==HARD_NUT && a.tr.can_hard()) return true;
            if (t==TALL_FRUIT && a.tr.can_tall()) return true;
        }
        return false;
    }
    bool adjacent_gate(const Agent &a) const {
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            if (!dx&&!dy) continue;
            int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
            if (grid[idx(nx,ny)]==GATE) return true;
        }
        return false;
    }
    int agent_at(int x,int y) const { int v=occ[idx(x,y)]; return v>0 ? v-1 : -1; }

    float harvest(Agent &a) {
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
            int t=grid[idx(nx,ny)];
            if (t==FOOD||t==OASIS){
                grid[idx(nx,ny)]=EMPTY; foods.push_back({nx,ny}); bucket[bidx(nx,ny)].push_back((int)foods.size()-1);
                if (food_regen) { regen_t[idx(nx,ny)]=(t==FOOD)?FOOD_REGEN:OASIS_REGEN; regen_type[idx(nx,ny)]=t; }
                // energy: survival always granted (so the agent never starves even
                // when regular-food reward is 0 in the Phase-2 env). reward: gated by
                // eat_gain_regular (default = eat_gain; Phase-2 sets it to 0).
                float egain=rp.eat_gain*(t==OASIS?OASIS_BONUS:1.0f);
                a.energy=std::min(E_MAX_F,a.energy+egain); a.inv++;
                return rp.eat_gain_regular*(t==OASIS?OASIS_BONUS:1.0f);
            }
            if (t==HARD_NUT && a.tr.can_hard()){
                grid[idx(nx,ny)]=EMPTY; foods.push_back({nx,ny}); bucket[bidx(nx,ny)].push_back((int)foods.size()-1);
                if (food_regen) { regen_t[idx(nx,ny)]=FOOD_REGEN; regen_type[idx(nx,ny)]=FOOD; }
                float gain=rp.eat_gain*(1.0f+HARD_BONUS); a.energy=std::min(E_MAX_F,a.energy+gain); a.inv++; return gain;
            }
            if (t==TALL_FRUIT && a.tr.can_tall()){
                grid[idx(nx,ny)]=EMPTY; foods.push_back({nx,ny}); bucket[bidx(nx,ny)].push_back((int)foods.size()-1);
                if (food_regen) { regen_t[idx(nx,ny)]=FOOD_REGEN; regen_type[idx(nx,ny)]=FOOD; }
                float gain=rp.eat_gain*(1.0f+TALL_BONUS); a.energy=std::min(E_MAX_F,a.energy+gain); a.inv++; return gain;
            }
        }
        return 0.0f;
    }

    float share(Agent &a, std::vector<float> &rew) {
        int best=-1; float beste=1e9;
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            if (!dx&&!dy) continue;
            int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
            int oi=agent_at(nx,ny); if (oi<0) continue;
            Agent &o=agents[oi]; if (!o.alive) continue;
            if (o.energy < a.energy-1.0f && o.energy<beste){ beste=o.energy; best=oi; }
        }
        if (best>=0){
            Agent &o=agents[best];
            float give=std::min(0.5f, a.energy-1.0f);
            if (give>0.05f){ a.energy=std::max(0.0f,a.energy-give); o.energy=std::min(E_MAX_F,o.energy+give); a.inv++; rew[o.idx]+=SHARE_GAIN; return SHARE_GAIN; }
        }
        return 0.0f;
    }

    void signal(Agent &a, std::vector<float> &rew) {
        std::vector<std::array<int,2>> cells={{a.x,a.y}};
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            if (!dx&&!dy) continue; int nx=a.x+dx, ny=a.y+dy;
            if (nx>=0&&ny>=0&&nx<W&&ny<H && (grid[idx(nx,ny)]==EMPTY||grid[idx(nx,ny)]==FOOD)) cells.push_back({nx,ny});
        }
        if (!cells.empty()){
            std::uniform_int_distribution<int> d(0,(int)cells.size()-1);
            auto &c=cells[d(rng)]; grid[idx(c[0],c[1])]=MARKER; regen_t[idx(c[0],c[1])]=1; regen_type[idx(c[0],c[1])]=MARKER;
        }
        a.energy=std::max(0.0f,a.energy-SIGNAL_ENERGY);
    }

    void mutate(Agent &a, int act, std::vector<float> &rew, bool near_gated=false) {
        if (a.cooldown>0) return;
        Traits &t=a.tr; bool bhard=t.can_hard(), btall=t.can_tall();
        if (act==8) t.strength=std::min(1.0f,t.strength+TRAIT_MUT);
        else if (act==9) t.strength=std::max(0.05f,t.strength-TRAIT_MUT);
        else if (act==10) t.reach=std::min(1.0f,t.reach+TRAIT_MUT);
        else if (act==11) t.reach=std::max(0.05f,t.reach-TRAIT_MUT);
        else if (act==12) t.speed=std::min(1.0f,t.speed+TRAIT_MUT);
        if (t.strength+t.speed>1.3f){ float s=1.3f/(t.strength+t.speed); t.strength*=s; t.speed*=s; }
        // A: mutation is FREE when adjacent to gated food (so entering the mutate->eat
        //    loop is not a net loss vs harvesting regular food). Otherwise normal pen.
        float pen = near_gated ? rp.trait_mut_pen_gated : rp.trait_mut_pen;
        a.energy=std::max(0.0f,a.energy-pen); rew[a.idx]-=pen;
        // instrumentation: trait-gain / trait-loss events
        if (!bhard && t.can_hard()) { rew[a.idx]+=1.0f; diag.trait_gain_events++; }   // gaining the trait is immediately rewarding
        if (!btall && t.can_tall()) { rew[a.idx]+=1.0f; diag.trait_gain_events++; }   // (mutate penalty is 1.0, so net ~0 unless it unlocks food)
        if (bhard && !t.can_hard()) diag.trait_loss_events++;
        if (btall && !t.can_tall()) diag.trait_loss_events++;
        a.cooldown=TRAIT_COOLDOWN;
    }

    void resolve_hazards(std::vector<float> &rew, std::vector<bool> &done) {
        for (auto &a:agents){ if (!a.alive) continue;
            if (grid[idx(a.x,a.y)]==HAZARD){ a.energy=std::max(0.0f,a.energy-HAZ_PEN);
                if (a.energy<=0){ a.alive=false; done[a.idx]=true; rew[a.idx]-=DEATH_PEN; occ[idx(a.x,a.y)]=0; } } }
    }
    void resolve_predators(std::vector<float> &rew, std::vector<bool> &done) {
        for (size_t p=0;p<predators.size();p++){
            int px=predators[p][0], py=predators[p][1];
            float s=0; std::vector<int> adj;
            for (auto &a:agents){ if (a.alive && abs(a.x-px)+abs(a.y-py)==1){ s+=a.tr.strength; adj.push_back(a.idx); } }
            if (s>=TH_PRED/100.0f){
                for (int ai:adj){ float share=agents[ai].tr.strength/std::max(1e-6f,s);
                    float gain=PRED_GAIN*(0.5f+share); agents[ai].energy=std::min(E_MAX_F,agents[ai].energy+gain); rew[ai]+=gain; }
                // relocate predator
                int tries=0; while(tries++<1000){ int x=rng()%(W-4)+2,y=rng()%(H-4)+2; if (grid[idx(x,y)]==EMPTY){ grid[idx(px,py)]=EMPTY; predators[p]={x,y}; break; } }
            }
        }
    }
    void resolve_gates(std::vector<float> &rew) {
        if (gate_cells.empty()) return;
        float s=0; std::vector<int> pushers;
        for (auto &a:agents){ if (!a.alive) continue;
            for (auto &g:gate_cells){ if (abs(a.x-g[0])+abs(a.y-g[1])==1){ s+=a.tr.strength; pushers.push_back(a.idx); break; } } }
        if (s>=TH_GATE/100.0f){
            for (auto &g:gate_cells){ if (grid[idx(g[0],g[1])]==GATE) grid[idx(g[0],g[1])]=EMPTY; }
            for (int ai:pushers){ agents[ai].energy=std::min(E_MAX_F,agents[ai].energy+rp.gate_gain); rew[ai]+=rp.gate_gain; }
        }
    }
    void regen_tiles() {
        auto rint=[&](int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng); };
        for (int i=0;i<W*H;i++){
            if (regen_t[i]>0){
                regen_t[i]--;
                if (regen_t[i]<=0){
                    int t=regen_type[i]; int x=i%W, y=i/W;
                    if (t==MARKER) grid[i]=EMPTY;
                    else if (std::find(oasis_cells.begin(),oasis_cells.end(),std::array<int,2>{x,y})!=oasis_cells.end()){ grid[i]=OASIS; foods.push_back({x,y}); bucket[bidx(x,y)].push_back((int)foods.size()-1); }
                    else {
                        // FOOD regen: mode 1 = in-place (original location, can pocket-feed
                        // the agent that ate it); mode 2 = random empty cell (food stays
                        // available but never regrows on top of the agent). mode 0 = no regen.
                        int rx=x, ry=y;
                        if (food_regen_mode==2){
                            for (int tries=0;tries<2000;tries++){ int tx=rint(1,W-2), ty=rint(1,H-2);
                                if (grid[idx(tx,ty)]==EMPTY && occ[idx(tx,ty)]==0){ rx=tx; ry=ty; break; } }
                            // The new food lives at (rx,ry), and regen of THAT cell is
                            // armed below. The ORIGINAL harvested source cell (i) must be
                            // disarmed, otherwise regen_t[i] stays <=0 and re-fires every
                            // tick -> the map floods with food forever. (mode 2 = "food
                            // stays available but moves", not "infinite spawn".)
                            regen_t[i]=0; regen_type[i]=MARKER;
                        }
                        grid[idx(rx,ry)]=FOOD; foods.push_back({rx,ry}); bucket[bidx(rx,ry)].push_back((int)foods.size()-1);
                        if (food_regen_mode==1) { regen_t[idx(rx,ry)]=FOOD_REGEN; regen_type[idx(rx,ry)]=FOOD; }
                    }
                }
            }
        }
    }
    void respawn_dead() {
        for (auto &a:agents){ if (a.alive) continue;
            for (int t=0;t<200;t++){ int x=rng()%(W-4)+2, y=rng()%(H-4)+2;
                if (grid[idx(x,y)]==EMPTY && occ[idx(x,y)]==0){ a.x=x; a.y=y; a.energy=0.3f*E_MAX_F; a.inv=0; a.alive=true; a.last_action=0; occ[idx(x,y)]=a.idx+1; break; } }
            // food_seed: keep a food tile near the (re)spawned agent so the
            // harvest->reward correlation persists across respawns (curriculum).
            if (food_seed>0) seed_food_ring(a, food_seed_dist);
        }
    }

    // drop exactly one FOOD tile at a random clear cell on the Manhattan ring of
    // radius d around agent a (curriculum: agent must walk d steps to reach it).
    void seed_food_ring(Agent &a, int d) {
        std::vector<std::array<int,2>> cand;
        for (int dx=-d;dx<=d;dx++) for (int dy=-d;dy<=d;dy++){
            if (abs(dx)+abs(dy)!=d) continue;
            int nx=a.x+dx, ny=a.y+dy;
            if (0<nx&&nx<W-1&&0<ny&&ny<H-1 && grid[idx(nx,ny)]==EMPTY && occ[idx(nx,ny)]==0)
                cand.push_back({nx,ny});
        }
        if (cand.empty()){ seed_food_near(a); return; }
        int pick = rng() % cand.size();
        grid[idx(cand[pick][0], cand[pick][1])] = FOOD;
        rebuild_food_index();
    }

    void seed_food_near(Agent &a) {
        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
            if (dx==0&&dy==0) continue;
            int nx=a.x+dx, ny=a.y+dy;
            if (0<nx&&nx<W-1&&0<ny&&ny<H-1 && grid[idx(nx,ny)]==EMPTY)
                { grid[idx(nx,ny)]=FOOD; rebuild_food_index(); return; }
        }
    }

    // returns python list of (obs ndarray, rewards list, dones list)
    py::tuple step(const std::vector<int> &actions) {
        step_count++;
        int n=n_agents;
        std::vector<float> rew(n,0.0f);
        std::vector<bool> done(n,false);
        std::vector<int> pre_dist(n);
        for (auto &a:agents) pre_dist[a.idx] = a.alive ? nearest_goal_dist(a.x,a.y) : 1e9;
        // order: by speed desc (deterministic)
        std::vector<int> order; for (auto &a:agents) if (a.alive) order.push_back(a.idx);
        std::sort(order.begin(),order.end(),[&](int i,int j){ return agents[i].tr.speed>agents[j].tr.speed; });
        std::vector<int> moved_set;
        for (int ai:order){
            Agent &a=agents[ai];
            int act = (ai<(int)actions.size())?actions[ai]:0;
            a.last_action=act;
            if (act>=1 && act<=4){
                int dx=0,dy=0;
                if (act==1) dy=-1; else if (act==2) dx=1; else if (act==3) dy=1; else dx=-1;
                int nx=a.x+dx, ny=a.y+dy;
                if (nx>=0&&ny>=0&&nx<W&&ny<H && tile_passable(a,grid[idx(nx,ny)])){
                    if (occ[idx(nx,ny)]==0){
                        occ[idx(a.x,a.y)]=0; a.x=nx; a.y=ny; occ[idx(nx,ny)]=a.idx+1; moved_set.push_back(ai);
                    }
                }
            }
            // NOTE: mutation (8-12), harvest (5), share (6), signal (7) reward is
            // handled in the per-agent reward section below (so each action is paid
            // ONCE -- mutate() changes traits there, not here). The move-loop only
            // moves the agent. (Previously mutate() was ALSO called here, double-
            // applying the trait change -- a latent bug.)
            // is added to the agent's own reward after PBRS).
        }
        for (auto &a:agents){
            if (!a.alive){ done[a.idx]=true; continue; }
            float r = 0.0f;
            // 0. PASSIVE HUNGER: energy decays every step regardless of action, and
            // the agent is penalized for being hungry. This removes the
            // "starving/camping is better" optimum -- there is NO policy that avoids
            // food and survives, so the agent is forced to harvest to stay alive.
            a.energy = std::max(0.0f, a.energy - STARVE_DECAY);
            // penalty grows as energy drops (hungrier = worse). At full energy
            // penalty ~0; near zero energy penalty ~STARVE_PEN.
            float hunger = 1.0f - std::min(1.0f, a.energy / E_MAX_F);
            r -= STARVE_PEN * hunger;
            int act = a.last_action;
            // 1. base step + stagnation penalty
            r -= STEP_PEN * a.tr.metabolism;
            if (act == 0) r -= STAGNATION_PEN;
            // 1b. blocked-move penalty: tried to move (1-4) but didn't (wall/agent).
            // Teaches the policy to route AROUND obstacles under BOTH stochastic
            // and greedy decode (otherwise greedy sticks against walls).
            if (act >= 1 && act <= 4 && std::find(moved_set.begin(), moved_set.end(), a.idx) == moved_set.end())
                r -= BLOCKED_PEN;
            // 2. potential-based navigation reward (Ng et al. PBRS)
            int dist_before = pre_dist[a.idx];
            int dist_after = nearest_goal_dist(a.x, a.y);
            if (dist_before > W + H) dist_before = W + H;
            if (dist_after  > W + H) dist_after  = W + H;
            // Skip the nav bonus once already adjacent (dist<=1): camping on food
            // shouldn't be rewarded, only the move-toward-food signal matters.
            if (dist_before > 1)
                r += rp.nav_alpha * (float)(dist_before - dist_after);
            // FOOD_PULL as a POTENTIAL (not a state reward): only pay on the step
            // the agent moves CLOSER to food, zero when stationary/adjacent. This
            // stops the agent from "orbiting" food to farm the pull -- eating
            // (+eat_gain, once) now strictly dominates camping (which yields nothing).
            if (dist_before > 1 && dist_after < dist_before)
                r += rp.food_pull * (float)(dist_before - dist_after) / (1.0f + (float)dist_after);
            // Trait-match shaping: reward being adjacent to a GATED tile the agent
            // can NOW eat (has can_hard/can_tall). This bridges the 3-step credit
            // gap in the mutate->eat loop: "I gained the trait AND I'm next to the
            // food it unlocks" becomes immediately rewarding, so the policy learns
            // to (a) mutate the right trait and (b) stay adjacent to harvest,
            // instead of mutating then wandering off. Not given for regular FOOD
            // (that's already handled by harvest + PBRS).
            if (a.tr.can_hard() || a.tr.can_tall()) {
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
                    int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
                    int t=grid[idx(nx,ny)];
                    if (t==HARD_NUT && a.tr.can_hard()) { r += rp.trait_match_bonus; break; }
                    if (t==TALL_FRUIT && a.tr.can_tall()) { r += rp.trait_match_bonus; break; }
                }
            }
            bool adj = adjacent_harvestable(a);
            // gated-food adjacency (for the pipeline funnel) -- computed once here
            bool adj_gated = false, adj_gated_unlock = false;
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
                int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
                int t=grid[idx(nx,ny)];
                if (t==HARD_NUT) { adj_gated=true; if (a.tr.can_hard()) adj_gated_unlock=true; }
                if (t==TALL_FRUIT) { adj_gated=true; if (a.tr.can_tall()) adj_gated_unlock=true; }
            }
            // G: dense gate-proximity shaping. Once the agent is STRONG enough to
            // open a gate (strength>=TH_GATE) and stands adjacent to a GATE cell,
            // pay a small continuous bonus every step. This bridges the final
            // credit gap: the one-shot gate_gain (0.8) only fires the instant the
            // gate opens, which is too sparse to learn "build up strength THEN go
            // to the gate". With gate_prox_bonus>0 the policy gets a steady signal
            // for the complete strong-at-gate state.
            if (rp.gate_prox_bonus != 0.0f && a.tr.strength >= TH_GATE / 100.0f) {
                for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
                    int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
                    if (grid[idx(nx,ny)]==GATE) { r += rp.gate_prox_bonus; break; }
                }
            }
            if (act == 5) {
                if (adj) { r += harvest(a); diag.harvest_valid++;
                          if (adj_gated_unlock) diag.harvested_gated++; }
                else     { r -= rp.invalid_harvest_pen; diag.harvest_invalid++; }
            } else if (act == 6) { r += share(a, rew); r -= ACT_COST_SHARE; }
            else if (act == 7) { signal(a, rew); r -= ACT_COST_SIGNAL; }
            else if (act >= 8 && act <= 12) {
                bool adj_gated_unlock_before = adj_gated_unlock;
                float rew_before_mut = rew[a.idx];   // mutate() writes reward here;
                mutate(a, act, rew, adj_gated); r -= ACT_COST_MUT; diag.mutate_steps++;
                // fold mutate()'s own reward writes (trait_mut_pen + trait-gain +1.0)
                // into r, since line 732 does rew[a.idx]=r and would otherwise DISCARD
                // them (latent bug: mutation penalty/gain was never delivered).
                r += (rew[a.idx] - rew_before_mut);
                if (adj_gated) {
                    diag.mutated_near_gated++;
                    // recompute whether this step's mutation unlocked the adjacent gated food
                    bool adj_gated_unlock_now = false;
                    for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
                        int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny<0||nx>=W||ny>=H) continue;
                        int t=grid[idx(nx,ny)];
                        if (t==HARD_NUT && a.tr.can_hard()) adj_gated_unlock_now=true;
                        if (t==TALL_FRUIT && a.tr.can_tall()) adj_gated_unlock_now=true;
                    }
                    if (!adj_gated_unlock_before && adj_gated_unlock_now) {
                        // C: gained the RIGHT trait -> shaped reward for the mutate->eat transition
                        diag.gained_right_trait++;
                        r += rp.mutate_gated_gain;
                    } else if (!adj_gated_unlock_before && !adj_gated_unlock_now) {
                        // C: mutated WRONG trait near gated food -> small penalty
                        diag.wrong_trait_mut++;
                        r -= rp.wrong_trait_pen;
                    }
                }
            }
            // ---- full instrumentation ----
            diag.steps++;
            // trait dynamics sampling (mean strength/reach over agent-steps)
            diag.sum_strength += a.tr.strength;
            diag.sum_reach += a.tr.reach;
            diag.trait_samples++;
            if (a.tr.strength > diag.max_strength) diag.max_strength = a.tr.strength;
            if (a.tr.strength >= TH_GATE/100.0f) diag.gate_chain_possible = 1;
            // ground-truth distances (not the obs proxy)
            int gf = nearest_food_dist(a.x, a.y);
            diag.dist_food_sum += (gf > W+H ? (float)(W+H) : (float)gf);
            diag.dist_food_samples++;
            int gg = nearest_gated_dist(a.x, a.y);
            if (gg < 1e9) { diag.dist_gated_sum += (float)gg; diag.dist_gated_samples++; }
            // reward probe: attribute this step's reward to the action taken
            if (act >= 0 && act < 13) diag.rew_by_action[act] += r;
            // gated adjacency already computed above (adj_gated / adj_gated_unlock)
            if (adj_gated) diag.reached_gated++;
            // (mutated_near_gated / wrong_trait_mut / gained_right_trait are counted
            //  in the act==5..12 branch above to avoid double-counting)
            // movement vs gated food (direction)
            if (act >= 1 && act <= 4) {
                if (gg < a.last_gated_dist) diag.moved_closer_gated++;
                else if (gg > a.last_gated_dist) diag.moved_away_gated++;
            }
            a.last_gated_dist = gg;
            // move-away/closer vs nearest food (existing)
            if (act >= 1 && act <= 4) {
                if (dist_after > dist_before) diag.move_away++;
                else if (dist_after < dist_before) diag.move_closer++;
            }
            // gate proximity
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++){
                int nx=a.x+dx, ny=a.y+dy; if (nx<0||ny>=W||ny>=H) continue;
                if (grid[idx(nx,ny)]==GATE) { diag.gate_adj++; if (a.tr.strength>=TH_GATE/100.0f) diag.gate_adj_strong++; break; }
            }
            // 5. death evaluation
            if (a.energy <= 0) { a.energy = 0; a.alive = false; done[a.idx] = true; r -= DEATH_PEN; occ[idx(a.x, a.y)] = 0; diag.dead++; }
            rew[a.idx] = r;
        }
        resolve_hazards(rew,done);
        resolve_predators(rew,done);
        resolve_gates(rew);
        regen_tiles();
        for (auto &a:agents) if (a.cooldown>0) a.cooldown--;
        if (respawn) respawn_dead();
        // build obs -- zero-fill first (py::array_t is NOT zero-initialized, unlike
        // np.zeros; unwritten onehot channels must be 0, not garbage).
        py::array_t<float> obs(std::vector<ssize_t>{n, (ssize_t)(H*W*CHAN+OWN_DIM)});
        auto o=obs.mutable_unchecked<2>();
        float *op = obs.mutable_data();
        std::fill(op, op + n*(ssize_t)(H*W*CHAN+OWN_DIM), 0.0f);
        for (int i=0;i<n;i++){
            const Agent &a=agents[i];
            int base=0;
            // grid channels: onehot 11 + occ 1, per cell
            for (int y=0;y<H;y++) for (int x=0;x<W;x++){
                int t=grid[idx(x,y)]; int cell=base+(y*W+x)*CHAN;
                if (t>=0 && t<=10) o(i,cell+t)=1.0f;
                if (occ[idx(x,y)]>0) o(i,cell+11)=1.0f;
            }
            base=H*W*CHAN;
            const Traits &tr=a.tr;
            o(i,base+0)=std::min(1.0f, a.energy/E_MAX_F); o(i,base+1)=a.inv/8.0f;
            o(i,base+2)=tr.strength; o(i,base+3)=tr.reach; o(i,base+4)=tr.speed;
            o(i,base+5)=tr.perception/4.0f; o(i,base+6)=tr.metabolism; o(i,base+7)=tr.social;
            o(i,base+8)=tr.can_small()?1.0f:0.0f; o(i,base+9)=a.last_action/7.0f;
            o(i,base+10)=tr.can_hard()?1.0f:0.0f; o(i,base+11)=tr.can_tall()?1.0f:0.0f;
            o(i,base+12)=a.x/(float)W; o(i,base+13)=a.y/(float)H;
            // LOCAL 11x11 patch around the agent (direct food perception + navigation;
            // the global ViT dilutes single food tiles to ~1/1024, so agents need a
            // large local sight to steer toward distant food).
            int lp=14;
            for (int dy=-5;dy<=5;dy++) for (int dx=-5;dx<=5;dx++){
                int cx=a.x+dx, cy=a.y+dy;
                int cell=base+14+(lp-14)*CHAN; lp++;
                if (0<=cx && cx<W && 0<=cy && cy<H){
                    int t=grid[idx(cx,cy)];
                    if (t>=0 && t<=10) o(i,cell+t)=1.0f;
                    if (occ[idx(cx,cy)]>0) o(i,cell+11)=1.0f;
                }
            }
            // compact food-direction signal: unit vector to nearest food + normalized dist
            { int bx=-1,by=-1,bd=999; for (int y=0;y<H;y++) for (int x=0;x<W;x++){ if (grid[idx(x,y)]==1){ int d=std::abs(x-a.x)+std::abs(y-a.y); if (d<bd){ bd=d; bx=x; by=y; } } }
              if (bx>=0){ float dxn=(float)(bx-a.x), dyn=(float)(by-a.y); float L=std::sqrt(dxn*dxn+dyn*dyn)+1e-6f;
                o(i,base+14+121*CHAN+0)=dxn/L; o(i,base+14+121*CHAN+1)=dyn/L; o(i,base+14+121*CHAN+2)=std::min(1.0f, bd/40.0f); } }
        }
        py::list rewlist; for (float r:rew) rewlist.append(r);
        py::list donelist; for (bool d:done) donelist.append(d);
        return py::make_tuple(obs, rewlist, donelist);
    }

    py::array_t<float> reset() {
        step_count=0;
        reset_diag();   // clear closed-loop diagnostics each episode
        std::fill(regen_t.begin(),regen_t.end(),0);
        std::fill(regen_type.begin(),regen_type.end(),0);
        std::fill(occ.begin(),occ.end(),0);
        build_grid();
        spawn_agents();
        rebuild_food_index();
        // build obs
        int n=n_agents;
        py::array_t<float> obs(std::vector<ssize_t>{n, (ssize_t)(H*W*CHAN+OWN_DIM)});
        auto o=obs.mutable_unchecked<2>();
        float *op = obs.mutable_data();
        std::fill(op, op + n*(ssize_t)(H*W*CHAN+OWN_DIM), 0.0f);
        for (int i=0;i<n;i++){
            const Agent &a=agents[i]; int base=0;
            for (int y=0;y<H;y++) for (int x=0;x<W;x++){
                int t=grid[idx(x,y)]; int cell=base+(y*W+x)*CHAN;
                if (t>=0 && t<=10) o(i,cell+t)=1.0f;
                if (occ[idx(x,y)]>0) o(i,cell+11)=1.0f;
            }
            base=H*W*CHAN; const Traits &tr=a.tr;
            o(i,base+0)=std::min(1.0f, a.energy/E_MAX_F); o(i,base+1)=a.inv/8.0f;
            o(i,base+2)=tr.strength; o(i,base+3)=tr.reach; o(i,base+4)=tr.speed;
            o(i,base+5)=tr.perception/4.0f; o(i,base+6)=tr.metabolism; o(i,base+7)=tr.social;
            o(i,base+8)=tr.can_small()?1.0f:0.0f; o(i,base+9)=a.last_action/7.0f;
            o(i,base+10)=tr.can_hard()?1.0f:0.0f; o(i,base+11)=tr.can_tall()?1.0f:0.0f;
            o(i,base+12)=a.x/(float)W; o(i,base+13)=a.y/(float)H;
            // LOCAL 11x11 patch around the agent (direct food perception + navigation;
            // the global ViT dilutes single food tiles to ~1/1024, so agents need a
            // large local sight to steer toward distant food).
            int lp=14;
            for (int dy=-5;dy<=5;dy++) for (int dx=-5;dx<=5;dx++){
                int cx=a.x+dx, cy=a.y+dy;
                int cell=base+14+(lp-14)*CHAN; lp++;
                if (0<=cx && cx<W && 0<=cy && cy<H){
                    int t=grid[idx(cx,cy)];
                    if (t>=0 && t<=10) o(i,cell+t)=1.0f;
                    if (occ[idx(cx,cy)]>0) o(i,cell+11)=1.0f;
                }
            }
            // compact food-direction signal: unit vector to nearest food + normalized dist
            { int bx=-1,by=-1,bd=999; for (int y=0;y<H;y++) for (int x=0;x<W;x++){ if (grid[idx(x,y)]==1){ int d=std::abs(x-a.x)+std::abs(y-a.y); if (d<bd){ bd=d; bx=x; by=y; } } }
              if (bx>=0){ float dxn=(float)(bx-a.x), dyn=(float)(by-a.y); float L=std::sqrt(dxn*dxn+dyn*dyn)+1e-6f;
                o(i,base+14+121*CHAN+0)=dxn/L; o(i,base+14+121*CHAN+1)=dyn/L; o(i,base+14+121*CHAN+2)=std::min(1.0f, bd/40.0f); } }
        }
        return obs;
    }

    int obs_dim() const { return H*W*CHAN+OWN_DIM; }

    // expose agent state as python list of dicts (for the env.py wrapper;
    // keeps train.py/render.py untouched).
    py::list dump_agents() {
        py::list out;
        for (auto &a:agents){
            py::dict d;
            d["idx"]=a.idx; d["x"]=a.x; d["y"]=a.y; d["energy"]=a.energy;
            d["inv"]=a.inv;
            d["alive"]=a.alive; d["last_action"]=a.last_action; d["cooldown"]=a.cooldown;
            d["strength"]=a.tr.strength; d["reach"]=a.tr.reach; d["speed"]=a.tr.speed;
            d["perception"]=a.tr.perception; d["metabolism"]=a.tr.metabolism; d["social"]=a.tr.social;
            d["size_small"]=a.tr.size_small;
            d["can_hard"]=a.tr.can_hard(); d["can_tall"]=a.tr.can_tall(); d["can_small"]=a.tr.can_small();
            out.append(d);
        }
        return out;
    }
};

PYBIND11_MODULE(cpp_sim, m) {
    py::class_<Agent>(m, "Agent")
        .def_readonly("idx", &Agent::idx)
        .def_readonly("x", &Agent::x)
        .def_readonly("y", &Agent::y)
        .def_readonly("energy", &Agent::energy)
        .def_readonly("inv", &Agent::inv)
        .def_readonly("alive", &Agent::alive)
        .def_readonly("last_action", &Agent::last_action)
        .def_readonly("cooldown", &Agent::cooldown);

    py::class_<Sim>(m, "Sim")
        .def(py::init<int,int,int,uint32_t,int,bool,int,int,int>())
        .def("set_food_seed", [](Sim&s,int v){ s.food_seed=v; })
        .def("set_food_seed_dist", [](Sim&s,int v){ s.food_seed_dist=v; })
        .def("set_food_density_div", [](Sim&s,int v){ s.food_density_div=v; })
        .def("set_food_regen", [](Sim&s,bool v){ s.food_regen=v; })
        .def("set_food_regen_mode", [](Sim&s,int v){ s.food_regen_mode=v; if (v==0) s.food_regen=false; else s.food_regen=true; })
        .def("set_gated_food", [](Sim&s,int v){ s.gated_food=v; })
        .def("set_reward_params", [](Sim&s, float food_pull, float nav_alpha, float eat_gain,
                                      float eat_gain_regular, float invalid_harvest_pen,
                                      float trait_mut_pen, float trait_mut_pen_gated,
                                      float gate_gain, float trait_match_bonus,
                                      float mutate_gated_gain, float wrong_trait_pen){
            s.rp.food_pull = food_pull;
            s.rp.nav_alpha = nav_alpha;
            s.rp.eat_gain = eat_gain;
            s.rp.eat_gain_regular = eat_gain_regular;
            s.rp.invalid_harvest_pen = invalid_harvest_pen;
            s.rp.trait_mut_pen = trait_mut_pen;
            s.rp.trait_mut_pen_gated = trait_mut_pen_gated;
            s.rp.gate_gain = gate_gain;
            s.rp.trait_match_bonus = trait_match_bonus;
            s.rp.mutate_gated_gain = mutate_gated_gain;
            s.rp.wrong_trait_pen = wrong_trait_pen;
        })
        .def("set_gate_prox_bonus", [](Sim&s, float v){ s.rp.gate_prox_bonus = v; })
        .def("set_step_frac", [](Sim&s,float f){ s.step_frac = f; })
        .def("get_diag", [](Sim&s){
            const auto &d = s.diag;
            return py::make_tuple(d.steps, d.harvest_invalid, d.harvest_valid,
                                  d.move_away, d.move_closer, d.mutate_steps,
                                  d.gate_adj, d.gate_adj_strong, d.dead);
        })
        .def("get_diag_full", [](Sim&s){
            const auto &d = s.diag;
            py::dict out;
            out["steps"] = d.steps;
            out["harvest_invalid"] = d.harvest_invalid;
            out["harvest_valid"] = d.harvest_valid;
            out["move_away"] = d.move_away;
            out["move_closer"] = d.move_closer;
            out["mutate_steps"] = d.mutate_steps;
            out["gate_adj"] = d.gate_adj;
            out["gate_adj_strong"] = d.gate_adj_strong;
            out["dead"] = d.dead;
            // pipeline funnel
            out["reached_gated"] = d.reached_gated;
            out["mutated_near_gated"] = d.mutated_near_gated;
            out["gained_right_trait"] = d.gained_right_trait;
            out["harvested_gated"] = d.harvested_gated;
            out["wrong_trait_mut"] = d.wrong_trait_mut;
            // trait dynamics
            out["trait_gain_events"] = d.trait_gain_events;
            out["trait_loss_events"] = d.trait_loss_events;
            out["mean_strength"] = (d.trait_samples>0) ? (float)(d.sum_strength/d.trait_samples) : 0.0f;
            out["mean_reach"] = (d.trait_samples>0) ? (float)(d.sum_reach/d.trait_samples) : 0.0f;
            // ground-truth distances
            out["mean_dist_food"] = (d.dist_food_samples>0) ? (float)(d.dist_food_sum/d.dist_food_samples) : -1.0f;
            out["mean_dist_gated"] = (d.dist_gated_samples>0) ? (float)(d.dist_gated_sum/d.dist_gated_samples) : -1.0f;
            out["moved_closer_gated"] = d.moved_closer_gated;
            out["moved_away_gated"] = d.moved_away_gated;
            // gate progress
            out["max_strength"] = d.max_strength;
            out["gate_chain_possible"] = d.gate_chain_possible;
            // reward probe
            py::list rb;
            for (int i=0;i<13;i++) rb.append(d.rew_by_action[i]);
            out["rew_by_action"] = rb;
            return out;
        })
        .def("set_tile", [](Sim&s,int x,int y,int t){ if (x>=0&&y>=0&&x<s.W&&y<s.H) s.grid[s.idx(x,y)]=t; })
        .def("get_tile", [](Sim&s,int x,int y){ return (x>=0&&y>=0&&x<s.W&&y<s.H)? s.grid[s.idx(x,y)] : -1; })
        .def("clear_food", [](Sim&s){ for (auto &v:s.grid) if (v==1) v=0; })
        .def("step", &Sim::step)
        .def("reset", &Sim::reset)
        .def("obs_dim", &Sim::obs_dim)
        .def("dump_agents", &Sim::dump_agents)
        .def_readonly("W", &Sim::W)
        .def_readonly("H", &Sim::H)
        .def_readonly("grid", &Sim::grid)
        .def_readonly("oasis_cells", &Sim::oasis_cells)
        .def_readonly("gate_cells", &Sim::gate_cells)
        .def_readonly("agents", &Sim::agents)
        .def("adjacent_harvestable", &Sim::adjacent_harvestable)
        .def("gate_threshold", [](Sim&s){ return (float)TH_GATE/100.0f; });
}
