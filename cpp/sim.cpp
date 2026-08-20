// EmergenceGrid C++ core (pybind11 module: cpp_sim).
// Drop-in replacement for the Python env. Same mechanics, but:
//   * food distance is NOT a grid-wide BFS/DT -- we keep a food list + coarse
//     bucket grid and answer nearest-food queries in O(local) per agent.
//   * occupancy grid (occ[W*H]) replaces the O(N^2) agent-collision scan.
//   * _agent_at / share / gate / predator use neighbor/bucket lookups.
// Obs is (N, 49166) float32 -- identical layout to the Python version, so the
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
static const int TH_STR=60, TH_REACH=60, TH_GATE=110, TH_PRED=130; // x100
static const int E_MAX=1000;       // energy in int x1000 internally? keep float
static const float E_MAX_F=10.0f;
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
    float nav_alpha=0.25f;         // PBRS navigation coefficient (annealed by curriculum)

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
        if      (curric >= 3) nav_alpha = 0.10f;
        else if (curric >= 1) nav_alpha = 0.15f;
        else                  nav_alpha = 0.25f;
    }

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
            a.energy=E_MAX_F; a.inv=0; a.alive=true; a.last_action=0; a.cooldown=0;
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
                    int d=abs(foods[fi][0]-x)+abs(foods[fi][1]-y);
                    if (d<best){ best=d; if (d<=r) {found=true;} }
                }
            }
            if (found && best<=r) break; // closest food in this ring is exact
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
                float gain=EAT_GAIN*(t==OASIS?OASIS_BONUS:1.0f);
                a.energy=std::min(E_MAX_F,a.energy+gain); a.inv++; return gain;
            }
            if (t==HARD_NUT && a.tr.can_hard()){
                grid[idx(nx,ny)]=EMPTY; foods.push_back({nx,ny}); bucket[bidx(nx,ny)].push_back((int)foods.size()-1);
                if (food_regen) { regen_t[idx(nx,ny)]=FOOD_REGEN; regen_type[idx(nx,ny)]=FOOD; }
                float gain=EAT_GAIN*(1.0f+HARD_BONUS); a.energy=std::min(E_MAX_F,a.energy+gain); a.inv++; return gain;
            }
            if (t==TALL_FRUIT && a.tr.can_tall()){
                grid[idx(nx,ny)]=EMPTY; foods.push_back({nx,ny}); bucket[bidx(nx,ny)].push_back((int)foods.size()-1);
                if (food_regen) { regen_t[idx(nx,ny)]=FOOD_REGEN; regen_type[idx(nx,ny)]=FOOD; }
                float gain=EAT_GAIN*(1.0f+TALL_BONUS); a.energy=std::min(E_MAX_F,a.energy+gain); a.inv++; return gain;
            }
        }
        return 0.0f;
    }

    void share(Agent &a, std::vector<float> &rew) {
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
            if (give>0.05f){ a.energy=std::max(0.0f,a.energy-give); o.energy=std::min(E_MAX_F,o.energy+give); a.inv++; rew[a.idx]+=SHARE_GAIN; rew[o.idx]+=SHARE_GAIN; }
        }
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

    void mutate(Agent &a, int act, std::vector<float> &rew) {
        if (a.cooldown>0) return;
        Traits &t=a.tr; bool bhard=t.can_hard(), btall=t.can_tall();
        if (act==8) t.strength=std::min(1.0f,t.strength+TRAIT_MUT);
        else if (act==9) t.strength=std::max(0.05f,t.strength-TRAIT_MUT);
        else if (act==10) t.reach=std::min(1.0f,t.reach+TRAIT_MUT);
        else if (act==11) t.reach=std::max(0.05f,t.reach-TRAIT_MUT);
        else if (act==12) t.speed=std::min(1.0f,t.speed+TRAIT_MUT);
        if (t.strength+t.speed>1.3f){ float s=1.3f/(t.strength+t.speed); t.strength*=s; t.speed*=s; }
        a.energy=std::max(0.0f,a.energy-TRAIT_MUT_PEN); rew[a.idx]-=TRAIT_MUT_PEN;
        if (!bhard && t.can_hard()) rew[a.idx]+=1.0f;   // gaining the trait is immediately rewarding
        if (!btall && t.can_tall()) rew[a.idx]+=1.0f;   // (mutate penalty is 1.0, so net ~0 unless it unlocks food)
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
            for (int ai:pushers){ agents[ai].energy=std::min(E_MAX_F,agents[ai].energy+GATE_GAIN); rew[ai]+=GATE_GAIN; }
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
                        }
                        grid[idx(rx,ry)]=FOOD; foods.push_back({rx,ry}); bucket[bidx(rx,ry)].push_back((int)foods.size()-1);
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
        for (auto &a:agents) pre_dist[a.idx] = a.alive ? nearest_food_dist(a.x,a.y) : 1e9;
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
            } else if (act>=8 && act<=12){ mutate(a,act,rew); rew[ai]-=ACT_COST_MUT; }
            // NOTE: harvest (act==5) reward is handled in the per-agent reward
            // section below (so valid/invalid harvest is paid once, and the gain
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
            int dist_after = nearest_food_dist(a.x, a.y);
            if (dist_before > W + H) dist_before = W + H;
            if (dist_after  > W + H) dist_after  = W + H;
            // Skip the nav bonus once already adjacent (dist<=1): camping on food
            // shouldn't be rewarded, only the move-toward-food signal matters.
            if (dist_before > 1)
                r += nav_alpha * (float)(dist_before - dist_after);
            // FOOD_PULL as a POTENTIAL (not a state reward): only pay on the step
            // the agent moves CLOSER to food, zero when stationary/adjacent. This
            // stops the agent from "orbiting" food to farm the pull -- eating
            // (+15, once) now strictly dominates camping (which yields nothing).
            if (dist_before > 1 && dist_after < dist_before)
                r += FOOD_PULL * (float)(dist_before - dist_after) / (1.0f + (float)dist_after);
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
                    if (t==HARD_NUT && a.tr.can_hard()) { r += TRAIT_MATCH_BONUS; break; }
                    if (t==TALL_FRUIT && a.tr.can_tall()) { r += TRAIT_MATCH_BONUS; break; }
                }
            }
            bool adj = adjacent_harvestable(a);
            if (act == 5) {
                if (adj) r += harvest(a);
                else     r -= INVALID_HARVEST_PEN;
            } else if (act == 6) { share(a, rew); r -= ACT_COST_SHARE; }
            else if (act == 7) { signal(a, rew); r -= ACT_COST_SIGNAL; }
            else if (act >= 8 && act <= 12) { mutate(a, act, rew); r -= ACT_COST_MUT; }
            // 5. death evaluation
            if (a.energy <= 0) { a.energy = 0; a.alive = false; done[a.idx] = true; r -= DEATH_PEN; occ[idx(a.x, a.y)] = 0; }
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
        .def("adjacent_harvestable", &Sim::adjacent_harvestable);
}
