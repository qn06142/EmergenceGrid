"""Sanity checks for EmergenceGrid.

Verifies the SIM behaves as expected: determinism, trait bounds,
obs shape consistency, harvest mechanics, and signal energy costs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from env import EmergenceGrid, FOOD, EMPTY


def test_obs_shape():
    g = EmergenceGrid(width=64, height=64, n_agents=16, seed=9)
    obs = g.reset()
    dim = g.obs_dim()
    assert obs.shape == (16, dim), f"obs shape {obs.shape} != (16, {dim})"
    print(f"[PASS] obs shape consistent: shape={obs.shape}, dim={dim}")


def test_deterministic_step():
    """Same seed + same actions -> identical state & observations."""
    def run(seed):
        g = EmergenceGrid(width=64, height=64, n_agents=4, seed=seed)
        obs0 = g.reset()
        obs_history = []
        for _ in range(10):
            o, r, d, info = g.step([1, 2, 3, 4])
            obs_history.append((o.copy(), list(r), list(d)))
        return obs_history

    r1 = run(42)
    r2 = run(42)
    for t in range(len(r1)):
        np.testing.assert_array_equal(r1[t][0], r2[t][0])
        assert r1[t][1] == r2[t][1]
        assert r1[t][2] == r2[t][2]
    print("[PASS] step is deterministic (seed-stable)")


def test_trait_bounds():
    """Verify that agent traits satisfy bounded trade-offs."""
    g = EmergenceGrid(width=64, height=64, n_agents=16, seed=123)
    for a in g.agents:
        t = a.traits
        assert t.strength + t.speed <= 1.31, f"Strength+Speed {t.strength}+{t.speed} > 1.3"
        assert t.perception / 4.0 + t.strength <= 1.31, f"Perception+Strength > 1.3"
        if t.social > 0.6:
            assert (t.strength + t.reach + t.speed) / 3.0 <= 0.46, "Social agent trait mean too high"
    print("[PASS] binding sampler: trade-offs hold across agents")


def test_signal_costs_energy():
    """Verify signal action (7) costs energy."""
    g = EmergenceGrid(width=64, height=64, n_agents=1, seed=11)
    g.reset()
    e0 = g.agents[0].energy
    g.step([7])
    e1 = g.agents[0].energy
    assert e1 < e0, f"Signal must cost energy: {e0} -> {e1}"
    print(f"[PASS] signal costs energy ({e0:.2f} -> {e1:.2f})")


def test_harvest_adjacent_food():
    """Verify harvest action (5) harvests adjacent food."""
    g = EmergenceGrid(width=64, height=64, n_agents=1, seed=15, food_seed=1, food_seed_dist=1)
    g.reset()
    # food_seed placed food adjacent (ring dist 1)
    e0 = g.agents[0].energy
    inv0 = g.agents[0].inv
    g.step([5])  # harvest
    e1 = g.agents[0].energy
    inv1 = g.agents[0].inv
    assert inv1 > inv0 or e1 > e0, f"Harvesting adjacent food should increase inv or energy ({inv0}->{inv1}, {e0}->{e1})"
    print(f"[PASS] harvest adjacent food succeeds: inv {inv0}->{inv1}, energy {e0:.2f}->{e1:.2f}")


if __name__ == '__main__':
    test_obs_shape()
    test_deterministic_step()
    test_trait_bounds()
    test_signal_costs_energy()
    test_harvest_adjacent_food()
    print("\nALL SANITY CHECKS PASSED")
