import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybeat.brain.model import BrainSim, Connectome, SimConfig, normalized_weights
from flybeat.brain.preprocess import hop_distance, top_k_inputs


def tiny_connectome() -> Connectome:
    """입력 2레인×1bin → 중간 뉴런 → 운동 뉴런 2개의 사슬 + 억제 뉴런 1개.

    0,1: 입력 (레인 0, 1)   2,3: 중간   4,5: 운동 (레인 0, 1)   6: 억제(레인1 입력 → 운동 0 억제)
    """
    n = 7
    edges = [(0, 2, 10), (2, 4, 10), (1, 3, 10), (3, 5, 10), (1, 6, 10), (6, 4, -10)]
    pre, post, w = zip(*edges, strict=True)
    W = sp.csr_matrix((np.array(w, np.float32), (post, pre)), shape=(n, n))
    groups = {
        "input": [[[0]], [[1]]],
        "output": [[4], [5]],
        "dan_reward": [],
        "dan_punish": [],
        "kc": [],
        "mbon": [],
    }
    e = np.array([], dtype=np.int64)
    return Connectome(W, pd.DataFrame({"bodyId": range(n)}), groups, e, e, e.astype(float), {})


CFG = SimConfig(dt_ms=1.0, tau_ms=5.0, gain=1.0, bias=0.0, normalize_alpha=1.0)


def test_input_propagates_to_matching_motor_lane():
    sim = BrainSim(tiny_connectome(), batch=2, cfg=CFG)
    u = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    for _ in range(10):
        sim.step(u, 10)
    m = sim.motor_lane_mean()
    assert m.shape == (2, 2)
    assert m[0, 0] > 0.3 and m[0, 1] < 0.01  # 레인 0 입력 → 운동 0
    assert m[1, 1] > 0.3 and m[1, 0] < 0.01  # 레인 1 입력 → 운동 1, 운동 0 은 억제


def test_activity_decays_without_input():
    sim = BrainSim(tiny_connectome(), batch=1, cfg=CFG)
    sim.step(np.array([[1.0, 1.0]], np.float32), 100)
    assert sim.r.max() > 0.3
    sim.step(np.zeros((1, 2), np.float32), 200)
    assert sim.r.max() < 0.01


def test_reset_cols():
    sim = BrainSim(tiny_connectome(), batch=2, cfg=CFG)
    sim.step(np.ones((2, 2), np.float32), 50)
    sim.reset_cols(np.array([True, False]))
    assert sim.r[:, 0].max() == 0 and sim.r[:, 1].max() > 0


def test_normalized_weights_rows():
    W = sp.csr_matrix(np.array([[0, 4, -4], [0, 0, 0], [9, 0, 0]], np.float32))
    N = normalized_weights(W, 1.0).toarray()
    np.testing.assert_allclose(N[0], [0, 0.5, -0.5])
    np.testing.assert_allclose(N[2], [1, 0, 0])
    N5 = normalized_weights(W, 0.5).toarray()
    np.testing.assert_allclose(N5[2], [3, 0, 0])


def test_hop_distance_and_top_k():
    adj = sp.csr_matrix((np.ones(3), ([0, 1, 2], [1, 2, 3])), shape=(5, 5))
    np.testing.assert_array_equal(hop_distance(adj, np.array([0]), 10), [0, 1, 2, 3, -1])
    np.testing.assert_array_equal(hop_distance(adj, np.array([0]), 2), [0, 1, 2, -1, -1])
    qi = np.array([0, 0, 0, 1, 1])
    w = np.array([1, 5, 3, 2, 1], np.float32)
    np.testing.assert_array_equal(top_k_inputs(qi, w, 2), [False, True, True, True, True])
    assert top_k_inputs(qi, w, 0).all()
