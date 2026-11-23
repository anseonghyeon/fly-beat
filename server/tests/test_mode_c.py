import numpy as np
import pytest

from flybeat.agents import checkpoint as ck
from flybeat.agents.base import eval_charts, evaluate


def test_mode_c_changes_only_kc_mbon_synapses(synthetic_data):
    from flybeat.agents.dopamine import DopamineAgent, DopamineConfig, train
    from flybeat.brain.model import Connectome

    conn = Connectome.load(synthetic_data)
    cid = train(DopamineConfig(profile=synthetic_data, episodes=2, n_envs=2, lr=1.0))
    tensors, meta = ck.load_checkpoint(cid)
    w, w0 = tensors["plastic.weight"], tensors["plastic.weight0"]
    assert meta["plasticity"]["changed"] > 0
    assert (w >= 0).all() and (w <= DopamineConfig().w_max * w0 + 1e-6).all()

    agent = DopamineAgent.load(cid)
    W = agent.brain.sim.W.toarray()
    from flybeat.brain.model import BrainSim

    W_orig = BrainSim(conn).W.toarray()
    diff = np.argwhere(~np.isclose(W, W_orig))
    allowed = set(zip(conn.plastic_post.tolist(), conn.plastic_pre.tolist(), strict=True))
    assert diff.size and all((int(r), int(c)) in allowed for r, c in diff)
    assert evaluate(agent, eval_charts())["accuracy"] == pytest.approx(meta["eval"]["accuracy"])


def test_leg_striker_tap_hold_and_refractory():
    from flybeat.agents.dopamine import DopamineConfig, LegStriker

    tick = 1000 / 60
    legs = LegStriker(DopamineConfig(lanes=1), batch=1, tick_ms=tick)  # 탭 3 tick, 불응기 7 tick
    zero, one = np.zeros((1, 1)), np.ones((1, 1))
    low = np.full((1, 1), 0.1)
    seq = []
    for _ in range(12):  # 확률 0.1, 난수 0 → 쉴 때마다 내리치려 함
        pressed, decided, _ = legs.step(low, zero)
        seq.append((int(pressed[0, 0]), bool(decided[0, 0])))
    # 0: 내리침, 1~2: 탭 유지, 3~6: 불응기, 7: 다시 내리침
    assert [x for x, _ in seq] == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert [d for _, d in seq] == [True, False, False, False, False, False, False, True] + [
        False
    ] * 4

    hold = LegStriker(DopamineConfig(lanes=1), 1, tick)
    out = [int(hold.step(np.full((1, 1), 0.9), zero)[0][0, 0]) for _ in range(20)]
    assert out == [1] * 20  # 확률이 hold_p 보다 높으면 계속 누름(롱노트)

    old = LegStriker(DopamineConfig(lanes=1, action_model="tick"), 1, tick)
    assert int(old.step(low, one)[0][0, 0]) == 0 and int(old.step(low, zero)[0][0, 0]) == 1
