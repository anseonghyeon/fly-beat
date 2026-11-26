import numpy as np
import pytest

from flymax.agents import checkpoint as ck
from flymax.agents.base import eval_charts, evaluate


def test_mode_c_changes_only_kc_mbon_synapses(synthetic_data):
    from flymax.agents.dopamine import DopamineAgent, DopamineConfig, train
    from flymax.brain.model import Connectome

    conn = Connectome.load(synthetic_data)
    cid = train(DopamineConfig(profile=synthetic_data, episodes=2, n_envs=2, lr=1.0))
    tensors, meta = ck.load_checkpoint(cid)
    w, w0 = tensors["plastic.weight"], tensors["plastic.weight0"]
    assert meta["plasticity"]["changed"] > 0
    assert (w >= 0).all() and (w <= DopamineConfig().w_max * w0 + 1e-6).all()

    agent = DopamineAgent.load(cid)
    W = agent.brain.sim.W.toarray()
    from flymax.brain.model import BrainSim

    W_orig = BrainSim(conn).W.toarray()
    diff = np.argwhere(~np.isclose(W, W_orig))
    allowed = set(zip(conn.plastic_post.tolist(), conn.plastic_pre.tolist(), strict=True))
    assert diff.size and all((int(r), int(c)) in allowed for r, c in diff)
    assert evaluate(agent, eval_charts())["accuracy"] == pytest.approx(meta["eval"]["accuracy"])
