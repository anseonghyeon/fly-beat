import pytest

from flymax.agents import checkpoint as ck
from flymax.agents.base import eval_charts, evaluate


def test_mode_b_train_save_load(synthetic_data):
    from flymax.agents.mapping import MappingAgent, MappingConfig, train

    cfg = MappingConfig(
        profile=synthetic_data, total_steps=512, n_envs=2, n_steps=128, batch_size=128
    )
    cid = train(cfg)
    meta = ck.load_meta(cid)
    agent = MappingAgent.load(cid)
    assert agent.activity()["motor"] is not None
    assert evaluate(agent, eval_charts())["accuracy"] == pytest.approx(meta["eval"]["accuracy"])
