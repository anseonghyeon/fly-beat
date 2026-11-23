import numpy as np
import pytest

from flybeat import paths
from flybeat.agents import checkpoint as ck
from flybeat.agents.base import eval_charts, evaluate


@pytest.fixture(autouse=True)
def tmp_ckpt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "A-1", ".hidden", "x" * 81])
def test_checkpoint_id_rejects_traversal(bad):
    with pytest.raises(ck.CheckpointError):
        ck.checkpoint_dir(bad)


def test_checkpoint_roundtrip(tmp_ckpt_dir):
    cid = ck.new_checkpoint_id("B")
    ck.save_checkpoint(cid, {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, {"mode": "B"})
    tensors, meta = ck.load_checkpoint(cid)
    np.testing.assert_array_equal(tensors["w"], np.arange(6).reshape(2, 3))
    assert meta["id"] == cid and meta["mode"] == "B"
    assert [m["id"] for m in ck.list_checkpoints()] == [cid]


def test_checkpoint_unknown_mode_rejected(tmp_ckpt_dir):
    ck.save_checkpoint("x-1", {"w": np.zeros(1, np.float32)}, {"mode": "Z"})
    with pytest.raises(ck.CheckpointError):
        ck.load_meta("x-1")
    assert ck.list_checkpoints() == []


def test_mode_a_train_save_load_roundtrip():
    from flybeat.agents.deep_rl import PPOAgent, PPOConfig, train

    progress = []
    cid = train(
        PPOConfig(total_steps=2048, n_envs=2, n_steps=256, batch_size=256),
        progress=progress.append,
    )
    assert progress and progress[-1].step >= 2048
    meta = ck.load_meta(cid)
    agent = PPOAgent.load(cid)
    again = evaluate(agent, eval_charts())
    assert again["accuracy"] == pytest.approx(meta["eval"]["accuracy"])


def test_mode_a_stop_flag():
    from flybeat.agents.deep_rl import PPOConfig, train

    seen = []
    train(
        PPOConfig(total_steps=100_000, n_envs=2, n_steps=128, batch_size=256),
        progress=seen.append,
        should_stop=lambda: True,
    )
    assert not seen or seen[-1].step < 100_000
