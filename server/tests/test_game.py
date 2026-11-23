import json
from pathlib import Path

import numpy as np
import pytest

from flybeat import paths
from flybeat.game.chart import ChartError, chart_from_dict, chart_path, list_charts, load_chart
from flybeat.game.engine import JudgeEngine, run_inputs
from flybeat.game.env import RhythmEnv, fixed_chart_source
from flybeat.game.generator import generate_chart

FIXTURES = sorted((paths.REPO_ROOT / "tests" / "fixtures" / "judge").glob("*.json"))


def _chart_dict(**kw):
    d = {"id": "t", "title": "t", "bpm": 120, "offsetMs": 0, "lanes": 4, "notes": []}
    d.update(kw)
    return d


# ---------------------------------------------------------------- chart
def test_chart_validation_rejects_bad_input():
    with pytest.raises(ChartError):
        chart_from_dict(_chart_dict(notes=[{"t": 0, "lane": 5, "type": "tap"}]))
    with pytest.raises(ChartError):
        chart_from_dict(_chart_dict(notes=[{"t": 0, "lane": 0, "type": "hold"}]))
    with pytest.raises(ChartError):
        chart_from_dict(_chart_dict(extra=1))
    with pytest.raises(ChartError):  # 같은 레인 겹침
        chart_from_dict(
            _chart_dict(
                notes=[
                    {"t": 0, "lane": 0, "type": "hold", "dur": 500},
                    {"t": 300, "lane": 0, "type": "tap"},
                ]
            )
        )


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", "A", "x" * 65, "..", "a.json"])
def test_chart_path_rejects_traversal(bad, tmp_path):
    with pytest.raises(ChartError):
        chart_path(bad, tmp_path)


def test_sample_charts_load():
    charts = list_charts()
    assert {c.id for c in charts} >= {"tutorial-steps", "easy-groove", "normal-flight"}
    assert load_chart("easy-groove").lanes == 4


def test_generator_deterministic_and_valid():
    a, b = generate_chart(7), generate_chart(7)
    assert a == b
    chart_from_dict(a.to_dict())  # 생성 결과가 검증을 통과해야 함


# ---------------------------------------------------------------- engine
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: Path(p).stem)
def test_golden_fixtures(path):
    case = json.loads(Path(path).read_text("utf-8"))
    chart = chart_from_dict(case["chart"])
    events, stats = run_inputs(chart, case["inputs"])
    assert [e.to_dict() for e in events] == case["expected"]["events"]
    assert stats.to_dict() == case["expected"]["stats"]


def test_tap_windows_expectations():
    case = json.loads((FIXTURES[0].parent / "tap-windows.json").read_text("utf-8"))
    got = [(e["lane"], e["judgment"]) for e in case["expected"]["events"]]
    assert got == [
        (1, "GREAT"),
        (0, "PERFECT"),
        (2, "GOOD"),
        (3, "MISS"),
        (3, "EMPTY"),
        (0, "EMPTY"),
        (0, "PERFECT"),
    ]


def test_hold_expectations():
    case = json.loads((FIXTURES[0].parent / "hold-notes.json").read_text("utf-8"))
    tails = {e["lane"]: e["judgment"] for e in case["expected"]["events"] if e["part"] == "tail"}
    assert tails == {0: "PERFECT", 1: "GREAT", 2: "MISS"}


def test_time_cannot_go_backwards():
    eng = JudgeEngine(generate_chart(1))
    eng.press(0, 5000)
    with pytest.raises(ValueError):
        eng.press(1, 4000)


# ---------------------------------------------------------------- env
def _oracle_action(env: RhythmEnv) -> np.ndarray:
    """정답 정책: 다음 노트 시각에 맞춰 누르고 롱노트는 끝까지 누른다."""
    eng = env.engine
    t = env.time_of(env.tick + 1)
    act = np.zeros(env.lanes, dtype=np.int8)
    half = 1000 / env.cfg.tick_hz / 2
    for lane in range(env.lanes):
        if eng.holding[lane] is not None:
            act[lane] = 1
            continue
        n = eng.next_note(lane)
        if n is not None and abs(t - n.t) <= half + 1:
            act[lane] = 1
    return act


def test_env_oracle_scores_high_and_random_scores_low():
    chart = load_chart("normal-flight")
    env = RhythmEnv(fixed_chart_source([chart]))
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, info = env.step(_oracle_action(env))
    assert info["stats"]["accuracy"] > 95
    assert info["stats"]["judged"] == info["stats"]["total"]

    env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, done, _, info = env.step(rng.integers(0, 2, env.lanes))
    assert info["stats"]["accuracy"] < 60


def test_env_observation_shape():
    env = RhythmEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
