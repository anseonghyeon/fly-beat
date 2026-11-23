"""Gymnasium 환경. 고정 tick 으로 렌더링 없이 실시간보다 빠르게 돈다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flybeat.game.chart import Chart
from flybeat.game.engine import JudgeEngine, JudgeEvent, JudgeWindows
from flybeat.game.generator import GeneratorConfig, generate_chart


@dataclass(frozen=True)
class RewardConfig:
    perfect: float = 1.0
    great: float = 0.7
    good: float = 0.3
    miss: float = -0.5
    empty: float = -0.1

    def of(self, ev: JudgeEvent) -> float:
        return {
            "PERFECT": self.perfect,
            "GREAT": self.great,
            "GOOD": self.good,
            "MISS": self.miss,
            "EMPTY": self.empty,
        }[ev.judgment]


@dataclass(frozen=True)
class EnvConfig:
    tick_hz: int = 60
    lookahead_ms: int = 600
    bins: int = 12
    windows: JudgeWindows = field(default_factory=JudgeWindows)
    reward: RewardConfig = field(default_factory=RewardConfig)


ChartSource = Callable[[np.random.Generator], Chart]


def random_chart_source(cfg: GeneratorConfig | None = None) -> ChartSource:
    def source(rng: np.random.Generator) -> Chart:
        return generate_chart(int(rng.integers(0, 2**31 - 1)), cfg)

    return source


def fixed_chart_source(charts: list[Chart]) -> ChartSource:
    if not charts:
        raise ValueError("charts 가 비어 있습니다")

    def source(rng: np.random.Generator) -> Chart:
        return charts[int(rng.integers(0, len(charts)))]

    return source


def obs_size(lanes: int, cfg: EnvConfig) -> int:
    return 2 * lanes * cfg.bins + 2 * lanes


def obs_head_grid(obs: np.ndarray, lanes: int, cfg: EnvConfig) -> np.ndarray:
    """관측에서 head 격자 (..., lanes, bins) 만 꺼낸다."""
    return obs[..., : lanes * cfg.bins].reshape(*obs.shape[:-1], lanes, cfg.bins)


def observe(engine: JudgeEngine, t: float, cfg: EnvConfig) -> np.ndarray:
    """관측: [head(lanes×bins), hold(lanes×bins), pressed(lanes), next(lanes)] 를 0~1 로 평탄화.

    bin 0 은 판정선에서 good 만큼 지난 시각, 마지막 bin 은 lookahead 끝이다.
    """
    lanes, bins = engine.lanes, cfg.bins
    good = engine.windows.good
    start = t - good
    span = cfg.lookahead_ms + good
    bin_ms = span / bins
    head = np.zeros((lanes, bins), dtype=np.float32)
    hold = np.zeros((lanes, bins), dtype=np.float32)
    notes = engine.chart.notes
    for lane in range(lanes):
        idxs = engine._lane_notes[lane]
        hi = engine.holding[lane]
        if hi is not None:
            n = notes[hi]
            b1 = int(min(bins - 1, max(0, (n.end - start) // bin_ms)))
            hold[lane, : b1 + 1] = 1.0
        for k in range(engine.head[lane], len(idxs)):
            n = notes[idxs[k]]
            if n.t >= start + span:
                break
            b0 = int(max(0, (n.t - start) // bin_ms))
            head[lane, b0] = 1.0
            if n.type == "hold":
                b1 = int(min(bins - 1, (n.end - start) // bin_ms))
                hold[lane, b0 : b1 + 1] = 1.0
    pressed = np.asarray(engine.pressed, dtype=np.float32)
    # 레인별 다음 노트까지 남은 시간 (판정선 기준, 0 = good 만큼 지남, 1 = lookahead 이상)
    nxt = np.ones(lanes, dtype=np.float32)
    for lane in range(lanes):
        n = engine.next_note(lane)
        if n is not None:
            nxt[lane] = min(1.0, max(0.0, (n.t - start) / span))
    return np.concatenate([head.ravel(), hold.ravel(), pressed, nxt])


class RhythmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        chart_source: ChartSource | None = None,
        config: EnvConfig | None = None,
        lanes: int = 4,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.lanes = lanes
        self.chart_source = chart_source or random_chart_source(GeneratorConfig(lanes=lanes))
        obs_dim = obs_size(lanes, self.cfg)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(lanes)
        self.engine: JudgeEngine | None = None
        self.tick = 0

    def time_of(self, tick: int) -> int:
        return round(tick * 1000 / self.cfg.tick_hz)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        chart = (options or {}).get("chart") or self.chart_source(self.np_random)
        if chart.lanes != self.lanes:
            raise ValueError(f"lanes 불일치: env={self.lanes}, chart={chart.lanes}")
        self.engine = JudgeEngine(chart, self.cfg.windows)
        self.tick = 0
        return observe(self.engine, 0, self.cfg), {"chart": chart.id}

    def step(self, action):
        assert self.engine is not None, "reset() 먼저 호출"
        eng = self.engine
        self.tick += 1
        t = self.time_of(self.tick)
        events = eng.advance(t)
        act = np.asarray(action).astype(bool)
        for lane in range(self.lanes):
            if act[lane] and not eng.pressed[lane]:
                events += eng.press(lane, t)
            elif not act[lane] and eng.pressed[lane]:
                events += eng.release(lane, t)
        terminated = t >= eng.end_time
        if terminated:
            events += eng.finish()
        reward = float(sum(self.cfg.reward.of(e) for e in events))
        obs = observe(eng, t, self.cfg)
        info = {"t": t, "events": events}
        if terminated:
            info["stats"] = eng.stats.to_dict()
        return obs, reward, terminated, False, info
