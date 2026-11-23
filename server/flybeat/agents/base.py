"""세 모드 공통 에이전트 인터페이스와 평가 함수."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from flybeat.game.chart import Chart
from flybeat.game.env import EnvConfig, RhythmEnv, fixed_chart_source
from flybeat.game.generator import GeneratorConfig, generate_chart


class Agent(Protocol):
    mode: str
    lanes: int

    def reset(self) -> None:
        """곡을 시작할 때 내부 상태(뇌 활동 등)를 초기화한다."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """관측 하나 → 레인별 누름(0/1) 배열."""

    def activity(self) -> dict[str, Any] | None:
        """시각화용 내부 활동 (없으면 None)."""


@dataclass
class TrainProgress:
    """학습 진행 상황 보고용."""

    step: int
    total: int
    episodes: int
    accuracy: float | None = None
    reward: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "total": self.total,
            "episodes": self.episodes,
            "accuracy": self.accuracy,
            "reward": self.reward,
            **self.extra,
        }


ProgressFn = Callable[[TrainProgress], None]
StopFn = Callable[[], bool]


def run_episode(
    agent: Agent,
    chart: Chart,
    env_cfg: EnvConfig | None = None,
    on_step: Callable[[int, np.ndarray, dict], None] | None = None,
) -> dict[str, Any]:
    """곡 하나를 끝까지 플레이하고 최종 통계를 반환한다."""
    env = RhythmEnv(fixed_chart_source([chart]), env_cfg, lanes=chart.lanes)
    obs, _ = env.reset(seed=0)
    agent.reset()
    done = False
    info: dict = {}
    while not done:
        action = agent.act(obs)
        obs, _, done, _, info = env.step(action)
        if on_step is not None:
            on_step(info["t"], action, info)
    return info["stats"]


def eval_charts(n: int = 4, seed: int = 10_000, cfg: GeneratorConfig | None = None) -> list[Chart]:
    """학습에 쓰지 않는 고정 seed 의 검증용 채보."""
    return [generate_chart(seed + i, cfg) for i in range(n)]


def evaluate(agent: Agent, charts: list[Chart], env_cfg: EnvConfig | None = None) -> dict:
    stats = [run_episode(agent, c, env_cfg) for c in charts]
    acc = [s["accuracy"] for s in stats]
    return {
        "accuracy": float(np.mean(acc)),
        "perCharts": [{"chart": c.id, **s} for c, s in zip(charts, stats, strict=True)],
    }
