"""학습된 에이전트 플레이 스트리밍 (WebSocket).

프로토콜
  클라이언트 → {"type": "start", "checkpoint": "<id>", "chart": "<id>"}  또는 "seed": <int>
  서버 → {"type": "ready", "mode", "chart": {...}, "leadMs"}
  서버 → {"type": "frames", "frames": [{"t", "inputs": [[lane, down], ...], "activity"?}, ...]}
  서버 → {"type": "end", "stats": {...}}  (서버 판정 결과, 클라이언트 TS 엔진 결과와 같아야 함)
  서버 → {"type": "error", "message"}

서버는 실제 시간보다 LEAD_MS 만큼 앞서 프레임을 보내고, 클라이언트는 프레임의 t 에 맞춰 재생한다.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from flymax.agents.checkpoint import CheckpointError, load_meta
from flymax.game.chart import Chart, ChartError, load_chart
from flymax.game.env import RhythmEnv, fixed_chart_source
from flymax.game.generator import generate_chart

LEAD_MS = 400
CHUNK_TICKS = 6
ACTIVITY_EVERY = 3


def load_agent(ckpt_id: str):
    meta = load_meta(ckpt_id)
    if meta["mode"] == "A":
        from flymax.agents.deep_rl import PPOAgent

        return PPOAgent.load(ckpt_id)
    if meta["mode"] == "B":
        from flymax.agents.mapping import MappingAgent

        return MappingAgent.load(ckpt_id)
    from flymax.agents.dopamine import DopamineAgent

    return DopamineAgent.load(ckpt_id)


def resolve_chart(msg: dict[str, Any]) -> Chart:
    if "seed" in msg:
        seed = msg["seed"]
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ChartError("seed 범위 오류")
        return generate_chart(seed)
    chart_id = msg.get("chart")
    if not isinstance(chart_id, str):
        raise ChartError("chart 가 필요합니다")
    return load_chart(chart_id)


class PlaySession:
    """에이전트로 곡 하나를 tick 단위로 진행한다 (동기, 스레드에서 호출)."""

    def __init__(self, agent, chart: Chart) -> None:
        self.agent = agent
        self.chart = chart
        self.env = RhythmEnv(fixed_chart_source([chart]), lanes=chart.lanes)
        self.obs, _ = self.env.reset(seed=0)
        agent.reset()
        self.prev = np.zeros(chart.lanes, dtype=np.int8)
        self.done = False
        self.stats: dict[str, Any] | None = None

    def run(self, ticks: int) -> list[dict[str, Any]]:
        frames = []
        for _ in range(ticks):
            if self.done:
                break
            action = np.asarray(self.agent.act(self.obs), dtype=np.int8)
            self.obs, _, self.done, _, info = self.env.step(action)
            changed = [[int(i), bool(action[i])] for i in np.flatnonzero(action != self.prev)]
            self.prev = action
            frame: dict[str, Any] = {"t": info["t"], "inputs": changed}
            if self.env.tick % ACTIVITY_EVERY == 0:
                act = self.agent.activity()
                if act is not None:
                    frame["activity"] = act
            frames.append(frame)
            if self.done:
                self.stats = info["stats"]
        return frames


async def stream(ws, session: PlaySession) -> None:
    start = time.monotonic() * 1000 + LEAD_MS
    await ws.send_json(
        {
            "type": "ready",
            "mode": session.agent.mode,
            "profile": getattr(session.agent, "profile", None),
            "chart": session.chart.to_dict(),
            "leadMs": LEAD_MS,
        }
    )
    while not session.done:
        frames = await asyncio.to_thread(session.run, CHUNK_TICKS)
        if frames:
            await ws.send_json({"type": "frames", "frames": frames})
            # 실제 시간보다 LEAD_MS 이상 앞서지 않게 기다린다
            ahead = frames[-1]["t"] - (time.monotonic() * 1000 - start) - LEAD_MS
            if ahead > 0:
                await asyncio.sleep(ahead / 1000)
    await ws.send_json({"type": "end", "stats": session.stats})


async def open_session(msg: dict[str, Any]) -> PlaySession:
    ckpt = msg.get("checkpoint")
    if not isinstance(ckpt, str):
        raise CheckpointError("checkpoint 가 필요합니다")
    chart = resolve_chart(msg)
    agent = await asyncio.to_thread(load_agent, ckpt)
    if agent.lanes != chart.lanes:
        raise ChartError("레인 수가 다릅니다")
    return PlaySession(agent, chart)
