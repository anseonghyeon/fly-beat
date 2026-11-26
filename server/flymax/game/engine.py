"""리듬게임 판정 엔진 (기준 구현).

web/src/game/engine.ts 가 이 파일과 같은 규칙을 따른다.
두 구현이 같은 결과를 내는지 tests/fixtures/judge/*.json 으로 검증한다.

규칙 요약
- 키를 누르면(press) 해당 레인의 다음 미판정 노트와 시간 차이를 비교해 판정한다.
  |차이| <= good 이 아니면 EMPTY(헛누름)이고 노트는 그대로 남는다.
- 노트 시각 + good 이 지나도록 누르지 않으면 MISS (롱노트는 머리·꼬리 모두 MISS).
- 롱노트는 머리를 친 뒤 끝 시각까지 누르고 있으면 꼬리 PERFECT.
  일찍 떼면 남은 시간으로 꼬리를 판정한다 (good 보다 많이 남으면 MISS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from flymax.game.chart import Chart, Note

Judgment = Literal["PERFECT", "GREAT", "GOOD", "MISS", "EMPTY"]
Part = Literal["head", "tail", "empty"]

JUDGMENTS: tuple[Judgment, ...] = ("PERFECT", "GREAT", "GOOD", "MISS")
SCORE_WEIGHT: dict[str, float] = {"PERFECT": 1.0, "GREAT": 0.7, "GOOD": 0.3, "MISS": 0.0}


@dataclass(frozen=True)
class JudgeWindows:
    perfect: int = 40
    great: int = 80
    good: int = 120

    def judge(self, diff: float) -> Judgment:
        a = abs(diff)
        if a <= self.perfect:
            return "PERFECT"
        if a <= self.great:
            return "GREAT"
        if a <= self.good:
            return "GOOD"
        return "MISS"


@dataclass(frozen=True, slots=True)
class JudgeEvent:
    t: int
    lane: int
    note: int  # chart.notes 인덱스, EMPTY 는 -1
    part: Part
    judgment: Judgment
    offset: int = 0  # 입력 시각 - 기준 시각 (head: 노트 시각, tail: 끝 시각)

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "lane": self.lane,
            "note": self.note,
            "part": self.part,
            "judgment": self.judgment,
            "offset": self.offset,
        }


@dataclass
class Stats:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(JUDGMENTS, 0))
    empty: int = 0
    combo: int = 0
    max_combo: int = 0
    score: float = 0.0
    judged: int = 0
    total: int = 0  # 판정 대상 개수 (tap 1, hold 2)

    @property
    def accuracy(self) -> float:
        """지금까지 판정된 것 기준 정확도(%)."""
        return 100.0 * self.score / self.judged if self.judged else 0.0

    def to_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "empty": self.empty,
            "combo": self.combo,
            "maxCombo": self.max_combo,
            "score": round(self.score, 6),
            "judged": self.judged,
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
        }


class JudgeEngine:
    def __init__(self, chart: Chart, windows: JudgeWindows | None = None) -> None:
        self.chart = chart
        self.windows = windows or JudgeWindows()
        self.lanes = chart.lanes
        self._lane_notes: list[list[int]] = [[] for _ in range(self.lanes)]
        for i, n in enumerate(chart.notes):
            self._lane_notes[n.lane].append(i)
        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self) -> None:
        self.head = [0] * self.lanes
        self.holding: list[int | None] = [None] * self.lanes
        self.pressed = [False] * self.lanes
        self.now = -math.inf
        self.stats = Stats(total=sum(2 if n.type == "hold" else 1 for n in self.chart.notes))
        self.finished = False

    def next_note(self, lane: int) -> Note | None:
        idxs = self._lane_notes[lane]
        h = self.head[lane]
        return self.chart.notes[idxs[h]] if h < len(idxs) else None

    @property
    def end_time(self) -> int:
        return self.chart.duration_ms + self.windows.good + 1

    # ------------------------------------------------------------------ 내부
    def _emit(self, out: list[JudgeEvent], ev: JudgeEvent) -> None:
        s = self.stats
        if ev.judgment == "EMPTY":
            s.empty += 1
        else:
            s.counts[ev.judgment] += 1
            s.judged += 1
            s.score += SCORE_WEIGHT[ev.judgment]
            if ev.judgment == "MISS":
                s.combo = 0
            else:
                s.combo += 1
                s.max_combo = max(s.max_combo, s.combo)
        out.append(ev)

    # ------------------------------------------------------------------ API
    def advance(self, t: float) -> list[JudgeEvent]:
        """시각 t 까지 시간을 흘려 MISS 와 롱노트 꼬리 자동 판정을 처리한다."""
        if t < self.now:
            raise ValueError(f"시간은 거꾸로 갈 수 없습니다: {t} < {self.now}")
        out: list[JudgeEvent] = []
        good = self.windows.good
        for lane in range(self.lanes):
            idxs = self._lane_notes[lane]
            while True:
                hi = self.holding[lane]
                if hi is not None:
                    n = self.chart.notes[hi]
                    if n.end <= t:
                        self._emit(out, JudgeEvent(n.end, lane, hi, "tail", "PERFECT", 0))
                        self.holding[lane] = None
                        continue
                    break
                h = self.head[lane]
                if h < len(idxs):
                    ni = idxs[h]
                    n = self.chart.notes[ni]
                    if n.t + good < t:
                        et = n.t + good
                        self._emit(out, JudgeEvent(et, lane, ni, "head", "MISS", 0))
                        if n.type == "hold":
                            self._emit(out, JudgeEvent(et, lane, ni, "tail", "MISS", 0))
                        self.head[lane] = h + 1
                        continue
                break
        self.now = max(self.now, t)
        return out

    def press(self, lane: int, t: int) -> list[JudgeEvent]:
        out = self.advance(t)
        if self.pressed[lane]:
            return out
        self.pressed[lane] = True
        idxs = self._lane_notes[lane]
        h = self.head[lane]
        if self.holding[lane] is None and h < len(idxs):
            ni = idxs[h]
            n = self.chart.notes[ni]
            diff = t - n.t
            if abs(diff) <= self.windows.good:
                self._emit(out, JudgeEvent(t, lane, ni, "head", self.windows.judge(diff), diff))
                self.head[lane] = h + 1
                if n.type == "hold":
                    self.holding[lane] = ni
                return out
        self._emit(out, JudgeEvent(t, lane, -1, "empty", "EMPTY", 0))
        return out

    def release(self, lane: int, t: int) -> list[JudgeEvent]:
        out = self.advance(t)
        if not self.pressed[lane]:
            return out
        self.pressed[lane] = False
        hi = self.holding[lane]
        if hi is not None:
            n = self.chart.notes[hi]
            remaining = n.end - t  # advance 에서 end <= t 는 이미 처리됨 → remaining > 0
            self._emit(
                out, JudgeEvent(t, lane, hi, "tail", self.windows.judge(remaining), -remaining)
            )
            self.holding[lane] = None
        return out

    def finish(self) -> list[JudgeEvent]:
        out = self.advance(max(self.end_time, self.now))
        self.finished = True
        return out


def run_inputs(
    chart: Chart, inputs: list[dict], windows: JudgeWindows | None = None
) -> tuple[list[JudgeEvent], Stats]:
    """입력 목록 [{t, lane, down}] 을 순서대로 넣고 끝까지 진행한다 (테스트·리플레이용)."""
    eng = JudgeEngine(chart, windows)
    events: list[JudgeEvent] = []
    for inp in inputs:
        if inp["down"]:
            events += eng.press(inp["lane"], inp["t"])
        else:
            events += eng.release(inp["lane"], inp["t"])
    events += eng.finish()
    return events, eng.stats
