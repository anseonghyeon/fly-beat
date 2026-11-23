"""채보(chart) 모델, 검증, 로딩."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import jsonschema

from flybeat import paths

CHART_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_CHART_BYTES = 2 * 1024 * 1024

NoteType = Literal["tap", "hold"]


class ChartError(ValueError):
    """채보 형식 오류."""


@dataclass(frozen=True, slots=True)
class Note:
    t: int
    lane: int
    type: NoteType = "tap"
    dur: int = 0

    @property
    def end(self) -> int:
        return self.t + self.dur if self.type == "hold" else self.t

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"t": self.t, "lane": self.lane, "type": self.type}
        if self.type == "hold":
            d["dur"] = self.dur
        return d


@dataclass(frozen=True, slots=True)
class Chart:
    id: str
    title: str
    bpm: float
    offset_ms: int
    lanes: int
    notes: tuple[Note, ...]
    artist: str = ""
    audio: str | None = None
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def duration_ms(self) -> int:
        return max((n.end for n in self.notes), default=0)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "bpm": self.bpm,
            "offsetMs": self.offset_ms,
            "lanes": self.lanes,
            "notes": [n.to_dict() for n in self.notes],
        }
        if self.artist:
            d["artist"] = self.artist
        if self.audio:
            d["audio"] = self.audio
        return d

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "bpm": self.bpm,
            "lanes": self.lanes,
            "notes": len(self.notes),
            "durationMs": self.duration_ms,
        }


@cache
def _schema() -> dict[str, Any]:
    text = resources.files("flybeat.game").joinpath("chart_schema.json").read_text("utf-8")
    return json.loads(text)


def chart_from_dict(data: Any) -> Chart:
    """JSON 객체를 검증하고 Chart로 바꾼다. 문제가 있으면 ChartError."""
    try:
        jsonschema.validate(data, _schema())
    except jsonschema.ValidationError as e:
        raise ChartError(f"채보 스키마 오류: {e.message}") from None

    lanes = data["lanes"]
    notes: list[Note] = []
    for raw in data["notes"]:
        if raw["lane"] >= lanes:
            raise ChartError(f"lane {raw['lane']} 이 lanes={lanes} 범위를 벗어납니다")
        if raw["type"] == "hold":
            if "dur" not in raw:
                raise ChartError("hold 노트에는 dur 가 필요합니다")
            notes.append(Note(raw["t"], raw["lane"], "hold", raw["dur"]))
        else:
            if "dur" in raw:
                raise ChartError("tap 노트에는 dur 를 쓸 수 없습니다")
            notes.append(Note(raw["t"], raw["lane"], "tap", 0))

    notes.sort(key=lambda n: (n.t, n.lane))
    last_end: dict[int, int] = {}
    for n in notes:
        prev = last_end.get(n.lane)
        if prev is not None and n.t <= prev:
            raise ChartError(f"lane {n.lane} 에서 노트가 겹칩니다 (t={n.t})")
        last_end[n.lane] = n.end

    return Chart(
        id=data["id"],
        title=data["title"],
        bpm=float(data["bpm"]),
        offset_ms=data["offsetMs"],
        lanes=lanes,
        notes=tuple(notes),
        artist=data.get("artist", ""),
        audio=data.get("audio"),
    )


def chart_path(chart_id: str, charts_dir: Path | None = None) -> Path:
    """ID로만 경로를 만든다. 경로 탈출을 막기 위해 ID 형식을 먼저 검사한다."""
    if not CHART_ID_RE.fullmatch(chart_id):
        raise ChartError("잘못된 채보 ID")
    base = (charts_dir or paths.CHARTS_DIR).resolve()
    p = (base / f"{chart_id}.json").resolve()
    if p.parent != base:
        raise ChartError("잘못된 채보 ID")
    return p


def load_chart(chart_id: str, charts_dir: Path | None = None) -> Chart:
    p = chart_path(chart_id, charts_dir)
    if not p.is_file():
        raise FileNotFoundError(chart_id)
    if p.stat().st_size > MAX_CHART_BYTES:
        raise ChartError("채보 파일이 너무 큽니다")
    data = json.loads(p.read_text("utf-8"))
    chart = chart_from_dict(data)
    if chart.id != chart_id:
        raise ChartError("파일 이름과 채보 ID가 다릅니다")
    return chart


def list_charts(charts_dir: Path | None = None) -> list[Chart]:
    base = (charts_dir or paths.CHARTS_DIR).resolve()
    if not base.is_dir():
        return []
    out: list[Chart] = []
    for p in sorted(base.glob("*.json")):
        try:
            out.append(load_chart(p.stem, base))
        except (ChartError, FileNotFoundError, json.JSONDecodeError):
            continue
    return out
