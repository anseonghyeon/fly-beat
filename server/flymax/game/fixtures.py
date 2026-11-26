"""판정 골든 테스트 픽스처를 만든다.

사용법: python -m flymax.game.fixtures
- tests/fixtures/judge/*.json : Python·TS 엔진 공용 판정 테스트
"""

from __future__ import annotations

import json

import numpy as np

from flymax import paths
from flymax.game.chart import Chart, Note
from flymax.game.engine import run_inputs
from flymax.game.generator import GeneratorConfig, generate_chart

FIXTURE_DIR = paths.REPO_ROOT / "tests" / "fixtures" / "judge"


def _chart(notes: list[Note], cid: str = "fixture") -> Chart:
    return Chart(cid, "fixture", 120.0, 0, 4, tuple(sorted(notes, key=lambda n: (n.t, n.lane))))


def _case(name: str, chart: Chart, inputs: list[dict]) -> dict:
    inputs = sorted(inputs, key=lambda x: x["t"])  # 안정 정렬: 같은 시각이면 적은 순서 유지
    events, stats = run_inputs(chart, inputs)
    return {
        "name": name,
        "chart": chart.to_dict(),
        "inputs": inputs,
        "expected": {"events": [e.to_dict() for e in events], "stats": stats.to_dict()},
    }


def _down(t: int, lane: int) -> dict:
    return {"t": t, "lane": lane, "down": True}


def _up(t: int, lane: int) -> dict:
    return {"t": t, "lane": lane, "down": False}


def _random_inputs(chart: Chart, seed: int) -> list[dict]:
    """노트 근처를 노이즈 섞어 누르는 가상 플레이어."""
    rng = np.random.default_rng(seed)
    inputs: list[dict] = []
    for n in chart.notes:
        if rng.random() < 0.1:
            continue  # 일부러 놓침
        t = int(n.t + rng.normal(0, 60))
        inputs.append(_down(t, n.lane))
        up = n.end + int(rng.normal(0, 80)) if n.type == "hold" else t + 60
        inputs.append(_up(max(up, t + 1), n.lane))
        if rng.random() < 0.05:
            et = t + int(rng.integers(200, 400))
            inputs += [_down(et, int(rng.integers(0, 4))), _up(et + 30, int(rng.integers(0, 4)))]
    inputs.sort(key=lambda x: (x["t"], x["lane"], not x["down"]))
    # 같은 레인에서 down/up 이 번갈아 나오도록 정리
    state = [False] * chart.lanes
    cleaned = []
    for x in inputs:
        if x["down"] != state[x["lane"]]:
            cleaned.append(x)
            state[x["lane"]] = x["down"]
    return cleaned


def build_fixtures() -> list[dict]:
    cases = [
        _case(
            "tap-windows",
            _chart([Note(1000, 0), Note(1000, 1), Note(1000, 2), Note(1000, 3), Note(2000, 0)]),
            [
                _down(1000, 0),
                _up(1050, 0),  # PERFECT (0)
                _down(940, 1),
                _up(990, 1),  # GREAT (-60)
                _down(1100, 2),
                _up(1150, 2),  # GOOD (+100)
                _down(1121, 3),
                _up(1150, 3),  # 범위 밖 → EMPTY, 이후 MISS
                _down(1500, 0),
                _up(1550, 0),  # 너무 이름 → EMPTY
                _down(1960, 0),
                _up(2000, 0),  # PERFECT (-40 경계)
            ],
        ),
        _case(
            "hold-notes",
            _chart(
                [Note(1000, 0, "hold", 500), Note(1000, 1, "hold", 500), Note(1000, 2, "hold", 500)]
            ),
            [
                _down(1010, 0),  # 끝까지 누름 → 꼬리 PERFECT
                _down(990, 1),
                _up(1430, 1),  # 70ms 일찍 뗌 → 꼬리 GREAT
                _down(1000, 2),
                _up(1200, 2),  # 300ms 일찍 뗌 → 꼬리 MISS
                _up(1600, 0),
            ],
        ),
        _case("all-miss", _chart([Note(500, 0), Note(800, 3, "hold", 300)]), []),
    ]
    for seed in (1, 2, 3):
        chart = generate_chart(seed, GeneratorConfig(length_beats=32, hold_prob=0.2))
        cases.append(_case(f"random-{seed}", chart, _random_inputs(chart, seed)))
    return cases


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for case in build_fixtures():
        p = FIXTURE_DIR / f"{case['name']}.json"
        p.write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print("fixture", p.relative_to(paths.REPO_ROOT))


if __name__ == "__main__":
    main()
