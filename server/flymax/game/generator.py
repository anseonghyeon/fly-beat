"""학습용 랜덤 채보 생성기. 같은 seed면 항상 같은 채보가 나온다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flymax.game.chart import Chart, Note


@dataclass(frozen=True)
class GeneratorConfig:
    lanes: int = 4
    bpm_min: float = 90.0
    bpm_max: float = 150.0
    length_beats: int = 64
    subdivision: int = 2  # 박자당 칸 수 (2 = 8분음표)
    density: float = 0.45  # 칸마다 노트가 나올 확률
    chord_prob: float = 0.1  # 동시 타격 확률
    hold_prob: float = 0.1
    hold_max_beats: float = 2.0
    lead_in_ms: int = 2000
    min_gap_ms: int = 150  # 같은 레인에서 이전 노트 끝과 다음 노트 사이 최소 간격


def generate_chart(
    seed: int, cfg: GeneratorConfig | None = None, chart_id: str | None = None
) -> Chart:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(seed)
    bpm = float(round(rng.uniform(cfg.bpm_min, cfg.bpm_max)))
    beat_ms = 60000.0 / bpm
    slot_ms = beat_ms / cfg.subdivision
    lane_free_at = [-(10**9)] * cfg.lanes
    notes: list[Note] = []

    for slot in range(cfg.length_beats * cfg.subdivision):
        if rng.random() >= cfg.density:
            continue
        t = int(round(cfg.lead_in_ms + slot * slot_ms))
        k = 2 if (cfg.lanes > 1 and rng.random() < cfg.chord_prob) else 1
        for lane in rng.permutation(cfg.lanes)[:k]:
            lane = int(lane)
            if t - lane_free_at[lane] < cfg.min_gap_ms:
                continue
            if rng.random() < cfg.hold_prob:
                beats = rng.integers(1, int(cfg.hold_max_beats * cfg.subdivision) + 1)
                dur = int(round(beats * slot_ms))
                notes.append(Note(t, lane, "hold", dur))
                lane_free_at[lane] = t + dur
            else:
                notes.append(Note(t, lane, "tap", 0))
                lane_free_at[lane] = t

    notes.sort(key=lambda n: (n.t, n.lane))
    return Chart(
        id=chart_id or f"random-{seed}",
        title=f"Random #{seed}",
        bpm=bpm,
        offset_ms=cfg.lead_in_ms,
        lanes=cfg.lanes,
        notes=tuple(notes),
        artist="generator",
    )
