"""음악적인 4레인 채보 생성 (autochart 2세대).

분석 결과(Analysis: 템포·박 위치·대역별 강세·멜로디 대역 밝기)로 채보를 짠다.

리듬
- 16분음표 격자, 4/4 마디. 저음 강세가 몰리는 박을 마디 첫 박(다운비트)으로 잡는다.
- 칸 점수 = 박 위치 가중치 × (킥·스네어·멜로디·하이햇 강세의 가중 합).
- 난이도별로 쓸 수 있는 칸: 쉬움 4분, 보통 8분, 어려움 16분.
- 마디마다 곡의 에너지에 비례해 노트 수 상한을 정하고(조용하면 적게, 신나면 많게),
  실제로 두드러진 강세가 있는 칸 중 점수가 높은 칸만 고른다 (상한을 채우려고 약한 소리에 놓지 않음).

레인 (맛)
- 멜로디 노트: 멜로디 대역이 밝아지면(음이 올라가면) 오른쪽, 어두워지면 왼쪽으로 한 칸씩 움직인다.
  비슷하면 옆 레인과 번갈아 치는 트릴. 16분 연속(어려움)은 같은 방향으로 이어지는 계단(롤).
- 드럼만 있는 칸: 킥은 바깥 레인(0·3), 스네어는 안쪽 레인(1·2)을 번갈아.
- 강한 다운비트에는 좌우 대칭 동시치기.
- 같은 레인 연타는 최소 간격보다 가깝게 두지 않는다.

구조
- 앞에서 나온 마디와 소리가 거의 같은 마디는 그 마디의 패턴을 그대로 다시 쓴다
  (어려움은 2번째 반복마다 좌우 반전).
- 멜로디가 길게 이어지고 뒤가 1박 이상 비면 롱노트.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flymax.game.chart import Chart, Note, chart_from_dict

LANES = 4
SLOTS_PER_BEAT = 4
SLOTS_PER_BAR = 16


@dataclass(frozen=True)
class Difficulty:
    name: str
    step: int  # 쓸 수 있는 칸 간격 (16분 단위): 4=4분, 2=8분, 1=16분
    nps: float  # 목표 초당 노트 수 (곡 평균)
    chord_top: float  # 다운비트 강세 상위 이 비율에 동시치기 (0 = 없음)
    hold_beats: float  # 롱노트가 되려면 뒤가 비어야 하는 박 수
    hold_ratio: float  # 롱노트 최대 비율
    min_gap_ms: int  # 같은 레인 최소 간격
    min_q: float  # 곡 전체 강세 분위수 기준: 이보다 약한 칸에는 놓지 않음
    rel: float  # 마디 안 가장 센 칸 대비 최소 비율


DIFFICULTIES = {
    "easy": Difficulty("easy", 4, 1.3, 0.0, 2.0, 0.12, 300, 0.35, 0.35),
    "normal": Difficulty("normal", 2, 2.4, 0.25, 1.0, 0.12, 200, 0.22, 0.25),
    "hard": Difficulty("hard", 1, 4.2, 0.35, 1.0, 0.10, 120, 0.12, 0.15),
}

POS_WEIGHT = {0: 1.0, 4: 0.85, 8: 0.92, 12: 0.85, 2: 0.6, 6: 0.6, 10: 0.6, 14: 0.6}
W = {"low": 0.9, "mid": 0.8, "mel": 1.0, "high": 0.25}


@dataclass
class Slot:
    i: int  # 곡 전체 16분 칸 번호 (마디 시작 = 0)
    t_ms: int
    pos: int  # 마디 안 위치 0~15
    low: float
    mid: float
    mel: float
    high: float
    bright: float  # 멜로디 대역 밝기 (로그 Hz)
    energy: float  # 멜로디 대역 로그 에너지
    score: float = 0.0


def _features(a, t_s: float) -> tuple[float, ...]:
    f = a.frame_of(t_s)
    lo, hi = int(max(0, f - 2)), int(min(len(a.onset), f + 3))
    if hi <= lo:
        return (0.0,) * 6
    b = a.bands
    # 밝기·에너지는 강세 직후(약 30ms)의 안정된 소리로 본다
    k = int(min(len(a.onset) - 1, max(0, f + 0.03 / a.frame_s)))
    return (
        float(b["low"][lo:hi].max()),
        float(b["mid"][lo:hi].max()),
        float(b["mel"][lo:hi].max()),
        float(b["high"][lo:hi].max()),
        float(np.log2(max(a.mel_centroid[k], 1.0))),
        float(a.mel_energy[k]),
    )


def _slots(a) -> tuple[list[Slot], int]:
    beat = 60.0 / a.bpm
    slot_s = beat / SLOTS_PER_BEAT
    # 다운비트: 저음 강세가 가장 몰리는 박 위상 (0~3 박)
    beats = np.arange(a.first_beat_s, a.duration_s - 0.2, beat)
    low_at = np.array([_features(a, t)[0] for t in beats])
    phase = int(np.argmax([low_at[o::4].sum() for o in range(4)])) if len(beats) >= 8 else 0
    start = a.first_beat_s + phase * beat
    # 곡 시작 앞쪽의 박도 쓰도록 한 마디씩 앞으로 당긴다 (1초 이전은 비움)
    while start - 4 * beat >= 1.0 - 1e-6:
        start -= 4 * beat
    slots = []
    i = 0
    t = start
    while t < a.duration_s - 0.4:
        if t >= 1.0:
            lo, mi, me, hi, br, en = _features(a, t)
            slots.append(Slot(i, int(round(t * 1000)), i % SLOTS_PER_BAR, lo, mi, me, hi, br, en))
        i += 1
        t = start + i * slot_s
    return slots, phase


def _bar_signature(bar: list[Slot]) -> np.ndarray:
    v = np.zeros(SLOTS_PER_BAR * 3)
    for s in bar:
        v[s.pos * 3 : s.pos * 3 + 3] = (s.low, s.mid, s.mel)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class _LanePicker:
    """레인 선택 상태 (직전 레인·방향·레인별 사용 가능 시각)."""

    def __init__(self, min_gap_ms: int) -> None:
        self.gap = min_gap_ms
        self.free = [-(10**9)] * LANES  # 레인이 다시 비는 시각
        self.last = 1
        self.dir = 1
        self.prev_bright: float | None = None
        self.prev_t = -(10**9)
        self.alt = {"kick": 0, "snare": 0}

    def ok(self, lane: int, t: int) -> bool:
        return 0 <= lane < LANES and t - self.free[lane] >= self.gap

    def nearest_ok(self, want: int, t: int, prefer_dir: int) -> int | None:
        for d in (0, prefer_dir, -prefer_dir, 2 * prefer_dir, -2 * prefer_dir, 3, -3):
            if self.ok(want + d, t):
                return want + d
        return None

    def melodic(self, s: Slot, slot_ms: float, run: bool) -> int | None:
        if self.prev_bright is None:
            move = 0
        else:
            delta = s.bright - self.prev_bright  # 옥타브 단위
            move = 1 if delta > 0.05 else -1 if delta < -0.05 else 0
        if run:  # 16분 연속: 같은 방향으로 계단
            move = self.dir
        if move == 0:  # 비슷한 높이: 트릴 (옆 레인과 번갈아)
            want = self.last + self.dir
        else:
            self.dir = move
            want = self.last + move
        if want < 0 or want >= LANES:  # 끝에서 튕기기
            self.dir = -self.dir
            want = self.last + self.dir
        lane = self.nearest_ok(want, s.t_ms, self.dir)
        self.prev_bright = s.bright
        return lane

    def drum(self, s: Slot) -> int | None:
        kind = "kick" if s.low >= s.mid else "snare"
        pair = (0, 3) if kind == "kick" else (1, 2)
        k = self.alt[kind]
        want = pair[k % 2]
        if want == self.last:
            want = pair[(k + 1) % 2]
        self.alt[kind] = k + 1
        return self.nearest_ok(want, s.t_ms, 1 if want < 2 else -1)

    def use(self, lane: int, t: int, end: int | None = None) -> None:
        self.free[lane] = end if end is not None else t
        self.last = lane
        self.prev_t = t


def make_chart(
    a,
    diff: Difficulty,
    chart_id: str,
    title: str,
    artist: str,
    audio: str | None,
) -> Chart:
    slots, _ = _slots(a)
    if not slots:
        raise ValueError("곡이 너무 짧습니다")
    beat_s = 60.0 / a.bpm
    slot_ms = beat_s * 1000 / SLOTS_PER_BEAT
    for s in slots:
        pw = POS_WEIGHT.get(s.pos, 0.35)
        s.score = pw * (W["low"] * s.low + W["mid"] * s.mid + W["mel"] * s.mel + W["high"] * s.high)

    # 마디로 묶기
    bars: dict[int, list[Slot]] = {}
    for s in slots:
        bars.setdefault(s.i // SLOTS_PER_BAR, []).append(s)
    bar_ids = sorted(bars)
    energies = np.array([np.mean([s.energy for s in bars[b]]) for b in bar_ids])
    e_lo, e_hi = np.percentile(energies, 15), np.percentile(energies, 85)
    k_base = diff.nps * 4 * beat_s  # 마디당 목표 노트 수
    chord_thr = (
        np.quantile([s.low + s.mid for s in slots if s.pos == 0], 1 - diff.chord_top)
        if diff.chord_top > 0 and any(s.pos == 0 for s in slots)
        else np.inf
    )
    silence = np.percentile([s.score for s in slots], 20)
    # 약한 소리에는 노트를 두지 않는다: 강세(대역 합)가 곡 전체 기준 이 값 이상인 칸만 후보
    strength = np.array([max(s.low, s.mid, s.mel, s.high) for s in slots if s.pos % diff.step == 0])
    min_strength = float(np.quantile(strength, diff.min_q)) if len(strength) else 0.0

    picker = _LanePicker(diff.min_gap_ms)
    notes: list[Note] = []
    history: list[tuple[np.ndarray, list[tuple[int, list[int]]]]] = []  # (서명, [(pos, lanes)])
    repeats: dict[int, int] = {}

    for bi, b in enumerate(bar_ids):
        bar = bars[b]
        cand = [s for s in bar if s.pos % diff.step == 0]
        if not cand or max(s.score for s in cand) <= silence:
            continue
        top = max(s.score for s in cand)
        # 마디에서 두드러진 강세만 (상대 0.35 이상, 절대 min_strength 이상). 다운비트는 항상 후보.
        cand = [
            s
            for s in cand
            if (s.score >= diff.rel * top and max(s.low, s.mid, s.mel, s.high) >= min_strength)
            or (s.pos == 0 and s.score >= 0.5 * top)
        ]
        if not cand:
            continue
        f = np.clip((energies[bi] - e_lo) / max(e_hi - e_lo, 1e-9), 0, 1)
        k = int(round(k_base * (0.5 + 0.8 * f)))
        k = max(1, min(k, len(cand)))
        sig = _bar_signature(bar)

        # 앞에서 나온 거의 같은 마디 → 같은 패턴
        reuse = None
        for hi_, (hsig, pattern) in enumerate(history):
            if float(sig @ hsig) > 0.93 and abs(len(pattern) - k) <= max(1, k // 4):
                reuse = (hi_, pattern)
                break
        by_pos = {s.pos: s for s in bar}
        placed: list[tuple[int, list[int]]] = []

        if reuse is not None:
            hi_, pattern = reuse
            repeats[hi_] = repeats.get(hi_, 0) + 1
            mirror = diff.name == "hard" and repeats[hi_] % 2 == 0
            for pos, lanes in pattern:
                s = by_pos.get(pos)
                if s is None:
                    continue
                got = []
                for ln in lanes:
                    ln = LANES - 1 - ln if mirror else ln
                    ln2 = ln if picker.ok(ln, s.t_ms) else picker.nearest_ok(ln, s.t_ms, 1)
                    if ln2 is None:
                        continue
                    notes.append(Note(s.t_ms, ln2, "tap", 0))
                    picker.use(ln2, s.t_ms)
                    got.append(ln2)
                if got:
                    placed.append((pos, got))
            history.append((sig, placed))
            continue

        chosen = sorted(sorted(cand, key=lambda s: -s.score)[:k], key=lambda s: s.i)
        prev_i = None
        for s in chosen:
            run = diff.step == 1 and prev_i is not None and s.i - prev_i == 1
            prev_i = s.i
            drum_only = s.mel < 0.35 * max(s.low, s.mid) and not run
            lane = picker.drum(s) if drum_only else picker.melodic(s, slot_ms, run)
            if lane is None:
                continue
            notes.append(Note(s.t_ms, lane, "tap", 0))
            picker.use(lane, s.t_ms)
            lanes = [lane]
            if s.pos == 0 and s.low + s.mid >= chord_thr:
                other = LANES - 1 - lane
                if other != lane and picker.ok(other, s.t_ms):
                    notes.append(Note(s.t_ms, other, "tap", 0))
                    picker.free[other] = s.t_ms
                    lanes.append(other)
            placed.append((s.pos, lanes))
        history.append((sig, placed))

    notes = _add_holds(notes, a, diff, slot_ms)
    notes.sort(key=lambda n: (n.t, n.lane))
    chart = Chart(
        id=chart_id,
        title=title,
        bpm=round(a.bpm, 2),
        offset_ms=int(round(a.first_beat_s * 1000)),
        lanes=LANES,
        notes=tuple(notes),
        artist=artist,
        audio=audio,
    )
    chart_from_dict(chart.to_dict())  # 스키마·겹침 검증
    return chart


def _add_holds(notes: list[Note], a, diff: Difficulty, slot_ms: float) -> list[Note]:
    """뒤가 비고 멜로디가 이어지는 단독 노트를 롱노트로 (최대 hold_ratio 비율)."""
    if diff.hold_ratio <= 0 or not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.t, n.lane))
    times = sorted({n.t for n in notes})
    count_at: dict[int, int] = {}
    for n in notes:
        count_at[n.t] = count_at.get(n.t, 0) + 1
    beat_ms = slot_ms * SLOTS_PER_BEAT
    cands = []
    for idx, t in enumerate(times[:-1]):
        nxt = times[idx + 1]
        if count_at[t] != 1 or nxt - t < diff.hold_beats * beat_ms:
            continue
        f0 = int(max(0, a.frame_of(t / 1000)))
        f1 = int(a.frame_of((nxt - slot_ms) / 1000))
        if f1 <= f0 + 2:
            continue
        e = a.mel_energy[f0:f1]
        sustain = float(e.min() - e[0])  # 0 에 가까울수록 소리가 이어짐
        if sustain > -0.8:
            cands.append((sustain, t, nxt))
    limit = int(len(notes) * diff.hold_ratio)
    hold_at = {t: nxt for _, t, nxt in sorted(cands, reverse=True)[:limit]}
    out = []
    for n in notes:
        if n.t in hold_at:
            dur = int(round(hold_at[n.t] - slot_ms - n.t))
            if dur >= 150:
                out.append(Note(n.t, n.lane, "hold", dur))
                continue
        out.append(n)
    return out


def chart_quality(chart: Chart, a) -> dict:
    """채보 품질 지표 (사람이 맛을 판단하는 데 참고)."""
    notes = chart.notes
    if not notes:
        return {}
    dur = max(chart.duration_ms, 1) / 1000
    lanes = np.bincount([n.lane for n in notes], minlength=LANES)
    p = lanes / lanes.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 2)  # 1 = 완전 균등
    by_t = sorted(notes, key=lambda n: (n.t, n.lane))
    jacks = sum(
        1 for x, y in zip(by_t, by_t[1:], strict=False) if x.lane == y.lane and y.t - x.end < 250
    )
    onset = a.onset
    strong = np.percentile(onset, 70)
    on_beat = np.mean(
        [
            onset[int(max(0, a.frame_of(n.t / 1000) - 2)) : int(a.frame_of(n.t / 1000) + 3)].max()
            >= strong
            for n in notes
        ]
    )
    moves = [abs(y.lane - x.lane) for x, y in zip(by_t, by_t[1:], strict=False) if y.t != x.t]
    return {
        "notes": len(notes),
        "nps": round(len(notes) / dur, 2),
        "lane_balance": round(entropy, 2),
        "jack_ratio": round(jacks / len(notes), 3),
        "on_strong_onset": round(float(on_beat), 2),
        "step_moves": round(float(np.mean([m == 1 for m in moves])) if moves else 0, 2),
        "holds": sum(n.type == "hold" for n in notes),
        "chords": len(notes) - len({n.t for n in notes}),
    }
