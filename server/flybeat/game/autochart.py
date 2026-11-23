"""음원 → 4레인 채보 자동 생성.

1. 음원을 모노로 읽고 스펙트럼 변화량(spectral flux)으로 강세(onset) 곡선을 만든다.
2. 강세 곡선으로 BPM 과 첫 박 위치(위상)를 찾는다 (일정한 템포 가정).
3. 난이도별 격자(쉬움 4분, 보통 8분, 어려움 16분 음표) 위에서 강세가 큰 칸에 노트를 놓는다.
4. 레인은 그 순간 소리의 높낮이(스펙트럼 무게중심)로 정한다: 낮은 소리 → 왼쪽, 높은 소리 → 오른쪽.
5. 보통·어려움은 강한 박에 동시치기, 뒤가 비는 강한 노트에 롱노트를 넣는다.

사용법:
  python -m flybeat.game.autochart song.wav --id my-song --title "My Song" --artist "Someone" \\
      [--audio-out charts/audio/my-song.ogg] [--difficulty easy normal hard]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybeat import paths
from flybeat.game.chart import CHART_ID_RE, Chart, Note, chart_from_dict

N_FFT = 2048
HOP = 256


@dataclass(frozen=True)
class Difficulty:
    name: str
    subdivision: int  # 박자당 칸 수
    keep: float  # 강세 상위 몇 % 칸에 노트를 놓을지
    chord_top: float  # 강세 상위 몇 % 에 동시치기 (0 = 없음)
    hold_prob: float  # 조건을 만족하는 노트 중 롱노트 비율
    min_gap_ms: int  # 같은 레인 최소 간격


DIFFICULTIES = {
    "easy": Difficulty("easy", 1, 0.55, 0.0, 0.0, 300),
    "normal": Difficulty("normal", 2, 0.45, 0.06, 0.15, 200),
    "hard": Difficulty("hard", 4, 0.40, 0.12, 0.12, 120),
}


@dataclass
class Analysis:
    sr: int
    duration_s: float
    frame_s: float
    onset: np.ndarray  # (T,) 0~1
    centroid: np.ndarray  # (T,) Hz
    rms: np.ndarray  # (T,)
    bpm: float
    first_beat_s: float
    t0: float = 0.0  # 프레임 i 의 시각 = t0 + i * frame_s

    def frame_of(self, ts: float) -> float:
        return (ts - self.t0) / self.frame_s


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def analyze(y: np.ndarray, sr: int, bpm_range: tuple[float, float] = (80, 190)) -> Analysis:
    n_frames = 1 + max(0, (len(y) - N_FFT) // HOP)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hanning(N_FFT).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    logmag = np.log1p(100 * mag)
    flux = np.maximum(0, np.diff(logmag, axis=0, prepend=logmag[:1])).sum(axis=1)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")  # 지역 평균 제거
    onset = np.maximum(flux, 0)
    onset /= max(np.percentile(onset, 99.5), 1e-9)
    onset = np.clip(onset, 0, 1)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    centroid = (mag * freqs).sum(axis=1) / np.maximum(mag.sum(axis=1), 1e-9)
    rms = np.sqrt((frames**2).mean(axis=1))
    frame_s = HOP / sr

    bpm, first = estimate_tempo(onset, frame_s, bpm_range)
    # 해닝 창에서 강세(에너지 증가)가 가장 크게 잡히는 곳은 창 시작에서 3/4 지점이다.
    # 프레임 시각을 창 시작으로 두면 노트가 실제 소리보다 앞서므로 그만큼 보정한다.
    t0 = 0.75 * N_FFT / sr
    return Analysis(sr, len(y) / sr, frame_s, onset, centroid, rms, bpm, first + t0, t0)


def _comb(onset: np.ndarray, period: float, n_phase: int = 48) -> tuple[float, float]:
    """박자 격자 대비 점수 = (격자 위 강세 평균 − 반 박 위치 강세 평균) 의 위상별 최대값.

    진짜 템포는 대비가 크고, 절반 템포(반 박에도 강세)·두 배 템포(격자 절반이 빈칸)는 작다.
    반환: (점수, 가장 좋은 위상[프레임])
    """
    t = np.arange(len(onset))
    best, best_ph = -np.inf, 0.0
    for ph in np.linspace(0, period, n_phase, endpoint=False):
        on = np.interp(np.arange(ph, len(onset) - 1, period), t, onset).mean()
        off = np.interp(np.arange(ph + period / 2, len(onset) - 1, period), t, onset).mean()
        if on - off > best:
            best, best_ph = on - off, ph
    return float(best), float(best_ph)


def estimate_tempo(
    onset: np.ndarray, frame_s: float, bpm_range: tuple[float, float]
) -> tuple[float, float]:
    """자기상관 후보(+ 2배·절반·3/2·2/3) 중 대비 점수가 큰 것을 고르고,
    ±2% 범위를 0.02 BPM 단위로 다듬는다."""
    o = onset - onset.mean()
    ac = np.correlate(o, o, mode="full")[len(o) - 1 :]
    bpms = np.arange(bpm_range[0], bpm_range[1] + 0.01, 0.1)
    score = np.interp(60.0 / bpms / frame_s, np.arange(len(ac)), ac)
    peaks = [
        i for i in range(1, len(score) - 1) if score[i] >= score[i - 1] and score[i] >= score[i + 1]
    ]
    cands = {float(bpms[i]) for i in sorted(peaks, key=lambda i: -score[i])[:6]}
    cands |= {c * m for c in list(cands) for m in (2.0, 0.5, 1.5, 2 / 3)}
    cands = {c for c in cands if bpm_range[0] <= c <= bpm_range[1]}

    def total(b: float) -> float:
        prior = np.exp(-0.5 * (np.log2(b / 130.0) / 1.2) ** 2)  # 아주 약한 선호
        return _comb(onset, 60.0 / b / frame_s, 24)[0] * prior

    coarse = max(cands, key=total)
    fine = np.arange(coarse * 0.98, coarse * 1.02, 0.02)
    bpm = float(max(fine, key=lambda b: _comb(onset, 60.0 / b / frame_s, 24)[0]))
    _, ph = _comb(onset, 60.0 / bpm / frame_s, 96)
    return round(bpm, 2), ph * frame_s


def make_chart(
    a: Analysis,
    diff: Difficulty,
    chart_id: str,
    title: str,
    artist: str,
    audio: str | None,
    lanes: int = 4,
    seed: int = 0,
) -> Chart:
    rng = np.random.default_rng(seed)
    beat_s = 60.0 / a.bpm
    slot_s = beat_s / diff.subdivision
    t_frames = np.arange(len(a.onset))
    slots = np.arange(a.first_beat_s, a.duration_s - 0.5, slot_s)
    slots = slots[slots >= 1.0]  # 시작 1초는 비워 둔다

    def strength(ts: float) -> float:
        f = a.frame_of(ts)
        lo, hi = int(max(0, f - 2)), int(min(len(a.onset), f + 3))
        return float(a.onset[lo:hi].max()) if hi > lo else 0.0

    s = np.array([strength(ts) for ts in slots])
    if len(s) == 0:
        raise ValueError("곡이 너무 짧습니다")
    thr = np.quantile(s, 1 - diff.keep)
    chosen = np.flatnonzero((s >= thr) & (s > 0.05))
    chord_thr = np.quantile(s, 1 - diff.chord_top) if diff.chord_top > 0 else np.inf

    # 레인: 스펙트럼 무게중심 분위수로 낮은 소리 → 왼쪽
    cents = np.interp(a.frame_of(slots[chosen]), t_frames, a.centroid)
    edges = np.quantile(cents, np.linspace(0, 1, lanes + 1)[1:-1]) if len(cents) else []
    lane_free = [-1e9] * lanes  # 레인이 다시 비는 시각 (ms)
    notes: list[Note] = []
    gap = diff.min_gap_ms

    def place(t_ms: int, want: int) -> int | None:
        for d in (0, 1, -1, 2, -2, 3, -3):
            ln = want + d
            if 0 <= ln < lanes and t_ms - lane_free[ln] >= gap:
                return ln
        return None

    for k, ci in enumerate(chosen):
        t_ms = int(round(slots[ci] * 1000))
        want = int(np.searchsorted(edges, cents[k]))
        lane = place(t_ms, want)
        if lane is None:
            continue
        # 롱노트: 다음 선택 칸까지 1박 이상 비고, 소리가 이어질 때
        nxt = slots[chosen[k + 1]] if k + 1 < len(chosen) else a.duration_s
        f0 = max(0, int(a.frame_of(slots[ci])))
        f1 = int(a.frame_of(min(nxt, a.duration_s - 0.2)))
        sustained = f1 > f0 and a.rms[f0:f1].min() > 0.5 * a.rms[f0]
        if (
            diff.hold_prob > 0
            and nxt - slots[ci] >= beat_s * 1.0
            and sustained
            and rng.random() < diff.hold_prob * 3
        ):
            dur = int(round((nxt - slots[ci] - slot_s) * 1000))
            notes.append(Note(t_ms, lane, "hold", max(dur, 100)))
            lane_free[lane] = t_ms + max(dur, 100)
        else:
            notes.append(Note(t_ms, lane, "tap", 0))
            lane_free[lane] = t_ms
        if s[ci] >= chord_thr:
            other = place(t_ms, lanes - 1 - lane)
            if other is not None and other != lane:
                notes.append(Note(t_ms, other, "tap", 0))
                lane_free[other] = t_ms

    notes.sort(key=lambda n: (n.t, n.lane))
    chart = Chart(
        id=chart_id,
        title=title,
        bpm=round(a.bpm, 2),
        offset_ms=int(round(a.first_beat_s * 1000)),
        lanes=lanes,
        notes=tuple(notes),
        artist=artist,
        audio=audio,
    )
    chart_from_dict(chart.to_dict())  # 스키마·겹침 검증
    return chart


def encode_ogg(y_path: Path, out: Path, quality: float = 0.4) -> None:
    """WAV 등 → OGG Vorbis (저장소 용량 절약)."""
    import soundfile as sf

    data, sr = sf.read(str(y_path), dtype="float32", always_2d=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        str(out),
        "w",
        samplerate=sr,
        channels=data.shape[1],
        format="OGG",
        subtype="VORBIS",
        compression_level=1.0 - quality,
    ) as f:
        f.write(data)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m flybeat.game.autochart")
    p.add_argument("audio", type=Path)
    p.add_argument("--id", required=True, help="채보 ID 접두사 (예: my-song → my-song-easy)")
    p.add_argument("--title", required=True)
    p.add_argument("--artist", default="")
    p.add_argument("--audio-out", type=Path, help="OGG 로 변환해 저장할 경로 (charts/audio/*.ogg)")
    p.add_argument(
        "--difficulty", nargs="+", default=list(DIFFICULTIES), choices=list(DIFFICULTIES)
    )
    p.add_argument("--out-dir", type=Path, default=paths.CHARTS_DIR)
    args = p.parse_args()

    if not CHART_ID_RE.fullmatch(args.id) or len(args.id) > 56:
        raise SystemExit("ID 는 소문자·숫자·하이픈만, 56자 이하")
    audio_name = None
    if args.audio_out:
        encode_ogg(args.audio, args.audio_out)
        audio_name = args.audio_out.name
        print(f"[audio] {args.audio_out} ({args.audio_out.stat().st_size / 1e6:.1f} MB)")

    y, sr = load_mono(args.audio)
    a = analyze(y, sr)
    print(f"[analyze] {a.duration_s:.1f}s  BPM {a.bpm:.2f}  첫 박 {a.first_beat_s * 1000:.0f}ms")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.difficulty:
        cid = f"{args.id}-{name}"
        chart = make_chart(
            a, DIFFICULTIES[name], cid, f"{args.title} ({name.upper()})", args.artist, audio_name
        )
        out = args.out_dir / f"{cid}.json"
        out.write_text(json.dumps(chart.to_dict(), ensure_ascii=False) + "\n", "utf-8")
        nps = len(chart.notes) / max(chart.duration_ms / 1000, 1)
        print(f"[chart] {out.name}: {len(chart.notes)} notes ({nps:.1f}/s)")


if __name__ == "__main__":
    main()
