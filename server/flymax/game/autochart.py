"""음원 분석 + 채보 생성 CLI.

분석(이 파일): 음원을 모노로 읽고 스펙트럼 변화량으로 강세 곡선과
대역별(킥·스네어·멜로디·하이햇) 강세, 멜로디 대역 밝기를 구한 뒤
BPM 과 첫 박 위치(위상)를 찾는다 (일정한 템포 가정).
채보 생성: flymax.game.charter (음악적인 레인 배정·반복·롱노트).

사용법:
  python -m flymax.game.autochart song.wav --id my-song --title "My Song" --artist "Someone" \\
      [--audio-out charts/audio/my-song.ogg] [--difficulty easy normal hard]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flymax import paths
from flymax.game.chart import AUDIO_NAME_RE, CHART_ID_RE

N_FFT = 2048
HOP = 256


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
    # 대역별 강세 (0~1): low=킥·베이스, mid=스네어, mel=멜로디 대역, high=하이햇
    bands: dict[str, np.ndarray] | None = None
    mel_centroid: np.ndarray | None = (
        None  # 멜로디 대역(250~4000Hz) 무게중심 (Hz, 로그 스케일 비교용)
    )
    mel_energy: np.ndarray | None = None  # 멜로디 대역 로그 에너지

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

    def band_onset(lo: float, hi: float) -> np.ndarray:
        m = (freqs >= lo) & (freqs < hi)
        fl = np.maximum(0, np.diff(logmag[:, m], axis=0, prepend=logmag[:1, m])).sum(axis=1)
        fl -= np.convolve(fl, np.ones(16) / 16, mode="same")
        fl = np.maximum(fl, 0)
        return np.clip(fl / max(np.percentile(fl, 99.5), 1e-9), 0, 1)

    bands = {
        "low": band_onset(20, 160),
        "mid": band_onset(160, 2500),
        "mel": band_onset(250, 4000),
        "high": band_onset(5000, min(16000, sr / 2)),
    }
    mm = (freqs >= 250) & (freqs < 4000)
    mel_mag = mag[:, mm]
    mel_centroid = (mel_mag * freqs[mm]).sum(axis=1) / np.maximum(mel_mag.sum(axis=1), 1e-9)
    mel_energy = np.log1p(100 * mel_mag.sum(axis=1))

    bpm, first = estimate_tempo(onset, frame_s, bpm_range)
    # 해닝 창에서 강세(에너지 증가)가 가장 크게 잡히는 곳은 창 시작에서 3/4 지점이다.
    # 프레임 시각을 창 시작으로 두면 노트가 실제 소리보다 앞서므로 그만큼 보정한다.
    t0 = 0.75 * N_FFT / sr
    return Analysis(
        sr,
        len(y) / sr,
        frame_s,
        onset,
        centroid,
        rms,
        bpm,
        first + t0,
        t0,
        bands,
        mel_centroid,
        mel_energy,
    )


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
    from flymax.game.charter import DIFFICULTIES, chart_quality, make_chart

    p = argparse.ArgumentParser(prog="python -m flymax.game.autochart")
    p.add_argument("audio", type=Path)
    p.add_argument("--id", required=True, help="채보 ID 접두사 (예: my-song → my-song-easy)")
    p.add_argument("--title", required=True)
    p.add_argument("--artist", default="")
    p.add_argument("--audio-out", type=Path, help="OGG 로 변환해 저장할 경로 (charts/audio/*.ogg)")
    p.add_argument(
        "--audio-name", help="이미 charts/audio 에 있는 음원 파일 이름 (변환 없이 채보에만 연결)"
    )
    p.add_argument(
        "--difficulty", nargs="+", default=list(DIFFICULTIES), choices=list(DIFFICULTIES)
    )
    p.add_argument("--out-dir", type=Path, default=paths.CHARTS_DIR)
    args = p.parse_args()

    if not CHART_ID_RE.fullmatch(args.id) or len(args.id) > 56:
        raise SystemExit("ID 는 소문자·숫자·하이픈만, 56자 이하")
    audio_name = args.audio_name
    if audio_name and not AUDIO_NAME_RE.fullmatch(audio_name):
        raise SystemExit("음원 이름 형식이 올바르지 않습니다 (예: my-song.ogg)")
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
        q = chart_quality(chart, a)
        print(
            f"[chart] {out.name}: {len(chart.notes)} notes ({nps:.1f}/s) "
            f"강세일치 {q['on_strong_onset']:.0%} 흐름 {q['step_moves']:.0%} "
            f"연타 {q['jack_ratio']:.1%} 롱노트 {q['holds']} 동시치기 {q['chords']}"
        )


if __name__ == "__main__":
    main()
