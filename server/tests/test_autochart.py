import numpy as np
import pytest
import soundfile as sf

from flymax.game.autochart import analyze, encode_ogg, load_mono
from flymax.game.chart import ChartError, audio_path
from flymax.game.charter import DIFFICULTIES, chart_quality, make_chart


def click_track(path, bpm=128.0, seconds=30.0, sr=22050, offset=0.35):
    """박마다 높은음/낮은음 클릭이 번갈아 나는 합성 음원."""
    y = np.zeros(int(seconds * sr), np.float32)
    beat = 60 / bpm
    t = np.arange(int(0.03 * sr)) / sr
    for k, start in enumerate(np.arange(offset, seconds - 0.1, beat)):
        f = 2000 if k % 2 else 300
        tone = np.sin(2 * np.pi * f * t) * np.exp(-t * 80)
        i = int(start * sr)
        y[i : i + len(tone)] += tone.astype(np.float32)
    sf.write(path, y, sr)
    return offset


def test_tempo_and_phase(tmp_path):
    p = tmp_path / "click.wav"
    offset = click_track(p, bpm=128.0)
    a = analyze(*load_mono(p))
    assert a.bpm == pytest.approx(128.0, abs=0.3)
    beat = 60 / 128.0
    phase_err = (a.first_beat_s - offset) % beat
    assert min(phase_err, beat - phase_err) < 0.02


def test_charts_valid_and_on_beat(tmp_path):
    p = tmp_path / "click.wav"
    offset = click_track(p, bpm=120.0)
    a = analyze(*load_mono(p))
    for name, d in DIFFICULTIES.items():
        c = make_chart(a, d, f"click-{name}", "Click", "test", "click.ogg")
        assert c.notes, name
        beat_ms = 60000 / 120.0
        # 쉬움은 모든 노트가 박 위에 있어야 한다
        if name == "easy":
            err = [((n.t - offset * 1000) % beat_ms) for n in c.notes]
            assert max(min(e, beat_ms - e) for e in err) < 20
    easy = make_chart(a, DIFFICULTIES["easy"], "e", "E", "", None)
    hard = make_chart(a, DIFFICULTIES["hard"], "h", "H", "", None)
    assert len(hard.notes) > len(easy.notes)


def test_encode_ogg(tmp_path):
    p = tmp_path / "click.wav"
    click_track(p, seconds=5)
    out = tmp_path / "audio" / "click.ogg"
    encode_ogg(p, out)
    assert out.stat().st_size < p.stat().st_size
    assert audio_path("click.ogg", tmp_path) == out.resolve()


@pytest.mark.parametrize("bad", ["../x.ogg", "a.exe", "A.ogg", "click", ".ogg"])
def test_audio_path_rejects(bad, tmp_path):
    with pytest.raises((ChartError, FileNotFoundError)):
        audio_path(bad, tmp_path)


def test_chart_quality_flow_and_no_jacks(tmp_path):
    p = tmp_path / "click.wav"
    click_track(p, bpm=120.0, seconds=40)
    a = analyze(*load_mono(p))
    for name, d in DIFFICULTIES.items():
        c = make_chart(a, d, f"q-{name}", "Q", "", None)
        q = chart_quality(c, a)
        assert q["on_strong_onset"] >= 0.9, (name, q)  # 소리가 있는 곳에 노트
        assert q["jack_ratio"] <= 0.05, (name, q)  # 같은 레인 연타 거의 없음
