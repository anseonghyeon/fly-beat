import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { SynthAudio } from "../game/audio";
import { JudgeEngine } from "../game/engine";
import { canvasSize, drawFrame } from "../game/renderer";
import type { Chart, ChartSummary, Judgment, Stats } from "../game/types";
import { ChartPicker } from "../components/ChartPicker";
import { StatsView } from "../components/StatsView";

const KEYS = ["KeyD", "KeyF", "KeyJ", "KeyK"];
const KEY_LABELS = ["D", "F", "J", "K"];
const LEAD_IN_S = 1.0;

export default function Play() {
  const [charts, setCharts] = useState<ChartSummary[]>([]);
  const [chartId, setChartId] = useState("tutorial-steps");
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [result, setResult] = useState<Stats | null>(null);
  const [speed, setSpeed] = useState(0.6);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .charts()
      .then(setCharts)
      .catch((e: Error) => setError(`채보 목록을 불러오지 못했습니다: ${e.message}`));
    return () => stopRef.current?.();
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setResult(null);
    let chart: Chart;
    try {
      chart = await api.chart(chartId);
    } catch (e) {
      setError(`채보를 불러오지 못했습니다: ${(e as Error).message}`);
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = new JudgeEngine(chart);
    const audio = new SynthAudio();
    const ctx = await audio.start();
    let music: AudioBuffer | null = null;
    if (chart.audio) {
      try {
        music = await audio.load(chart.audio);
      } catch (e) {
        setError(`${(e as Error).message} — 합성음으로 진행합니다`);
      }
    }
    const zeroAt = ctx.currentTime + LEAD_IN_S;
    await audio.playChart(chart, zeroAt, music);
    const songTime = () => (ctx.currentTime - zeroAt) * 1000;
    let last: { judgment: Judgment; at: number } | null = null;
    let raf = 0;
    let stopped = false;

    const onEvents = (evs: ReturnType<JudgeEngine["advance"]>) => {
      const real = evs.filter((e) => e.judgment !== "EMPTY");
      const lastEv = real[real.length - 1];
      if (lastEv) last = { judgment: lastEv.judgment, at: songTime() };
    };
    const onKey = (down: boolean) => (e: KeyboardEvent) => {
      const lane = KEYS.indexOf(e.code);
      if (lane < 0 || e.repeat) return;
      e.preventDefault();
      const t = Math.max(Math.round(songTime()), engine.now);
      onEvents(down ? engine.press(lane, t) : engine.release(lane, t));
      if (down) audio.hit();
    };
    const kd = onKey(true);
    const ku = onKey(false);
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);

    const stop = () => {
      if (stopped) return;
      stopped = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
      void audio.stop();
      setPlaying(false);
      stopRef.current = null;
    };
    stopRef.current = stop;

    const frame = () => {
      const now = songTime();
      if (now > engine.now) onEvents(engine.advance(now));
      drawFrame(canvas, engine, now, { speed, keyLabels: KEY_LABELS, lastJudgment: last });
      if (now >= engine.endTime) {
        onEvents(engine.finish());
        setResult({ ...engine.stats });
        stop();
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    setPlaying(true);
    raf = requestAnimationFrame(frame);
  }, [chartId, speed]);

  const { width, height } = canvasSize(4);

  return (
    <div className="row">
      <div className="stage">
        <canvas ref={canvasRef} width={width} height={height} />
      </div>
      <div className="grow" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <section className="panel">
          <h2>직접 플레이</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            키보드 <code>D</code> <code>F</code> <code>J</code> <code>K</code> 로 4개 레인을 칩니다.
            롱노트는 끝까지 누르고 있으세요.
          </p>
          <div className="controls">
            <ChartPicker charts={charts} value={chartId} onChange={setChartId} disabled={playing} />
            <label className="field">
              노트 속도
              <input
                type="number"
                min={0.2}
                max={2}
                step={0.1}
                value={speed}
                disabled={playing}
                onChange={(e) =>
                  setSpeed(Math.min(2, Math.max(0.2, Number(e.target.value) || 0.6)))
                }
              />
            </label>
            {playing ? (
              <button onClick={() => stopRef.current?.()}>중지</button>
            ) : (
              <button className="primary" onClick={() => void start()}>
                시작
              </button>
            )}
          </div>
          {error && <p className="error">{error}</p>}
        </section>
        {result && (
          <section className="panel">
            <h2>결과</h2>
            <StatsView stats={result} />
          </section>
        )}
      </div>
    </div>
  );
}
