import { useCallback, useEffect, useRef, useState } from "react";
import { api, wsUrl } from "../api/client";
import type { Activity, CheckpointSummary, Frame, Mode, PlayMessage } from "../api/types";
import { BrainPanel } from "../components/BrainPanel";
import { ChartPicker } from "../components/ChartPicker";
import { StatsView } from "../components/StatsView";
import { SynthAudio } from "../game/audio";
import { JudgeEngine } from "../game/engine";
import { canvasSize, drawFrame } from "../game/renderer";
import type { ChartSummary, Judgment, Stats } from "../game/types";

const MODE_NAME: Record<Mode, string> = {
  A: "A · MLP PPO",
  B: "B · 커넥톰 매핑",
  C: "C · 도파민 가소성",
};

export default function Watch() {
  const [charts, setCharts] = useState<ChartSummary[]>([]);
  const [ckpts, setCkpts] = useState<CheckpointSummary[]>([]);
  const [chartId, setChartId] = useState("easy-groove");
  const [ckptId, setCkptId] = useState("");
  const [sound, setSound] = useState(true);
  const [status, setStatus] = useState<"idle" | "loading" | "playing" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode | null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [result, setResult] = useState<{ client: Stats; server: Stats } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .charts()
      .then(setCharts)
      .catch(() => setError("채보 목록을 불러오지 못했습니다"));
    api
      .get<CheckpointSummary[]>("/api/checkpoints")
      .then((cs) => {
        setCkpts(cs);
        if (cs[0]) setCkptId(cs[0].id);
      })
      .catch(() => setError("체크포인트 목록을 불러오지 못했습니다"));
    return () => stopRef.current?.();
  }, []);

  const start = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas || !ckptId) return;
    setError(null);
    setResult(null);
    setActivity(null);
    setStatus("loading");

    const ws = new WebSocket(wsUrl("/ws/play"));
    const audio = new SynthAudio();
    let engine: JudgeEngine | null = null;
    const queue: Frame[] = [];
    let readyAt = 0;
    let leadMs = 400;
    let raf = 0;
    let last: { judgment: Judgment; at: number } | null = null;
    let serverStats: Stats | null = null;
    let closed = false;

    const stop = () => {
      if (closed) return;
      closed = true;
      cancelAnimationFrame(raf);
      ws.close();
      void audio.stop();
      stopRef.current = null;
    };
    stopRef.current = stop;

    const nowMs = () => performance.now() - readyAt - leadMs;

    const frame = () => {
      if (!engine) return;
      const now = nowMs();
      while (queue.length && queue[0]!.t <= now) {
        const f = queue.shift()!;
        for (const [lane, down] of f.inputs) {
          const evs = down ? engine.press(lane, f.t) : engine.release(lane, f.t);
          const hit = evs.filter((e) => e.judgment !== "EMPTY").pop();
          if (hit) last = { judgment: hit.judgment, at: f.t };
        }
        const evs = engine.advance(f.t);
        const miss = evs.pop();
        if (miss) last = { judgment: miss.judgment, at: f.t };
        if (f.activity) setActivity(f.activity);
      }
      drawFrame(canvas, engine, Math.max(now, engine.now), {
        speed: 0.6,
        keyLabels: ["D", "F", "J", "K"],
        lastJudgment: last,
      });
      if (serverStats && queue.length === 0) {
        engine.finish();
        setResult({ client: { ...engine.stats }, server: serverStats });
        setStatus("done");
        stop();
        return;
      }
      raf = requestAnimationFrame(frame);
    };

    ws.onopen = () =>
      ws.send(JSON.stringify({ type: "start", checkpoint: ckptId, chart: chartId }));
    ws.onerror = () => setError("서버에 연결하지 못했습니다");
    ws.onmessage = async (ev) => {
      let msg: PlayMessage;
      try {
        msg = JSON.parse(String(ev.data)) as PlayMessage;
      } catch {
        return;
      }
      if (msg.type === "error") {
        setError(msg.message);
        setStatus("idle");
        stop();
      } else if (msg.type === "ready") {
        engine = new JudgeEngine(msg.chart);
        leadMs = msg.leadMs;
        readyAt = performance.now();
        setMode(msg.mode);
        setStatus("playing");
        if (sound) {
          const ctx = await audio.start();
          audio.schedule(msg.chart, ctx.currentTime + leadMs / 1000);
        }
        raf = requestAnimationFrame(frame);
      } else if (msg.type === "frames") {
        queue.push(...msg.frames);
      } else if (msg.type === "end") {
        serverStats = msg.stats;
      }
    };
  }, [chartId, ckptId, sound]);

  const { width, height } = canvasSize(4);
  const selected = ckpts.find((c) => c.id === ckptId);
  const busy = status === "loading" || status === "playing";

  return (
    <div className="row">
      <div className="stage">
        <canvas ref={canvasRef} width={width} height={height} />
      </div>
      <div className="grow" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <section className="panel">
          <h2>초파리 플레이 관전</h2>
          {ckpts.length === 0 ? (
            <p className="muted">
              체크포인트가 없습니다. <a href="/train">Train</a> 화면에서 먼저 학습하세요.
            </p>
          ) : (
            <div className="controls">
              <label className="field">
                체크포인트
                <select value={ckptId} disabled={busy} onChange={(e) => setCkptId(e.target.value)}>
                  {ckpts.map((c) => (
                    <option key={c.id} value={c.id}>
                      [{c.mode}] {c.id} · {c.accuracy?.toFixed(1) ?? "-"}%
                    </option>
                  ))}
                </select>
              </label>
              <ChartPicker charts={charts} value={chartId} onChange={setChartId} disabled={busy} />
              <label className="field check">
                <input
                  type="checkbox"
                  checked={sound}
                  disabled={busy}
                  onChange={(e) => setSound(e.target.checked)}
                />
                소리
              </label>
              {busy ? (
                <button onClick={() => (stopRef.current?.(), setStatus("idle"))}>중지</button>
              ) : (
                <button className="primary" onClick={() => void start()}>
                  재생
                </button>
              )}
            </div>
          )}
          {selected && (
            <p className="muted small">
              모드 {MODE_NAME[selected.mode]} · 검증 정확도 {selected.accuracy?.toFixed(2) ?? "-"}%
              {status === "loading" && " · 뇌 모델 불러오는 중…"}
            </p>
          )}
          {error && <p className="error">{error}</p>}
        </section>
        <section className="panel">
          <h2>뇌 활동</h2>
          <BrainPanel mode={mode ?? selected?.mode ?? null} activity={activity} />
        </section>
        {result && (
          <section className="panel">
            <h2>결과</h2>
            <StatsView stats={result.client} />
            <p
              className={
                result.client.score === result.server.score ? "muted small" : "error small"
              }
            >
              {result.client.score === result.server.score
                ? "브라우저(TS) 판정과 서버(Python) 판정이 일치합니다."
                : `판정 불일치: 브라우저 ${result.client.accuracy}% / 서버 ${result.server.accuracy}%`}
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
