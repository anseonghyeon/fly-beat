/**
 * 사람과 초파리가 같은 채보를 동시에 플레이한다.
 * 왼쪽: 내 레일 · 가운데: 설정 + 초파리 몸과 뇌 + 뇌 활동 수치 · 오른쪽: 초파리 레일
 *
 * 시계는 하나(AudioContext)다. 초파리 프레임은 서버가 실제 시간에 맞춰 보내고,
 * 브라우저는 LEAD_IN 만큼 늦게 재생해 버퍼로 쓴다.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { api, wsUrl } from "../api/client";
import type { Activity, Frame, Mode, PlayMessage } from "../api/types";
import { BrainPanel } from "../components/BrainPanel";
import { ResultModal, type MatchResult } from "../components/ResultModal";
import { SynthAudio } from "../game/audio";
import { JudgeEngine } from "../game/engine";
import { LANE_KEY_CODES, LANE_KEY_LABELS, LANE_KEYS_TEXT } from "../game/keys";
import { OPPONENTS, SOLO } from "../game/opponents";
import { FlyRenderer, REGION_COLORS, type Layout, type ViewName } from "../game/flyRenderer";
import { canvasSize, drawFrame, HitEffects, scoreOf, type LastJudgment } from "../game/renderer";
import { loadSettings, saveRecord, type PlaySettings } from "../game/settings";
import type { Chart, JudgeEvent, Stats } from "../game/types";
import { VersusHud, type HudState } from "../components/VersusHud";

const KEYS: readonly string[] = LANE_KEY_CODES;
const KEY_LABELS = [...LANE_KEY_LABELS];
const LEAD_IN_S = 1.0;

const MODE_NAME: Record<Mode, string> = {
  A: OPPONENTS.A.name,
  B: OPPONENTS.B.name,
  C: OPPONENTS.C.name,
};

type Status = "idle" | "loading" | "playing";

/** WebSocket 으로 초파리 세션을 열고 ready 메시지를 기다린다. */
function openFly(
  ckptId: string,
  chartId: string,
  onMessage: (m: PlayMessage) => void,
): Promise<{ ws: WebSocket; mode: Mode }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/ws/play"));
    let ready = false;
    ws.onopen = () =>
      ws.send(JSON.stringify({ type: "start", checkpoint: ckptId, chart: chartId }));
    ws.onerror = () => !ready && reject(new Error("서버에 연결하지 못했습니다"));
    ws.onmessage = (ev) => {
      let msg: PlayMessage;
      try {
        msg = JSON.parse(String(ev.data)) as PlayMessage;
      } catch {
        return;
      }
      if (!ready) {
        if (msg.type === "ready") {
          ready = true;
          resolve({ ws, mode: msg.mode });
        } else if (msg.type === "error") {
          reject(new Error(msg.message));
          ws.close();
        }
        return;
      }
      onMessage(msg);
    };
  });
}

export default function Play() {
  // 로비에서 넘겨준 설정 (새로고침이면 마지막으로 저장된 설정)
  const location = useLocation();
  const [settings] = useState<PlaySettings | null>(
    () => (location.state as PlaySettings | null) ?? loadSettings(),
  );
  if (!settings) return <Navigate to="/" replace />;
  return <PlaySession settings={settings} />;
}

function PlaySession({ settings }: { settings: PlaySettings }) {
  const { chartId, ckptId, speed, sound } = settings;
  const navigate = useNavigate();
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [flyMode, setFlyMode] = useState<Mode | null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [result, setResult] = useState<MatchResult | null>(null);
  const [hud, setHud] = useState<HudState | null>(null);
  const [layout, setLayout] = useState<Layout | null>(null);
  const [view, setView] = useState<ViewName>("top");

  const humanCanvas = useRef<HTMLCanvasElement>(null);
  const flyCanvas = useRef<HTMLCanvasElement>(null);
  const bodyCanvas = useRef<HTMLCanvasElement>(null);
  const flyRenderer = useRef<FlyRenderer | null>(null);
  const stopRef = useRef<(() => void) | null>(null);
  /** 세션 번호. 바뀌면 진행 중이던 준비(await)를 그만둔다 (중지·다시 하기·페이지 이탈). */
  const sessionRef = useRef(0);

  useEffect(() => {
    api
      .get<Layout>("/api/brain/layout/compact")
      .then((l) => {
        setLayout(l);
        flyRenderer.current = new FlyRenderer(l);
      })
      .catch(() => undefined);
    return () => stopRef.current?.();
  }, []);

  // 대기 중에도 초파리 몸을 그려 둔다
  useEffect(() => {
    if (flyRenderer.current) flyRenderer.current.view = view;
    const c = bodyCanvas.current;
    if (c && flyRenderer.current && status === "idle")
      flyRenderer.current.draw(c, null, [false, false, false, false], 0);
  }, [view, layout, status]);

  const stop = useCallback(() => {
    sessionRef.current++;
    stopRef.current?.();
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    sessionRef.current++;
    stopRef.current?.();
    const session = sessionRef.current;
    const alive = () => sessionRef.current === session;
    setError(null);
    setResult(null);
    setActivity(null);
    setStatus("loading");

    let chart: Chart;
    try {
      chart = await api.chart(chartId);
    } catch (e) {
      setError(`채보를 불러오지 못했습니다: ${(e as Error).message}`);
      setStatus("idle");
      return;
    }
    if (!alive()) return;
    const hCanvas = humanCanvas.current;
    const fCanvas = flyCanvas.current;
    if (!hCanvas || !fCanvas) {
      setStatus("idle");
      return;
    }

    const audio = new SynthAudio();
    const ctx = await audio.start();
    let music: AudioBuffer | null = null;
    if (sound && chart.audio) {
      music = await audio.load(chart.audio).catch(() => {
        setError("음원을 불러오지 못해 합성음으로 진행합니다");
        return null;
      });
    }
    if (!alive()) {
      await audio.stop();
      return;
    }

    // 초파리 세션 (선택)
    const human = new JudgeEngine(chart);
    const fly = ckptId ? new JudgeEngine(chart) : null;
    const queue: Frame[] = [];
    let flyServerStats: Stats | null = null;
    let ws: WebSocket | null = null;
    let mode: Mode | null = null;
    if (ckptId) {
      try {
        const opened = await openFly(ckptId, chartId, (m) => {
          if (m.type === "frames") queue.push(...m.frames);
          else if (m.type === "end") flyServerStats = m.stats;
          else if (m.type === "error") setError(`초파리: ${m.message}`);
        });
        ws = opened.ws;
        mode = opened.mode;
      } catch (e) {
        setError(`초파리를 불러오지 못했습니다: ${(e as Error).message}`);
        await audio.stop();
        setStatus("idle");
        return;
      }
      if (!alive()) {
        ws.close();
        await audio.stop();
        return;
      }
    }
    setFlyMode(mode);

    // 공통 시계: 채보 시각 0 = zeroAt (서버는 ready 직후부터 실제 시간에 맞춰 프레임을 보낸다)
    const zeroAt = ctx.currentTime + LEAD_IN_S;
    if (sound) await audio.playChart(chart, zeroAt, music);
    const songTime = () => (ctx.currentTime - zeroAt) * 1000;

    let lastHuman: LastJudgment | null = null;
    let lastFly: LastJudgment | null = null;
    const humanFx = new HitEffects();
    const flyFx = new HitEffects();
    let latest: Activity | null = null;
    let raf = 0;
    let closed = false;
    let lastPerf = performance.now();
    let lastHud = 0;
    const duration = human.endTime;
    /** 판정 이벤트 → 타격 효과 + 마지막 판정 표시 */
    const show = (evs: JudgeEvent[], fx: HitEffects): LastJudgment | null => {
      let last: LastJudgment | null = null;
      for (const e of evs) {
        if (e.judgment === "EMPTY") continue;
        fx.add(e);
        last = { judgment: e.judgment, shownAt: performance.now(), offset: e.offset };
      }
      return last;
    };

    const onKey = (down: boolean) => (e: KeyboardEvent) => {
      const lane = KEYS.indexOf(e.code);
      if (lane < 0 || e.repeat) return;
      e.preventDefault();
      const t = Math.max(Math.round(songTime()), human.now);
      lastHuman = show(down ? human.press(lane, t) : human.release(lane, t), humanFx) ?? lastHuman;
      if (down && sound) audio.hit();
    };
    const kd = onKey(true);
    const ku = onKey(false);
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);

    const cleanup = () => {
      if (closed) return;
      closed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
      ws?.close();
      void audio.stop();
      stopRef.current = null;
      setStatus("idle");
      setHud(null);
    };
    stopRef.current = cleanup;

    const finish = () => {
      human.finish();
      fly?.finish();
      if (human.stats.judged > 0)
        saveRecord(chart.id, {
          score: scoreOf(human),
          rate: human.stats.accuracy,
          maxCombo: human.stats.maxCombo,
          fullCombo: human.stats.counts.MISS === 0,
        });
      setResult({
        chartTitle: chart.title,
        artist: chart.artist,
        bpm: chart.bpm,
        human: { ...human.stats },
        fly:
          fly && mode
            ? { name: MODE_NAME[mode], client: { ...fly.stats }, server: flyServerStats }
            : null,
      });
      cleanup();
    };

    const frame = () => {
      const now = songTime();
      // 나
      const progress = Math.max(0, now) / duration;
      if (now > human.now) lastHuman = show(human.advance(now), humanFx) ?? lastHuman;
      drawFrame(hCanvas, human, now, {
        speed,
        keyLabels: KEY_LABELS,
        lastJudgment: lastHuman,
        effects: humanFx,
        progress,
      });

      // 초파리: 도착한 프레임 중 지금 시각까지 재생
      if (fly) {
        while (queue.length && queue[0]!.t <= now) {
          const f = queue.shift()!;
          for (const [lane, down] of f.inputs) {
            lastFly = show(down ? fly.press(lane, f.t) : fly.release(lane, f.t), flyFx) ?? lastFly;
          }
          lastFly = show(fly.advance(f.t), flyFx) ?? lastFly;
          if (f.activity) {
            latest = f.activity;
            setActivity(f.activity);
          }
        }
        drawFrame(fCanvas, fly, Math.max(now, fly.now), {
          speed,
          keyLabels: KEY_LABELS,
          lastJudgment: lastFly,
          effects: flyFx,
          progress,
        });
      }
      const perf = performance.now();
      const body = bodyCanvas.current;
      if (body && flyRenderer.current)
        flyRenderer.current.draw(
          body,
          latest,
          fly ? fly.pressed : [false, false, false, false],
          perf - lastPerf,
        );
      lastPerf = perf;

      // 대결 상황판은 초당 10번만 갱신
      if (perf - lastHud > 100) {
        lastHud = perf;
        setHud({
          title: chart.title,
          artist: chart.artist ?? "",
          bpm: chart.bpm,
          progress: Math.min(1, progress),
          human: {
            score: scoreOf(human),
            rate: human.stats.accuracy,
            combo: human.stats.combo,
          },
          fly:
            fly && mode
              ? {
                  name: MODE_NAME[mode],
                  score: scoreOf(fly),
                  rate: fly.stats.accuracy,
                  combo: fly.stats.combo,
                }
              : null,
        });
      }

      const humanDone = now >= human.endTime;
      const flyDone = !fly || (flyServerStats !== null && queue.length === 0);
      if (humanDone && flyDone) {
        finish();
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    setStatus("playing");
    raf = requestAnimationFrame(frame);
  }, [chartId, ckptId, speed, sound]);

  // 들어오자마자 바로 시작
  useEffect(() => {
    // 다음 틱에 시작 (개발 모드 StrictMode 의 마운트-해제-마운트에서 한 번만 시작되도록)
    const t = window.setTimeout(() => void start(), 0);
    return () => {
      window.clearTimeout(t);
      stop();
    };
    // 설정은 이 화면에서 바뀌지 않으므로 처음 한 번만
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toLobby = useCallback(() => navigate("/"), [navigate]);
  const { width, height } = canvasSize(4);
  const flyLabel = !ckptId ? SOLO.name : flyMode ? MODE_NAME[flyMode] : "불러오는 중…";

  return (
    <div className="versus">
      <section className="rail" aria-label="내 레일">
        <div className="rail-head">
          <b>나</b>
          <span className="muted small">{LANE_KEYS_TEXT}</span>
        </div>
        <div className="stage">
          <canvas ref={humanCanvas} width={width} height={height} />
        </div>
      </section>

      <section className="center">
        {status === "playing" && hud && (
          <div className="panel hud-panel">
            <VersusHud hud={hud} onStop={() => (stop(), toLobby())} />
          </div>
        )}
        {!(status === "playing" && hud) && (
          <div className="panel hud-panel play-status">
            {status === "loading" ? (
              <p className="loading-text">
                준비 중… {ckptId ? "초파리 뇌 모델을 불러오고 있습니다" : ""}
              </p>
            ) : (
              <div className="play-actions">
                <span className="muted">대결이 끝났습니다</span>
                <button className="primary" onClick={() => void start()}>
                  다시 하기
                </button>
                <button onClick={toLobby}>로비로</button>
              </div>
            )}
            {error && <p className="error">{error}</p>}
          </div>
        )}

        <div className="panel">
          <div className="panel-head">
            <h2>초파리 몸과 뇌</h2>
            <div className="seg">
              <button className={view === "top" ? "on" : ""} onClick={() => setView("top")}>
                위에서
              </button>
              <button className={view === "side" ? "on" : ""} onClick={() => setView("side")}>
                옆에서
              </button>
            </div>
          </div>
          {layout ? (
            <>
              <canvas ref={bodyCanvas} className="fly-canvas" />
              <div className="legend" style={{ marginTop: 8 }}>
                {layout.regions.map((r) => (
                  <span key={r.key}>
                    <i style={{ background: REGION_COLORS[r.key] }} />
                    {r.name}
                  </span>
                ))}
              </div>
              <p className="muted small">
                점 = 뉴런 세포체(MaleCNS 실측 좌표) · 빛남 = 활동 · 빛나는 다리 = 누름 · 몸 윤곽은
                도식
              </p>
            </>
          ) : (
            <p className="muted">뉴런 배치를 불러오는 중… (전처리가 필요할 수 있습니다)</p>
          )}
        </div>

        <div className="panel">
          <h2>뇌 활동 수치</h2>
          <BrainPanel mode={flyMode} activity={activity} />
        </div>
      </section>

      <section className="rail" aria-label="초파리 레일">
        <div className="rail-head">
          <b>{flyMode && !OPPONENTS[flyMode].isFly ? "상대" : "초파리"}</b>
          <span className="muted small">{flyLabel}</span>
        </div>
        <div className={`stage ${ckptId ? "" : "dim"}`}>
          <canvas ref={flyCanvas} width={width} height={height} />
        </div>
      </section>

      {result && (
        <ResultModal
          result={result}
          onClose={() => setResult(null)}
          onRetry={() => {
            setResult(null);
            void start();
          }}
          onLobby={toLobby}
        />
      )}
    </div>
  );
}
