/**
 * 레일 렌더러 (아케이드 리듬게임 연출, fly-beat 고유 스타일).
 * - 뒤에 비치는 큰 콤보 숫자, 빛나는 노트·롱노트, 누른 레인의 빛줄기
 * - 타격 효과: 겹눈을 닮은 육각형 파편 (HitEffects)
 * - 판정 + FAST/SLOW + 점수·정확도, 오른쪽 곡 진행 게이지
 */
import type { JudgeEngine } from "./engine";
import { type JudgeEvent, type Judgment, noteEnd } from "./types";

export interface LastJudgment {
  judgment: Judgment;
  /** 표시 시작 시각 (performance.now) */
  shownAt: number;
  /** 입력 시각 − 기준 시각 (ms). 음수 = 빠름 */
  offset: number;
}

export interface RenderOptions {
  /** 판정선까지 노트가 떨어지는 속도 (px/ms) */
  speed: number;
  keyLabels?: string[];
  lastJudgment?: LastJudgment | null;
  effects?: HitEffects;
  /** 0~1 곡 진행 */
  progress?: number;
}

const LANE_W = 84;
const GAUGE_W = 10;
const JUDGE_LINE_FROM_BOTTOM = 96;
const MAX_SCORE = 1_000_000;

/** 레일 색 (어두운 무대 전용, 화면 테마와 무관하게 고정) */
const C = {
  bg: "#0b0a08",
  laneA: "#14120e",
  laneB: "#100f0c",
  laneLine: "#2a261d",
  judge: "#f2a93b",
  tapTop: "#fff6dc",
  tapBottom: "#f2c14e",
  holdA: "rgba(242, 193, 78, 0.55)",
  holdB: "rgba(242, 169, 59, 0.15)",
  beam: "242, 169, 59",
  text: "#ece6d8",
  muted: "#8a8272",
  perfect: "#ffd54a",
  great: "#7bd88f",
  good: "#6ab0ff",
  miss: "#ff6b6b",
  fast: "#6ab0ff",
  slow: "#ff9a62",
};

const JUDGE_COLOR: Record<Judgment, string> = {
  PERFECT: C.perfect,
  GREAT: C.great,
  GOOD: C.good,
  MISS: C.miss,
  EMPTY: C.muted,
};

export function canvasSize(lanes: number): { width: number; height: number } {
  return { width: lanes * LANE_W + GAUGE_W + 2, height: 620 };
}

/** 0 ~ 1,000,000 점수 (판정 점수 합 / 전체 판정 대상) */
export function scoreOf(engine: JudgeEngine): number {
  return Math.round((MAX_SCORE * engine.stats.score) / Math.max(1, engine.stats.total));
}

// ------------------------------------------------------------------ 타격 효과
interface Hit {
  lane: number;
  judgment: Judgment;
  start: number;
  seed: number;
}

/** 타격 순간의 빛과 육각형 파편. 시간은 performance.now 기준. */
export class HitEffects {
  private hits: Hit[] = [];
  private n = 0;

  add(ev: JudgeEvent, now = performance.now()): void {
    if (ev.judgment === "MISS" || ev.judgment === "EMPTY") return;
    this.hits.push({ lane: ev.lane, judgment: ev.judgment, start: now, seed: ++this.n * 9301 });
    if (this.hits.length > 64) this.hits.shift();
  }

  draw(ctx: CanvasRenderingContext2D, now: number, judgeY: number): void {
    this.hits = this.hits.filter((h) => now - h.start < 520);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const h of this.hits) {
      const age = (now - h.start) / 520; // 0 → 1
      const cx = 1 + h.lane * LANE_W + LANE_W / 2;
      const color = JUDGE_COLOR[h.judgment];
      // 판정선 섬광
      const flash = ctx.createRadialGradient(cx, judgeY, 2, cx, judgeY, LANE_W * (0.6 + age));
      flash.addColorStop(0, `rgba(255,255,255,${0.75 * (1 - age)})`);
      flash.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = flash;
      ctx.fillRect(cx - LANE_W * 1.6, judgeY - LANE_W * 1.6, LANE_W * 3.2, LANE_W * 3.2);
      // 위로 솟는 빛기둥
      const col = ctx.createLinearGradient(0, judgeY - 260, 0, judgeY);
      col.addColorStop(0, "rgba(255,255,255,0)");
      col.addColorStop(1, `rgba(255,236,190,${0.45 * (1 - age) ** 2})`);
      ctx.fillStyle = col;
      ctx.fillRect(cx - LANE_W / 2 + 4, judgeY - 260, LANE_W - 8, 260);
      // 겹눈 육각형 파편
      let s = h.seed;
      const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
      const count = h.judgment === "PERFECT" ? 14 : 9;
      ctx.fillStyle = color;
      for (let i = 0; i < count; i++) {
        const ang = -Math.PI * (0.1 + 0.8 * rand());
        const speed = 60 + 140 * rand();
        const d = speed * age * (1.2 - 0.4 * age);
        const r = (2.5 + 4 * rand()) * (1 - age * 0.7);
        ctx.globalAlpha = (1 - age) * 0.9;
        hexagon(ctx, cx + Math.cos(ang) * d, judgeY + Math.sin(ang) * d, r);
        ctx.fill();
      }
      // 퍼지는 육각 고리
      ctx.globalAlpha = (1 - age) * 0.8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      hexagon(ctx, cx, judgeY, 10 + 46 * age);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }
}

function hexagon(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 3) * k + Math.PI / 6;
    if (k === 0) ctx.moveTo(x + r * Math.cos(a), y + r * Math.sin(a));
    else ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a));
  }
  ctx.closePath();
}

function pill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

// ------------------------------------------------------------------ 그리기
/** 엔진 상태와 현재 시각(ms)으로 한 프레임을 그린다. */
export function drawFrame(
  canvas: HTMLCanvasElement,
  engine: JudgeEngine,
  now: number,
  opts: RenderOptions,
): void {
  const dpr = window.devicePixelRatio || 1;
  const { width, height } = canvasSize(engine.lanes);
  if (canvas.width !== width * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const perf = performance.now();
  const railW = engine.lanes * LANE_W + 2;
  const judgeY = height - JUDGE_LINE_FROM_BOTTOM;
  const stats = engine.stats;

  // 배경 + 레인
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, width, height);
  for (let lane = 0; lane < engine.lanes; lane++) {
    const x = 1 + lane * LANE_W;
    ctx.fillStyle = lane % 2 ? C.laneB : C.laneA;
    ctx.fillRect(x, 0, LANE_W, judgeY);
    ctx.fillStyle = C.laneLine;
    ctx.fillRect(x + LANE_W - 1, 0, 1, height);
  }
  ctx.fillStyle = C.laneLine;
  ctx.fillRect(0, 0, 1, height);

  // 뒤에 비치는 큰 콤보
  if (stats.combo >= 2) {
    ctx.save();
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(242, 169, 59, 0.55)";
    ctx.font = "700 13px system-ui, sans-serif";
    ctx.fillText("◆ COMBO ◆", railW / 2, 70);
    ctx.fillStyle = "rgba(236, 230, 216, 0.13)";
    ctx.font = "800 96px system-ui, sans-serif";
    ctx.fillText(String(stats.combo), railW / 2, 160);
    ctx.restore();
  }

  // 누른 레인 빛줄기
  for (let lane = 0; lane < engine.lanes; lane++) {
    if (!engine.pressed[lane]) continue;
    const x = 1 + lane * LANE_W;
    const g = ctx.createLinearGradient(0, judgeY - 320, 0, judgeY);
    g.addColorStop(0, `rgba(${C.beam},0)`);
    g.addColorStop(1, `rgba(${C.beam},0.38)`);
    ctx.fillStyle = g;
    ctx.fillRect(x, judgeY - 320, LANE_W - 1, 320);
  }

  // 노트
  const yOf = (t: number) => judgeY - (t - now) * opts.speed;
  const drawHoldBody = (x: number, w: number, yTop: number, yBottom: number, active: boolean) => {
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, C.holdB);
    g.addColorStop(0.5, active ? "rgba(255, 220, 140, 0.85)" : C.holdA);
    g.addColorStop(1, C.holdB);
    ctx.fillStyle = g;
    ctx.fillRect(x + 10, yTop, w - 20, Math.max(0, yBottom - yTop));
    if (active) {
      ctx.save();
      ctx.shadowColor = "#f2c14e";
      ctx.shadowBlur = 18;
      ctx.fillStyle = "rgba(255, 220, 140, 0.25)";
      ctx.fillRect(x + 10, yTop, w - 20, Math.max(0, yBottom - yTop));
      ctx.restore();
    }
  };
  const drawTap = (x: number, w: number, y: number) => {
    const g = ctx.createLinearGradient(0, y - 9, 0, y + 9);
    g.addColorStop(0, C.tapTop);
    g.addColorStop(1, C.tapBottom);
    ctx.save();
    ctx.shadowColor = "rgba(242, 193, 78, 0.7)";
    ctx.shadowBlur = 10;
    ctx.fillStyle = g;
    pill(ctx, x + 2, y - 8, w - 4, 16, 4);
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillRect(x + 8, y - 6, w - 16, 2);
  };
  for (let lane = 0; lane < engine.lanes; lane++) {
    const x = 1 + lane * LANE_W;
    const w = LANE_W - 1;
    const hi = engine.holding[lane];
    if (hi !== null && hi !== undefined) {
      const n = engine.notes[hi]!;
      drawHoldBody(x, w, yOf(noteEnd(n)), judgeY, true);
    }
    const idxs = engine.laneNotes[lane]!;
    for (let k = engine.head[lane]!; k < idxs.length; k++) {
      const n = engine.notes[idxs[k]!]!;
      const y = yOf(n.t);
      if (y < -40) break;
      if (n.type === "hold") drawHoldBody(x, w, yOf(noteEnd(n)), y, false);
      drawTap(x, w, y);
    }
  }

  // 판정선
  ctx.save();
  ctx.shadowColor = C.judge;
  ctx.shadowBlur = 14;
  ctx.fillStyle = C.judge;
  ctx.fillRect(0, judgeY - 2, railW, 4);
  ctx.restore();
  ctx.fillStyle = "#0d0c09";
  ctx.fillRect(0, judgeY + 2, railW, height - judgeY - 2);

  // 타격 효과
  opts.effects?.draw(ctx, perf, judgeY);

  // 키 라벨 (누르면 빛남)
  ctx.textAlign = "center";
  for (let lane = 0; lane < engine.lanes; lane++) {
    const cx = 1 + lane * LANE_W + LANE_W / 2;
    const on = engine.pressed[lane];
    ctx.fillStyle = on ? "rgba(242, 169, 59, 0.25)" : "rgba(255,255,255,0.04)";
    pill(ctx, cx - 30, height - 62, 60, 34, 8);
    ctx.fill();
    ctx.fillStyle = on ? C.judge : C.muted;
    ctx.font = "700 15px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(opts.keyLabels?.[lane] ?? String(lane), cx, height - 40);
  }

  // 판정 · FAST/SLOW · 점수
  const lj = opts.lastJudgment;
  const midY = height * 0.5;
  if (lj && perf - lj.shownAt < 600) {
    const age = (perf - lj.shownAt) / 600;
    const pop = 1 + 0.25 * Math.max(0, 1 - age * 6);
    ctx.save();
    ctx.globalAlpha = Math.min(1, 1.6 - age * 1.6);
    ctx.translate(railW / 2, midY);
    ctx.scale(pop, pop);
    ctx.textAlign = "center";
    ctx.shadowColor = JUDGE_COLOR[lj.judgment];
    ctx.shadowBlur = 16;
    ctx.fillStyle = JUDGE_COLOR[lj.judgment];
    ctx.font = "900 34px system-ui, sans-serif";
    ctx.fillText(lj.judgment, 0, 0);
    ctx.restore();
    if (lj.judgment !== "MISS" && lj.judgment !== "PERFECT" && lj.offset !== 0) {
      ctx.globalAlpha = Math.min(1, 1.6 - age * 1.6);
      ctx.textAlign = "center";
      ctx.font = "800 14px system-ui, sans-serif";
      ctx.fillStyle = lj.offset < 0 ? C.fast : C.slow;
      ctx.fillText(lj.offset < 0 ? "FAST" : "SLOW", railW / 2, midY - 34);
      ctx.globalAlpha = 1;
    }
  }
  ctx.textAlign = "center";
  ctx.fillStyle = C.text;
  ctx.font = "800 20px system-ui, sans-serif";
  ctx.fillText(scoreOf(engine).toLocaleString(), railW / 2, midY + 36);
  ctx.fillStyle = C.muted;
  ctx.font = "600 11px system-ui, sans-serif";
  ctx.fillText(
    `RATE ${stats.accuracy.toFixed(2)}%   ·   MAX COMBO ${stats.maxCombo}`,
    railW / 2,
    midY + 54,
  );

  // 곡 진행 게이지 (오른쪽)
  const gx = railW + 1;
  ctx.fillStyle = "#151310";
  ctx.fillRect(gx, 0, GAUGE_W - 1, height);
  const p = Math.min(1, Math.max(0, opts.progress ?? 0));
  const gh = (height - 8) * p;
  const g = ctx.createLinearGradient(0, height, 0, 0);
  g.addColorStop(0, "#f2a93b");
  g.addColorStop(1, "#ffe6a6");
  ctx.fillStyle = g;
  ctx.fillRect(gx + 2, height - 4 - gh, GAUGE_W - 5, gh);
}
