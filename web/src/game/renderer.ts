import type { JudgeEngine } from "./engine";
import { type Judgment, noteEnd } from "./types";

export interface RenderOptions {
  /** 판정선까지 노트가 떨어지는 속도 (px/ms) */
  speed: number;
  keyLabels?: string[];
  lastJudgment?: { judgment: Judgment; at: number } | null;
}

const LANE_W = 84;
const JUDGE_LINE_FROM_BOTTOM = 96;

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function canvasSize(lanes: number): { width: number; height: number } {
  return { width: lanes * LANE_W + 2, height: 620 };
}

const JUDGE_COLOR: Record<Judgment, string> = {
  PERFECT: "--judge-perfect",
  GREAT: "--judge-great",
  GOOD: "--judge-good",
  MISS: "--judge-miss",
  EMPTY: "--fg-muted",
};

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
}

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

  const bg = cssVar("--stage-bg");
  const laneLine = cssVar("--stage-line");
  const tapColor = cssVar("--note-tap");
  const holdColor = cssVar("--note-hold");
  const accent = cssVar("--accent");
  const fg = cssVar("--stage-fg");

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  const judgeY = height - JUDGE_LINE_FROM_BOTTOM;

  for (let lane = 0; lane < engine.lanes; lane++) {
    const x = 1 + lane * LANE_W;
    if (engine.pressed[lane]) {
      const grad = ctx.createLinearGradient(0, judgeY - 220, 0, judgeY);
      grad.addColorStop(0, "transparent");
      grad.addColorStop(1, accent + "55");
      ctx.fillStyle = grad;
      ctx.fillRect(x, judgeY - 220, LANE_W, 220);
    }
    ctx.fillStyle = laneLine;
    ctx.fillRect(x + LANE_W - 1, 0, 1, height);
  }
  ctx.fillStyle = laneLine;
  ctx.fillRect(0, 0, 1, height);

  // 노트: 미판정 노트 + 누르고 있는 롱노트
  const yOf = (t: number) => judgeY - (t - now) * opts.speed;
  for (let lane = 0; lane < engine.lanes; lane++) {
    const x = 1 + lane * LANE_W + 6;
    const w = LANE_W - 12;
    const hi = engine.holding[lane];
    if (hi !== null && hi !== undefined) {
      const n = engine.notes[hi]!;
      const yEnd = yOf(noteEnd(n));
      ctx.fillStyle = holdColor;
      roundRect(ctx, x + 8, yEnd, w - 16, judgeY - yEnd, 6);
    }
    const idxs = engine.laneNotes[lane]!;
    for (let k = engine.head[lane]!; k < idxs.length; k++) {
      const n = engine.notes[idxs[k]!]!;
      const y = yOf(n.t);
      if (y < -40) break;
      if (n.type === "hold") {
        const yEnd = yOf(noteEnd(n));
        ctx.fillStyle = holdColor;
        roundRect(ctx, x + 8, yEnd, w - 16, y - yEnd, 6);
      }
      ctx.fillStyle = tapColor;
      roundRect(ctx, x, y - 9, w, 18, 9);
    }
  }

  // 판정선
  ctx.fillStyle = accent;
  ctx.fillRect(0, judgeY - 1, width, 3);

  // 키 라벨
  ctx.font = "600 14px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  for (let lane = 0; lane < engine.lanes; lane++) {
    const cx = 1 + lane * LANE_W + LANE_W / 2;
    ctx.fillStyle = engine.pressed[lane] ? accent : cssVar("--fg-muted");
    ctx.fillText(opts.keyLabels?.[lane] ?? String(lane), cx, height - 40);
  }

  // 판정 / 콤보
  const lj = opts.lastJudgment;
  if (lj && now - lj.at < 500) {
    ctx.globalAlpha = 1 - (now - lj.at) / 500;
    ctx.fillStyle = cssVar(JUDGE_COLOR[lj.judgment]);
    ctx.font = "800 30px system-ui, sans-serif";
    ctx.fillText(lj.judgment, width / 2, height * 0.42);
    ctx.globalAlpha = 1;
  }
  if (engine.stats.combo >= 2) {
    ctx.fillStyle = fg;
    ctx.font = "700 22px system-ui, sans-serif";
    ctx.fillText(`${engine.stats.combo} COMBO`, width / 2, height * 0.42 + 34);
  }
}
