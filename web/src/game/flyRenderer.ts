/**
 * 초파리 몸 + 실제 뉴런 위치(세포체 좌표) 위에 뇌 활동을 그린다.
 *
 * - top 보기: 위에서 본 초파리. 뇌는 머리, 복부신경삭(VNC)은 가슴 안에 실제 배치대로 들어간다.
 *   다리는 해당 다리 운동 뉴런들의 위치에서 뻗어 나온다.
 * - side 보기: 왼쪽에서 본 중추신경계(몸 윤곽 없이 뉴런만).
 * 몸 윤곽은 뉴런 분포에서 계산한 단순한 도식이다 (실제 외골격 형태가 아님).
 */
import type { Activity } from "../api/types";
import { LANE_KEY_LABELS } from "./keys";

export interface Layout {
  n: number;
  quant: number;
  regions: { key: string; name: string }[];
  region: number[];
  views: Record<ViewName, { x: number[]; y: number[]; aspect: number }>;
  groups: { input: number[][][]; output: number[][] };
}

export type ViewName = "top" | "side";

/** 영역 색 (어두운 배경용, 검증된 범주형 팔레트의 다크 단계). 0번(기타)은 중립 회색. */
export const REGION_COLORS: Record<string, string> = {
  other: "#8a857a",
  visual: "#3987e5",
  mb: "#199e70",
  dan: "#d55181",
  dn: "#c98500",
  motor: "#d95926",
  vnc: "#9085e9",
};

const BG = "#06070d";
const BODY_FILL = "#121630";
const BODY_LINE = "#2f3866";
const DOT_DIM = "rgba(200, 215, 255, 0.12)";
const TEXT = "#e8ecf7";
const TEXT_MUTED = "#8b93ad";
const LEG_IDLE = "#4a5378";

interface Pt {
  x: number;
  y: number;
}

const LEG_NAMES = ["왼 앞다리", "왼 가운뎃다리", "오른 가운뎃다리", "오른 앞다리"];

export class FlyRenderer {
  readonly layout: Layout;
  view: ViewName = "top";
  private nx: Record<ViewName, Float32Array>;
  private ny: Record<ViewName, Float32Array>;
  private glow: Float32Array;
  private staticLayer: HTMLCanvasElement | null = null;
  private staticKey = "";
  private anat: ReturnType<FlyRenderer["anatomy"]> | null = null;
  private lastActive: Activity["active"] | null = null;
  private tapAge = [1e9, 1e9, 1e9, 1e9];
  private prevPressed = [false, false, false, false];

  constructor(layout: Layout) {
    this.layout = layout;
    const q = layout.quant;
    const dec = (a: number[]) => Float32Array.from(a, (v) => v / q);
    this.nx = { top: dec(layout.views.top.x), side: dec(layout.views.side.x) };
    this.ny = { top: dec(layout.views.top.y), side: dec(layout.views.side.y) };
    this.glow = new Float32Array(layout.n);
  }

  // ------------------------------------------------------------ 좌표 변환
  private frame(w: number, h: number) {
    const v = this.view;
    const xs = this.nx[v];
    const ys = this.ny[v];
    let x0 = Infinity,
      x1 = -Infinity,
      y0 = Infinity,
      y1 = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      x0 = Math.min(x0, xs[i]!);
      x1 = Math.max(x1, xs[i]!);
      y0 = Math.min(y0, ys[i]!);
      y1 = Math.max(y1, ys[i]!);
    }
    // top 보기는 다리·배가 들어갈 여백을 둔다
    const padX = v === "top" ? 0.42 : 0.08;
    const padTop = v === "top" ? 0.22 : 0.08;
    const padBottom = v === "top" ? 0.62 : 0.08;
    const cw = (x1 - x0) * (1 + 2 * padX);
    const ch = (y1 - y0) * (1 + padTop + padBottom);
    const s = Math.min(w / cw, h / ch);
    const ox = (w - (x1 - x0) * s) / 2 - x0 * s;
    const oy = (y1 - y0) * padTop * s + (h - ch * s) / 2 - y0 * s;
    return { s, ox, oy, toPx: (x: number, y: number): Pt => ({ x: ox + x * s, y: oy + y * s }) };
  }

  private centroid(idx: number[], toPx: (x: number, y: number) => Pt): Pt {
    const xs = this.nx[this.view];
    const ys = this.ny[this.view];
    let sx = 0,
      sy = 0;
    for (const i of idx) {
      sx += xs[i]!;
      sy += ys[i]!;
    }
    return toPx(sx / Math.max(1, idx.length), sy / Math.max(1, idx.length));
  }

  /** 영역 코드 집합에 속한 점들의 (백분위로 자른) 경계 상자 */
  private bbox(codes: number[], toPx: (x: number, y: number) => Pt) {
    const xs: number[] = [];
    const ys: number[] = [];
    const { region } = this.layout;
    for (let i = 0; i < region.length; i++) {
      if (!codes.includes(region[i]!)) continue;
      const p = toPx(this.nx[this.view][i]!, this.ny[this.view][i]!);
      xs.push(p.x);
      ys.push(p.y);
    }
    const pick = (a: number[], q: number) => {
      const s = [...a].sort((u, v) => u - v);
      return s[Math.floor((s.length - 1) * q)] ?? 0;
    };
    return { x0: pick(xs, 0.01), x1: pick(xs, 0.99), y0: pick(ys, 0.01), y1: pick(ys, 0.99) };
  }

  private code(key: string): number {
    return this.layout.regions.findIndex((r) => r.key === key);
  }

  // ------------------------------------------------------------ 정적 레이어
  private anatomy(w: number, h: number) {
    const f = this.frame(w, h);
    const brainCodes = ["other", "visual", "mb", "dan", "dn"].map((k) => this.code(k));
    const vncCodes = ["vnc", "motor"].map((k) => this.code(k));
    const brain = this.bbox(brainCodes, f.toPx);
    const vnc = this.bbox(vncCodes, f.toPx);
    const out = this.layout.groups.output;
    const legBase = out.map((idx) => this.centroid(idx, f.toPx));
    return { f, brain, vnc, legBase };
  }

  private buildStatic(w: number, h: number, dpr: number): HTMLCanvasElement {
    const c = document.createElement("canvas");
    c.width = w * dpr;
    c.height = h * dpr;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    const { f, brain, vnc } = this.anatomy(w, h);

    if (this.view === "top") {
      const cx = (brain.x0 + brain.x1) / 2;
      ctx.fillStyle = BODY_FILL;
      ctx.strokeStyle = BODY_LINE;
      ctx.lineWidth = 1.5;
      const ell = (x: number, y: number, rx: number, ry: number) => {
        ctx.beginPath();
        ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      };
      // 날개 (반투명)
      const thY = (vnc.y0 + vnc.y1) / 2;
      const thRx = ((vnc.x1 - vnc.x0) / 2) * 1.55;
      const thRy = ((vnc.y1 - vnc.y0) / 2) * 1.12;
      ctx.save();
      ctx.globalAlpha = 0.35;
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.ellipse(
          cx + side * thRx * 0.9,
          thY + thRy * 1.5,
          thRx * 0.55,
          thRy * 1.5,
          side * -0.25,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
      // 배, 가슴, 머리
      ell(cx, vnc.y1 + thRy * 1.25, thRx * 0.8, thRy * 1.35);
      ell(cx, thY, thRx, thRy);
      const hdRx = ((brain.x1 - brain.x0) / 2) * 1.04;
      const hdRy = ((brain.y1 - brain.y0) / 2) * 1.25;
      const hdY = (brain.y0 + brain.y1) / 2;
      ell(cx, hdY, hdRx, hdRy);
      // 겹눈 (시각엽 위치)
      ctx.save();
      ctx.globalAlpha = 0.6;
      ctx.fillStyle = "#3a1530";
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.ellipse(cx + side * hdRx * 0.78, hdY, hdRx * 0.3, hdRy * 0.85, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      // 머리 방향 표시
      ctx.fillStyle = TEXT_MUTED;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("▲ 머리 (위에서 본 모습)", cx, 12);
    } else {
      ctx.fillStyle = TEXT_MUTED;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("◀ 앞   (왼쪽에서 본 중추신경계)", 10, 14);
      ctx.textAlign = "center";
      ctx.fillText("뇌", (brain.x0 + brain.x1) / 2, brain.y0 - 6);
      ctx.fillText("복부신경삭", (vnc.x0 + vnc.x1) / 2, vnc.y1 + 14);
    }

    // 전체 뉴런 (흐린 점)
    const xs = this.nx[this.view];
    const ys = this.ny[this.view];
    ctx.fillStyle = DOT_DIM;
    for (let i = 0; i < xs.length; i++) {
      const p = f.toPx(xs[i]!, ys[i]!);
      ctx.fillRect(p.x - 0.6, p.y - 0.6, 1.2, 1.2);
    }
    return c;
  }

  // ------------------------------------------------------------ 그리기
  /**
   * @param activity 최신 뇌 활동 (모드 A 는 null)
   * @param pressed  레인별 현재 누름 상태
   * @param dt       직전 프레임 이후 시간(ms), 빛 잔상 감쇠용
   */
  draw(canvas: HTMLCanvasElement, activity: Activity | null, pressed: boolean[], dt: number): void {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 440;
    const h = canvas.clientHeight || 560;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const key = `${this.view}:${w}x${h}@${dpr}`;
    if (key !== this.staticKey) {
      this.staticLayer = this.buildStatic(w, h, dpr);
      this.anat = this.anatomy(w, h);
      this.staticKey = key;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx || !this.staticLayer || !this.anat) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.staticLayer, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { f, brain, vnc, legBase } = this.anat;

    // 활동 잔상: 새 프레임이면 갱신, 아니면 감쇠
    const decay = Math.exp(-dt / 120);
    for (let i = 0; i < this.glow.length; i++) this.glow[i]! *= decay;
    if (activity?.active && activity.active !== this.lastActive) {
      const [idx, lvl] = activity.active;
      for (let k = 0; k < idx.length; k++) {
        const i = idx[k]!;
        this.glow[i] = Math.max(this.glow[i]!, lvl[k]! / 255);
      }
      this.lastActive = activity.active;
    }

    // 활성 뉴런 (영역 색, 가산 합성)
    const xs = this.nx[this.view];
    const ys = this.ny[this.view];
    const colors = this.layout.regions.map((r) => REGION_COLORS[r.key] ?? "#ffffff");
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < this.glow.length; i++) {
      const g = this.glow[i]!;
      if (g < 0.02) continue;
      const p = f.toPx(xs[i]!, ys[i]!);
      ctx.globalAlpha = Math.min(1, 0.25 + g);
      ctx.fillStyle = colors[this.layout.region[i]!]!;
      const r = 1.2 + 2.4 * Math.sqrt(g);
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    if (this.view === "top") {
      this.drawLegs(ctx, legBase, vnc, activity, pressed, dt);
      this.drawEyes(ctx, brain, activity);
    }
  }

  private drawLegs(
    ctx: CanvasRenderingContext2D,
    base: Pt[],
    vnc: { x0: number; x1: number; y0: number; y1: number },
    activity: Activity | null,
    pressed: boolean[],
    dt: number,
  ) {
    const scale = (vnc.x1 - vnc.x0) * 0.95;
    const motor = activity?.motor ?? [0, 0, 0, 0];
    const mMax = Math.max(0.02, ...motor);
    // 레인 → (다리 기준점, 좌/우, 앞(-1)/가운데(0))
    const legs = [
      { lane: 0, side: -1, fore: -1 },
      { lane: 1, side: -1, fore: 0 },
      { lane: 2, side: 1, fore: 0 },
      { lane: 3, side: 1, fore: -1 },
    ];
    // 뒷다리(학습에 안 쓰임): 가운뎃다리에서 앞다리 반대쪽으로
    const hind = [
      { b: base[1], o: base[0], side: -1 },
      { b: base[2], o: base[3], side: 1 },
    ];
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const hl of hind) {
      if (!hl.b || !hl.o) continue;
      const b = { x: hl.b.x, y: hl.b.y + (hl.b.y - hl.o.y) * 0.9 };
      this.leg(ctx, b, hl.side, 1, scale, LEG_IDLE, 2, 0);
    }
    for (const L of legs) {
      const b = base[L.lane];
      if (!b) continue;
      const isDown = !!pressed[L.lane];
      if (isDown && !this.prevPressed[L.lane]) this.tapAge[L.lane] = 0;
      this.prevPressed[L.lane] = isDown;
      this.tapAge[L.lane]! += dt;
      const a = Math.min(1, (motor[L.lane] ?? 0) / mMax);
      const color = isDown ? "#2fd3ff" : mix(LEG_IDLE, REGION_COLORS.motor!, a * 0.8);
      const tip = this.leg(
        ctx,
        b,
        L.side,
        L.fore,
        scale,
        color,
        isDown ? 4 : 2.5 + a,
        isDown ? 1 : 0,
      );
      // 누르는 순간 물결
      const age = this.tapAge[L.lane]!;
      if (age < 350) {
        ctx.save();
        ctx.strokeStyle = "#2fd3ff";
        ctx.globalAlpha = 1 - age / 350;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 6 + age / 18, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      ctx.fillStyle = isDown ? "#2fd3ff" : TEXT_MUTED;
      ctx.font = `${isDown ? 700 : 500} 11px system-ui, sans-serif`;
      ctx.textAlign = L.side < 0 ? "right" : "left";
      ctx.fillText(
        `${LEG_NAMES[L.lane]} · ${LANE_KEY_LABELS[L.lane]}`,
        tip.x + L.side * 8,
        tip.y + (L.fore < 0 ? -6 : 4),
      );
    }
  }

  /** 다리 하나: 기준점 → 무릎 → 발끝. 반환값은 발끝 위치. */
  private leg(
    ctx: CanvasRenderingContext2D,
    b: Pt,
    side: number,
    fore: number,
    scale: number,
    color: string,
    width: number,
    press: number,
  ): Pt {
    const reach = 1 + 0.18 * press; // 누르면 다리를 뻗는다
    const knee = { x: b.x + side * scale * 0.9, y: b.y + fore * scale * 0.55 - scale * 0.05 };
    const tip = {
      x: knee.x + side * scale * 0.55 * reach,
      y: knee.y + (fore < 0 ? -1 : fore > 0 ? 1 : 0.35) * scale * 0.75 * reach,
    };
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(knee.x, knee.y);
    ctx.lineTo(tip.x, tip.y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, width * 0.9, 0, Math.PI * 2);
    ctx.fill();
    return tip;
  }

  /** 겹눈 위에 레인별 시야(세로 4칸)를 육각형으로 표시 */
  private drawEyes(
    ctx: CanvasRenderingContext2D,
    brain: { x0: number; x1: number; y0: number; y1: number },
    activity: Activity | null,
  ) {
    const input = activity?.input;
    const cx = (brain.x0 + brain.x1) / 2;
    const hdRx = ((brain.x1 - brain.x0) / 2) * 1.04;
    const hdY = (brain.y0 + brain.y1) / 2;
    const max = Math.max(0.05, ...(input?.flat() ?? [0]));
    const r = Math.max(4, hdRx * 0.075);
    // 레인 0,1 = 왼눈(바깥, 안쪽), 레인 2,3 = 오른눈(안쪽, 바깥)
    const cols = [
      { lane: 0, x: cx - hdRx * 0.9 },
      { lane: 1, x: cx - hdRx * 0.66 },
      { lane: 2, x: cx + hdRx * 0.66 },
      { lane: 3, x: cx + hdRx * 0.9 },
    ];
    for (const col of cols) {
      const bins = input?.[col.lane] ?? [0, 0, 0, 0];
      bins.forEach((v, b) => {
        const y = hdY + (b - (bins.length - 1) / 2) * r * 1.8;
        const a = Math.min(1, v / max);
        hex(ctx, col.x, y, r);
        ctx.fillStyle =
          a > 0.05 ? `rgba(47, 211, 255, ${0.25 + 0.75 * a})` : "rgba(200,215,255,0.08)";
        ctx.fill();
      });
    }
    ctx.fillStyle = TEXT;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("겹눈 시야", cx - hdRx * 0.78, hdY + r * 5);
    ctx.fillText("겹눈 시야", cx + hdRx * 0.78, hdY + r * 5);
  }
}

function hex(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 3) * k + Math.PI / 6;
    const px = x + r * Math.cos(a);
    const py = y + r * Math.sin(a);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function mix(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ch = (s: number) =>
    Math.round(((pa >> s) & 255) * (1 - t) + ((pb >> s) & 255) * t)
      .toString(16)
      .padStart(2, "0");
  return `#${ch(16)}${ch(8)}${ch(0)}`;
}
