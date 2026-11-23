/**
 * 리듬게임 판정 엔진 (사람 플레이용 TS 구현).
 * 기준 구현은 server/flybeat/game/engine.py 이며, 규칙을 바꿀 때는 두 파일을 함께 바꾼다.
 * tests/fixtures/judge/*.json 으로 두 구현의 결과가 같은지 검증한다.
 */
import {
  type Chart,
  type JudgeEvent,
  type Judgment,
  type Note,
  type Stats,
  noteEnd,
} from "./types";

export interface JudgeWindows {
  perfect: number;
  great: number;
  good: number;
}

export const DEFAULT_WINDOWS: JudgeWindows = { perfect: 40, great: 80, good: 120 };

const SCORE_WEIGHT = { PERFECT: 1.0, GREAT: 0.7, GOOD: 0.3, MISS: 0.0 } as const;

export function judge(diff: number, w: JudgeWindows): Judgment {
  const a = Math.abs(diff);
  if (a <= w.perfect) return "PERFECT";
  if (a <= w.great) return "GREAT";
  if (a <= w.good) return "GOOD";
  return "MISS";
}

/** Python 의 round(x, 6) 과 같은 값을 내기 위한 반올림 (fixture 비교용). */
const round6 = (x: number): number => Math.round(x * 1e6) / 1e6;

export class JudgeEngine {
  readonly chart: Chart;
  readonly notes: Note[];
  readonly windows: JudgeWindows;
  readonly lanes: number;
  readonly laneNotes: number[][];
  head: number[] = [];
  holding: (number | null)[] = [];
  pressed: boolean[] = [];
  now = -Infinity;
  finished = false;
  stats!: Stats;
  private score = 0;

  constructor(chart: Chart, windows: JudgeWindows = DEFAULT_WINDOWS) {
    this.chart = chart;
    this.notes = [...chart.notes].sort((a, b) => a.t - b.t || a.lane - b.lane);
    this.windows = windows;
    this.lanes = chart.lanes;
    this.laneNotes = Array.from({ length: this.lanes }, () => []);
    this.notes.forEach((n, i) => this.laneNotes[n.lane]!.push(i));
    this.reset();
  }

  reset(): void {
    this.head = Array(this.lanes).fill(0);
    this.holding = Array(this.lanes).fill(null);
    this.pressed = Array(this.lanes).fill(false);
    this.now = -Infinity;
    this.finished = false;
    this.score = 0;
    this.stats = {
      counts: { PERFECT: 0, GREAT: 0, GOOD: 0, MISS: 0 },
      empty: 0,
      combo: 0,
      maxCombo: 0,
      score: 0,
      judged: 0,
      total: this.notes.reduce((s, n) => s + (n.type === "hold" ? 2 : 1), 0),
      accuracy: 0,
    };
  }

  get durationMs(): number {
    return this.notes.reduce((m, n) => Math.max(m, noteEnd(n)), 0);
  }

  get endTime(): number {
    return this.durationMs + this.windows.good + 1;
  }

  private emit(out: JudgeEvent[], ev: JudgeEvent): void {
    const s = this.stats;
    if (ev.judgment === "EMPTY") {
      s.empty += 1;
    } else {
      s.counts[ev.judgment] += 1;
      s.judged += 1;
      this.score += SCORE_WEIGHT[ev.judgment];
      if (ev.judgment === "MISS") s.combo = 0;
      else {
        s.combo += 1;
        s.maxCombo = Math.max(s.maxCombo, s.combo);
      }
      s.score = round6(this.score);
      s.accuracy = s.judged ? round6((100 * this.score) / s.judged) : 0;
    }
    out.push(ev);
  }

  advance(t: number): JudgeEvent[] {
    if (t < this.now) throw new Error(`시간은 거꾸로 갈 수 없습니다: ${t} < ${this.now}`);
    const out: JudgeEvent[] = [];
    const good = this.windows.good;
    for (let lane = 0; lane < this.lanes; lane++) {
      const idxs = this.laneNotes[lane]!;
      for (;;) {
        const hi = this.holding[lane];
        if (hi !== null && hi !== undefined) {
          const n = this.notes[hi]!;
          const end = noteEnd(n);
          if (end <= t) {
            this.emit(out, {
              t: end,
              lane,
              note: hi,
              part: "tail",
              judgment: "PERFECT",
              offset: 0,
            });
            this.holding[lane] = null;
            continue;
          }
          break;
        }
        const h = this.head[lane]!;
        if (h < idxs.length) {
          const ni = idxs[h]!;
          const n = this.notes[ni]!;
          if (n.t + good < t) {
            const et = n.t + good;
            this.emit(out, { t: et, lane, note: ni, part: "head", judgment: "MISS", offset: 0 });
            if (n.type === "hold")
              this.emit(out, { t: et, lane, note: ni, part: "tail", judgment: "MISS", offset: 0 });
            this.head[lane] = h + 1;
            continue;
          }
        }
        break;
      }
    }
    this.now = Math.max(this.now, t);
    return out;
  }

  press(lane: number, t: number): JudgeEvent[] {
    const out = this.advance(t);
    if (this.pressed[lane]) return out;
    this.pressed[lane] = true;
    const idxs = this.laneNotes[lane]!;
    const h = this.head[lane]!;
    if (this.holding[lane] === null && h < idxs.length) {
      const ni = idxs[h]!;
      const n = this.notes[ni]!;
      const diff = t - n.t;
      if (Math.abs(diff) <= this.windows.good) {
        this.emit(out, {
          t,
          lane,
          note: ni,
          part: "head",
          judgment: judge(diff, this.windows),
          offset: diff,
        });
        this.head[lane] = h + 1;
        if (n.type === "hold") this.holding[lane] = ni;
        return out;
      }
    }
    this.emit(out, { t, lane, note: -1, part: "empty", judgment: "EMPTY", offset: 0 });
    return out;
  }

  release(lane: number, t: number): JudgeEvent[] {
    const out = this.advance(t);
    if (!this.pressed[lane]) return out;
    this.pressed[lane] = false;
    const hi = this.holding[lane];
    if (hi !== null && hi !== undefined) {
      const remaining = noteEnd(this.notes[hi]!) - t;
      this.emit(out, {
        t,
        lane,
        note: hi,
        part: "tail",
        judgment: judge(remaining, this.windows),
        offset: -remaining,
      });
      this.holding[lane] = null;
    }
    return out;
  }

  finish(): JudgeEvent[] {
    const out = this.advance(Math.max(this.endTime, this.now));
    this.finished = true;
    return out;
  }
}

export interface InputEvent {
  t: number;
  lane: number;
  down: boolean;
}

export function runInputs(
  chart: Chart,
  inputs: InputEvent[],
): { events: JudgeEvent[]; stats: Stats } {
  const eng = new JudgeEngine(chart);
  const events: JudgeEvent[] = [];
  for (const inp of inputs) {
    events.push(...(inp.down ? eng.press(inp.lane, inp.t) : eng.release(inp.lane, inp.t)));
  }
  events.push(...eng.finish());
  return { events, stats: eng.stats };
}
