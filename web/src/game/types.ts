export type NoteType = "tap" | "hold";

export interface Note {
  t: number;
  lane: number;
  type: NoteType;
  dur?: number;
}

export interface Chart {
  id: string;
  title: string;
  artist?: string;
  bpm: number;
  offsetMs: number;
  lanes: number;
  audio?: string;
  notes: Note[];
}

export interface ChartSummary {
  id: string;
  title: string;
  artist: string;
  bpm: number;
  lanes: number;
  notes: number;
  durationMs: number;
}

export type Judgment = "PERFECT" | "GREAT" | "GOOD" | "MISS" | "EMPTY";
export type Part = "head" | "tail" | "empty";

export interface JudgeEvent {
  t: number;
  lane: number;
  note: number;
  part: Part;
  judgment: Judgment;
  offset: number;
}

export interface Stats {
  counts: Record<"PERFECT" | "GREAT" | "GOOD" | "MISS", number>;
  empty: number;
  combo: number;
  maxCombo: number;
  score: number;
  judged: number;
  total: number;
  accuracy: number;
}

export const noteEnd = (n: Note): number => (n.type === "hold" ? n.t + (n.dur ?? 0) : n.t);
