import type { Chart, Stats } from "../game/types";

export type Mode = "A" | "B" | "C";
export type JobStatus = "queued" | "running" | "done" | "failed" | "stopped";

export interface Progress {
  step: number;
  total: number;
  episodes: number;
  accuracy: number | null;
  reward: number | null;
  [key: string]: unknown;
}

export interface Job {
  id: string;
  mode: Mode;
  request: { mode: Mode; steps: number | null; episodes: number | null; seed: number };
  status: JobStatus;
  created: number;
  finished: number | null;
  checkpoint: string | null;
  error: string | null;
  latest: Progress | null;
  history?: Progress[];
}

export interface CheckpointSummary {
  id: string;
  mode: Mode;
  savedAt: string;
  accuracy: number | null;
  trainedSteps: number | null;
  trainedEpisodes: number | null;
  plasticity: { synapses: number; changed: number; meanAbsChange: number } | null;
}

export interface Activity {
  input: number[][];
  motor: number[];
  kc: number;
  mbon: number;
  danReward: number;
  danPunish: number;
  activeFraction: number;
  pressProb?: number[];
  /** 활발한 뉴런: [인덱스 목록, 세기 0~255 목록] */
  active?: [number[], number[]];
}

export interface Frame {
  t: number;
  inputs: [number, boolean][];
  activity?: Activity;
}

export type PlayMessage =
  | { type: "ready"; mode: Mode; profile: string | null; chart: Chart; leadMs: number }
  | { type: "frames"; frames: Frame[] }
  | { type: "end"; stats: Stats }
  | { type: "error"; message: string };

export interface BrainSummary {
  available: boolean;
  profile: string;
  neurons?: number;
  synapses?: number;
  kcMbonSynapses?: number;
  groups?: {
    input: number[][];
    output: number[];
    dan_reward: number;
    dan_punish: number;
    kc: number;
    mbon: number;
  };
}

export interface PlasticityRow {
  kc: string;
  mbon: string;
  dw: number;
  rel: number;
}

export interface Plasticity {
  checkpoint: string;
  synapses: number;
  changed: number;
  strengthened: number;
  weakened: number;
  totalRatio: number;
  histogram: { edges: number[]; counts: number[] };
  topUp: PlasticityRow[];
  topDown: PlasticityRow[];
  laneBias: number[];
}
