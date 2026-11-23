/**
 * 음원 파일 없이 WebAudio 로 박자(메트로놈)와 노트 소리를 합성한다.
 * 저작권 걱정 없는 기본 사운드.
 */
import type { Chart } from "./types";

export class SynthAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;

  async start(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.35;
      this.master.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    return this.ctx;
  }

  private blip(at: number, freq: number, dur: number, gain: number, type: OscillatorType) {
    const ctx = this.ctx!;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, at);
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(gain, at + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    osc.connect(g).connect(this.master!);
    osc.start(at);
    osc.stop(at + dur + 0.02);
  }

  /** 채보 시각 0 이 AudioContext 시각 `zeroAt` 이 되도록 박자와 노트 소리를 예약한다. */
  schedule(chart: Chart, zeroAt: number): void {
    if (!this.ctx) return;
    const beatMs = 60000 / chart.bpm;
    const end = chart.notes.reduce((m, n) => Math.max(m, n.t + (n.dur ?? 0)), 0) + 1000;
    for (let i = 0, t = chart.offsetMs; t <= end; i++, t = chart.offsetMs + i * beatMs) {
      const strong = i % 4 === 0;
      this.blip(zeroAt + t / 1000, strong ? 110 : 82, 0.18, strong ? 0.9 : 0.6, "sine");
      this.blip(zeroAt + t / 1000, 2400, 0.03, 0.12, "square");
    }
    const laneFreq = [523.25, 587.33, 659.25, 783.99, 880, 987.77, 1046.5, 1174.66];
    for (const n of chart.notes) {
      this.blip(zeroAt + n.t / 1000, laneFreq[n.lane] ?? 660, 0.09, 0.18, "triangle");
    }
  }

  hit(): void {
    if (!this.ctx) return;
    this.blip(this.ctx.currentTime, 1800, 0.04, 0.15, "square");
  }

  async stop(): Promise<void> {
    if (this.ctx) {
      await this.ctx.close();
      this.ctx = null;
      this.master = null;
    }
  }
}
