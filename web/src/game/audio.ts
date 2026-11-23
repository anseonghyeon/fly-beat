/**
 * 채보에 음원(chart.audio)이 있으면 그 음원을 재생하고,
 * 없으면 WebAudio 로 박자(메트로놈)와 노트 소리를 합성한다.
 */
import type { Chart } from "./types";

/**
 * 앱 전체가 함께 쓰는 AudioContext. 브라우저는 사용자 동작(클릭 등) 안에서 만든 오디오만 재생을 허락하므로
 * 로비의 START 클릭에서 unlockAudio() 를 불러 두면, 화면이 바뀐 뒤에도 바로 소리를 낼 수 있다.
 */
let shared: AudioContext | null = null;

export function unlockAudio(): AudioContext {
  if (!shared || shared.state === "closed") shared = new AudioContext();
  if (shared.state === "suspended") void shared.resume();
  return shared;
}

export class SynthAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private nodes: AudioScheduledSourceNode[] = [];

  async start(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = unlockAudio();
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
    this.nodes.push(osc);
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

  /** 서버의 음원(/api/audio/<name>)을 받아 디코딩한다. */
  async load(name: string): Promise<AudioBuffer> {
    const ctx = await this.start();
    const res = await fetch(`/api/audio/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`음원을 불러오지 못했습니다 (${res.status})`);
    return await ctx.decodeAudioData(await res.arrayBuffer());
  }

  /** 채보 시각 0 이 AudioContext 시각 zeroAt 이 되도록 음원을 재생한다 (이미 지났으면 그만큼 건너뜀). */
  playBuffer(buffer: AudioBuffer, zeroAt: number): void {
    if (!this.ctx || !this.master) return;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    const gain = this.ctx.createGain();
    gain.gain.value = 2.2; // master(0.35) 를 상쇄해 원음에 가깝게
    src.connect(gain).connect(this.master);
    this.nodes.push(src);
    const now = this.ctx.currentTime;
    if (zeroAt >= now) src.start(zeroAt);
    else src.start(now, now - zeroAt);
  }

  /** 음원이 있으면 음원을, 없으면 합성음을 예약한다. */
  async playChart(chart: Chart, zeroAt: number, buffer?: AudioBuffer | null): Promise<void> {
    if (buffer) this.playBuffer(buffer, zeroAt);
    else this.schedule(chart, zeroAt);
  }

  hit(): void {
    if (!this.ctx) return;
    this.blip(this.ctx.currentTime, 1800, 0.04, 0.15, "square");
  }

  /** 이 세션이 예약한 소리를 모두 멈춘다 (공유 AudioContext 는 닫지 않는다). */
  async stop(): Promise<void> {
    for (const n of this.nodes) {
      try {
        n.stop();
      } catch {
        /* 이미 끝난 소리 */
      }
    }
    this.nodes = [];
    this.master?.disconnect();
    this.master = null;
    this.ctx = null;
  }
}
