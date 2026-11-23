/**
 * 결과 화면 (아케이드 연출, fly-beat 고유 스타일).
 * 등급 문자 · 점수 카운트업 · 판정 막대 · FULL COMBO / ALL PERFECT 배지 · 승패
 */
import { useEffect, useRef, useState } from "react";
import { LANE_KEYS_TEXT } from "../game/keys";
import type { Stats } from "../game/types";

export interface MatchResult {
  chartTitle: string;
  artist?: string;
  bpm?: number;
  human: Stats;
  fly: { name: string; client: Stats; server: Stats | null } | null;
}

const MAX_SCORE = 1_000_000;
const DIFF_RE = /\((EASY|NORMAL|HARD)\)\s*$/;
const JUDGES = [
  { key: "PERFECT", color: "#ffd54a" },
  { key: "GREAT", color: "#7bd88f" },
  { key: "GOOD", color: "#6ab0ff" },
  { key: "MISS", color: "#ff6b6b" },
] as const;

export const scoreFromStats = (s: Stats) =>
  Math.round((MAX_SCORE * s.score) / Math.max(1, s.total));

/** fly-beat 등급 (정확도 기준) */
export function gradeOf(acc: number): string {
  if (acc >= 99) return "S+";
  if (acc >= 95) return "S";
  if (acc >= 90) return "A";
  if (acc >= 80) return "B";
  if (acc >= 70) return "C";
  return "D";
}

function useCountUp(target: number, ms = 1100): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const dur = reduced ? 0 : ms;
    let raf = 0;
    const t0 = performance.now();
    const tick = () => {
      const p = dur ? Math.min(1, (performance.now() - t0) / dur) : 1;
      setV(Math.round(target * (1 - (1 - p) ** 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

type Outcome = "WIN" | "LOSE" | "DRAW" | null;

function PlayerCard({
  who,
  sub,
  stats,
  outcome,
  accent,
}: {
  who: string;
  sub?: string;
  stats: Stats;
  outcome: Outcome;
  accent: "me" | "fly";
}) {
  const score = useCountUp(scoreFromStats(stats));
  const grade = gradeOf(stats.accuracy);
  const judged = Math.max(1, stats.judged);
  const fullCombo = stats.judged > 0 && stats.counts.MISS === 0;
  const allPerfect = fullCombo && stats.counts.PERFECT === stats.judged;
  return (
    <section className={`rcard ${accent}`}>
      <header className="rcard-head">
        <div>
          <div className="rcard-who">{who}</div>
          {sub && <div className="rcard-sub">{sub}</div>}
        </div>
        {outcome && <span className={`outcome ${outcome.toLowerCase()}`}>{outcome}</span>}
      </header>

      <div className="rcard-main">
        <div
          className={`grade g-${grade.replace("+", "p").toLowerCase()}`}
          aria-label={`등급 ${grade}`}
        >
          {grade}
        </div>
        <div className="rcard-score">
          <div className="label">SCORE</div>
          <div className="score">{score.toLocaleString()}</div>
          <div className="rate">
            <span className="rate-tag">RATE</span>
            <b>{stats.accuracy.toFixed(2)}%</b>
          </div>
        </div>
      </div>

      {(allPerfect || fullCombo) && (
        <div className={`badge-fc ${allPerfect ? "ap" : ""}`}>
          {allPerfect ? "ALL PERFECT" : "FULL COMBO"}
        </div>
      )}

      <ul className="judges">
        {JUDGES.map((j) => {
          const n = stats.counts[j.key];
          return (
            <li key={j.key}>
              <span className="jname" style={{ color: j.color }}>
                {j.key}
              </span>
              <span className="jbar">
                <i style={{ width: `${(100 * n) / judged}%`, background: j.color }} />
              </span>
              <span className="jnum">{n}</span>
            </li>
          );
        })}
      </ul>
      <div className="rcard-foot">
        <span>
          MAX COMBO <b>{stats.maxCombo}</b>
        </span>
      </div>
    </section>
  );
}

/** 결과 화면 본문 (결과 페이지에서 사용). 다시 하기 버튼에 포커스가 있어 Enter 로 바로 다시 시작 */
export function ResultView({
  result,
  onRetry,
  onLobby,
}: {
  result: MatchResult;
  onRetry: () => void;
  onLobby: () => void;
}) {
  const retryRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    retryRef.current?.focus();
  }, []);

  const { human, fly } = result;
  const diff = result.chartTitle.match(DIFF_RE)?.[1] ?? null;
  const title = result.chartTitle.replace(DIFF_RE, "").trim();
  const hs = scoreFromStats(human);
  const fs = fly ? scoreFromStats(fly.client) : 0;
  const mine: Outcome = !fly ? null : hs > fs ? "WIN" : hs < fs ? "LOSE" : "DRAW";
  const theirs: Outcome = !fly ? null : mine === "WIN" ? "LOSE" : mine === "LOSE" ? "WIN" : "DRAW";

  return (
    <div className="result-card" aria-labelledby="result-title">
      <header className="result-head">
        <span className="result-tag">RESULT</span>
        <h2 id="result-title">{title}</h2>
        {diff && <span className={`diff ${diff.toLowerCase()}`}>{diff}</span>}
        {(result.artist || result.bpm) && (
          <span className="result-meta">
            {result.artist}
            {result.bpm ? ` · ${Math.round(result.bpm)} BPM` : ""}
          </span>
        )}
      </header>

      <div className={`result-cols ${fly ? "" : "solo"}`}>
        <PlayerCard
          who="나"
          sub={`키 ${LANE_KEYS_TEXT}`}
          stats={human}
          outcome={mine}
          accent="me"
        />
        {fly && (
          <PlayerCard who="상대" sub={fly.name} stats={fly.client} outcome={theirs} accent="fly" />
        )}
      </div>

      <footer className="result-foot">
        <button className="result-secondary" onClick={onLobby}>
          로비로
        </button>
        <button ref={retryRef} className="result-close" onClick={onRetry}>
          다시 하기
        </button>
      </footer>
    </div>
  );
}
