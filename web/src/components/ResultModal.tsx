import { useEffect, useRef } from "react";
import type { Stats } from "../game/types";
import { StatsView } from "./StatsView";

export interface MatchResult {
  chartTitle: string;
  human: Stats;
  fly: { name: string; client: Stats; server: Stats | null } | null;
}

export function ResultModal({ result, onClose }: { result: MatchResult; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const { human, fly } = result;
  const verdict = !fly
    ? null
    : human.accuracy > fly.client.accuracy
      ? "내가 이겼어요!"
      : human.accuracy < fly.client.accuracy
        ? "초파리가 이겼어요!"
        : "무승부예요!";
  const consistent = fly?.server ? fly.server.score === fly.client.score : null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="result-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="result-title">결과 · {result.chartTitle}</h2>
        {verdict && <p className="verdict">{verdict}</p>}
        <div className="result-cols">
          <section>
            <h3>나</h3>
            <StatsView stats={human} />
          </section>
          {fly && (
            <section>
              <h3>초파리 · {fly.name}</h3>
              <StatsView stats={fly.client} />
              {consistent !== null && (
                <p className={consistent ? "muted small" : "error small"}>
                  {consistent
                    ? "브라우저(TS) 판정과 서버(Python) 판정이 일치합니다."
                    : `판정 불일치: 브라우저 ${fly.client.accuracy}% / 서버 ${fly.server?.accuracy}%`}
                </p>
              )}
            </section>
          )}
        </div>
        <div className="modal-actions">
          <button ref={closeRef} className="primary" onClick={onClose}>
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
