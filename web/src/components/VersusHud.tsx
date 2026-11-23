/**
 * 플레이 중 대결 상황판: 곡 정보, 점수 차 줄다리기 막대, 실시간 정확도 비교.
 */
export interface HudState {
  title: string;
  artist: string;
  bpm: number;
  progress: number;
  human: { score: number; rate: number; combo: number };
  fly: { name: string; score: number; rate: number; combo: number } | null;
}

const DIFF_RE = /\((EASY|NORMAL|HARD)\)\s*$/;

export function VersusHud({ hud, onStop }: { hud: HudState; onStop?: () => void }) {
  const diff = hud.title.match(DIFF_RE)?.[1] ?? null;
  const title = hud.title.replace(DIFF_RE, "").trim();
  const lead = hud.fly ? hud.human.score - hud.fly.score : 0;
  // 줄다리기: 점수 차 ±200,000 을 막대 끝으로
  const share = hud.fly ? Math.max(-1, Math.min(1, lead / 200_000)) : 0;

  return (
    <div className="hud">
      <div className="hud-song">
        <div className="hud-title">
          <b>{title}</b>
          {diff && <span className={`diff ${diff.toLowerCase()}`}>{diff}</span>}
          <span className="muted small">
            {hud.artist || "—"} · {Math.round(hud.bpm)} BPM
          </span>
          {onStop && (
            <button className="hud-stop" onClick={onStop}>
              중지
            </button>
          )}
        </div>
        <div className="hud-progress" aria-label="곡 진행">
          <span style={{ width: `${Math.round(hud.progress * 100)}%` }} />
        </div>
      </div>

      {hud.fly ? (
        <>
          <div className="hud-lead">
            <div className="hud-lead-label">
              {lead === 0 ? "동점" : lead > 0 ? "내가 앞서는 중" : "상대가 앞서는 중"}
              <b>{Math.abs(lead).toLocaleString()}</b>
            </div>
            <div className="tug" role="img" aria-label={`점수 차 ${lead}`}>
              <span className="tug-me" style={{ width: `${50 + share * 50}%` }} />
              <span className="tug-fly" style={{ width: `${50 - share * 50}%` }} />
              <i className="tug-mid" />
            </div>
          </div>
          <div className="hud-rates">
            <div className="me">
              <span className="who">나</span>
              <b>{hud.human.rate.toFixed(2)}%</b>
              <span className="muted small">{hud.human.score.toLocaleString()}</span>
            </div>
            <span className="rate-tag">RATE</span>
            <div className="fly">
              <span className="who">{hud.fly.name}</span>
              <b>{hud.fly.rate.toFixed(2)}%</b>
              <span className="muted small">{hud.fly.score.toLocaleString()}</span>
            </div>
          </div>
        </>
      ) : (
        <div className="hud-rates solo">
          <div className="me">
            <span className="who">나</span>
            <b>{hud.human.rate.toFixed(2)}%</b>
            <span className="muted small">{hud.human.score.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
