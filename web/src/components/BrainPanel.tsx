import type { Activity, Mode } from "../api/types";
import { LANE_KEY_LABELS } from "../game/keys";

const LANE_KEYS = LANE_KEY_LABELS;
const LEG = ["왼 앞다리", "왼 가운뎃다리", "오른 가운뎃다리", "오른 앞다리"];

/** 0~max 값을 순차(단일 색상) 농도로. */
function Cell({ v, max, title }: { v: number; max: number; title: string }) {
  const a = Math.min(1, v / (max || 1));
  return (
    <div
      className="heat"
      title={title}
      style={{
        background: `color-mix(in oklab, var(--seq) ${Math.round(a * 100)}%, var(--seq-0))`,
      }}
    />
  );
}

function Bar({ label, v, max, hint }: { label: string; v: number; max: number; hint?: string }) {
  return (
    <div className="hbar" title={hint ?? `${label}: ${v.toFixed(4)}`}>
      <span className="hbar-label">{label}</span>
      <span className="hbar-track">
        <span
          className="hbar-fill"
          style={{ width: `${Math.min(100, (v / (max || 1)) * 100)}%` }}
        />
      </span>
      <span className="hbar-value mono">{v.toFixed(3)}</span>
    </div>
  );
}

export function BrainPanel({ mode, activity }: { mode: Mode | null; activity: Activity | null }) {
  if (mode === "A") {
    return (
      <p className="muted">모드 A 는 커넥톰을 쓰지 않는 일반 신경망(MLP)이라 뇌 활동이 없습니다.</p>
    );
  }
  if (!activity) return <p className="muted">재생을 시작하면 초파리 뇌 활동이 표시됩니다.</p>;
  const inMax = Math.max(0.05, ...activity.input.flat());
  const moMax = Math.max(0.02, ...activity.motor);
  const grpMax = Math.max(0.02, activity.kc, activity.mbon, activity.danReward, activity.danPunish);
  return (
    <div className="brain-panel">
      <div>
        <h3>시각 입력 (L1/L2)</h3>
        <p className="muted small">레인별 시야 위(먼 노트) → 아래(판정선)</p>
        <div className="heat-grid">
          {activity.input.map((lane, l) => (
            <div key={l} className="heat-col">
              {lane.map((v, b) => (
                <Cell
                  key={b}
                  v={v}
                  max={inMax}
                  title={`레인 ${LANE_KEYS[l]} · 칸 ${b}: ${v.toFixed(3)}`}
                />
              ))}
              <span className="mono small">{LANE_KEYS[l]}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h3>다리 운동 뉴런</h3>
        {activity.motor.map((v, l) => (
          <Bar key={l} label={LEG[l] ?? `레인 ${l}`} v={v} max={moMax} />
        ))}
      </div>
      <div>
        <h3>버섯체 · 도파민</h3>
        <Bar label="Kenyon cell" v={activity.kc} max={grpMax} />
        <Bar label="MBON" v={activity.mbon} max={grpMax} />
        <Bar label="DAN 보상 (PAM)" v={activity.danReward} max={grpMax} />
        <Bar label="DAN 처벌 (PPL1)" v={activity.danPunish} max={grpMax} />
        <p className="muted small">활성 뉴런 비율 {(activity.activeFraction * 100).toFixed(1)}%</p>
      </div>
      {activity.pressProb && (
        <div>
          <h3>누름 확률</h3>
          {activity.pressProb.map((v, l) => (
            <Bar key={l} label={`레인 ${LANE_KEYS[l]}`} v={v} max={1} />
          ))}
        </div>
      )}
    </div>
  );
}
