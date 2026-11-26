import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { BrainSummary, CheckpointSummary, Plasticity, PlasticityRow } from "../api/types";

function Tile({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v">{typeof v === "number" ? v.toLocaleString() : v}</div>
    </div>
  );
}

/** log2(변화 후/전) 분포 막대그래프. 0 기준 왼쪽=약화, 오른쪽=강화 (발산형 두 색). */
function Histogram({ h }: { h: Plasticity["histogram"] }) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...h.counts);
  const W = 560;
  const H = 180;
  const bw = W / h.counts.length;
  return (
    <div className="chart">
      <div className="legend">
        <span>
          <i style={{ background: "var(--div-neg)" }} />
          약화 (log₂ 비율 &lt; 0)
        </span>
        <span>
          <i style={{ background: "var(--div-pos)" }} />
          강화 (log₂ 비율 &gt; 0)
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H + 24}`} width="100%" role="img" aria-label="가중치 변화 분포">
        {h.counts.map((c, i) => {
          const mid = (h.edges[i]! + h.edges[i + 1]!) / 2;
          const bh = (c / max) * H;
          return (
            <g key={i} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={i * bw} y={0} width={bw} height={H} fill="transparent" />
              <rect
                x={i * bw + 1}
                y={H - bh}
                width={bw - 2}
                height={bh}
                rx={2}
                fill={mid < 0 ? "var(--div-neg)" : "var(--div-pos)"}
                opacity={hover === null || hover === i ? 1 : 0.55}
              />
            </g>
          );
        })}
        <line x1={W / 2} x2={W / 2} y1={0} y2={H} className="crosshair" />
        {[-3, -2, -1, 0, 1, 2, 3].map((v) => (
          <text key={v} x={((v + 3) / 6) * W} y={H + 16} className="tick" textAnchor="middle">
            {v === 0 ? "×1" : v > 0 ? `×${2 ** v}` : `×1/${2 ** -v}`}
          </text>
        ))}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: Math.min(hover * bw, W - 180), top: 30 }}>
          {h.edges[hover]!.toFixed(2)} ~ {h.edges[hover + 1]!.toFixed(2)} (log₂):{" "}
          <b>{h.counts[hover]} 시냅스</b>
        </div>
      )}
    </div>
  );
}

function PairTable({ rows, title }: { rows: PlasticityRow[]; title: string }) {
  return (
    <div className="grow">
      <h3>{title}</h3>
      <table>
        <thead>
          <tr>
            <th>KC 유형</th>
            <th>MBON 유형</th>
            <th>변화량</th>
            <th>상대 변화</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.kc}-${r.mbon}`}>
              <td className="mono">{r.kc}</td>
              <td className="mono">{r.mbon}</td>
              <td className="mono">{r.dw.toFixed(4)}</td>
              <td className="mono">
                {r.rel > 0 ? "+" : ""}
                {(r.rel * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Brain() {
  const [summary, setSummary] = useState<BrainSummary | null>(null);
  const [ckpts, setCkpts] = useState<CheckpointSummary[]>([]);
  const [ckptId, setCkptId] = useState("");
  const [plast, setPlast] = useState<Plasticity | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<BrainSummary>("/api/brain")
      .then(setSummary)
      .catch(() => setError("서버 연결 실패"));
    api
      .get<CheckpointSummary[]>("/api/checkpoints")
      .then((cs) => {
        const c = cs.filter((x) => x.mode === "C");
        setCkpts(c);
        if (c[0]) setCkptId(c[0].id);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!ckptId) return;
    api
      .get<Plasticity>(`/api/brain/plasticity/${encodeURIComponent(ckptId)}`)
      .then(setPlast)
      .catch((e) =>
        setError(e instanceof ApiError ? e.message : "가소성 정보를 불러오지 못했습니다"),
      );
  }, [ckptId]);

  const g = summary?.groups;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <section className="panel">
        <h2>시뮬레이션에 쓰는 초파리 커넥톰</h2>
        {!summary ? (
          <p className="muted">불러오는 중…</p>
        ) : !summary.available ? (
          <p className="muted">
            전처리 결과가 없습니다. <code>python -m flymax.brain.preprocess</code> 를 먼저
            실행하세요.
          </p>
        ) : (
          <>
            <div className="stat-grid">
              <Tile k="프로필" v={summary.profile} />
              <Tile k="뉴런" v={summary.neurons ?? 0} />
              <Tile k="연결" v={summary.synapses ?? 0} />
              <Tile k="KC→MBON (가소성 후보)" v={summary.kcMbonSynapses ?? 0} />
            </div>
            {g && (
              <table style={{ marginTop: 12 }}>
                <tbody>
                  <tr>
                    <th>시각 입력 (L1/L2)</th>
                    <td className="mono">
                      {g.input
                        .map((l, i) => `레인${i}: ${l.reduce((a, b) => a + b, 0)}`)
                        .join(" · ")}
                    </td>
                  </tr>
                  <tr>
                    <th>다리 운동 뉴런</th>
                    <td className="mono">{g.output.join(" · ")}</td>
                  </tr>
                  <tr>
                    <th>Kenyon cell / MBON</th>
                    <td className="mono">
                      {g.kc} / {g.mbon}
                    </td>
                  </tr>
                  <tr>
                    <th>도파민 뉴런 PAM (보상) / PPL1 (처벌)</th>
                    <td className="mono">
                      {g.dan_reward} / {g.dan_punish}
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
            <p className="muted small">
              데이터: MaleCNS v1.0 (Google Research · HHMI Janelia, CC-BY 4.0)
            </p>
          </>
        )}
      </section>

      <section className="panel">
        <h2>모드 C 가 바꾼 시냅스</h2>
        {ckpts.length === 0 ? (
          <p className="muted">모드 C 체크포인트가 없습니다.</p>
        ) : (
          <>
            <div className="controls">
              <label className="field">
                체크포인트
                <select value={ckptId} onChange={(e) => setCkptId(e.target.value)}>
                  {ckpts.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} · {c.accuracy?.toFixed(1) ?? "-"}%
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {plast && plast.checkpoint === ckptId && (
              <>
                <div className="stat-grid" style={{ marginTop: 12 }}>
                  <Tile k="가소성 시냅스" v={plast.synapses} />
                  <Tile k="바뀐 시냅스" v={plast.changed} />
                  <Tile k="강화" v={plast.strengthened} />
                  <Tile k="약화" v={plast.weakened} />
                  <Tile k="전체 가중치 비율" v={`×${plast.totalRatio.toFixed(3)}`} />
                </div>
                <h3 style={{ marginTop: 16 }}>바뀐 시냅스의 변화 비율 분포</h3>
                <Histogram h={plast.histogram} />
                <div className="row" style={{ marginTop: 12 }}>
                  <PairTable rows={plast.topUp} title="가장 강해진 연결 (KC 유형 → MBON 유형)" />
                  <PairTable rows={plast.topDown} title="가장 약해진 연결" />
                </div>
              </>
            )}
          </>
        )}
        {error && <p className="error">{error}</p>}
      </section>
    </div>
  );
}
