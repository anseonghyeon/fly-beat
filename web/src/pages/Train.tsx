import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, wsUrl } from "../api/client";
import type { CheckpointSummary, Job, Mode, Progress } from "../api/types";
import { LineChart } from "../components/LineChart";

const MODES: { mode: Mode; name: string; desc: string; unit: "steps" | "episodes"; def: number }[] =
  [
    {
      mode: "A",
      name: "A · 일반 딥러닝",
      desc: "게임 화면을 MLP 정책망에 바로 넣고 SB3 PPO 로 학습합니다. 비교 기준선입니다.",
      unit: "steps",
      def: 300_000,
    },
    {
      mode: "B",
      name: "B · 감각-행동 매핑",
      desc: "커넥톰은 고정하고, 운동·하행 뉴런 활동 → 키 입력 리드아웃만 PPO 로 학습합니다.",
      unit: "steps",
      def: 1_000_000,
    },
    {
      mode: "C",
      name: "C · 도파민 가소성",
      desc: "보상을 도파민 뉴런 자극으로 주고, 3요소 규칙으로 KC→MBON 시냅스만 바꿉니다. 역전파 없음.",
      unit: "episodes",
      def: 64,
    },
  ];

const MODE_COLOR: Record<Mode, string> = { A: "--mode-a", B: "--mode-b", C: "--mode-c" };

export default function Train() {
  const [mode, setMode] = useState<Mode>("A");
  const [amount, setAmount] = useState<number>(300_000);
  const [seed, setSeed] = useState(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [ckpts, setCkpts] = useState<CheckpointSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [hist, setHist] = useState<{ id: string | null; items: Progress[] }>({
    id: null,
    items: [],
  });
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const spec = MODES.find((m) => m.mode === mode)!;

  const refresh = useCallback(() => {
    api
      .get<Job[]>("/api/train")
      .then(setJobs)
      .catch(() => undefined);
    api
      .get<CheckpointSummary[]>("/api/checkpoints")
      .then(setCkpts)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  // 선택한 작업의 진행 상황을 WebSocket 으로 받는다
  useEffect(() => {
    wsRef.current?.close();
    if (!selected) return;
    api
      .get<Job>(`/api/train/${encodeURIComponent(selected)}`)
      .then((j) => setHist({ id: selected, items: j.history ?? [] }))
      .catch(() => undefined);
    const ws = new WebSocket(wsUrl(`/ws/train/${encodeURIComponent(selected)}`));
    wsRef.current = ws;
    let first = true;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(String(ev.data)) as { type: string; job?: Job; progress?: Progress[] };
      if (msg.type !== "update" || !msg.progress) return;
      const items = msg.progress;
      if (first) {
        first = false; // 첫 메시지는 REST 로 받은 history 와 겹친다
        setHist({ id: selected, items });
      } else setHist((h) => ({ id: selected, items: [...h.items, ...items] }));
      if (msg.job && msg.job.status !== "running") refresh();
    };
    return () => ws.close();
  }, [selected, refresh]);

  const start = async () => {
    setError(null);
    try {
      const body =
        spec.unit === "steps" ? { mode, steps: amount, seed } : { mode, episodes: amount, seed };
      const job = await api.post<Job>("/api/train", body);
      setSelected(job.id);
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "학습을 시작하지 못했습니다");
    }
  };

  const stop = async (id: string) => {
    await api.post(`/api/train/${encodeURIComponent(id)}/stop`, {}).catch(() => undefined);
    refresh();
  };

  const running = jobs.some((j) => j.status === "running" || j.status === "queued");
  const job = jobs.find((j) => j.id === selected);
  const history = hist.id === selected ? hist.items : [];
  const points = history
    .filter((p) => p.accuracy !== null)
    .map((p) => ({ x: p.step, y: p.accuracy as number }));
  const best: Partial<Record<Mode, CheckpointSummary>> = {};
  for (const c of ckpts)
    if (c.accuracy !== null && (!best[c.mode] || c.accuracy > (best[c.mode]!.accuracy ?? 0)))
      best[c.mode] = c;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="row">
        <section className="panel grow">
          <h2>새 학습</h2>
          <div className="mode-pick">
            {MODES.map((m) => (
              <label key={m.mode} className={`mode-card ${mode === m.mode ? "on" : ""}`}>
                <input
                  type="radio"
                  name="mode"
                  checked={mode === m.mode}
                  onChange={() => {
                    setMode(m.mode);
                    setAmount(m.def);
                  }}
                />
                <span className="swatch" style={{ background: `var(${MODE_COLOR[m.mode]})` }} />
                <b>{m.name}</b>
                <span className="muted small">{m.desc}</span>
              </label>
            ))}
          </div>
          <div className="controls">
            <label className="field">
              {spec.unit === "steps" ? "학습 스텝" : "학습 에피소드(곡 수)"}
              <input
                type="number"
                value={amount}
                min={spec.unit === "steps" ? 1000 : 1}
                max={spec.unit === "steps" ? 5_000_000 : 1000}
                onChange={(e) => setAmount(Number(e.target.value) || spec.def)}
              />
            </label>
            <label className="field">
              seed
              <input
                type="number"
                value={seed}
                min={0}
                onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
              />
            </label>
            <button className="primary" disabled={running} onClick={() => void start()}>
              학습 시작
            </button>
          </div>
          {running && <p className="muted small">한 번에 하나의 학습만 실행할 수 있습니다.</p>}
          {error && <p className="error">{error}</p>}
        </section>
        <section className="panel grow">
          <h2>모드별 최고 검증 정확도</h2>
          <div className="stat-grid">
            {(["A", "B", "C"] as Mode[]).map((m) => (
              <div className="stat" key={m}>
                <div className="k">
                  <span className="swatch" style={{ background: `var(${MODE_COLOR[m]})` }} /> 모드{" "}
                  {m}
                </div>
                <div className="v">
                  {best[m]?.accuracy !== undefined && best[m]?.accuracy !== null
                    ? `${best[m]!.accuracy!.toFixed(1)}%`
                    : "–"}
                </div>
              </div>
            ))}
          </div>
          <p className="muted small">
            학습에 쓰지 않은 랜덤 채보 4곡 기준. PERFECT 100% · GREAT 70% · GOOD 30% · MISS 0%.
          </p>
        </section>
      </div>

      <section className="panel">
        <h2>
          학습 곡선{" "}
          {job && (
            <span className="muted small">
              · {job.id} (모드 {job.mode})
            </span>
          )}
        </h2>
        {!selected ? (
          <p className="muted">아래 목록에서 학습 작업을 선택하세요.</p>
        ) : points.length < 2 ? (
          <p className="muted">진행 데이터를 기다리는 중…</p>
        ) : (
          <LineChart
            series={[
              { name: `모드 ${job?.mode ?? ""}`, color: MODE_COLOR[job?.mode ?? "A"], points },
            ]}
            xLabel={job?.mode === "C" ? "에피소드" : "스텝"}
            yLabel="정확도 (최근 20판, %)"
            yDomain={[0, 100]}
          />
        )}
      </section>

      <div className="row">
        <section className="panel grow">
          <h2>학습 작업</h2>
          {jobs.length === 0 ? (
            <p className="muted">아직 없습니다.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>작업</th>
                  <th>모드</th>
                  <th>상태</th>
                  <th>진행</th>
                  <th>정확도</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr
                    key={j.id}
                    className={j.id === selected ? "selected" : ""}
                    onClick={() => setSelected(j.id)}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="mono">{j.id}</td>
                    <td>{j.mode}</td>
                    <td>
                      <span className={`badge ${j.status}`}>{j.status}</span>
                      {j.error && <div className="error small">{j.error}</div>}
                    </td>
                    <td className="mono">
                      {j.latest
                        ? `${j.latest.step.toLocaleString()}/${j.latest.total.toLocaleString()}`
                        : "-"}
                    </td>
                    <td>{j.latest?.accuracy?.toFixed(1) ?? "-"}</td>
                    <td>
                      {j.status === "running" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            void stop(j.id);
                          }}
                        >
                          중지
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <section className="panel grow">
          <h2>체크포인트</h2>
          {ckpts.length === 0 ? (
            <p className="muted">아직 없습니다.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>모드</th>
                  <th>검증 정확도</th>
                  <th>학습량</th>
                </tr>
              </thead>
              <tbody>
                {ckpts.map((c) => (
                  <tr key={c.id}>
                    <td className="mono">{c.id}</td>
                    <td>{c.mode}</td>
                    <td>{c.accuracy?.toFixed(2) ?? "-"}%</td>
                    <td className="mono">
                      {c.trainedEpisodes != null
                        ? `${c.trainedEpisodes} 판`
                        : `${(c.trainedSteps ?? 0).toLocaleString()} 스텝`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </div>
  );
}
