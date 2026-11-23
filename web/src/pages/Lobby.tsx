/**
 * 로비: 곡·난이도, 상대 초파리, 옵션을 고르고 START 로 대결을 시작한다.
 */
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { CheckpointSummary, Mode } from "../api/types";
import { Logo } from "../components/Logo";
import { unlockAudio } from "../game/audio";
import { LANE_KEYS_TEXT } from "../game/keys";
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type PlaySettings } from "../game/settings";
import type { ChartSummary } from "../game/types";

const DIFF_RE = /\((EASY|NORMAL|HARD)\)\s*$/;
const DIFF_ORDER = ["EASY", "NORMAL", "HARD", "—"];

const MODE_INFO: Record<Mode, { name: string; desc: string }> = {
  A: { name: "A · MLP", desc: "일반 인공신경망 (초파리 아님, 비교 기준)" },
  B: { name: "B · 커넥톰 매핑", desc: "초파리 뇌는 그대로, 해석기만 학습" },
  C: { name: "C · 도파민 가소성", desc: "도파민으로 뇌 시냅스가 직접 학습" },
};

interface Song {
  title: string;
  artist: string;
  bpm: number;
  audio: boolean;
  charts: { id: string; diff: string; notes: number }[];
}

function groupSongs(charts: ChartSummary[]): Song[] {
  const map = new Map<string, Song>();
  for (const c of charts) {
    const diff = c.title.match(DIFF_RE)?.[1] ?? "—";
    const title = c.title.replace(DIFF_RE, "").trim();
    const key = `${title}|${c.artist}`;
    const song = map.get(key) ?? {
      title,
      artist: c.artist,
      bpm: c.bpm,
      audio: c.audio,
      charts: [],
    };
    song.charts.push({ id: c.id, diff, notes: c.notes });
    map.set(key, song);
  }
  const songs = [...map.values()];
  for (const s of songs)
    s.charts.sort((a, b) => DIFF_ORDER.indexOf(a.diff) - DIFF_ORDER.indexOf(b.diff));
  // 음원 있는 곡 먼저
  return songs.sort((a, b) => Number(b.audio) - Number(a.audio) || a.title.localeCompare(b.title));
}

export default function Lobby() {
  const navigate = useNavigate();
  const [charts, setCharts] = useState<ChartSummary[]>([]);
  const [ckpts, setCkpts] = useState<CheckpointSummary[]>([]);
  const [settings, setSettings] = useState<PlaySettings>(() => loadSettings() ?? DEFAULT_SETTINGS);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .charts()
      .then(setCharts)
      .catch(() => setError("채보 목록을 불러오지 못했습니다. 서버가 켜져 있는지 확인하세요."));
    api
      .get<CheckpointSummary[]>("/api/checkpoints")
      .then((cs) => {
        setCkpts(cs);
        // 처음이거나 저장된 상대가 사라졌으면 B(없으면 첫 번째)를 기본으로.
        // 사용자가 직접 "혼자 플레이"를 골라 저장해 둔 경우는 그대로 둔다.
        const savedSolo = loadSettings()?.ckptId === "";
        setSettings((s) =>
          (s.ckptId === "" && savedSolo) || cs.some((c) => c.id === s.ckptId)
            ? s
            : { ...s, ckptId: (cs.find((c) => c.mode === "B") ?? cs[0])?.id ?? "" },
        );
      })
      .catch(() => undefined);
  }, []);

  const songs = useMemo(() => groupSongs(charts), [charts]);
  const set = (patch: Partial<PlaySettings>) => setSettings((s) => ({ ...s, ...patch }));
  const chartOk = charts.some((c) => c.id === settings.chartId);

  const start = () => {
    if (!chartOk) return;
    unlockAudio(); // 이 클릭 안에서 오디오를 열어 둬야 플레이 화면에서 바로 소리가 난다
    saveSettings(settings);
    navigate("/play", { state: settings });
  };

  return (
    <div className="lobby">
      <div className="lobby-hero">
        <Logo />
        <p className="muted">실제 초파리 뇌 배선으로 만든 AI 와 리듬게임 대결</p>
      </div>

      <div className="lobby-grid">
        <section className="panel lobby-songs">
          <h2>곡 선택</h2>
          {error && <p className="error">{error}</p>}
          <ul className="song-list">
            {songs.map((s) => {
              const active = s.charts.some((c) => c.id === settings.chartId);
              return (
                <li key={`${s.title}|${s.artist}`} className={`song ${active ? "on" : ""}`}>
                  <div className="song-meta">
                    <b>{s.title}</b>
                    <span className="muted small">
                      {s.audio ? "♪ " : ""}
                      {s.artist || "—"} · {Math.round(s.bpm)} BPM
                    </span>
                  </div>
                  <div className="song-diffs">
                    {s.charts.map((c) => (
                      <button
                        key={c.id}
                        className={`diff-btn ${c.diff.toLowerCase()} ${
                          settings.chartId === c.id ? "on" : ""
                        }`}
                        onClick={() => set({ chartId: c.id })}
                        aria-pressed={settings.chartId === c.id}
                      >
                        <span>{c.diff === "—" ? "PLAY" : c.diff}</span>
                        <small>{c.notes} notes</small>
                      </button>
                    ))}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        <aside className="lobby-side">
          <section className="panel">
            <h2>상대</h2>
            <div className="opponents">
              <button
                className={`opponent ${settings.ckptId === "" ? "on" : ""}`}
                onClick={() => set({ ckptId: "" })}
                aria-pressed={settings.ckptId === ""}
              >
                <b>혼자 플레이</b>
                <span className="muted small">초파리 없이 연습</span>
              </button>
              {ckpts.map((c) => (
                <button
                  key={c.id}
                  className={`opponent mode-${c.mode.toLowerCase()} ${
                    settings.ckptId === c.id ? "on" : ""
                  }`}
                  onClick={() => set({ ckptId: c.id })}
                  aria-pressed={settings.ckptId === c.id}
                >
                  <b>{MODE_INFO[c.mode].name}</b>
                  <span className="muted small">{MODE_INFO[c.mode].desc}</span>
                  <span className="opp-acc">
                    검증 정확도 <b>{c.accuracy?.toFixed(1) ?? "-"}%</b>
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>옵션</h2>
            <label className="field">
              노트 속도 <b className="speed-val">{settings.speed.toFixed(1)}</b>
              <input
                type="range"
                min={0.3}
                max={1.5}
                step={0.1}
                value={settings.speed}
                onChange={(e) => set({ speed: Number(e.target.value) })}
              />
            </label>
            <label className="field check">
              <input
                type="checkbox"
                checked={settings.sound}
                onChange={(e) => set({ sound: e.target.checked })}
              />
              소리
            </label>
            <p className="muted small">
              키: <kbd>{LANE_KEYS_TEXT}</kbd> · 롱노트는 끝까지 누르고 있으세요
            </p>
          </section>

          <button className="start-btn" onClick={start} disabled={!chartOk}>
            START
          </button>
        </aside>
      </div>
    </div>
  );
}
