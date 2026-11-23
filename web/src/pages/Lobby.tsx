/**
 * 로비 (곡 선택 화면).
 * 왼쪽: 선택한 곡 카드 + 내 최고 기록 + 옵션 + START · 오른쪽: 분류 탭·정렬·곡 목록 · 아래: 키 안내
 * 키보드: ↑↓ 곡, ←→ 난이도, Tab 분류, O 상대, [ ] 속도, Enter 시작
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { CheckpointSummary, Mode } from "../api/types";
import { OpponentIcon } from "../components/OpponentIcon";
import { Cover } from "../components/Cover";
import { Logo } from "../components/Logo";
import { gradeOf } from "../components/ResultView";
import { unlockAudio } from "../game/audio";
import { LANE_KEYS_TEXT } from "../game/keys";
import { OPPONENTS, strengthOf } from "../game/opponents";
import {
  DEFAULT_SETTINGS,
  loadRecords,
  loadSettings,
  saveSettings,
  type BestRecord,
  type PlaySettings,
} from "../game/settings";
import type { ChartSummary } from "../game/types";

const DIFF_RE = /\((EASY|NORMAL|HARD)\)\s*$/;
const DIFFS = ["EASY", "NORMAL", "HARD"] as const;
type Diff = (typeof DIFFS)[number] | "—";

/** 상대 목록 순서: ROOKIE FLY(C) → WILD FLY(B) → MACHINE(A) */
const OPP_ORDER: Record<Mode, number> = { C: 1, B: 2, A: 3 };

type SortKey = "title" | "bpm" | "notes";
const SORTS: { key: SortKey; label: string }[] = [
  { key: "title", label: "곡 제목 (A→Z)" },
  { key: "bpm", label: "BPM (낮은 순)" },
  { key: "notes", label: "노트 수 (적은 순)" },
];

interface SongChart {
  id: string;
  diff: Diff;
  notes: number;
  durationMs: number;
}
interface Song {
  key: string;
  title: string;
  artist: string;
  bpm: number;
  audio: boolean;
  charts: SongChart[];
}

function groupSongs(charts: ChartSummary[]): Song[] {
  const map = new Map<string, Song>();
  for (const c of charts) {
    const diff = (c.title.match(DIFF_RE)?.[1] ?? "—") as Diff;
    const title = c.title.replace(DIFF_RE, "").trim();
    const key = `${title}|${c.artist}`;
    const song = map.get(key) ?? {
      key,
      title,
      artist: c.artist,
      bpm: c.bpm,
      audio: c.audio,
      charts: [],
    };
    song.charts.push({ id: c.id, diff, notes: c.notes, durationMs: c.durationMs });
    map.set(key, song);
  }
  const order = (d: Diff) => (d === "—" ? 0 : DIFFS.indexOf(d));
  for (const s of map.values()) s.charts.sort((a, b) => order(a.diff) - order(b.diff));
  return [...map.values()];
}

const categoryOf = (s: Song) => (s.audio ? s.artist || "음원" : "기본 채보");
const fmtTime = (ms: number) =>
  `${Math.floor(ms / 60000)}:${String(Math.round((ms % 60000) / 1000)).padStart(2, "0")}`;
const diffsOf = (s: Song): readonly Diff[] => (s.charts.length > 1 ? DIFFS : (["—"] as const));

function RecordMark({ rec }: { rec?: BestRecord }) {
  if (!rec) return <span className="mark none">–</span>;
  const g = gradeOf(rec.rate);
  return (
    <span className={`mark g-${g.replace("+", "p").toLowerCase()} ${rec.fullCombo ? "fc" : ""}`}>
      {g}
    </span>
  );
}

export default function Lobby() {
  const navigate = useNavigate();
  const [charts, setCharts] = useState<ChartSummary[]>([]);
  const [ckpts, setCkpts] = useState<CheckpointSummary[]>([]);
  const [ckptsLoaded, setCkptsLoaded] = useState(false);
  const [settings, setSettings] = useState<PlaySettings>(() => loadSettings() ?? DEFAULT_SETTINGS);
  const [records] = useState(loadRecords);
  const [category, setCategory] = useState("전체");
  const [sort, setSort] = useState<SortKey>("title");
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    api
      .charts()
      .then(setCharts)
      .catch(() => setError("채보 목록을 불러오지 못했습니다. 서버가 켜져 있는지 확인하세요."));
    api
      .get<CheckpointSummary[]>("/api/checkpoints")
      .then((cs) => {
        setCkpts([...cs].sort((a, b) => OPP_ORDER[a.mode] - OPP_ORDER[b.mode]));
        // 처음이거나 저장된 상대가 사라졌으면 WILD FLY(B), 없으면 첫 번째 상대
        setSettings((s) =>
          cs.some((c) => c.id === s.ckptId)
            ? s
            : { ...s, ckptId: (cs.find((c) => c.mode === "B") ?? cs[0])?.id ?? "" },
        );
      })
      .catch(() => undefined)
      .finally(() => setCkptsLoaded(true));
  }, []);

  const songs = useMemo(() => groupSongs(charts), [charts]);
  const categories = useMemo(() => ["전체", ...new Set(songs.map(categoryOf))], [songs]);
  const listFor = useCallback(
    (cat: string) => {
      const list = songs.filter((s) => cat === "전체" || categoryOf(s) === cat);
      const notes = (s: Song) => Math.min(...s.charts.map((c) => c.notes));
      return list.sort((a, b) =>
        sort === "bpm"
          ? a.bpm - b.bpm
          : sort === "notes"
            ? notes(a) - notes(b)
            : a.title.localeCompare(b.title),
      );
    },
    [songs, sort],
  );
  const visible = useMemo(() => listFor(category), [listFor, category]);

  // 선택한 채보 (저장된 채보가 없으면 목록 첫 곡)
  const selected = songs.find((s) => s.charts.some((c) => c.id === settings.chartId)) ?? visible[0];
  const current = selected;
  const chart = current?.charts.find((c) => c.id === settings.chartId) ?? current?.charts[0];
  const record = chart ? records[chart.id] : undefined;
  const set = useCallback(
    (patch: Partial<PlaySettings>) => setSettings((s) => ({ ...s, ...patch })),
    [],
  );

  // 곡을 바꾸면 같은 난이도를 유지 (없으면 첫 난이도)
  const pickSong = useCallback(
    (song: Song) => {
      const same = song.charts.find((c) => c.diff === chart?.diff) ?? song.charts[0];
      if (same) set({ chartId: same.id });
    },
    [chart?.diff, set],
  );

  const cycleOpponent = useCallback(() => {
    if (ckpts.length === 0) return;
    const opps = ckpts.map((c) => c.id);
    set({ ckptId: opps[(opps.indexOf(settings.ckptId) + 1) % opps.length]! });
  }, [ckpts, settings.ckptId, set]);

  const changeSpeed = useCallback(
    (delta: number) =>
      set({ speed: Math.round(Math.min(1.5, Math.max(0.3, settings.speed + delta)) * 10) / 10 }),
    [settings.speed, set],
  );

  const ready = !!chart && ckpts.some((c) => c.id === settings.ckptId);
  const start = useCallback(() => {
    if (!chart || !ready) return;
    unlockAudio(); // 이 사용자 동작 안에서 오디오를 열어 둬야 플레이 화면에서 바로 소리가 난다
    const s = { ...settings, chartId: chart.id };
    saveSettings(s);
    navigate("/play", { state: s });
  }, [chart, ready, settings, navigate]);

  // 분류를 바꿨는데 선택 곡이 새 목록에 없으면 첫 곡을 고른다
  const changeCategory = useCallback(
    (cat: string) => {
      setCategory(cat);
      const list = listFor(cat);
      if (list.length && !list.some((x) => x.key === current?.key)) pickSong(list[0]!);
    },
    [listFor, current?.key, pickSong],
  );

  // 키보드 조작
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if ((e.target as HTMLElement | null)?.tagName === "SELECT") return;
      const idx = visible.findIndex((s) => s.key === current?.key);
      switch (e.key) {
        case "ArrowDown":
        case "ArrowUp": {
          e.preventDefault();
          const step = e.key === "ArrowDown" ? 1 : -1;
          const next = visible[(idx + step + visible.length) % visible.length];
          if (next) pickSong(next);
          break;
        }
        case "ArrowLeft":
        case "ArrowRight": {
          if (!current || !chart) return;
          e.preventDefault();
          const i = current.charts.indexOf(chart) + (e.key === "ArrowRight" ? 1 : -1);
          const c = current.charts[Math.max(0, Math.min(current.charts.length - 1, i))];
          if (c) set({ chartId: c.id });
          break;
        }
        case "Tab": {
          e.preventDefault();
          const ci = categories.indexOf(category);
          const n = categories.length;
          changeCategory(categories[(ci + (e.shiftKey ? -1 : 1) + n) % n]!);
          break;
        }
        case "o":
        case "O":
          cycleOpponent();
          break;
        case "[":
          changeSpeed(-0.1);
          break;
        case "]":
          changeSpeed(0.1);
          break;
        case "Enter":
          e.preventDefault();
          start();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    visible,
    current,
    chart,
    categories,
    category,
    set,
    pickSong,
    changeCategory,
    cycleOpponent,
    changeSpeed,
    start,
  ]);

  // 선택한 줄이 보이도록 스크롤
  useEffect(() => {
    listRef.current?.querySelector(".srow.on")?.scrollIntoView({ block: "nearest" });
  }, [settings.chartId, category, sort]);

  return (
    <div className="lobby2">
      <header className="lobby-top">
        <div className="lobby-mode">
          <Logo />
          <span className="mode-tag">VERSUS</span>
        </div>
        <label className="sort">
          <span>≡ 정렬</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            {SORTS.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
      </header>

      <div className="lobby-body">
        {/* 왼쪽: 선택한 곡 */}
        <aside className="song-card">
          {current && chart ? (
            <>
              <div className="song-card-head">
                <span className="lanes-badge">
                  <b>4</b> LANES
                </span>
                <span className="muted small">{categoryOf(current)}</span>
              </div>
              <div className="song-hero">
                <Cover title={current.title} size={112} />
                <div className="song-hero-meta">
                  <h2>{current.title}</h2>
                  <div className="muted">{current.artist || "—"}</div>
                  <div className="bpm">
                    BPM <b>{Math.round(current.bpm)}</b> · {fmtTime(chart.durationMs)} ·{" "}
                    {chart.notes} notes
                  </div>
                  {current.audio && <span className="audio-tag">♪ 음원</span>}
                </div>
              </div>
              <div className="diff-tabs" role="tablist">
                {diffsOf(current).map((d) => {
                  const c = current.charts.find((x) => x.diff === d);
                  const on = c?.id === chart.id;
                  return (
                    <button
                      key={d}
                      role="tab"
                      aria-selected={on}
                      disabled={!c}
                      className={`dtab ${d === "—" ? "play" : d.toLowerCase()} ${on ? "on" : ""}`}
                      onClick={() => c && set({ chartId: c.id })}
                    >
                      {d === "—" ? "PLAY" : d}
                    </button>
                  );
                })}
              </div>
              <dl className="best">
                <div>
                  <dt>Best Score</dt>
                  <dd>{record ? record.score.toLocaleString() : "—"}</dd>
                </div>
                <div>
                  <dt>Rate</dt>
                  <dd>{record ? `${record.rate.toFixed(2)}%` : "—"}</dd>
                </div>
                <div>
                  <dt>Max Combo</dt>
                  <dd>{record ? record.maxCombo : "—"}</dd>
                </div>
                <div className="best-grade">
                  <RecordMark rec={record} />
                  {record?.fullCombo && <span className="fc-tag">FULL COMBO</span>}
                </div>
              </dl>
            </>
          ) : (
            <p className="muted">{error ?? "곡을 불러오는 중…"}</p>
          )}

          <div className="opt-title">옵션</div>
          <div className="opt-tiles one">
            <button
              className="tile speed"
              onClick={() => changeSpeed(settings.speed >= 1.5 ? -1.2 : 0.1)}
            >
              <span className="tile-k">SPEED</span>
              <b>{settings.speed.toFixed(1)}</b>
            </button>
          </div>

          <div className="opt-title">대결 상대</div>
          <div className="opp-list" role="radiogroup" aria-label="대결 상대">
            {ckpts.map((c) => {
              const info = OPPONENTS[c.mode];
              const on = settings.ckptId === c.id;
              const lv = strengthOf(c.accuracy);
              return (
                <button
                  key={c.id}
                  role="radio"
                  aria-checked={on}
                  className={`opp-row ${on ? "on" : ""} ${info.isFly ? "fly" : "robot"}`}
                  onClick={() => set({ ckptId: c.id })}
                >
                  <span className="opp-icon">
                    <OpponentIcon kind={info.icon} size={20} />
                  </span>
                  <span className="opp-text">
                    <b>{info.name}</b>
                  </span>
                  <span
                    className="opp-power"
                    title={`검증 정확도 ${c.accuracy?.toFixed(1) ?? "-"}%`}
                  >
                    <span className="pips" aria-label={`실력 ${lv}/5`}>
                      {[1, 2, 3, 4, 5].map((i) => (
                        <i key={i} className={i <= lv ? "lit" : ""} />
                      ))}
                    </span>
                    <small>{c.accuracy?.toFixed(0) ?? "-"}%</small>
                  </span>
                </button>
              );
            })}
          </div>
          {ckptsLoaded && ckpts.length === 0 && (
            <p className="muted small">
              대결할 상대가 없어요. Esc 메뉴의 학습에서 상대를 먼저 학습시켜 주세요.
            </p>
          )}
          <button className="start-btn" onClick={start} disabled={!ready}>
            START
          </button>
        </aside>

        {/* 오른쪽: 분류 + 목록 */}
        <section className="song-browser">
          <div className="cat-tabs" role="tablist">
            {categories.map((c) => (
              <button
                key={c}
                role="tab"
                aria-selected={c === category}
                className={`ctab ${c === category ? "on" : ""}`}
                onClick={() => changeCategory(c)}
              >
                {c}
              </button>
            ))}
            <span className="cat-hint">
              <kbd>Tab</kbd>
            </span>
          </div>
          <ul className="song-rows" ref={listRef}>
            {visible.map((s) => {
              const on = s.key === current?.key;
              return (
                <li key={s.key}>
                  <button
                    className={`srow ${on ? "on" : ""}`}
                    onClick={() => pickSong(s)}
                    onDoubleClick={start}
                  >
                    <Cover title={s.title} size={44} />
                    <span className="srow-text">
                      <b>{s.title}</b>
                      <small>
                        {s.artist || "—"} · {Math.round(s.bpm)} BPM
                      </small>
                    </span>
                    <span className="srow-marks">
                      {diffsOf(s).map((d) => {
                        const c = s.charts.find((x) => x.diff === d);
                        return (
                          <span
                            key={d}
                            className={`mark-wrap ${d === "—" ? "" : d.toLowerCase()}`}
                            title={d === "—" ? "기록" : `${d} 기록`}
                          >
                            <RecordMark rec={c ? records[c.id] : undefined} />
                          </span>
                        );
                      })}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      </div>

      <footer className="key-hints">
        <span>
          <kbd>↑</kbd>
          <kbd>↓</kbd> 곡 선택
        </span>
        <span>
          <kbd>←</kbd>
          <kbd>→</kbd> 난이도
        </span>
        <span>
          <kbd>Tab</kbd> 분류
        </span>
        <span>
          <kbd>[</kbd>
          <kbd>]</kbd> 속도
        </span>
        <span className="sp" />
        <span>
          플레이 키 <kbd>{LANE_KEYS_TEXT}</kbd>
        </span>
        <span>
          <kbd>Enter</kbd> 시작
        </span>
        <span>
          <kbd>Esc</kbd> 메뉴
        </span>
      </footer>
    </div>
  );
}
