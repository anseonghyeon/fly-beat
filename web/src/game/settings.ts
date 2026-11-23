/** 로비에서 정한 대결 설정. 마지막 설정은 브라우저에 기억해 둔다 (실패해도 동작에는 지장 없음). */
export interface PlaySettings {
  chartId: string;
  /** 대결 상대 체크포인트 ID (빈 문자열 = 아직 고르지 않음) */
  ckptId: string;
  speed: number;
}

const KEY = "flybeat.playSettings";

export const DEFAULT_SETTINGS: PlaySettings = {
  chartId: "junkala-title-easy",
  ckptId: "",
  speed: 1.1,
};

export function loadSettings(): PlaySettings | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const v = JSON.parse(raw) as Partial<PlaySettings>;
    if (typeof v.chartId !== "string" || typeof v.ckptId !== "string") return null;
    return {
      chartId: v.chartId,
      ckptId: v.ckptId,
      speed: typeof v.speed === "number" ? Math.min(2, Math.max(0.2, v.speed)) : 1.1,
    };
  } catch {
    return null;
  }
}

export function saveSettings(s: PlaySettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* 저장 불가(사생활 보호 모드 등) — 무시 */
  }
}

/** 채보별 내 최고 기록 (브라우저에만 저장) */
export interface BestRecord {
  score: number;
  rate: number;
  maxCombo: number;
  fullCombo: boolean;
}

const RECORDS_KEY = "flybeat.records";

export function loadRecords(): Record<string, BestRecord> {
  try {
    const raw = localStorage.getItem(RECORDS_KEY);
    const v = raw ? (JSON.parse(raw) as unknown) : null;
    return v && typeof v === "object" ? (v as Record<string, BestRecord>) : {};
  } catch {
    return {};
  }
}

/** 기존 기록보다 좋으면 저장하고, 새 기록이면 true */
export function saveRecord(chartId: string, rec: BestRecord): boolean {
  const all = loadRecords();
  const prev = all[chartId];
  if (prev && prev.score >= rec.score) return false;
  all[chartId] = {
    score: rec.score,
    rate: rec.rate,
    maxCombo: Math.max(rec.maxCombo, prev?.maxCombo ?? 0),
    fullCombo: rec.fullCombo || (prev?.fullCombo ?? false),
  };
  try {
    localStorage.setItem(RECORDS_KEY, JSON.stringify(all));
  } catch {
    /* 저장 불가 — 무시 */
  }
  return true;
}

/** 방금 끝난 대결 결과 (새로고침해도 결과 페이지를 다시 볼 수 있도록 탭 저장소에 둔다) */
const RESULT_KEY = "flybeat.lastResult";

export function saveLastResult(v: unknown): void {
  try {
    sessionStorage.setItem(RESULT_KEY, JSON.stringify(v));
  } catch {
    /* 저장 불가 — 무시 */
  }
}

export function loadLastResult<T>(): T | null {
  try {
    const raw = sessionStorage.getItem(RESULT_KEY);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}
