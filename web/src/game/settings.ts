/** 로비에서 정한 대결 설정. 마지막 설정은 브라우저에 기억해 둔다 (실패해도 동작에는 지장 없음). */
export interface PlaySettings {
  chartId: string;
  /** 빈 문자열 = 초파리 없이 혼자 */
  ckptId: string;
  speed: number;
  sound: boolean;
}

const KEY = "flybeat.playSettings";

export const DEFAULT_SETTINGS: PlaySettings = {
  chartId: "tutorial-steps",
  ckptId: "",
  speed: 0.6,
  sound: true,
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
      speed: typeof v.speed === "number" ? Math.min(2, Math.max(0.2, v.speed)) : 0.6,
      sound: v.sound !== false,
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
