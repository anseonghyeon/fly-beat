/**
 * 대결 상대(학습 모드) 이름. 화면에는 기술 용어 대신 알기 쉬운 이름을 쓴다.
 *   A = 일반 인공신경망(MLP, 초파리 아님) · B = 커넥톰 고정 + 해석기 학습 · C = 도파민 3요소 학습
 */
import type { Mode } from "../api/types";

export interface OpponentInfo {
  name: string;
  /** 로비 목록의 한 줄 설명 */
  desc: string;
  /** 학습 페이지 등에서 함께 보여 줄 기술 이름 */
  tech: string;
  icon: "robot" | "fly";
  isFly: boolean;
}

export const OPPONENTS: Record<Mode, OpponentInfo> = {
  A: {
    name: "AI 로봇",
    desc: "초파리가 아닌 일반 AI. 비교용이라 제일 잘 쳐요",
    tech: "모드 A · MLP + PPO",
    icon: "robot",
    isFly: false,
  },
  B: {
    name: "본능 초파리",
    desc: "타고난 뇌 배선 그대로 노트에 반응해요",
    tech: "모드 B · 커넥톰 고정 + 해석기 학습",
    icon: "fly",
    isFly: true,
  },
  C: {
    name: "학습 초파리",
    desc: "칭찬(도파민)을 받으며 뇌가 스스로 배웠어요",
    tech: "모드 C · 도파민 3요소 학습",
    icon: "fly",
    isFly: true,
  },
};

/** 정확도(%) → 실력 단계 1~5 */
export function strengthOf(acc: number | null | undefined): number {
  if (acc == null) return 0;
  return acc >= 90 ? 5 : acc >= 80 ? 4 : acc >= 70 ? 3 : acc >= 60 ? 2 : 1;
}
