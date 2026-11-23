/**
 * 레인별 키 배정 (사람 플레이용). 초파리는 레인 번호로 누르므로 키를 바꿔도 다시 학습할 필요가 없다.
 * KeyboardEvent.code 를 쓰므로 한글 입력 상태에서도 같은 물리 키로 동작한다.
 */
export const LANE_KEY_CODES = ["KeyA", "KeyS", "Semicolon", "Quote"] as const;
export const LANE_KEY_LABELS = ["A", "S", ";", "'"] as const;
export const LANE_KEYS_TEXT = LANE_KEY_LABELS.join(" ");
