"""게임 관측 ↔ 초파리 뇌 연결 (모드 B·C 공통).

감각(입력): 관측의 노트 격자(레인 × 시간 bin) → 레인 × 세로칸(vertical_bins) 시각 입력 채널.
  화면 위쪽(먼 노트) = 위쪽 시야 = 채널 bin 0, 판정선 근처 = 마지막 채널.
행동(출력): 운동 뉴런 활동을 리드아웃해 레인별 누름을 정한다 (리드아웃은 모드마다 다름).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybeat.game.env import EnvConfig


@dataclass
class SensoryEncoder:
    """관측 → 시각 입력 채널 세기 (B, lanes*vbins).

    weights: (vbins, bins) 행렬. 채널 v 의 세기 = Σ_b weights[v, b] · (head[b] + hold_w · hold[b]).
    기본값은 망막 위치 대응: 시간 bin 을 vbins 개 구간으로 나눠 해당 구간만 1.
    """

    lanes: int
    bins: int
    vbins: int
    weights: np.ndarray
    hold_weight: float = 0.5

    @classmethod
    def retinotopic(cls, lanes: int, env_cfg: EnvConfig, vbins: int) -> SensoryEncoder:
        bins = env_cfg.bins
        w = np.zeros((vbins, bins), dtype=np.float32)
        # 관측 bin 0 = 판정선(가장 가까움) → 채널 vbins-1 (시야 아래쪽)
        for v, chunk in enumerate(np.array_split(np.arange(bins), vbins)):
            w[vbins - 1 - v, chunk] = 1.0
        return cls(lanes, bins, vbins, w)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        n = self.lanes * self.bins
        head = obs[:, :n].reshape(-1, self.lanes, self.bins)
        hold = obs[:, n : 2 * n].reshape(-1, self.lanes, self.bins)
        grid = np.maximum(head, self.hold_weight * hold)  # (B, lanes, bins)
        u = np.einsum("vb,klb->klv", self.weights, grid)  # (B, lanes, vbins)
        return np.clip(u, 0.0, None).reshape(len(obs), -1).astype(np.float32)

    def pressed(self, obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        n = 2 * self.lanes * self.bins
        return obs[:, n : n + self.lanes]
