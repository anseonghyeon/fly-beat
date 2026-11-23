"""커넥톰 기반 rate 모델 시뮬레이터.

τ dr/dt = −r + f(gain · W r + I_ext + bias),  f(x) = tanh(relu(x))  (0 ≤ r < 1)

W[post, pre] 는 부호 × 시냅스 수를 post 뉴런의 입력 합^alpha 로 나눈 값이다.
배치(B)를 지원해 여러 환경을 한 번에 시뮬레이션한다. 상태 r 의 모양은 (N, B).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybeat import paths
from flybeat.brain.config import load_brain_config

SUPPORTED_FORMAT = 3


@dataclass
class Connectome:
    """전처리 결과물 (data/processed/<profile>)."""

    W: sp.csr_matrix
    neurons: pd.DataFrame
    groups: dict
    dan_pre: np.ndarray
    dan_post: np.ndarray
    dan_weight: np.ndarray
    meta: dict
    plastic_pre: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    plastic_post: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def lanes(self) -> int:
        return len(self.groups["output"])

    @property
    def vertical_bins(self) -> int:
        return len(self.groups["input"][0])

    @classmethod
    def load(cls, profile: str = "compact", root: Path | None = None) -> Connectome:
        d = (root or paths.PROCESSED_DIR) / profile
        if not (d / "meta.json").is_file():
            raise FileNotFoundError(
                f"전처리 결과가 없습니다: {d}\n"
                f"먼저 python -m flybeat.brain.preprocess --profile {profile} 를 실행하세요."
            )
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        if meta.get("format") != SUPPORTED_FORMAT:
            raise ValueError("전처리 형식 버전이 다릅니다. preprocess 를 다시 실행하세요.")
        # allow_pickle=False: 신뢰할 수 없는 npz 로 코드가 실행되지 않게 한다.
        with np.load(d / "W.npz", allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["shape"])
            W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
        with np.load(d / "dan.npz", allow_pickle=False) as z:
            dan = (z["pre"], z["post"], z["weight"])
        with np.load(d / "plastic.npz", allow_pickle=False) as z:
            plastic = (z["pre"], z["post"])
        return cls(
            W=W,
            neurons=pd.read_feather(d / "neurons.feather"),
            groups=json.loads((d / "groups.json").read_text("utf-8")),
            dan_pre=dan[0],
            dan_post=dan[1],
            dan_weight=dan[2],
            meta=meta,
            plastic_pre=plastic[0],
            plastic_post=plastic[1],
        )

    def save(self, profile: str, root: Path | None = None) -> Path:
        """전처리 결과와 같은 형식으로 저장한다 (테스트·도구용)."""
        d = (root or paths.PROCESSED_DIR) / profile
        d.mkdir(parents=True, exist_ok=True)
        W = self.W.tocsr()
        np.savez_compressed(
            d / "W.npz", indptr=W.indptr, indices=W.indices, data=W.data, shape=np.array(W.shape)
        )
        np.savez_compressed(
            d / "dan.npz", pre=self.dan_pre, post=self.dan_post, weight=self.dan_weight
        )
        np.savez_compressed(d / "plastic.npz", pre=self.plastic_pre, post=self.plastic_post)
        self.neurons.reset_index(drop=True).to_feather(d / "neurons.feather")
        (d / "groups.json").write_text(json.dumps(self.groups), "utf-8")
        meta = {**self.meta, "format": SUPPORTED_FORMAT, "profile": profile, "neurons": self.n}
        (d / "meta.json").write_text(json.dumps(meta), "utf-8")
        return d

    @cached_property
    def input_matrix(self) -> sp.csr_matrix:
        """(N, lanes*vbins): 입력 채널 → 해당 뉴런들에 같은 전류."""
        rows, cols = [], []
        for lane, bins in enumerate(self.groups["input"]):
            for b, idxs in enumerate(bins):
                rows += idxs
                cols += [lane * self.vertical_bins + b] * len(idxs)
        return sp.csr_matrix(
            (np.ones(len(rows), np.float32), (rows, cols)),
            shape=(self.n, self.lanes * self.vertical_bins),
        )

    @cached_property
    def output_indices(self) -> np.ndarray:
        """모든 출력(운동) 뉴런 인덱스를 레인 순서로 이어 붙인 것."""
        return np.array([i for lane in self.groups["output"] for i in lane], dtype=np.int64)

    @cached_property
    def readout_indices(self) -> np.ndarray:
        """행동 리드아웃 대상: 다리 운동 뉴런 + 하행 뉴런(뇌 → 복부신경삭)."""
        dn = np.flatnonzero(self.neurons["superclass"].to_numpy() == "descending_neuron")
        return np.concatenate([self.output_indices, dn]).astype(np.int64)

    @cached_property
    def output_lane_of(self) -> np.ndarray:
        return np.array(
            [lane for lane, idxs in enumerate(self.groups["output"]) for _ in idxs], dtype=np.int64
        )


@dataclass
class SimConfig:
    dt_ms: float = 2.0
    tau_ms: float = 10.0
    gain: float = 1.5
    input_gain: float = 1.0
    bias: float = 0.0
    normalize_alpha: float = 1.0  # W / (입력 시냅스 합)^alpha
    kc_sparsity: float = 0.0  # >0 이면 매 순간 상위 이 비율의 KC 만 활성 (APL 억제 근사)
    kc_gain: float = 1.0  # KC 입력 가중치 배율 (KC 는 약한 입력을 많이 받음)
    kc_bias: float = 0.0  # KC 에만 더하는 전류 (전역 bias 를 상쇄)

    @classmethod
    def from_yaml(cls) -> SimConfig:
        return cls(**load_brain_config()["simulation"])


def normalized_weights(W: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    """post 뉴런(행)마다 |입력 시냅스| 합의 alpha 제곱으로 나눈다."""
    in_sum = np.asarray(abs(W).sum(axis=1)).ravel()
    scale = 1.0 / np.maximum(in_sum, 1.0) ** alpha
    return sp.csr_matrix(sp.diags(scale.astype(np.float32)) @ W, dtype=np.float32)


class BrainSim:
    """numpy/scipy 구현. (희소 행렬 곱은 scipy 가 torch CSR 보다 빠르다.)"""

    def __init__(self, conn: Connectome, batch: int = 1, cfg: SimConfig | None = None) -> None:
        self.conn = conn
        self.cfg = cfg or SimConfig.from_yaml()
        self.n = conn.n
        self.W = normalized_weights(conn.W, self.cfg.normalize_alpha)
        self.P = conn.input_matrix
        self.out_idx = conn.output_indices
        self.out_lane = conn.output_lane_of
        self.readout_idx = conn.readout_indices
        self.kc_idx = np.asarray(conn.groups["kc"], dtype=np.int64)
        self._kc_k = int(round(self.cfg.kc_sparsity * len(self.kc_idx)))
        if self.cfg.kc_gain != 1.0 and len(self.kc_idx):
            d = np.ones(self.n, np.float32)
            d[self.kc_idx] = self.cfg.kc_gain
            self.W = sp.csr_matrix(sp.diags(d) @ self.W, dtype=np.float32)
        self._bias = np.full((self.n, 1), self.cfg.bias, np.float32)
        self._bias[self.kc_idx] += self.cfg.kc_bias
        self.batch = batch
        self.reset()

    @property
    def n_input_channels(self) -> int:
        return self.conn.lanes * self.conn.vertical_bins

    @property
    def n_out(self) -> int:
        return len(self.out_idx)

    def reset(self, batch: int | None = None) -> None:
        if batch is not None:
            self.batch = batch
        self.r = np.zeros((self.n, self.batch), dtype=np.float32)

    def reset_cols(self, mask: np.ndarray) -> None:
        """배치 중 일부(에피소드가 끝난 환경)만 상태를 0 으로."""
        if mask.any():
            self.r[:, mask] = 0.0

    def drive_of(self, u: np.ndarray) -> np.ndarray:
        """u: (B, lanes*vbins) → 입력 전류 (N, B)."""
        return np.asarray(self.P @ (u.T.astype(np.float32) * self.cfg.input_gain))

    def step(self, u: np.ndarray, ms: float, extra: np.ndarray | None = None) -> np.ndarray:
        """ms 동안 적분하고 r (N, B) 를 반환. extra: (N, B) 추가 전류."""
        c = self.cfg
        drive = self.drive_of(u)
        if extra is not None:
            drive = drive + extra
        drive = drive + self._bias
        steps = max(1, round(ms / c.dt_ms))
        a = np.float32(c.dt_ms / c.tau_ms)
        g = np.float32(c.gain)
        r = self.r
        for _ in range(steps):
            x = g * (self.W @ r) + drive
            if self._kc_k > 0:
                self._kc_wta(x)
            np.maximum(x, 0.0, out=x)
            np.tanh(x, out=x)
            r += a * (x - r)
        return r

    @property
    def n_readout(self) -> int:
        return len(self.readout_idx)

    def readout(self) -> np.ndarray:
        """리드아웃 뉴런(운동 + 하행) 활동 (B, n_readout)."""
        return self.r[self.readout_idx].T

    def _kc_wta(self, x: np.ndarray) -> None:
        """KC 중 상위 k 개만 남기고 나머지를 억제 (열마다 k 번째 값을 문턱으로)."""
        xk = x[self.kc_idx]
        k = self._kc_k
        if k >= len(xk):
            return
        # (k+1) 번째로 큰 값을 문턱으로 → 상위 k 개만 양수로 남는다
        idx = len(xk) - k - 1
        thr = np.partition(xk, idx, axis=0)[idx]
        x[self.kc_idx] = xk - np.maximum(thr, 0.0)

    def motor(self) -> np.ndarray:
        """출력(운동) 뉴런 활동 (B, n_out)."""
        return self.r[self.out_idx].T

    def motor_lane_mean(self) -> np.ndarray:
        """레인별 운동 뉴런 평균 활동 (B, lanes)."""
        lanes = self.conn.lanes
        m = self.r[self.out_idx]
        s = np.zeros((lanes, self.batch), dtype=np.float32)
        np.add.at(s, self.out_lane, m)
        cnt = np.maximum(np.bincount(self.out_lane, minlength=lanes), 1)[:, None]
        return (s / cnt).T
