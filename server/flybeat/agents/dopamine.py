"""모드 C: 도파민 보상으로 커넥톰의 KC→MBON 시냅스만 직접 바꾸는 강화학습.

역전파를 쓰지 않는다. 학습 규칙은 3요소 규칙(three-factor rule)이다.

  e_ij ← e_ij · exp(−Δt/τ_e) + KC_i · (a_l − p_l)          (적격 흔적, j 는 레인 l 의 MBON)
  Δw_ij = η · e_ij · DA_j                                   (DA_j: MBON j 구획의 도파민 신호)

- 행동: MBON 을 레인에 나눠 배정하고, 레인 l 의 MBON 평균 활동으로 누름 확률 p_l 을 정한다.
  학습 중에는 p_l 로 확률적으로 누르고(a_l), 평가·플레이 때는 p_l > 0.5 이면 누른다.
- 보상 → 도파민: 보상 예측 오차 δ = r − 기대 보상. δ > 0 이면 보상 계열 DAN(PAM)을,
  δ < 0 이면 처벌 계열 DAN(PPL1)을 전류로 자극한다 (시뮬레이션에도 반영되어 시각화에 보인다).
  도파민 신호는 자극된 DAN 이 실제로 연결된(DAN→MBON) 구획에만 전달된다:
  DA_j = δ · (δ>0 이면 PAM 연결 비율, δ<0 이면 PPL1 연결 비율). DAN 입력이 없는 MBON 은 δ 그대로.
  (시뮬레이션된 DAN 발화율을 그대로 쓰면 네트워크 자체 활동이 보상과 무관한 잡음이 되어
  학습이 안 됐다.)
- 가중치는 부호를 지킨다 (KC 는 흥분성이므로 0 ≤ w ≤ w_max · 원래 값).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from flybeat.agents.base import ProgressFn, StopFn, TrainProgress, eval_charts, evaluate
from flybeat.agents.brain_io import SensoryEncoder
from flybeat.agents.checkpoint import load_checkpoint, new_checkpoint_id, save_checkpoint
from flybeat.agents.mapping import brain_activity
from flybeat.brain.model import BrainSim, Connectome
from flybeat.game.env import EnvConfig, RhythmEnv, random_chart_source
from flybeat.game.generator import GeneratorConfig


@dataclass
class DopamineConfig:
    profile: str = "compact"
    episodes: int = 64
    n_envs: int = 8
    lr: float = 0.2  # η (누름 로짓 단위)
    tau_elig_ms: float = 150.0  # 적격 흔적 시간 상수
    readout_gain: float = 1.0  # β: 표준화된 MBON 입력 → 누름 로짓
    press_bias: float = -2.5  # 초기 누름 로짓 (가끔만 누르게)
    bias_lr: float = 0.01  # 레인별 흥분성(bias) 학습률
    da_current: float = 1.0  # 보상 1 당 DAN 자극 전류
    baseline_decay: float = 0.01  # 기대 보상 이동 평균 속도
    w_max: float = 4.0  # 원래 시냅스 수 대비 최대 배율
    # 학습 신호용 보상 (평가 지표인 정확도에는 영향 없음). None 이면 게임 기본 보상.
    reward_miss: float | None = None
    reward_empty: float | None = None
    seed: int = 0
    lanes: int = 4


def assign_mbon_lanes(conn: Connectome, lanes: int) -> np.ndarray:
    """MBON 을 (type, side) 순서로 정렬해 레인에 돌아가며 배정한다. 결과: 각 MBON 의 레인."""
    mbon = np.asarray(conn.groups["mbon"], dtype=np.int64)
    df = conn.neurons.iloc[mbon]
    order = np.lexsort((df["somaSide"].fillna("").to_numpy(), df["type"].fillna("").to_numpy()))
    lane = np.empty(len(mbon), dtype=np.int64)
    lane[order] = np.arange(len(mbon)) % lanes
    return lane


class DopamineBrain:
    """뇌 시뮬레이터 + 가소성 KC→MBON 시냅스 + MBON 리드아웃 (배치 B)."""

    def __init__(self, conn: Connectome, cfg: DopamineConfig, batch: int, tick_ms: float):
        self.conn = conn
        self.cfg = cfg
        self.tick_ms = tick_ms
        self.sim = BrainSim(conn, batch=batch)
        self.lanes = cfg.lanes
        g = conn.groups
        self.kc = np.asarray(g["kc"], dtype=np.int64)
        self.mbon = np.asarray(g["mbon"], dtype=np.int64)
        self.pam = np.asarray(g["dan_reward"], dtype=np.int64)
        self.ppl1 = np.asarray(g["dan_punish"], dtype=np.int64)
        self.mbon_lane = assign_mbon_lanes(conn, cfg.lanes)

        # 가소성 시냅스: W(post=MBON, pre=KC) 의 CSR data 위치
        W = self.sim.W
        pos = []
        for pre, post in zip(conn.plastic_pre, conn.plastic_post, strict=True):
            lo, hi = W.indptr[post], W.indptr[post + 1]
            k = lo + np.searchsorted(W.indices[lo:hi], pre)
            if k >= hi or W.indices[k] != pre:
                raise ValueError(
                    "가소성 시냅스가 가중치 행렬에 없습니다. preprocess 를 다시 실행하세요."
                )
            pos.append(k)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.w0 = W.data[self.pos].copy()  # 원래 (정규화된) 가중치
        self.w = self.w0.copy()

        mbon_index = {int(m): i for i, m in enumerate(self.mbon)}
        kc_index = {int(k): i for i, k in enumerate(self.kc)}
        self.syn_mbon = np.array([mbon_index[int(p)] for p in conn.plastic_post], dtype=np.int64)
        self.syn_kc = np.array([kc_index[int(p)] for p in conn.plastic_pre], dtype=np.int64)
        self.syn_lane = self.mbon_lane[self.syn_mbon]

        # DAN → MBON 구획 행렬 (n_mbon × n_pam), (n_mbon × n_ppl1), 행 정규화
        self.da_pam = self._dan_matrix(self.pam, mbon_index)
        self.da_ppl1 = self._dan_matrix(self.ppl1, mbon_index)
        # MBON 구획마다 PAM / PPL1 입력 유무 (둘 다 없으면 둘 다 1)
        has_pam = self.da_pam.sum(1) > 0
        has_ppl1 = self.da_ppl1.sum(1) > 0
        none = ~has_pam & ~has_ppl1
        self.pam_share = (has_pam | none).astype(np.float32)
        self.ppl1_share = (has_ppl1 | none).astype(np.float32)

        # 시냅스 → MBON → 레인 합산 행렬
        n_syn = len(self.pos)
        self.S = sp.csr_matrix(
            (np.ones(n_syn, np.float32), (self.syn_mbon, np.arange(n_syn))),
            shape=(len(self.mbon), n_syn),
        )
        cnt = np.maximum(np.bincount(self.mbon_lane, minlength=self.lanes), 1)
        self.L = sp.csr_matrix(
            (1.0 / cnt[self.mbon_lane], (self.mbon_lane, np.arange(len(self.mbon)))),
            shape=(self.lanes, len(self.mbon)),
            dtype=np.float32,
        )
        # 레인 입력 표준화 (calibrate 에서 정함): logit = β (z − μ) / σ + bias
        self.mu = np.zeros(self.lanes, np.float32)
        self.sd = np.ones(self.lanes, np.float32)

        self.bias = np.full(cfg.lanes, cfg.press_bias, np.float32)
        self.elig = np.zeros((n_syn, batch), np.float32)
        self.elig_b = np.zeros((cfg.lanes, batch), np.float32)
        self.last_p = np.zeros((batch, cfg.lanes), np.float32)

    def _dan_matrix(self, dans: np.ndarray, mbon_index: dict[int, int]) -> np.ndarray:
        dan_index = {int(d): i for i, d in enumerate(dans)}
        M = np.zeros((len(self.mbon), len(dans)), np.float32)
        for pre, post, w in zip(
            self.conn.dan_pre, self.conn.dan_post, self.conn.dan_weight, strict=True
        ):
            if int(pre) in dan_index and int(post) in mbon_index:
                M[mbon_index[int(post)], dan_index[int(pre)]] += w
        s = M.sum(axis=1, keepdims=True)
        return np.divide(M, s, out=np.zeros_like(M), where=s > 0)

    @property
    def batch(self) -> int:
        return self.sim.batch

    def reset(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.sim.reset(self.batch)
            self.elig[:] = 0
            self.elig_b[:] = 0
        else:
            self.sim.reset_cols(mask)
            self.elig[:, mask] = 0
            self.elig_b[:, mask] = 0

    def lane_drive(self) -> np.ndarray:
        """(lanes, B) 레인별 MBON 의 가소성 KC 입력 전류 평균. w 에 대해 선형이다."""
        kc_r = self.sim.r[self.kc]  # (n_kc, B)
        contrib = self.w[:, None] * kc_r[self.syn_kc]  # (n_syn, B)
        return np.asarray(self.L @ (self.S @ contrib))

    def press_prob(self) -> np.ndarray:
        """(B, lanes) 레인별 누름 확률."""
        z = (self.lane_drive().T - self.mu) / self.sd
        logit = self.cfg.readout_gain * z + self.bias
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    def calibrate(self, drives: np.ndarray) -> None:
        """학습 전 레인 입력 분포 (T, lanes) 로 표준화 통계를 정한다."""
        self.mu = drives.mean(axis=0).astype(np.float32)
        floor = max(float(drives.std(axis=0).mean()) * 0.1, 1e-3)
        self.sd = np.maximum(drives.std(axis=0), floor).astype(np.float32)

    def step(self, u: np.ndarray, dan_current: np.ndarray | None) -> np.ndarray:
        """감각 입력(+ 직전 보상의 DAN 자극)으로 1 tick 진행하고 누름 확률을 반환."""
        self.sim.step(u, self.tick_ms, extra=dan_current)
        self.last_p = self.press_prob()
        return self.last_p

    def dan_current(self, delta: np.ndarray) -> np.ndarray:
        """보상 예측 오차 δ (B,) → DAN 자극 전류 (N, B)."""
        cur = np.zeros((self.sim.n, self.batch), np.float32)
        c = self.cfg.da_current
        cur[self.pam] = np.maximum(delta, 0) * c
        cur[self.ppl1] = np.maximum(-delta, 0) * c
        return cur

    def dopamine(self, delta: np.ndarray) -> np.ndarray:
        """(n_mbon, B) MBON 구획별 도파민 신호."""
        pos = np.maximum(delta, 0)[None, :] * self.pam_share[:, None]
        neg = np.minimum(delta, 0)[None, :] * self.ppl1_share[:, None]
        return (pos + neg).astype(np.float32)

    def learn(self, action: np.ndarray, delta: np.ndarray) -> None:
        """적격 흔적 갱신 + 도파민으로 가중치 업데이트 (배치 평균).

        e_ij += KC_i · (a_l − p_l) · ∂logit_l/∂(KC 입력) 이므로, 도파민이 보상 예측 오차일 때
        Δw = η · e · DA 는 정책 경사(REINFORCE)와 같다.
        """
        cfg = self.cfg
        decay = np.float32(np.exp(-self.tick_ms / cfg.tau_elig_ms))
        kc_r = self.sim.r[self.kc]  # (n_kc, B)
        post_term = (action - self.last_p).T  # (lanes, B)
        g = (cfg.readout_gain / self.sd)[:, None]  # ∂logit/∂z
        lane_n = np.maximum(np.bincount(self.mbon_lane, minlength=self.lanes), 1)[:, None]
        self.elig *= decay
        self.elig += kc_r[self.syn_kc] * (post_term * g / lane_n)[self.syn_lane]
        self.elig_b *= decay
        self.elig_b += post_term
        da = self.dopamine(delta)  # (n_mbon, B)
        dw = (self.elig * da[self.syn_mbon]).mean(axis=1)
        self.w = np.clip(self.w + cfg.lr * dw, 0.0, cfg.w_max * self.w0)
        self.sim.W.data[self.pos] = self.w
        # 레인별 흥분성(bias)도 같은 규칙 (전체 도파민)
        self.bias += cfg.bias_lr * (self.elig_b * delta[None, :]).mean(axis=1)

    def set_weights(self, w: np.ndarray, bias: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> None:
        self.w = w.astype(np.float32)
        self.sim.W.data[self.pos] = self.w
        self.bias = bias.astype(np.float32)
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)


class DopamineAgent:
    """학습된 모드 C 에이전트 (플레이·평가용, 배치 1).

    학습한 그대로 확률적으로 누르되, 곡마다 같은 시드로 시작해 결과를 재현할 수 있게 한다.
    """

    mode = "C"

    def __init__(self, brain: DopamineBrain, encoder: SensoryEncoder) -> None:
        self.brain = brain
        self.encoder = encoder
        self.lanes = brain.lanes
        self.rng = np.random.default_rng(0)

    @property
    def profile(self) -> str:
        return self.brain.cfg.profile

    def reset(self) -> None:
        self.brain.reset()
        self.rng = np.random.default_rng(self.brain.cfg.seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        p = self.brain.step(self.encoder.encode(obs), None)
        return (self.rng.random(self.lanes) < p[0]).astype(np.int8)

    def activity(self) -> dict[str, Any]:
        a = brain_activity(self.brain.sim)
        a["pressProb"] = [float(x) for x in self.brain.last_p[0]]
        return a

    @classmethod
    def build(cls, conn: Connectome, cfg: DopamineConfig) -> DopamineAgent:
        env_cfg = EnvConfig()
        brain = DopamineBrain(conn, cfg, 1, 1000 / env_cfg.tick_hz)
        return cls(brain, SensoryEncoder.retinotopic(cfg.lanes, env_cfg, conn.vertical_bins))

    @classmethod
    def load(cls, ckpt_id: str) -> DopamineAgent:
        tensors, meta = load_checkpoint(ckpt_id)
        if meta["mode"] != "C":
            raise ValueError("모드 C 체크포인트가 아닙니다")
        known = {f.name for f in dataclasses.fields(DopamineConfig)}
        cfg = DopamineConfig(**{k: v for k, v in meta["config"].items() if k in known})
        conn = Connectome.load(cfg.profile)
        if not (
            np.array_equal(conn.plastic_pre, tensors["plastic.pre"])
            and np.array_equal(conn.plastic_post, tensors["plastic.post"])
        ):
            raise ValueError("체크포인트와 전처리 결과의 KC→MBON 시냅스가 다릅니다")
        agent = cls.build(conn, cfg)
        agent.brain.set_weights(
            tensors["plastic.weight"],
            tensors["lane.bias"],
            tensors["readout.mu"],
            tensors["readout.sd"],
        )
        return agent


def train(
    cfg: DopamineConfig,
    gen: GeneratorConfig | None = None,
    env_cfg: EnvConfig | None = None,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
    ckpt_id: str | None = None,
) -> str:
    gen = gen or GeneratorConfig(lanes=cfg.lanes)
    env_cfg = env_cfg or EnvConfig()
    reward = env_cfg.reward
    if cfg.reward_miss is not None:
        reward = dataclasses.replace(reward, miss=cfg.reward_miss)
    if cfg.reward_empty is not None:
        reward = dataclasses.replace(reward, empty=cfg.reward_empty)
    train_env_cfg = dataclasses.replace(env_cfg, reward=reward)
    conn = Connectome.load(cfg.profile)
    tick_ms = 1000 / env_cfg.tick_hz
    brain = DopamineBrain(conn, cfg, cfg.n_envs, tick_ms)
    encoder = SensoryEncoder.retinotopic(cfg.lanes, env_cfg, conn.vertical_bins)
    rng = np.random.default_rng(cfg.seed)
    envs = [
        RhythmEnv(random_chart_source(gen), train_env_cfg, lanes=cfg.lanes)
        for _ in range(cfg.n_envs)
    ]
    obs = np.stack([e.reset(seed=cfg.seed + i)[0] for i, e in enumerate(envs)])
    seed = cfg.seed + cfg.n_envs
    brain.reset()

    # 레인 입력 표준화: 누르지 않고 15초 동안 본 MBON 입력 분포
    calib_env = RhythmEnv(random_chart_source(gen), env_cfg, lanes=cfg.lanes)
    calib = DopamineBrain(conn, cfg, 1, tick_ms)
    o, _ = calib_env.reset(seed=cfg.seed + 999)
    drives = []
    for _ in range(int(15000 / tick_ms)):
        calib.sim.step(encoder.encode(o), tick_ms)
        drives.append(calib.lane_drive()[:, 0])
        o, *_ = calib_env.step(np.zeros(cfg.lanes, np.int8))
    brain.calibrate(np.array(drives))

    baseline = 0.0
    delta = np.zeros(cfg.n_envs, np.float32)
    episodes, step = 0, 0
    recent_acc: list[float] = []
    ep_reward = np.zeros(cfg.n_envs)
    recent_rew: list[float] = []
    last_report = time.time()

    while episodes < cfg.episodes and not (should_stop and should_stop()):
        p = brain.step(encoder.encode(obs), brain.dan_current(delta))
        action = (rng.random(p.shape) < p).astype(np.float32)
        rewards = np.zeros(cfg.n_envs, np.float32)
        dones = np.zeros(cfg.n_envs, bool)
        for i, env in enumerate(envs):
            o, r, done, _, info = env.step(action[i].astype(np.int8))
            rewards[i], dones[i] = r, done
            ep_reward[i] += r
            if done:
                episodes += 1
                recent_acc = (recent_acc + [info["stats"]["accuracy"]])[-20:]
                recent_rew = (recent_rew + [float(ep_reward[i])])[-20:]
                ep_reward[i] = 0
                o, _ = env.reset(seed=seed)
                seed += 1
            obs[i] = o
        # 보상 예측 오차 → 다음 tick 의 DAN 자극, 동시에 현재 적격 흔적으로 학습
        delta = rewards - baseline
        baseline += cfg.baseline_decay * (float(rewards.mean()) - baseline)
        brain.learn(action, delta)
        if dones.any():
            brain.reset(dones)
            delta[dones] = 0
        step += 1
        if progress and (time.time() - last_report > 1.0 or episodes >= cfg.episodes):
            last_report = time.time()
            changed = brain.w != brain.w0
            progress(
                TrainProgress(
                    step=episodes,
                    total=cfg.episodes,
                    episodes=episodes,
                    accuracy=float(np.mean(recent_acc)) if recent_acc else None,
                    reward=float(np.mean(recent_rew)) if recent_rew else None,
                    extra={
                        "ticks": step * cfg.n_envs,
                        "changedSynapses": int(changed.sum()),
                        "weightRatio": float(brain.w.sum() / max(brain.w0.sum(), 1e-9)),
                        "laneBias": [round(float(b), 3) for b in brain.bias],
                    },
                )
            )

    agent = DopamineAgent.build(conn, cfg)
    agent.brain.set_weights(brain.w, brain.bias, brain.mu, brain.sd)
    result = evaluate(agent, eval_charts(cfg=gen), env_cfg)
    ckpt_id = ckpt_id or new_checkpoint_id("C")
    save_checkpoint(
        ckpt_id,
        {
            "plastic.weight": brain.w,
            "plastic.weight0": brain.w0,
            "plastic.pre": conn.plastic_pre.astype(np.int64),
            "plastic.post": conn.plastic_post.astype(np.int64),
            "lane.bias": brain.bias,
            "readout.mu": brain.mu,
            "readout.sd": brain.sd,
        },
        {
            "mode": "C",
            "config": dataclasses.asdict(cfg),
            "generator": dataclasses.asdict(gen),
            "connectome": {k: conn.meta.get(k) for k in ("profile", "neurons", "synapses")},
            "trainedEpisodes": episodes,
            "plasticity": {
                "synapses": int(len(brain.w)),
                "changed": int((np.abs(brain.w - brain.w0) > 1e-9).sum()),
                "meanAbsChange": float(np.abs(brain.w - brain.w0).mean()),
            },
            "eval": result,
        },
    )
    return ckpt_id
