"""모드 B: 커넥톰 고정 + 감각·행동 매핑 강화학습.

- 커넥톰 가중치는 그대로 둔다 (저수지처럼 사용).
- 감각 매핑: SensoryEncoder (기본: 망막 위치 대응). 선택적으로 진화 전략(ES)으로 가중치를 탐색한다.
- 행동 매핑: 리드아웃 뉴런(다리 운동 뉴런 + 하행 뉴런) 활동과 다리 누름 상태 → 레인별 누름 확률.
  선형층이며 SB3 PPO 로 학습한다. 입력은 VecNormalize 로 표준화하고 통계도 체크포인트에 저장한다.

환경 여러 개가 뇌 시뮬레이터 하나를 배치로 공유하도록 VecEnv 를 직접 구현한다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from flybeat.agents.base import ProgressFn, StopFn, TrainProgress, eval_charts, evaluate
from flybeat.agents.brain_io import SensoryEncoder
from flybeat.agents.checkpoint import load_checkpoint, new_checkpoint_id, save_checkpoint
from flybeat.brain.model import BrainSim, Connectome, SimConfig
from flybeat.game.env import EnvConfig, RhythmEnv, random_chart_source
from flybeat.game.generator import GeneratorConfig


@dataclass
class MappingConfig:
    profile: str = "compact"
    total_steps: int = 200_000
    n_envs: int = 16
    n_steps: int = 128
    batch_size: int = 512
    learning_rate: float = 1e-3
    gamma: float = 0.95
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=list)  # [] = 선형 리드아웃
    es_iters: int = 0  # >0 이면 감각 매핑을 ES 로 탐색
    es_pop: int = 4
    es_sigma: float = 0.3
    seed: int = 0
    lanes: int = 4


def readout_features(sim: BrainSim, pressed: np.ndarray) -> np.ndarray:
    """(B, n_readout + lanes): 리드아웃 뉴런 활동, 다리 누름 상태."""
    return np.concatenate([sim.readout(), pressed], axis=1).astype(np.float32)


class BrainVecEnv(VecEnv):
    """게임 환경 B개 + 배치 B 짜리 뇌 시뮬레이터 1개. 관측 = 운동 뉴런 활동."""

    def __init__(
        self,
        conn: Connectome,
        encoder: SensoryEncoder,
        n_envs: int,
        gen: GeneratorConfig,
        env_cfg: EnvConfig,
        sim_cfg: SimConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.envs = [
            RhythmEnv(random_chart_source(gen), env_cfg, lanes=gen.lanes) for _ in range(n_envs)
        ]
        self.sim = BrainSim(conn, batch=n_envs, cfg=sim_cfg)
        self.encoder = encoder
        self.tick_ms = 1000 / env_cfg.tick_hz
        self._seed = seed
        self._obs = np.zeros((n_envs, self.envs[0].observation_space.shape[0]), np.float32)
        self._actions: np.ndarray | None = None
        obs_space = spaces.Box(0.0, 1.0, shape=(self.sim.n_readout + gen.lanes,), dtype=np.float32)
        super().__init__(n_envs, obs_space, spaces.MultiBinary(gen.lanes))

    def _features(self) -> np.ndarray:
        return readout_features(self.sim, self.encoder.pressed(self._obs))

    def _advance_brain(self) -> None:
        self.sim.step(self.encoder.encode(self._obs), self.tick_ms)

    def reset(self):
        self.sim.reset(self.num_envs)
        for i, env in enumerate(self.envs):
            self._obs[i], _ = env.reset(seed=self._seed + i)
        self._seed += self.num_envs
        self._advance_brain()
        return self._features()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        rewards = np.zeros(self.num_envs, np.float32)
        dones = np.zeros(self.num_envs, bool)
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            obs, r, done, _, info = env.step(self._actions[i])
            rewards[i] = r
            dones[i] = done
            info = {k: v for k, v in info.items() if k != "events"}
            if done:
                self._obs[i], _ = env.reset(seed=self._seed)
                self._seed += 1
            else:
                self._obs[i] = obs
            infos.append(info)
        self.sim.reset_cols(dones)
        self._advance_brain()
        feats = self._features()
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = feats[i].copy()
        return feats, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(e, attr_name) for e in self._sel(indices)]

    def set_attr(self, attr_name, value, indices=None) -> None:
        for e in self._sel(indices):
            setattr(e, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(e, method_name)(*args, **kwargs) for e in self._sel(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [isinstance(e, wrapper_class) for e in self._sel(indices)]

    def _sel(self, indices) -> list[gym.Env]:
        if indices is None:
            return self.envs
        if isinstance(indices, int):
            indices = [indices]
        return [self.envs[i] for i in indices]


def _make_model(cfg: MappingConfig, venv: VecEnv) -> PPO:
    return PPO(
        "MlpPolicy",
        venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": {"pi": list(cfg.net_arch), "vf": [64]}},
        seed=cfg.seed,
        device="cpu",
        verbose=0,
    )


@dataclass
class ObsNorm:
    """VecNormalize 의 관측 통계 (평가·플레이 때 같은 표준화를 적용)."""

    mean: np.ndarray
    var: np.ndarray
    clip: float = 10.0
    eps: float = 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


class MappingAgent:
    """학습된 모드 B 에이전트 (플레이·평가용, 배치 1)."""

    mode = "B"

    def __init__(
        self,
        conn: Connectome,
        encoder: SensoryEncoder,
        policy,
        norm: ObsNorm,
        cfg: MappingConfig,
    ) -> None:
        self.conn = conn
        self.encoder = encoder
        self.policy = policy
        self.norm = norm
        self.cfg = cfg
        self.lanes = cfg.lanes
        self.sim = BrainSim(conn, batch=1)
        self.tick_ms = 1000 / EnvConfig().tick_hz
        self.rng = np.random.default_rng(0)
        self.last_p = np.zeros(self.lanes, np.float32)

    @property
    def profile(self) -> str:
        return self.cfg.profile

    def reset(self) -> None:
        # 정책은 확률적(학습한 그대로)이고, 곡마다 같은 시드로 시작해 결과를 재현할 수 있게 한다.
        self.sim.reset(1)
        self.rng = np.random.default_rng(self.cfg.seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        # 관측 → 시각 입력 → 뇌 1 tick → 리드아웃 뉴런 → 선형 리드아웃 → 레인별 누름 확률
        self.sim.step(self.encoder.encode(obs), self.tick_ms)
        feats = self.norm(readout_features(self.sim, self.encoder.pressed(obs)))
        with torch.no_grad():
            dist = self.policy.get_distribution(torch.from_numpy(feats))
            p = dist.distribution.probs[0].numpy()
        self.last_p = p
        return (self.rng.random(self.lanes) < p).astype(np.int8)

    def activity(self) -> dict[str, Any]:
        a = brain_activity(self.sim)
        a["pressProb"] = [float(x) for x in self.last_p]
        return a

    @classmethod
    def load(cls, ckpt_id: str) -> MappingAgent:
        tensors, meta = load_checkpoint(ckpt_id)
        if meta["mode"] != "B":
            raise ValueError("모드 B 체크포인트가 아닙니다")
        known = {f.name for f in dataclasses.fields(MappingConfig)}
        cfg = MappingConfig(**{k: v for k, v in meta["config"].items() if k in known})
        conn = Connectome.load(cfg.profile)
        enc = SensoryEncoder.retinotopic(cfg.lanes, EnvConfig(), conn.vertical_bins)
        enc.weights = tensors.pop("encoder.weights")
        norm = ObsNorm(tensors.pop("norm.mean"), tensors.pop("norm.var"))
        venv = BrainVecEnv(conn, enc, 1, GeneratorConfig(lanes=cfg.lanes), EnvConfig())
        model = _make_model(cfg, venv)
        model.policy.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})
        model.policy.set_training_mode(False)
        return cls(conn, enc, model.policy, norm, cfg)


def brain_activity(sim: BrainSim) -> dict[str, Any]:
    """시각화용 요약: 그룹별 평균 활동과 레인별 운동 뉴런 평균."""
    g = sim.conn.groups
    r = sim.r[:, 0]

    def mean(idxs: list[int]) -> float:
        return float(r[idxs].mean()) if idxs else 0.0

    return {
        "input": [[mean(b) for b in lane] for lane in g["input"]],
        "motor": [mean(lane) for lane in g["output"]],
        "kc": mean(g["kc"]),
        "mbon": mean(g["mbon"]),
        "danReward": mean(g["dan_reward"]),
        "danPunish": mean(g["dan_punish"]),
        "activeFraction": float((r > 0.01).mean()),
        "active": top_active(r),
    }


def top_active(r: np.ndarray, k: int = 600, threshold: float = 0.01) -> list[list[int]]:
    """가장 활발한 뉴런 최대 k 개: [[인덱스...], [세기 0~255...]] (뇌 지도 시각화용)."""
    idx = np.flatnonzero(r > threshold)
    if len(idx) > k:
        idx = idx[np.argpartition(r[idx], len(idx) - k)[-k:]]
    level = np.clip(np.round(r[idx] * 255), 1, 255).astype(int)
    return [idx.tolist(), level.tolist()]


def _es_search(
    conn: Connectome,
    enc: SensoryEncoder,
    model: PPO,
    norm: ObsNorm,
    cfg: MappingConfig,
    gen: GeneratorConfig,
    progress: ProgressFn | None,
) -> SensoryEncoder:
    """감각 매핑 가중치를 간단한 (1+λ) 진화 전략으로 탐색 (리드아웃은 고정)."""
    rng = np.random.default_rng(cfg.seed + 1)
    charts = eval_charts(2, seed=20_000, cfg=gen)

    def score(w: np.ndarray) -> float:
        e = dataclasses.replace(enc, weights=w.astype(np.float32))
        agent = MappingAgent(conn, e, model.policy, norm, cfg)
        return evaluate(agent, charts)["accuracy"]

    best_w, best = enc.weights.copy(), score(enc.weights)
    for it in range(cfg.es_iters):
        for _ in range(cfg.es_pop):
            w = np.clip(best_w + rng.normal(0, cfg.es_sigma, best_w.shape), 0, 2)
            s = score(w)
            if s > best:
                best_w, best = w, s
        if progress:
            progress(
                TrainProgress(
                    step=cfg.total_steps,
                    total=cfg.total_steps,
                    episodes=0,
                    accuracy=best,
                    extra={"phase": "es", "esIter": it + 1},
                )
            )
    return dataclasses.replace(enc, weights=best_w.astype(np.float32))


def train(
    cfg: MappingConfig,
    gen: GeneratorConfig | None = None,
    env_cfg: EnvConfig | None = None,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
    ckpt_id: str | None = None,
) -> str:
    from flybeat.agents.deep_rl import _ProgressCallback

    gen = gen or GeneratorConfig(lanes=cfg.lanes)
    env_cfg = env_cfg or EnvConfig()
    conn = Connectome.load(cfg.profile)
    enc = SensoryEncoder.retinotopic(cfg.lanes, env_cfg, conn.vertical_bins)
    raw = BrainVecEnv(conn, enc, cfg.n_envs, gen, env_cfg, None, cfg.seed)
    venv = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = _make_model(cfg, venv)
    model.learn(cfg.total_steps, callback=_ProgressCallback(cfg.total_steps, progress, should_stop))
    norm = ObsNorm(
        venv.obs_rms.mean.astype(np.float32), venv.obs_rms.var.astype(np.float32), clip=10.0
    )

    if cfg.es_iters > 0 and not (should_stop and should_stop()):
        enc = _es_search(conn, enc, model, norm, cfg, gen, progress)

    agent = MappingAgent(conn, enc, model.policy, norm, cfg)
    result = evaluate(agent, eval_charts(cfg=gen), env_cfg)
    ckpt_id = ckpt_id or new_checkpoint_id("B")
    tensors = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    tensors["encoder.weights"] = enc.weights
    tensors["norm.mean"] = norm.mean
    tensors["norm.var"] = norm.var
    save_checkpoint(
        ckpt_id,
        tensors,
        {
            "mode": "B",
            "config": dataclasses.asdict(cfg),
            "generator": dataclasses.asdict(gen),
            "connectome": conn.meta | {"source": None},
            "trainedSteps": int(model.num_timesteps),
            "eval": result,
        },
    )
    return ckpt_id
