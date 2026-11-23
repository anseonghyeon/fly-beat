"""모드 A: 일반 딥러닝 레이어(MLP) + Stable-Baselines3 PPO 베이스라인.

SB3 기본 저장 형식(.zip)은 pickle 을 포함하므로 쓰지 않는다.
정책망 state_dict 만 safetensors 로 저장하고, 불러올 때 같은 구조로 만든 뒤 가중치만 넣는다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from flybeat.agents.base import ProgressFn, StopFn, TrainProgress, eval_charts, evaluate
from flybeat.agents.checkpoint import load_checkpoint, new_checkpoint_id, save_checkpoint
from flybeat.game.env import EnvConfig, RhythmEnv, random_chart_source
from flybeat.game.generator import GeneratorConfig


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    n_envs: int = 8
    n_steps: int = 256
    batch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.95
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    seed: int = 0
    lanes: int = 4


def _policy_kwargs(cfg: PPOConfig) -> dict[str, Any]:
    return {"net_arch": list(cfg.net_arch), "activation_fn": torch.nn.ReLU}


def _make_model(cfg: PPOConfig, env) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        policy_kwargs=_policy_kwargs(cfg),
        seed=cfg.seed,
        device="cpu",
        verbose=0,
    )


class _ProgressCallback(BaseCallback):
    def __init__(self, total: int, progress: ProgressFn | None, should_stop: StopFn | None):
        super().__init__()
        self.total = total
        self.progress = progress
        self.should_stop = should_stop
        self.episodes = 0
        self.recent_acc: list[float] = []
        self.recent_rew: list[float] = []
        self._ep_rew: np.ndarray | None = None

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        if self._ep_rew is None:
            self._ep_rew = np.zeros_like(rewards, dtype=np.float64)
        self._ep_rew += rewards
        for i, (done, info) in enumerate(
            zip(self.locals["dones"], self.locals["infos"], strict=True)
        ):
            if done and "stats" in info:
                self.episodes += 1
                self.recent_acc = (self.recent_acc + [info["stats"]["accuracy"]])[-20:]
                self.recent_rew = (self.recent_rew + [float(self._ep_rew[i])])[-20:]
                self._ep_rew[i] = 0
        return not (self.should_stop and self.should_stop())

    def _on_rollout_end(self) -> None:
        if self.progress:
            self.progress(
                TrainProgress(
                    step=self.num_timesteps,
                    total=self.total,
                    episodes=self.episodes,
                    accuracy=float(np.mean(self.recent_acc)) if self.recent_acc else None,
                    reward=float(np.mean(self.recent_rew)) if self.recent_rew else None,
                )
            )


def train(
    cfg: PPOConfig,
    gen: GeneratorConfig | None = None,
    env_cfg: EnvConfig | None = None,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
    ckpt_id: str | None = None,
) -> str:
    gen = gen or GeneratorConfig(lanes=cfg.lanes)
    env_cfg = env_cfg or EnvConfig()

    def make(i: int):
        def _f():
            return RhythmEnv(random_chart_source(gen), env_cfg, lanes=cfg.lanes)

        return _f

    venv = DummyVecEnv([make(i) for i in range(cfg.n_envs)])
    venv.seed(cfg.seed)
    model = _make_model(cfg, venv)
    model.learn(cfg.total_steps, callback=_ProgressCallback(cfg.total_steps, progress, should_stop))

    agent = PPOAgent(model.policy, cfg)
    result = evaluate(agent, eval_charts(cfg=gen), env_cfg)
    ckpt_id = ckpt_id or new_checkpoint_id("A")
    tensors = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    save_checkpoint(
        ckpt_id,
        tensors,
        {
            "mode": "A",
            "config": dataclasses.asdict(cfg),
            "generator": dataclasses.asdict(gen),
            "trainedSteps": int(model.num_timesteps),
            "eval": result,
        },
    )
    venv.close()
    return ckpt_id


class PPOAgent:
    mode = "A"

    def __init__(self, policy, cfg: PPOConfig) -> None:
        self.policy = policy
        self.lanes = cfg.lanes

    @classmethod
    def load(cls, ckpt_id: str) -> PPOAgent:
        tensors, meta = load_checkpoint(ckpt_id)
        if meta["mode"] != "A":
            raise ValueError("모드 A 체크포인트가 아닙니다")
        known = {f.name for f in dataclasses.fields(PPOConfig)}
        cfg = PPOConfig(**{k: v for k, v in meta["config"].items() if k in known})
        env = DummyVecEnv([lambda: RhythmEnv(lanes=cfg.lanes)])
        model = _make_model(cfg, env)
        state = {k: torch.from_numpy(v) for k, v in tensors.items()}
        model.policy.load_state_dict(state, strict=True)
        model.policy.set_training_mode(False)
        return cls(model.policy, cfg)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.policy.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.int8)

    def activity(self) -> dict[str, Any] | None:
        return None
