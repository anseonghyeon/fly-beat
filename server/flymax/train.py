"""학습 CLI (서버 없이 실행).

예)
  python -m flymax.train --mode A --steps 300000
  python -m flymax.train --mode B --steps 200000
  python -m flymax.train --mode C --episodes 60
"""

from __future__ import annotations

import argparse
import time

from flymax.agents.base import TrainProgress
from flymax.agents.checkpoint import load_meta


def _printer():
    t0 = time.time()

    def show(p: TrainProgress) -> None:
        d = p.to_dict()
        acc = "-" if d["accuracy"] is None else f"{d['accuracy']:.1f}%"
        rew = "-" if d["reward"] is None else f"{d['reward']:.1f}"
        print(
            f"[{time.time() - t0:7.1f}s] step {d['step']:>9,}/{d['total']:,}"
            f"  episodes {d['episodes']:>5}  acc {acc:>7}  reward {rew}",
            flush=True,
        )

    return show


def run(mode: str, args: argparse.Namespace) -> str:
    if mode == "A":
        from flymax.agents.deep_rl import PPOConfig, train

        cfg = PPOConfig(total_steps=args.steps or 300_000, seed=args.seed)
        return train(cfg, progress=_printer())
    if mode == "B":
        from flymax.agents.mapping import MappingConfig
        from flymax.agents.mapping import train as train_b

        cfg_b = MappingConfig(
            total_steps=args.steps or 1_000_000, profile=args.profile, seed=args.seed
        )
        return train_b(cfg_b, progress=_printer())
    from flymax.agents.dopamine import DopamineConfig
    from flymax.agents.dopamine import train as train_c

    cfg_c = DopamineConfig(episodes=args.episodes or 64, profile=args.profile, seed=args.seed)
    return train_c(cfg_c, progress=_printer())


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m flymax.train")
    p.add_argument("--mode", choices=["A", "B", "C"], required=True)
    p.add_argument("--steps", type=int, default=None, help="학습 스텝 수 (모드 A·B)")
    p.add_argument("--episodes", type=int, default=None, help="학습 에피소드 수 (모드 C)")
    p.add_argument("--profile", default="compact", help="커넥톰 프로필 (모드 B·C)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ckpt = run(args.mode, args)
    meta = load_meta(ckpt)
    print(f"\n체크포인트: {ckpt}")
    print(f"검증 정확도: {meta['eval']['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
