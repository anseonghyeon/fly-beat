"""뇌 시뮬레이터 벤치마크와 응답 확인.

사용법: python -m flybeat.brain.bench [--profile compact] [--gain 0.25 --bias -0.15]
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from flybeat.brain.model import BrainSim, Connectome, SimConfig

TICK_MS = 1000 / 60


def response_trace(conn: Connectome, cfg: SimConfig) -> None:
    """레인별로 167ms 켜고 167ms 끄는 입력을 주고 운동 뉴런 평균 활동을 출력한다."""
    lanes, vb = conn.lanes, conn.vertical_bins
    sim = BrainSim(conn, batch=1, cfg=cfg)
    print("tick  입력레인  운동평균  활성비율   레인별 운동 평균")
    for k in range(80):
        u = np.zeros((1, lanes * vb), np.float32)
        lane = (k // 20) % lanes
        on = (k // 10) % 2 == 0
        if on:
            u[0, lane * vb : (lane + 1) * vb] = 1.0
        sim.step(u, TICK_MS)
        if k % 4 == 3:
            m = sim.motor_lane_mean()[0]
            print(
                f"{k:4d}  {lane if on else '-':>6}  {sim.motor().mean():8.4f}"
                f"  {(sim.r > 0.01).mean():7.3f}   " + " ".join(f"{x:.3f}" for x in m)
            )


def throughput(conn: Connectome, cfg: SimConfig) -> None:
    for batch in (1, 8, 32):
        sim = BrainSim(conn, batch=batch, cfg=cfg)
        u = np.random.default_rng(0).random((batch, sim.n_input_channels), dtype=np.float32)
        n = 60
        t = time.perf_counter()
        for _ in range(n):
            sim.step(u, TICK_MS)
        dt = time.perf_counter() - t
        print(f"batch {batch:3d}: {batch * n / dt / 60:6.1f} game-sec / wall-sec")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m flybeat.brain.bench")
    p.add_argument("--profile", default="compact")
    for f in dataclasses.fields(SimConfig):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=None)
    args = p.parse_args()
    cfg = SimConfig.from_yaml()
    for f in dataclasses.fields(SimConfig):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    conn = Connectome.load(args.profile)
    print(f"profile={args.profile} neurons={conn.n:,} synapses={conn.W.nnz:,}")
    print(cfg)
    response_trace(conn, cfg)
    throughput(conn, cfg)


if __name__ == "__main__":
    main()
