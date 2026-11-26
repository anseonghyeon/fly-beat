"""학습 작업 관리. 학습은 별도 프로세스에서 돌려 API 응답을 막지 않는다."""

from __future__ import annotations

import multiprocessing as mp
import queue
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["A", "B", "C"]
Status = Literal["queued", "running", "done", "failed", "stopped"]
MAX_HISTORY = 2000


class TrainRequest(BaseModel):
    """학습 요청. 자원을 과하게 쓰지 않도록 범위를 제한한다."""

    mode: Mode
    steps: int | None = Field(default=None, ge=1_000, le=5_000_000)
    episodes: int | None = Field(default=None, ge=1, le=1_000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _check(self) -> TrainRequest:
        if self.mode == "C" and self.steps is not None:
            raise ValueError("모드 C 는 steps 대신 episodes 를 씁니다")
        if self.mode in ("A", "B") and self.episodes is not None:
            raise ValueError("모드 A·B 는 episodes 대신 steps 를 씁니다")
        return self


def _worker(req: dict[str, Any], ckpt_id: str, q: mp.Queue, stop: Any) -> None:
    """자식 프로세스: 학습하고 진행 상황을 큐로 보낸다."""
    try:
        from flymax.agents.base import TrainProgress

        def progress(p: TrainProgress) -> None:
            q.put(("progress", p.to_dict()))

        def should_stop() -> bool:
            return stop.is_set()

        mode = req["mode"]
        if mode == "A":
            from flymax.agents.deep_rl import PPOConfig, train

            cfg = PPOConfig(total_steps=req.get("steps") or 300_000, seed=req["seed"])
        elif mode == "B":
            from flymax.agents.mapping import MappingConfig, train

            cfg = MappingConfig(total_steps=req.get("steps") or 1_000_000, seed=req["seed"])
        else:
            from flymax.agents.dopamine import DopamineConfig, train

            cfg = DopamineConfig(episodes=req.get("episodes") or 64, seed=req["seed"])
        train(cfg, progress=progress, should_stop=should_stop, ckpt_id=ckpt_id)
        q.put(("done", ckpt_id))
    except FileNotFoundError as e:
        q.put(("failed", str(e).splitlines()[0]))
    except Exception as e:  # noqa: BLE001 - 자식 프로세스의 모든 오류를 부모에 알린다
        traceback.print_exc()
        q.put(("failed", f"{type(e).__name__}: 학습 중 오류가 발생했습니다 (서버 로그 참고)"))


@dataclass
class Job:
    id: str
    request: TrainRequest
    status: Status = "queued"
    created: float = field(default_factory=time.time)
    finished: float | None = None
    checkpoint: str | None = None
    error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    _proc: Any = None
    _stop: Any = None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.request.mode,
            "request": self.request.model_dump(),
            "status": self.status,
            "created": self.created,
            "finished": self.finished,
            "checkpoint": self.checkpoint,
            "error": self.error,
            "latest": self.history[-1] if self.history else None,
        }


class JobLimitError(RuntimeError):
    pass


class TrainingManager:
    def __init__(self, max_jobs: int = 1) -> None:
        self.max_jobs = max_jobs
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("queued", "running")]

    def start(self, req: TrainRequest) -> Job:
        from flymax.agents.checkpoint import new_checkpoint_id

        with self._lock:
            if len(self.running()) >= self.max_jobs:
                raise JobLimitError(f"동시에 실행할 수 있는 학습은 {self.max_jobs}개입니다")
            job = Job(id=secrets.token_hex(6), request=req)
            q: mp.Queue = self._ctx.Queue()
            job._stop = self._ctx.Event()
            ckpt_id = new_checkpoint_id(req.mode)
            job._proc = self._ctx.Process(
                target=_worker, args=(req.model_dump(), ckpt_id, q, job._stop), daemon=True
            )
            job._proc.start()
            job.status = "running"
            self.jobs[job.id] = job
        threading.Thread(target=self._pump, args=(job, q), daemon=True).start()
        return job

    def _pump(self, job: Job, q: mp.Queue) -> None:
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except queue.Empty:
                if not job._proc.is_alive():
                    if job.status == "running":
                        job.status = "failed"
                        job.error = "학습 프로세스가 비정상 종료되었습니다"
                        job.finished = time.time()
                    return
                continue
            if kind == "progress":
                job.history.append(payload)
                if len(job.history) > MAX_HISTORY:
                    job.history = job.history[::2]
            elif kind == "done":
                job.checkpoint = payload
                job.status = "stopped" if job._stop.is_set() else "done"
                job.finished = time.time()
                return
            elif kind == "failed":
                job.status = "failed"
                job.error = str(payload)[:300]
                job.finished = time.time()
                return

    def stop(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.status == "running":
            job._stop.set()
        return job

    def shutdown(self) -> None:
        for job in self.running():
            job._stop.set()
            if job._proc is not None:
                job._proc.join(timeout=5)
                if job._proc.is_alive():
                    job._proc.terminate()
