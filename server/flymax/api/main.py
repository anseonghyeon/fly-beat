"""FastAPI 앱.

로컬 실행 전용이지만 공개 저장소이므로 기본값을 안전하게 둔다.
- Host 헤더 제한 (DNS 리바인딩 방지)
- 상태를 바꾸는 요청(POST)과 WebSocket 은 허용된 Origin 에서만 받는다 (다른 사이트가 브라우저를 통해
  로컬 서버를 조작하는 것 방지). POST 는 JSON 만 받는다.
- 경로는 클라이언트 입력으로 받지 않고 ID 로만 찾는다.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse

from flymax import __version__
from flymax.agents.checkpoint import CheckpointError, list_checkpoints, load_meta
from flymax.api import brain_info
from flymax.api.jobs import JobLimitError, TrainingManager, TrainRequest
from flymax.api.play import open_session, stream
from flymax.api.settings import Settings, settings
from flymax.game.chart import AUDIO_TYPES, ChartError, audio_path, list_charts, load_chart
from flymax.game.generator import generate_chart

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]


def _checkpoint_summary(m: dict) -> dict:
    ev = m.get("eval") or {}
    return {
        "id": m.get("id"),
        "mode": m.get("mode"),
        "savedAt": m.get("savedAt"),
        "accuracy": ev.get("accuracy"),
        "trainedSteps": m.get("trainedSteps"),
        "trainedEpisodes": m.get("trainedEpisodes"),
        "plasticity": m.get("plasticity"),
    }


def create_app(cfg: Settings = settings) -> FastAPI:
    manager = TrainingManager(cfg.max_train_jobs)
    play_slots = asyncio.Semaphore(cfg.max_play_sessions)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="flymax-respect-v",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS + cfg.extra_hosts)

    def origin_ok(origin: str | None) -> bool:
        # Origin 이 없으면 브라우저가 아닌 클라이언트(curl 등) → 허용
        return origin is None or origin in cfg.cors_origins

    @app.middleware("http")
    async def guard_writes(request: Request, call_next):
        if request.method == "POST":
            if not origin_ok(request.headers.get("origin")):
                return JSONResponse({"detail": "허용되지 않은 Origin"}, status_code=403)
            ctype = request.headers.get("content-type", "")
            if not ctype.startswith("application/json"):
                return JSONResponse({"detail": "Content-Type 은 application/json"}, 415)
        return await call_next(request)

    # ------------------------------------------------------------------ 기본
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    # ------------------------------------------------------------------ 채보
    @app.get("/api/charts")
    def charts() -> list[dict]:
        return [c.summary() for c in list_charts()]

    @app.get("/api/charts/random/{seed}")
    def random_chart(seed: int) -> dict:
        if not 0 <= seed < 2**31:
            raise HTTPException(422, "seed 범위 오류")
        return generate_chart(seed).to_dict()

    @app.get("/api/charts/{chart_id}")
    def chart(chart_id: str) -> dict:
        try:
            return load_chart(chart_id).to_dict()
        except (ChartError, FileNotFoundError):
            raise HTTPException(404, "채보를 찾을 수 없습니다") from None

    @app.get("/api/audio/{name}")
    def audio(name: str) -> FileResponse:
        try:
            p = audio_path(name)
        except (ChartError, FileNotFoundError):
            raise HTTPException(404, "음원을 찾을 수 없습니다") from None
        return FileResponse(
            p,
            media_type=AUDIO_TYPES[p.suffix.lstrip(".")],
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # ------------------------------------------------------------------ 학습
    @app.post("/api/train", status_code=201)
    def start_train(req: TrainRequest) -> dict:
        try:
            return manager.start(req).summary()
        except JobLimitError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/train")
    def jobs() -> list[dict]:
        return [j.summary() for j in sorted(manager.jobs.values(), key=lambda j: -j.created)]

    def _job(job_id: str):
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "학습 작업을 찾을 수 없습니다")
        return job

    @app.get("/api/train/{job_id}")
    def job(job_id: str) -> dict:
        j = _job(job_id)
        return {**j.summary(), "history": j.history}

    @app.post("/api/train/{job_id}/stop")
    def stop_job(job_id: str) -> dict:
        _job(job_id)
        return manager.stop(job_id).summary()

    @app.websocket("/ws/train/{job_id}")
    async def ws_train(ws: WebSocket, job_id: str) -> None:
        if not origin_ok(ws.headers.get("origin")):
            await ws.close(code=1008)
            return
        await ws.accept()
        sent = 0
        try:
            while True:
                j = manager.jobs.get(job_id)
                if j is None:
                    await ws.send_json({"type": "error", "message": "작업 없음"})
                    break
                new = j.history[sent:]
                sent = len(j.history)
                await ws.send_json({"type": "update", "job": j.summary(), "progress": new})
                if j.status not in ("queued", "running"):
                    break
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return
        await ws.close()

    # ------------------------------------------------------------------ 체크포인트·뇌
    @app.get("/api/checkpoints")
    def checkpoints() -> list[dict]:
        return [_checkpoint_summary(m) for m in list_checkpoints()]

    @app.get("/api/checkpoints/{ckpt_id}")
    def checkpoint(ckpt_id: str) -> dict:
        try:
            return load_meta(ckpt_id)
        except (CheckpointError, FileNotFoundError):
            raise HTTPException(404, "체크포인트를 찾을 수 없습니다") from None

    @app.get("/api/brain")
    def brain() -> dict:
        return brain_info.connectome_summary()

    @app.get("/api/brain/layout/{profile}")
    def layout(profile: str) -> dict:
        if profile not in ("compact", "medium", "full"):
            raise HTTPException(404, "알 수 없는 프로필")
        try:
            return brain_info.layout(profile)
        except FileNotFoundError:
            raise HTTPException(404, "전처리 결과가 없습니다") from None

    @app.get("/api/brain/plasticity/{ckpt_id}")
    def plasticity(ckpt_id: str) -> dict:
        try:
            return brain_info.plasticity_summary(ckpt_id)
        except (CheckpointError, FileNotFoundError):
            raise HTTPException(404, "체크포인트를 찾을 수 없습니다") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    # ------------------------------------------------------------------ 관전
    @app.websocket("/ws/play")
    async def ws_play(ws: WebSocket) -> None:
        if not origin_ok(ws.headers.get("origin")):
            await ws.close(code=1008)
            return
        await ws.accept()
        if play_slots.locked():
            await ws.send_json({"type": "error", "message": "동시 관전 세션이 너무 많습니다"})
            await ws.close()
            return
        async with play_slots:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if len(raw) > cfg.max_ws_message_bytes:
                    raise ValueError("메시지가 너무 큽니다")
                msg = json.loads(raw)
                if not isinstance(msg, dict) or msg.get("type") != "start":
                    raise ValueError("start 메시지가 필요합니다")
                session = await open_session(msg)
            except WebSocketDisconnect:
                return
            except (ValueError, ChartError, CheckpointError, FileNotFoundError, TimeoutError) as e:
                detail = str(e) if isinstance(e, (ValueError, ChartError)) else "찾을 수 없음"
                await ws.send_json({"type": "error", "message": detail[:200]})
                await ws.close()
                return
            try:
                await stream(ws, session)
                await ws.close()
            except WebSocketDisconnect:
                return

    return app


app = create_app()
