"""FastAPI 앱."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from flybeat import __version__
from flybeat.api.settings import settings
from flybeat.game.chart import ChartError, list_charts, load_chart
from flybeat.game.generator import generate_chart


def create_app() -> FastAPI:
    app = FastAPI(title="fly-beat", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

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

    return app


app = create_app()
