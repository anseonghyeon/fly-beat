"""서버 설정. 기본값은 로컬 전용(127.0.0.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("FLYMAX_HOST", "127.0.0.1")
    port: int = int(os.environ.get("FLYMAX_PORT", "8000"))
    cors_origins: list[str] = field(
        default_factory=lambda: _csv(
            "FLYMAX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        )
    )
    max_train_jobs: int = int(os.environ.get("FLYMAX_MAX_TRAIN_JOBS", "1"))
    max_play_sessions: int = int(os.environ.get("FLYMAX_MAX_PLAY_SESSIONS", "4"))
    max_ws_message_bytes: int = 4096
    # 127.0.0.1/localhost 외에 허용할 Host (외부 바인딩 시). 쉼표 구분.
    extra_hosts: list[str] = field(default_factory=lambda: _csv("FLYMAX_ALLOWED_HOSTS", ""))


settings = Settings()
