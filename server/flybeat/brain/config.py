"""뇌 모델 설정 로딩."""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

import yaml


@cache
def load_brain_config() -> dict[str, Any]:
    text = resources.files("flybeat.config").joinpath("brain.yaml").read_text("utf-8")
    return yaml.safe_load(text)


def profile_config(profile: str) -> dict[str, Any]:
    cfg = load_brain_config()
    if profile not in cfg["profiles"]:
        raise ValueError(f"알 수 없는 프로필: {profile} (가능: {', '.join(cfg['profiles'])})")
    return cfg["profiles"][profile]
