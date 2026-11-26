"""프로젝트 경로. 환경 변수로 위치를 바꿀 수 있다."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("FLYMAX_DATA_DIR", REPO_ROOT / "data")).resolve()
PROCESSED_DIR = DATA_DIR / "processed"
CHARTS_DIR = Path(os.environ.get("FLYMAX_CHARTS_DIR", REPO_ROOT / "charts")).resolve()
CHECKPOINTS_DIR = Path(
    os.environ.get("FLYMAX_CHECKPOINTS_DIR", REPO_ROOT / "checkpoints")
).resolve()

ANNOTATIONS_FILE = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILE = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def require_raw_data() -> None:
    missing = [
        p.name for p in (ANNOTATIONS_FILE, NEUROTRANSMITTERS_FILE, WEIGHTS_FILE) if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "원본 데이터가 없습니다: "
            + ", ".join(missing)
            + "\n먼저 ./scripts/download_data.sh 를 실행하세요."
        )
