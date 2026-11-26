"""체크포인트 저장/로딩.

- 텐서는 safetensors 로만 저장한다 (pickle 미사용 → 파일을 열어도 코드가 실행되지 않음).
- 설정·지표는 meta.json 에 둔다.
- 체크포인트는 ID 로만 찾고, CHECKPOINTS_DIR 밖으로 나가는 경로는 거부한다.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from flymax import paths

CKPT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
MODES = ("A", "B", "C")
MAX_META_BYTES = 1 << 20


class CheckpointError(ValueError):
    pass


def new_checkpoint_id(mode: str) -> str:
    return f"{mode.lower()}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def checkpoint_dir(ckpt_id: str, root: Path | None = None) -> Path:
    if not CKPT_ID_RE.fullmatch(ckpt_id):
        raise CheckpointError("잘못된 체크포인트 ID")
    base = (root or paths.CHECKPOINTS_DIR).resolve()
    d = (base / ckpt_id).resolve()
    if d.parent != base:
        raise CheckpointError("잘못된 체크포인트 ID")
    return d


def save_checkpoint(
    ckpt_id: str,
    tensors: dict[str, np.ndarray],
    meta: dict[str, Any],
    root: Path | None = None,
) -> Path:
    d = checkpoint_dir(ckpt_id, root)
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, tmp / "model.safetensors")
    meta = {**meta, "id": ckpt_id, "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    return d


def load_meta(ckpt_id: str, root: Path | None = None) -> dict[str, Any]:
    p = checkpoint_dir(ckpt_id, root) / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(ckpt_id)
    if p.stat().st_size > MAX_META_BYTES:
        raise CheckpointError("meta.json 이 너무 큽니다")
    meta = json.loads(p.read_text("utf-8"))
    if meta.get("mode") not in MODES:
        raise CheckpointError("알 수 없는 모드")
    return meta


def load_checkpoint(
    ckpt_id: str, root: Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta = load_meta(ckpt_id, root)
    tensors = load_file(checkpoint_dir(ckpt_id, root) / "model.safetensors")
    return tensors, meta


def list_checkpoints(root: Path | None = None) -> list[dict[str, Any]]:
    base = (root or paths.CHECKPOINTS_DIR).resolve()
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        if d.is_dir() and CKPT_ID_RE.fullmatch(d.name):
            try:
                out.append(load_meta(d.name, base))
            except (CheckpointError, FileNotFoundError, json.JSONDecodeError):
                continue
    return out
