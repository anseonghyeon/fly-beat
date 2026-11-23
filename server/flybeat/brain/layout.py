"""시각화용 뉴런 배치: 세포체 실제 3D 좌표(somaLocation) → 2D 투영.

MaleCNS 좌표축 (8nm 복셀): x = 좌우 (값이 클수록 초파리의 왼쪽), y = 등(작음)→배(큼),
z = 앞(작음)→뒤(큼, 복부신경삭 쪽).

- top  (위에서 본 모습, 머리가 위): 화면 x = 초파리 왼쪽이 화면 왼쪽, 화면 y = z
- side (왼쪽에서 본 모습, 머리가 왼쪽): 화면 x = z, 화면 y = y

좌표가 없는 뉴런(주로 시각엽 내부 뉴런)은 연결된 이웃 뉴런 좌표의 평균으로 채운다.
결과는 data/processed/<profile>/layout.json 에 캐시한다.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybeat import paths

QUANT = 4095  # 좌표를 0..QUANT 정수로 양자화 (전송량 절약)

# 시각화 영역 (색과 범례에 쓰임). 순서 = 코드 값.
REGIONS = [
    ("other", "기타 중추 뇌"),
    ("visual", "시각계"),
    ("mb", "버섯체 (KC·MBON)"),
    ("dan", "도파민 뉴런"),
    ("dn", "하행 뉴런"),
    ("motor", "다리 운동 뉴런"),
    ("vnc", "복부신경삭"),
]
_CODE = {k: i for i, (k, _) in enumerate(REGIONS)}


def region_codes(neurons: pd.DataFrame, groups: dict[str, Any]) -> np.ndarray:
    sc = neurons["superclass"].fillna("").to_numpy().astype(str)
    code = np.full(len(neurons), _CODE["other"], dtype=np.int8)
    code[np.char.startswith(sc, "vnc") | (sc == "ascending_neuron")] = _CODE["vnc"]
    code[np.char.startswith(sc, "ol_") | np.char.startswith(sc, "visual")] = _CODE["visual"]
    code[sc == "descending_neuron"] = _CODE["dn"]
    code[groups["kc"] + groups["mbon"]] = _CODE["mb"]
    code[groups["dan_reward"] + groups["dan_punish"]] = _CODE["dan"]
    code[[i for lane in groups["output"] for i in lane]] = _CODE["motor"]
    return code


def fill_missing(
    xyz: np.ndarray, known: np.ndarray, W: sp.csr_matrix, iters: int = 8
) -> np.ndarray:
    """좌표 없는 뉴런을 이웃(입출력 연결) 좌표 평균으로 반복해서 채운다."""
    A = (abs(W) + abs(W).T).tocsr()
    A.data[:] = 1.0
    xyz = xyz.copy()
    have = known.copy()
    for _ in range(iters):
        if have.all():
            break
        s = A @ (xyz * have[:, None])
        c = np.asarray(A @ have.astype(np.float64)).ravel()
        new = (~have) & (c > 0)
        xyz[new] = s[new] / c[new, None]
        have = have | new
    if (~have).any():  # 그래도 없으면 전체 중심
        xyz[~have] = xyz[have].mean(axis=0)
    return xyz


def _quantize(v: np.ndarray, lo: float, hi: float) -> list[int]:
    return np.round((v - lo) / max(hi - lo, 1e-9) * QUANT).astype(int).clip(0, QUANT).tolist()


def build_layout(profile: str = "compact") -> dict[str, Any]:
    from flybeat.brain.model import Connectome

    conn = Connectome.load(profile)
    ann = pd.read_feather(paths.ANNOTATIONS_FILE, columns=["bodyId", "somaLocation"])
    df = conn.neurons[["bodyId"]].merge(ann, on="bodyId", how="left")
    known = df["somaLocation"].notna().to_numpy()
    xyz = np.zeros((len(df), 3), dtype=np.float64)
    xyz[known] = np.stack(df.loc[known, "somaLocation"].to_numpy()).astype(np.float64)
    xyz = fill_missing(xyz, known, conn.W)
    # 같은 자리에 겹치지 않게 살짝 흩뜨림 (재현 가능)
    rng = np.random.default_rng(0)
    xyz[~known] += rng.normal(0, 600, size=((~known).sum(), 3))

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = float((hi - lo).max())  # 세 축 같은 축척
    top_x = (hi[0] - x) / span  # 초파리 왼쪽(x 큼) → 화면 왼쪽
    top_y = (z - lo[2]) / span
    side_x = (z - lo[2]) / span
    side_y = (y - lo[1]) / span
    return {
        "profile": profile,
        "n": int(len(df)),
        "quant": QUANT,
        "knownFraction": float(known.mean()),
        "regions": [{"key": k, "name": n} for k, n in REGIONS],
        "region": region_codes(conn.neurons, conn.groups).tolist(),
        "views": {
            "top": {
                "x": _quantize(top_x, 0, 1),
                "y": _quantize(top_y, 0, 1),
                "aspect": float((hi[0] - lo[0]) / (hi[2] - lo[2])),
            },
            "side": {
                "x": _quantize(side_x, 0, 1),
                "y": _quantize(side_y, 0, 1),
                "aspect": float((hi[2] - lo[2]) / (hi[1] - lo[1])),
            },
        },
        "groups": {
            "input": conn.groups["input"],
            "output": conn.groups["output"],
        },
    }


def load_layout(profile: str = "compact") -> dict[str, Any]:
    """캐시가 있으면 읽고, 없으면 만들어 저장한다."""
    p = paths.PROCESSED_DIR / profile / "layout.json"
    meta = paths.PROCESSED_DIR / profile / "meta.json"
    if p.is_file() and (not meta.is_file() or p.stat().st_mtime >= meta.stat().st_mtime):
        return json.loads(p.read_text("utf-8"))
    layout = build_layout(profile)
    p.write_text(json.dumps(layout, separators=(",", ":")), "utf-8")
    return layout
