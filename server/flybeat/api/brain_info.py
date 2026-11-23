"""Brain 화면용 정보: 커넥톰 요약, 모드 C 가소성 분석."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from flybeat import paths
from flybeat.agents.checkpoint import load_checkpoint


def connectome_summary(profile: str = "compact") -> dict[str, Any]:
    p = paths.PROCESSED_DIR / profile / "meta.json"
    if not p.is_file():
        return {"available": False, "profile": profile}
    meta = json.loads(p.read_text("utf-8"))
    return {
        "available": True,
        "profile": profile,
        "neurons": meta.get("neurons"),
        "synapses": meta.get("synapses"),
        "kcMbonSynapses": meta.get("kc_mbon_edges"),
        "groups": meta.get("groups"),
        "profileConfig": meta.get("profile_config"),
    }


@cache
def _neuron_types(profile: str) -> pd.Series:
    df = pd.read_feather(paths.PROCESSED_DIR / profile / "neurons.feather", columns=["type"])
    return df["type"].fillna("?")


def plasticity_summary(ckpt_id: str, top: int = 12) -> dict[str, Any]:
    tensors, meta = load_checkpoint(ckpt_id)
    if meta["mode"] != "C":
        raise ValueError("모드 C 체크포인트만 가소성 정보가 있습니다")
    w, w0 = tensors["plastic.weight"], tensors["plastic.weight0"]
    pre, post = tensors["plastic.pre"], tensors["plastic.post"]
    profile = meta["config"].get("profile", "compact")
    types = _neuron_types(profile)

    ratio = np.divide(w, w0, out=np.ones_like(w), where=w0 > 0)
    changed = np.abs(w - w0) > 1e-9
    log2 = np.log2(np.clip(ratio[changed], 1e-3, None))
    edges = np.linspace(-3, 3, 25)
    hist, _ = np.histogram(np.clip(log2, -3, 3), bins=edges)

    df = pd.DataFrame(
        {
            "kc": types.iloc[pre].to_numpy(),
            "mbon": types.iloc[post].to_numpy(),
            "dw": (w - w0).astype(float),
            "w0": w0.astype(float),
        }
    )
    agg = df.groupby(["kc", "mbon"], as_index=False).agg(dw=("dw", "sum"), w0=("w0", "sum"))
    agg["rel"] = agg["dw"] / agg["w0"].clip(lower=1e-9)
    agg = agg[agg["dw"] != 0]
    up = agg.nlargest(top, "dw")
    down = agg.nsmallest(top, "dw")

    def rows(d: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {"kc": r.kc, "mbon": r.mbon, "dw": float(r.dw), "rel": float(r.rel)}
            for r in d.itertuples()
        ]

    return {
        "checkpoint": ckpt_id,
        "synapses": int(len(w)),
        "changed": int(changed.sum()),
        "strengthened": int((w > w0 + 1e-9).sum()),
        "weakened": int((w < w0 - 1e-9).sum()),
        "totalRatio": float(w.sum() / max(w0.sum(), 1e-9)),
        "histogram": {"edges": edges.tolist(), "counts": hist.tolist()},
        "topUp": rows(up[up["dw"] > 0]),
        "topDown": rows(down[down["dw"] < 0]),
        "laneBias": tensors["lane.bias"].tolist(),
    }
