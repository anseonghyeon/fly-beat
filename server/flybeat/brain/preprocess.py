"""MaleCNS 원본 데이터 → 시뮬레이션용 희소 행렬과 뉴런 그룹.

사용법: python -m flybeat.brain.preprocess [--profile compact|medium|full]

출력 (data/processed/<profile>/)
- W.npz          : 부호 × 시냅스 수 CSR 행렬 W[post, pre] (float32)
- neurons.feather: 행렬 인덱스 ↔ bodyId, type, class, side, nt, sign
- groups.json    : 입력/출력/DAN/KC/MBON 뉴런 인덱스
- dan.npz        : DAN → KC·MBON 연결 (모드 C 의 도파민 구획 정보)
- meta.json      : 크기, 설정, 원본 파일 정보
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from flybeat import paths
from flybeat.brain.config import load_brain_config, profile_config

FORMAT_VERSION = 2


def log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


# --------------------------------------------------------------------------- 뉴런
def load_neurons(cfg: dict[str, Any]) -> pd.DataFrame:
    ann = pd.read_feather(paths.ANNOTATIONS_FILE)
    ann = ann[ann["status"].isin(cfg["neurons"]["status"])].copy()
    nt = pd.read_feather(paths.NEUROTRANSMITTERS_FILE, columns=["body", "consensus_nt"])
    ann = ann.merge(nt, left_on="bodyId", right_on="body", how="left").drop(columns="body")
    ann["nt"] = ann["consensus_nt"].fillna("missing")
    signs = cfg["signs"]
    ann["sign"] = ann["nt"].map(lambda x: signs.get(x, signs["missing"])).astype(np.int8)
    ann = ann.sort_values("bodyId").reset_index(drop=True)
    return ann


def _split_half(values: np.ndarray, half: str) -> np.ndarray:
    med = np.median(values)
    return values >= med if half == "outer" else values < med


def build_groups(neu: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """전체 Traced 뉴런 테이블 기준 인덱스로 그룹을 만든다."""
    icfg = cfg["input"]
    vb = icfg["vertical_bins"]
    a1, a2 = icfg["azimuth"]
    e1, e2 = icfg["elevation"]
    vis = neu[neu["type"].isin(icfg["types"]) & neu["assignedOlHex1"].notna()]
    h1 = vis["assignedOlHex1"].to_numpy()
    h2 = vis["assignedOlHex2"].to_numpy()
    az = a1 * h1 + a2 * h2
    el = e1 * h1 + e2 * h2
    side = vis["somaSide"].to_numpy()

    inputs: list[list[list[int]]] = []
    for lane_cfg in icfg["lanes"]:
        on_side = side == lane_cfg["side"]
        in_half = np.zeros(len(vis), dtype=bool)
        in_half[on_side] = _split_half(az[on_side], lane_cfg["half"])
        idx = np.where(in_half)[0]
        # 고도 순으로 나눔: bin 0 = 위(먼 노트), 마지막 = 아래(판정선 근처)
        order = idx[np.argsort(-el[idx], kind="stable")]
        bins = np.array_split(order, vb)
        inputs.append([sorted(vis.index[b].tolist()) for b in bins])

    outputs = []
    for o in cfg["output"]["lanes"]:
        m = (
            (neu["superclass"] == "vnc_motor")
            & (neu["exitNerve"] == o["exitNerve"])
            & (neu["somaSide"] == o["side"])
        )
        outputs.append(neu.index[m].tolist())

    mb = cfg["mushroom_body"]
    is_dan = neu["class"] == "DAN"
    t = neu["type"].fillna("")
    return {
        "input": inputs,
        "output": outputs,
        "dan_reward": neu.index[is_dan & t.str.startswith(mb["reward_dan_prefix"])].tolist(),
        "dan_punish": neu.index[is_dan & t.str.startswith(mb["punish_dan_prefix"])].tolist(),
        "kc": neu.index[neu["class"] == "Kenyon_Cell"].tolist(),
        "mbon": neu.index[neu["class"] == "MBON"].tolist(),
    }


# --------------------------------------------------------------------------- 연결
def load_edges(body_ids: np.ndarray, min_weight: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """양 끝이 모두 body_ids(정렬됨)에 있는 연결만 (pre_idx, post_idx, weight) 로 반환."""
    tb = feather.read_table(paths.WEIGHTS_FILE, memory_map=True)
    w = tb["weight"].to_numpy()
    keep = w >= min_weight
    pre = tb["body_pre"].to_numpy()[keep]
    post = tb["body_post"].to_numpy()[keep]
    w = w[keep]
    del tb, keep

    def to_idx(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.searchsorted(body_ids, b)
        pos_c = np.minimum(pos, len(body_ids) - 1)
        return pos_c, body_ids[pos_c] == b

    pi, okp = to_idx(pre)
    qi, okq = to_idx(post)
    ok = okp & okq & (pi != qi)
    return pi[ok].astype(np.int32), qi[ok].astype(np.int32), w[ok].astype(np.float32)


def hop_distance(adj: sp.csr_matrix, sources: np.ndarray, max_hops: int) -> np.ndarray:
    """adj[i, j] != 0 이면 i → j. sources 로부터의 최단 홉 수 (도달 못 하면 -1)."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int16)
    dist[sources] = 0
    frontier = np.zeros(n, dtype=np.float32)
    frontier[sources] = 1
    adj_t = adj.T.tocsr()
    for h in range(1, max_hops + 1):
        nxt = (adj_t @ frontier) > 0
        nxt &= dist < 0
        if not nxt.any():
            break
        dist[nxt] = h
        frontier = nxt.astype(np.float32)
    return dist


def select_subnetwork(
    n: int, pi: np.ndarray, qi: np.ndarray, groups: dict[str, Any], prof: dict[str, Any]
) -> np.ndarray:
    keep = np.zeros(n, dtype=bool)
    inputs = np.array(sorted({i for lane in groups["input"] for b in lane for i in b}))
    outputs = np.array(sorted({i for lane in groups["output"] for i in lane}))
    keep[inputs] = True
    keep[outputs] = True
    H = prof["max_path"]
    if H <= 0:
        keep[:] = True
        return keep

    adj = sp.csr_matrix((np.ones(len(pi), np.float32), (pi, qi)), shape=(n, n))
    df = hop_distance(adj, inputs, H)
    db = hop_distance(adj.T.tocsr(), outputs, H)
    keep |= (df >= 0) & (db >= 0) & (df + db <= H)

    if prof["include_mushroom_body"]:
        mb_set = np.array(
            groups["kc"] + groups["mbon"] + groups["dan_reward"] + groups["dan_punish"]
        )
        keep[mb_set] = True
        m = prof["mb_path"]
        if m > 0:
            kc = np.array(groups["kc"])
            mbon = np.array(groups["mbon"])
            d_to_kc = hop_distance(adj.T.tocsr(), kc, m)
            keep |= (df >= 0) & (d_to_kc >= 0) & (df + d_to_kc <= m)
            d_from_mbon = hop_distance(adj, mbon, m)
            keep |= (d_from_mbon >= 0) & (db >= 0) & (d_from_mbon + db <= m)
    return keep


def top_k_inputs(qi: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """post 뉴런마다 가중치가 큰 입력 k 개만 남기는 마스크."""
    if k <= 0:
        return np.ones(len(qi), dtype=bool)
    order = np.lexsort((-w, qi))
    q_sorted = qi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    mask = np.zeros(len(qi), dtype=bool)
    mask[order[rank < k]] = True
    return mask


# --------------------------------------------------------------------------- 메인
def preprocess(profile: str, out_root: Path | None = None) -> Path:
    paths.require_raw_data()
    cfg = load_brain_config()
    prof = profile_config(profile)
    t0 = time.time()

    neu = load_neurons(cfg)
    log(f"뉴런 {len(neu):,}개 ({', '.join(cfg['neurons']['status'])})")
    groups = build_groups(neu, cfg)

    min_w = max(int(prof["min_weight"]), 1)
    pi, qi, w = load_edges(neu["bodyId"].to_numpy(), min_w)
    log(f"연결 {len(w):,}개 (weight ≥ {min_w}), {time.time() - t0:.1f}s")

    keep = select_subnetwork(len(neu), pi, qi, groups, prof)
    log(f"서브네트워크 뉴런 {keep.sum():,}개")

    # 새 인덱스로 다시 매김
    new_idx = np.full(len(neu), -1, dtype=np.int64)
    new_idx[keep] = np.arange(keep.sum())
    e = keep[pi] & keep[qi]
    pi, qi, w = new_idx[pi[e]], new_idx[qi[e]], w[e]
    sub = neu[keep].reset_index(drop=True)

    def remap(ids: list[int]) -> list[int]:
        return [int(new_idx[i]) for i in ids if new_idx[i] >= 0]

    groups = {
        "input": [[remap(b) for b in lane] for lane in groups["input"]],
        "output": [remap(lane) for lane in groups["output"]],
        **{k: remap(groups[k]) for k in ("dan_reward", "dan_punish", "kc", "mbon")},
    }

    # DAN → KC/MBON 연결은 부호(0)와 상관없이 따로 저장 (도파민 구획 정보)
    sign = sub["sign"].to_numpy()
    dan_all = np.array(groups["dan_reward"] + groups["dan_punish"], dtype=np.int64)
    mb_targets = np.zeros(len(sub), dtype=bool)
    mb_targets[groups["kc"] + groups["mbon"]] = True
    is_dan = np.zeros(len(sub), dtype=bool)
    is_dan[dan_all] = True
    de = is_dan[pi] & mb_targets[qi]
    dan_pre, dan_post, dan_w = pi[de], qi[de], w[de]

    # 빠른 전달 행렬: 부호 0 제외, post 마다 상위 k 입력
    fast = sign[pi] != 0
    pi, qi, w = pi[fast], qi[fast], w[fast]
    tk = top_k_inputs(qi, w, prof["max_inputs_per_neuron"])
    pi, qi, w = pi[tk], qi[tk], w[tk]
    n = len(sub)
    # 부호 × 시냅스 수 (정규화는 시뮬레이터에서 normalize_alpha 로 한다)
    data = (sign[pi] * w).astype(np.float32)
    W = sp.csr_matrix((data, (qi, pi)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    W.sort_indices()
    log(f"최종 행렬 {n:,} × {n:,}, nnz {W.nnz:,}")

    out = (out_root or paths.PROCESSED_DIR) / profile
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "W.npz", indptr=W.indptr, indices=W.indices, data=W.data, shape=np.array(W.shape)
    )
    np.savez_compressed(out / "dan.npz", pre=dan_pre, post=dan_post, weight=dan_w)
    cols = ["bodyId", "type", "class", "superclass", "somaSide", "exitNerve", "nt", "sign"]
    sub[cols].reset_index(drop=True).to_feather(out / "neurons.feather")
    (out / "groups.json").write_text(json.dumps(groups), "utf-8")
    meta = {
        "format": FORMAT_VERSION,
        "profile": profile,
        "profile_config": prof,
        "neurons": n,
        "synapses": int(W.nnz),
        "min_weight": int(min_w),
        "groups": {
            "input": [[len(b) for b in lane] for lane in groups["input"]],
            "output": [len(x) for x in groups["output"]],
            **{k: len(groups[k]) for k in ("dan_reward", "dan_punish", "kc", "mbon")},
        },
        "dan_edges": int(len(dan_w)),
        "source": {p.name: p.stat().st_size for p in (paths.ANNOTATIONS_FILE, paths.WEIGHTS_FILE)},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    log(f"저장: {out} ({meta['seconds']}s)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m flybeat.brain.preprocess")
    p.add_argument("--profile", default="compact", choices=list(load_brain_config()["profiles"]))
    args = p.parse_args()
    preprocess(args.profile)


if __name__ == "__main__":
    main()
