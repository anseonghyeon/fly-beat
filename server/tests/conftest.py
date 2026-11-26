"""테스트용 작은 합성 커넥톰 (실제 데이터 없이 모드 B·C 를 검증)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flymax import paths
from flymax.brain.model import Connectome


def synthetic_connectome(seed: int = 0) -> Connectome:
    """입력(4레인×4칸) → KC → MBON → DN → 운동, 입력 → DN 지름길, DAN → MBON."""
    rng = np.random.default_rng(seed)
    rows = []  # (type, class, superclass, side, exitNerve, nt)
    groups = {"input": [[[] for _ in range(4)] for _ in range(4)], "output": [[] for _ in range(4)]}
    for lane in range(4):
        for b in range(4):
            groups["input"][lane][b].append(len(rows))
            rows.append(("L1", None, "ol_intrinsic", "L", None, "acetylcholine"))
    kc = list(range(len(rows), len(rows) + 16))
    rows += [("KCg-d", "Kenyon_Cell", "cb_intrinsic", "L", None, "acetylcholine")] * 16
    mbon = list(range(len(rows), len(rows) + 8))
    rows += [("MBON01", "MBON", "cb_intrinsic", "L", None, "acetylcholine")] * 8
    dn = list(range(len(rows), len(rows) + 4))
    rows += [("DNp01", None, "descending_neuron", "L", None, "acetylcholine")] * 4
    for lane in range(4):
        groups["output"][lane].append(len(rows))
        rows.append(("Ti flexor MN", None, "vnc_motor", "L", "ProLN", "acetylcholine"))
    pam = list(range(len(rows), len(rows) + 2))
    rows += [("PAM01", "DAN", "cb_intrinsic", "L", None, "dopamine")] * 2
    ppl = list(range(len(rows), len(rows) + 2))
    rows += [("PPL101", "DAN", "cb_intrinsic", "L", None, "dopamine")] * 2
    n = len(rows)

    edges = []
    inputs = [i for lane in groups["input"] for b in lane for i in b]
    for k in kc:
        for i in rng.choice(inputs, 4, replace=False):
            edges.append((int(i), k, 5))
    plastic = []
    for m in mbon:
        for k in rng.choice(kc, 6, replace=False):
            edges.append((int(k), m, 3))
            plastic.append((int(k), m))
    for lane in range(4):
        edges.append((mbon[2 * lane], dn[lane], 5))
        edges.append((groups["input"][lane][3][0], dn[lane], 5))
        edges.append((dn[lane], groups["output"][lane][0], 8))
    pre, post, w = (np.array(x) for x in zip(*edges, strict=True))
    W = sp.csr_matrix((w.astype(np.float32), (post, pre)), shape=(n, n))
    W.sum_duplicates()
    W.sort_indices()
    dan_edges = [(d, m, 4.0) for d in pam for m in mbon[:4]] + [
        (d, m, 4.0) for d in ppl for m in mbon[4:]
    ]
    dp, dq, dw = (np.array(x) for x in zip(*dan_edges, strict=True))
    neurons = pd.DataFrame(
        rows, columns=["type", "class", "superclass", "somaSide", "exitNerve", "nt"]
    )
    neurons.insert(0, "bodyId", np.arange(n) + 1000)
    neurons["sign"] = np.where(neurons["nt"] == "dopamine", 0, 1).astype(np.int8)
    groups |= {"dan_reward": pam, "dan_punish": ppl, "kc": kc, "mbon": mbon}
    order = np.lexsort(([p for p, _ in plastic], [q for _, q in plastic]))
    pp = np.array([plastic[i][0] for i in order])
    pq = np.array([plastic[i][1] for i in order])
    return Connectome(W, neurons, groups, dp, dq, dw, {"synapses": int(W.nnz)}, pp, pq)


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch):
    """data/processed/test 에 합성 커넥톰을 두고, 체크포인트도 임시 폴더로."""
    processed = tmp_path / "processed"
    synthetic_connectome().save("test", processed)
    monkeypatch.setattr(paths, "PROCESSED_DIR", processed)
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path / "ckpt")
    return "test"
