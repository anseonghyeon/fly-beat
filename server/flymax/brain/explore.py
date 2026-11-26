"""MaleCNS 데이터 요약 출력.

사용법: python -m flymax.brain.explore
"""

from __future__ import annotations

import pandas as pd

from flymax import paths


def main() -> None:
    paths.require_raw_data()
    ann = pd.read_feather(paths.ANNOTATIONS_FILE)
    nt = pd.read_feather(paths.NEUROTRANSMITTERS_FILE, columns=["body", "consensus_nt"])
    traced = ann[ann["status"] == "Traced"]

    print(f"annotation 행 수: {len(ann):,}  (Traced: {len(traced):,})")
    print("\n[status]")
    print(ann["status"].value_counts(dropna=False).to_string())
    print("\n[superclass, Traced]")
    print(traced["superclass"].value_counts().head(20).to_string())

    merged = traced.merge(nt, left_on="bodyId", right_on="body", how="left")
    print("\n[consensus_nt, Traced]")
    print(merged["consensus_nt"].value_counts(dropna=False).to_string())

    groups = {
        "시각 입력 L1/L2 (hex 좌표)": traced["type"].isin(["L1", "L2"])
        & traced["assignedOlHex1"].notna(),
        "앞다리 운동 뉴런 (ProLN)": (traced["superclass"] == "vnc_motor")
        & (traced["exitNerve"] == "ProLN"),
        "가운뎃다리 운동 뉴런 (MesoLN)": (traced["superclass"] == "vnc_motor")
        & (traced["exitNerve"] == "MesoLN"),
        "DAN PAM": (traced["class"] == "DAN") & traced["type"].str.startswith("PAM", na=False),
        "DAN PPL1": (traced["class"] == "DAN") & traced["type"].str.startswith("PPL1", na=False),
        "Kenyon cell": traced["class"] == "Kenyon_Cell",
        "MBON": traced["class"] == "MBON",
        "하행 뉴런 (DN)": traced["superclass"] == "descending_neuron",
    }
    print("\n[주요 뉴런 그룹, Traced]")
    for name, mask in groups.items():
        sub = traced[mask]
        sides = sub["somaSide"].value_counts().to_dict()
        print(f"  {name}: {len(sub):,}  side={sides}")


if __name__ == "__main__":
    main()
