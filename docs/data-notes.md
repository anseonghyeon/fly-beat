# MaleCNS v1.0 데이터 조사 노트 (Phase 0)

조사 스크립트: `python -m flymax.brain.explore`

## 1. 파일 구조

### `body-annotations-male-cns-v1.0-minconf-0.5.feather`: 211,577행, 36컬럼

주요 컬럼:

| 컬럼 | 설명 |
|---|---|
| `bodyId` | 뉴런(segment) ID. connectome-weights의 `body_pre`, `body_post`와 같은 값이다 |
| `status` | `Traced` 165,122 / `Orphan` 15,925 / `Glia` 11,864 / `Unimportant` 10,751 / `Assign` 1,832 / `Anchor` 611 / NaN 5,472 |
| `statusLabel` | `status`보다 세분화된 값 (`Reviewed`, `Roughly traced` 등) |
| `superclass` | `ol_intrinsic`, `cb_intrinsic`, `visual_projection`, `descending_neuron`, `vnc_motor`, `ol_sensory` 등 |
| `class` | `Kenyon_Cell`, `DAN`, `MBON`, `visual`, `olfactory` 등 (대부분 NaN) |
| `type`, `instance` | 세포 유형 이름 (`L1`, `PAM01`, `MBON10`, `KCg-m`, `Ti flexor MN` 등) |
| `somaSide` | `L` / `R` / `M` |
| `assignedOlHex1`, `assignedOlHex2` | 시각엽 컬럼의 육각 격자 좌표 (23,720개 뉴런만 있음, 1~36 / 1~39) |
| `entryNerve`, `exitNerve` | 감각 뉴런이 들어오는 신경, 운동 뉴런이 나가는 신경 |

### `body-neurotransmitters-male-cns-v1.0.feather`: 1,835,518행

- `body`, `consensus_nt`, `predicted_nt`, `predicted_nt_confidence`, `celltype_predicted_nt` 등.
- Traced 뉴런의 99.7%가 들어 있다.
- Traced 뉴런 기준 `consensus_nt` 분포:

  | 전달물질 | 뉴런 수 |
  |---|---|
  | acetylcholine | 103,718 |
  | glutamate | 29,296 |
  | gaba | 22,055 |
  | histamine | 5,910 |
  | unclear | 3,100 |
  | (없음) | 502 |
  | dopamine | 392 |
  | octopamine | 101 |
  | serotonin | 48 |

### `connectome-weights-male-cns-v1.0-minconf-0.5.feather`: 151,856,684행

- 컬럼은 `body_pre`, `body_post`, `weight`(시냅스 수) 3개뿐이다.
- 전체 시냅스 합은 311,833,243개다(postsynaptic 기준).
- **Traced 뉴런끼리의 연결만 남기면 25,563,197개 연결, 시냅스 124,025,046개**다.

  | 최소 시냅스 수 | 남는 연결 수 |
  |---|---|
  | ≥ 1 | 25,563,197 |
  | ≥ 3 | 10,511,038 |
  | ≥ 5 | 6,235,682 |
  | ≥ 10 | 2,749,407 |

- 메모리: 전체를 numpy로 읽으면 약 4.5GB(최대 RSS)를 쓴다. 16GB 머신에서 전처리할 수 있다.

> `body-stats` 파일은 쓰지 않는다. 담긴 정보(status/class/type, 시냅스 수)는 annotations와 connectome-weights로 대신할 수 있다.

## 2. 뉴런 선택 (Traced 기준)

| 역할 | 선택 기준 | 개수 |
|---|---|---|
| **시각 입력** | `type ∈ {L1, L2}`이고 `assignedOlHex1/2`가 있음 | 3,534 (L 1,749 / R 1,785) |
| **출력: 앞다리** | `superclass == vnc_motor`, `exitNerve == ProLN` | 81 (L 41 / R 40) |
| **출력: 가운뎃다리** | `superclass == vnc_motor`, `exitNerve == MesoLN` | 116 (L 58 / R 58) |
| 보상 도파민 | `class == DAN`, `type`이 `PAM`으로 시작 | 316 |
| 처벌 도파민 | `class == DAN`, `type`이 `PPL1`로 시작 | 16 |
| Kenyon cell | `class == Kenyon_Cell` | 4,064 |
| MBON | `class == MBON` | 97 |
| 하행 뉴런 | `superclass == descending_neuron` | 1,314 (480 types) |

### 왜 광수용체(R1–R6)가 아니라 L1/L2인가

- R1–R6(3,377개, histamine)에는 시각엽 hex 좌표가 없어서 망막 위치(= 레인)를 정할 수 없다.
- 광수용체 바로 다음 단계인 라미나 뉴런 L1(glutamate)과 L2(acetylcholine)는 컬럼마다 하나씩 있고 hex 좌표가 있다.
- 그래서 **"노트가 보인다" = 해당 레인 방향 컬럼의 L1/L2에 입력 전류**로 단순화한다.

### 레인 매핑 (4레인)

| 레인 | 시각 입력 (L1/L2) | 출력 (다리 운동 뉴런) |
|---|---|---|
| 0 | 왼쪽 눈, 바깥쪽 절반 | 왼쪽 앞다리 (ProLN, L) |
| 1 | 왼쪽 눈, 안쪽 절반 | 왼쪽 가운뎃다리 (MesoLN, L) |
| 2 | 오른쪽 눈, 안쪽 절반 | 오른쪽 가운뎃다리 (MesoLN, R) |
| 3 | 오른쪽 눈, 바깥쪽 절반 | 오른쪽 앞다리 (ProLN, R) |

- 눈 안쪽/바깥쪽은 hex 좌표로 나눈 **근사**다. 축 정의(`hex1 − hex2` 등)는 `brain/config/mapping.yaml`에서 바꿀 수 있다.
- 출력은 운동 뉴런 집단의 평균 활동이 임계값을 넘으면 키를 누른 것으로 본다(모드 B·C는 학습된 리드아웃 사용).

## 3. 신경전달물질 → 시냅스 부호

| consensus_nt | 부호 | 비고 |
|---|---|---|
| acetylcholine | +1 | 흥분성 |
| gaba | −1 | 억제성 |
| glutamate | −1 | 초파리 중추에서는 주로 GluCl을 통한 억제성. 설정으로 바꿀 수 있음 |
| histamine | −1 | 광수용체 → 라미나 (억제성) |
| dopamine, octopamine, serotonin | 0 | 빠른 전달에서 뺀다. 모드 C에서 도파민은 가소성 신호로만 쓴다 |
| unclear / 없음 | +1 | 기본값. 설정으로 0으로 바꿀 수 있음 |

## 4. 연결성 확인 (weight ≥ 5)

- 시각 입력(L1/L2)에서 다리 운동 뉴런까지 **최단 경로는 대부분 4홉**이다(3홉 17개, 4홉 155개, 5홉 21개, 도달 불가 4개).
- Kenyon cell은 시각 입력에서 2~4홉 안에 도달한다.
- 입력→출력 경로 위 뉴런만 남기는 서브네트워크 크기:

  | 최대 경로 길이 H | 남는 뉴런 수 |
  |---|---|
  | 4 | 14,728 |
  | 5 | 71,521 |
  | 6 | 123,821 |
  | 7 | 148,092 |

- 전체 Traced 네트워크(16.5만 뉴런, weight≥5 연결 624만 개)는 메모리에는 충분히 올라가지만, 학습할 때 실시간보다 빠르게 돌리기는 어렵다. 그래서 **기본값은 서브네트워크**로 하고, Phase 2에서 벤치마크해 확정한다.

## 5. 체크섬

`scripts/checksums.sha256`에 있다. 2026-11-23에 공식 URL에서 받은 파일 기준이다.
