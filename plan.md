# fly-beat 프로젝트 계획

> 실제 초파리 커넥톰(MaleCNS v1.0)으로 만든 뇌 모델에게 자체 제작 건반형 리듬게임을 강화학습으로 가르치는 프로젝트.
> 배포하지 않으며, GitHub에 공개해 누구나 clone 후 **로컬에서** 서버와 웹 UI를 실행할 수 있게 한다.

---

## 1. 목표

1. 웹 기반 건반형(세로 낙하 노트) 리듬게임을 직접 만든다. 특정 상용 게임의 이름·디자인·곡·에셋은 사용하지 않는다.
2. 같은 게임을 세 가지 방식의 에이전트가 학습·플레이하도록 구현하고 성능을 비교한다.
   - **모드 A: Deep RL 베이스라인.** 일반적인 딥러닝 레이어(MLP 등)를 쓰는 표준 강화학습.
   - **모드 B: 감각-행동 매핑 RL.** 커넥톰은 고정하고, 게임 화면을 감각 뉴런에 넣는 입력 매핑과 운동 뉴런을 키 입력으로 바꾸는 출력 매핑만 학습한다.
   - **모드 C: 도파민 보상 가소성 RL.** 보상을 도파민 뉴런 자극으로 주고, 3요소 학습 규칙(three-factor learning rule)으로 커넥톰 시냅스 가중치의 **일부만** 직접 업데이트한다.
3. 웹 UI에서 사람 플레이, 에이전트 플레이 관전, 학습 진행 상황, 뉴런 활동 시각화를 볼 수 있게 한다.

## 2. 데이터

### 2.1 사용 데이터 (`data/`)

| 파일 | 크기 | 용도 |
|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | ~1.05GB | 뉴런 간 연결 그래프. 시냅스 수를 가중치로 쓴다 |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | ~14MB | 뉴런 class/type/side. 입력·출력·도파민 뉴런을 선택할 때 쓴다 |
| `body-neurotransmitters-male-cns-v1.0.feather` | ~43MB | 신경전달물질 예측값. 시냅스 부호(흥분/억제)를 결정한다 |

- 출처: https://male-cns.janelia.org/download/ (Google Research · HHMI Janelia, *Cell* 2026)
- 라이선스: **CC-BY 4.0**. README에 출처와 인용 정보를 반드시 표기한다.
- 원본 데이터는 git에 올리지 않는다(`.gitignore`에 `data/` 추가). 대신 `scripts/download_data.sh`를 제공한다.
  - 다운로드 후 **SHA-256 체크섬을 검증**한다. 체크섬 목록은 저장소에 커밋한다.

### 2.2 전처리 (`data/processed/`, git 제외)

1. annotations의 `status`/`statusLabel`로 **정식 뉴런만 남긴다.** 시냅스가 적은 미분류 조각은 버린다. 뉴런별 입출력 시냅스 수가 필요하면 connectome-weights에서 직접 합산한다.
2. neurotransmitter 예측으로 시냅스 부호를 정한다.
   - ACh는 흥분성(+), GABA는 억제성(−)으로 둔다.
   - Glu는 초파리에서 주로 억제성이라 기본값을 −로 하고, 설정으로 바꿀 수 있게 한다.
   - DA, 5-HT, OA 같은 조절성 전달물질은 별도로 처리한다.
3. 부호를 붙인 가중치 행렬 `W[post, pre] = sign(pre) × synapse_count × scale`을 **희소 행렬**(CSR)로 만들어 `.npz`로 캐시한다.
4. 뉴런 ID와 행렬 인덱스를 잇는 매핑, 그리고 입력·출력·도파민 뉴런 그룹 정의를 JSON으로 저장한다.
5. **서브네트워크 추출 옵션.** 전체 뉴런 약 16.7만 개로 실시간 시뮬레이션하기는 부담이 클 수 있다. 그래서 입력 뉴런에서 출력 뉴런까지 N홉 이내의 뉴런과 버섯체(mushroom body) 회로만 남기는 축소 모드를 제공한다.

> **Phase 0에서 먼저 확인할 것:** 각 feather의 실제 컬럼 이름, 전체 연결 수, 그리고 아래 4장의 뉴런 타입(시각 뉴런, 다리 운동 뉴런, PAM·PPL1 도파민 뉴런, Kenyon cell, MBON)이 annotation에 어떤 이름으로 들어 있는지.

## 3. 리듬게임 설계

### 3.1 규칙

- **4레인** 세로 낙하 노트 게임(기본값, 확정). 코드는 레인 수를 하드코딩하지 않고 설정값으로 둔다.
- 노트 종류는 탭(tap)과 롱노트(hold, 시작과 끝을 판정)로 한다.
- 판정은 타격 시각과 노트 시각의 차이로 정한다.

  | 판정 | 범위 | 점수 |
  |---|---|---|
  | PERFECT | ±40ms | 100% |
  | GREAT | ±80ms | 70% |
  | GOOD | ±120ms | 30% |
  | MISS | 그 밖, 또는 미입력 | 0 |

  범위 값은 설정으로 바꿀 수 있다.
- 콤보, 정확도(%), 총점을 기록한다.

### 3.2 채보 포맷 (`charts/*.json`)

```json
{
  "id": "sample-001",
  "title": "Sample Beat",
  "bpm": 120,
  "offsetMs": 0,
  "lanes": 4,
  "audio": "sample-001.ogg",
  "notes": [
    { "t": 1000, "lane": 0, "type": "tap" },
    { "t": 1500, "lane": 2, "type": "hold", "dur": 500 }
  ]
}
```

- JSON Schema로 검증한다. 서버는 검증을 통과한 채보만 불러온다.
- 기본 제공 곡은 **저작권 문제가 없도록** 직접 만든 음원이나 절차적으로 생성한 비트(메트로놈, 드럼 패턴)만 쓴다.
- 학습용 **랜덤 채보 생성기**를 제공한다. 난이도(노트 밀도, BPM, 동시 타격 비율)를 조절할 수 있게 한다.

### 3.3 게임 엔진 구조

- **Python 게임 엔진(`server/game/`)을 기준 구현으로 삼는다.** 고정 tick(기본 60Hz)으로 결정론적으로 동작하며, 학습할 때는 렌더링 없이 실시간보다 빠르게 돌린다.
  - Gymnasium `Env` 인터페이스를 따른다.
  - **관측(observation):** 레인마다 판정선 앞쪽 일정 구간을 K개 구간으로 나눈 노트 존재·거리 격자(`lanes × K`), 그리고 현재 누르고 있는 키 상태.
  - **행동(action):** 레인별 누름/뗌(`MultiBinary(lanes)`).
  - **보상(reward):** 판정 점수. 헛누름에는 작은 패널티를 주고, MISS에는 음의 보상을 준다.
- **TypeScript 엔진(`web/src/game/`)은 사람이 브라우저에서 플레이할 때 쓴다.** 입력 지연을 없애려고 클라이언트에서 돌린다.
- 두 엔진의 판정 결과가 같은지 **공용 골든 테스트 픽스처**(`tests/fixtures/judge/*.json`)로 양쪽 테스트에서 검증한다.

## 4. 초파리 뇌 모델

### 4.1 뉴런 시뮬레이터 (`server/brain/`)

- 기본은 **rate 모델**이다: `τ dr/dt = −r + f(W·r + I_ext + b)`, `f = ReLU` 또는 `tanh`.
  - 선택 옵션으로 LIF 스파이킹 모델을 둔다(나중 단계).
- PyTorch 희소 행렬 연산을 쓰고, CPU / CUDA / Apple MPS를 지원한다.
- 게임 1 tick(약 16.7ms)마다 뉴런을 여러 번 적분한다(dt=1~2ms, 설정 가능).
- **성능 목표:** 학습할 때 실시간보다 빠르게 돌릴 수 있어야 한다. 전체 뇌로 목표에 못 미치면 2.2절의 서브네트워크 모드를 기본값으로 쓴다. Phase 2에서 벤치마크한 뒤 결정한다.

### 4.2 입력·출력 매핑 (Driving Fly 방식 참고)

- **입력(감각):** 시각계 뉴런을 쓴다. 시야를 레인 수만큼 수평으로 나누어, 각 레인을 해당 위치의 시각 컬럼 뉴런(광수용체 또는 초기 시각 뉴런)에 대응시킨다. 노트가 판정선에 가까울수록 입력 전류를 키운다.
  - 선택 옵션: 리듬 박자를 청각 뉴런(Johnston's organ 계열)에 추가 입력.
- **출력(행동):** 4레인을 다리 4개에 대응시킨다. 앞다리 좌/우는 레인 0·1, 가운뎃다리 좌/우는 레인 2·3이다. 해당 다리 운동 뉴런 또는 하행 뉴런(descending neuron) 집단의 활동이 임계값을 넘으면 키를 누른 것으로 본다.
- 구체적인 뉴런 타입은 Phase 0의 annotation 조사 결과로 확정하고, `brain/config/mapping.yaml`로 관리한다.

## 5. 세 가지 학습 모드

### 모드 A: Deep RL 베이스라인 (`server/agents/deep_rl/`)

- 커넥톰을 쓰지 않는 기준 모델이다.
- 정책망은 관측 격자를 MLP(2~3층)에 넣는 구조다. 선택 옵션으로 1D Conv나 GRU를 둔다.
- 알고리즘은 **Stable-Baselines3의 PPO**를 쓴다(확정). 행동 공간은 `MultiBinary(4)`다.
- SB3 기본 저장 형식(`model.save()`의 `.zip`)에는 pickle 데이터가 섞여 있어 쓰지 않는다. 정책망 `state_dict`만 꺼내 `safetensors`로 저장하고, 하이퍼파라미터는 JSON으로 따로 저장한다. 불러올 때는 같은 설정으로 PPO 객체를 만든 뒤 가중치만 넣는다.
- 의미: "일반 신경망이면 이 정도 성능이 나온다"는 상한선을 정한다.

### 모드 B: 감각-행동 매핑 RL (`server/agents/mapping/`)

- 커넥톰 가중치는 **고정**하고 저수지(reservoir)처럼 쓴다.
- 학습하는 것은 두 가지뿐이다.
  1. **입력 매핑:** 관측 격자를 입력 뉴런 전류로 바꾸는 작은 선형층.
  2. **출력 리드아웃:** 출력 뉴런 활동을 레인별 누름 확률로 바꾸는 선형층.
- 알고리즘은 PPO 또는 REINFORCE다. 커넥톰 시뮬레이션을 통한 역전파는 계산량이 크므로, 기본은 리드아웃 쪽만 그래디언트로 학습하고 입력 매핑은 진화 전략(ES) 같은 그래디언트 없는 방법도 비교한다.
- 의미: 커넥톰 자체는 그대로 두고, 감각과 행동을 잇는 매핑만으로 어디까지 가능한지 본다.

### 모드 C: 도파민 보상 가소성 RL (`server/agents/dopamine/`)

- 보상 신호를 **도파민 뉴런 자극으로** 바꾼다.
  - 좋은 판정(PERFECT/GREAT)이면 보상 계열 도파민 뉴런(PAM 군)을 자극한다.
  - MISS나 헛누름이면 처벌 계열 도파민 뉴런(PPL1 군)을 자극한다.
- 학습 규칙은 **3요소 학습 규칙**이다: `Δw_ij = η · e_ij · DA(t)`.
  - `e_ij`: 시냅스 전·후 뉴런 활동으로 만드는 적격 흔적(eligibility trace). 지수 감쇠한다.
  - `DA(t)`: 해당 시냅스 구획에 투사하는 도파민 뉴런의 활동.
- **업데이트 대상은 일부 시냅스로 제한한다.** 기본은 실제 초파리에서 도파민 가소성이 일어나는 **Kenyon cell → MBON 시냅스**(버섯체)다. 설정으로 대상 구획을 넓힐 수 있다.
- 안정화 장치를 둔다.
  - 가중치 범위 클리핑
  - 부호 보존(흥분성 시냅스가 억제성으로 바뀌지 않게)
  - 학습률 스케줄
  - 원본 가중치와의 차이를 체크포인트에 저장
- 의미: 역전파 없이 생물학적 학습 규칙만으로 리듬게임을 배울 수 있는지 본다.

### 비교 지표

- 학습 스텝별 정확도(%), 판정 분포, 평균 타이밍 오차(ms), 최대 콤보
- 처음 보는 채보(검증 세트)에 대한 일반화 성능
- 학습 시간과 시뮬레이션 처리량(game-seconds / wall-second)

## 6. 서버 (`server/`)

- **기술:** Python 3.11+, FastAPI, Uvicorn, PyTorch, Gymnasium, Stable-Baselines3, safetensors, pyarrow, scipy. 패키지 관리는 표준 `venv` + `pip`(`pyproject.toml`)로 하고, `server/requirements.lock`으로 버전을 고정한다.
- **REST API**

  | 메서드 | 경로 | 설명 |
  |---|---|---|
  | GET | `/api/charts` | 채보 목록 |
  | GET | `/api/charts/{id}` | 채보 상세 |
  | POST | `/api/train` | 학습 시작 (`mode: A/B/C`, 하이퍼파라미터) |
  | GET | `/api/train/{job_id}` | 학습 상태 |
  | POST | `/api/train/{job_id}/stop` | 학습 중지 |
  | GET | `/api/checkpoints` | 체크포인트 목록 |
  | POST | `/api/play` | 체크포인트로 에이전트 플레이 세션 시작 |

- **WebSocket**
  - `/ws/play/{session_id}`: 에이전트 플레이 중 게임 상태와 선택된 뉴런 활동을 스트리밍한다(30~60fps).
  - `/ws/train/{job_id}`: 학습 지표(보상, 정확도)를 실시간으로 보낸다.
- 학습은 **별도 프로세스**에서 돌려 API 응답을 막지 않는다. 동시 학습 작업 수는 설정으로 제한한다(기본 1).
- 서버 없이 학습만 돌릴 수 있는 CLI도 제공한다: `python -m flybeat.train --mode C --chart random`.

## 7. 웹 UI (`web/`)

- **기술:** Vite, React, TypeScript. 게임 화면은 Canvas 2D로 그린다. 차트는 경량 라이브러리를 쓴다.
- **화면 구성**
  1. **Play:** 사람이 키보드(D F J K 등, 바꿀 수 있음)로 직접 플레이한다.
  2. **Watch:** 학습된 초파리(또는 모드 A 에이전트)가 플레이하는 모습을 관전한다. 옆 패널에 입력·출력·도파민 뉴런 활동 히트맵을 보여 준다.
  3. **Train:** 모드와 하이퍼파라미터를 골라 학습을 시작하고, 학습 곡선과 세 모드 비교 그래프를 본다.
  4. **Brain:** 사용 중인 서브네트워크 정보(뉴런 수, 연결 수)와 모드 C에서 가중치가 바뀐 시냅스 분포를 보여 준다.
- 디자인은 독자적인 스타일로 만든다. 초파리를 테마로 하고 특정 상용 게임의 UI를 모방하지 않는다.

## 8. 보안 (로컬 실행이지만 공개 저장소이므로 기본값을 안전하게)

- 서버는 기본으로 **`127.0.0.1`에만 바인딩**한다. 외부 바인딩은 명시적 옵션으로만 허용하고, 켜면 경고를 출력한다.
- CORS는 로컬 개발 출처(`http://localhost:5173` 등)만 허용한다.
- 모든 요청과 WebSocket 메시지는 Pydantic으로 검증한다. 하이퍼파라미터는 범위를 제한하고, WebSocket 메시지 크기와 연결 수도 제한한다.
- **파일 경로를 클라이언트 입력으로 받지 않는다.** 채보와 체크포인트는 ID로만 지정하고, 허용 디렉터리 안에서만 찾는다(경로 탈출 방지).
- 체크포인트는 **`safetensors`** 형식으로 저장하고, 메타데이터는 JSON으로 둔다. pickle 기반 로딩은 쓰지 않는다. 불가피하게 `torch.load`를 쓸 때는 `weights_only=True`로 한다.
- 다운로드 스크립트는 HTTPS와 SHA-256 체크섬으로 무결성을 검증한다.
- 의존성 버전을 고정하고(`requirements.lock`, `package-lock.json`), Dependabot이나 `pip-audit`/`npm audit` CI를 둔다.
- 비밀값은 필요 없도록 설계한다. 필요해지면 `.env`로만 받고 `.env.example`만 커밋한다.

## 9. 저장소 구조 (안)

```
fly-beat/
├── plan.md
├── README.md                 # 소개, 데이터 출처/라이선스, 실행 방법
├── LICENSE                   # 코드 라이선스 (MIT 등, 결정 필요)
├── .gitignore                # data/, checkpoints/, node_modules/ 등
├── scripts/
│   ├── download_data.sh      # 데이터 다운로드 + 체크섬 검증
│   └── checksums.sha256
├── data/                     # (git 제외) 원본 feather + processed/
├── charts/                   # 기본 제공 채보 + 음원
├── server/
│   ├── pyproject.toml
│   └── flybeat/
│       ├── api/              # FastAPI 라우터, WebSocket
│       ├── game/             # 게임 엔진, Gymnasium Env, 채보 생성기
│       ├── brain/            # 전처리, 희소 행렬, 뉴런 시뮬레이터, 매핑 설정
│       ├── agents/
│       │   ├── deep_rl/      # 모드 A
│       │   ├── mapping/      # 모드 B
│       │   └── dopamine/     # 모드 C
│       ├── train.py          # 학습 CLI
│       └── config/
│   └── tests/
├── web/
│   ├── package.json
│   └── src/
│       ├── game/             # TS 게임 엔진 (사람 플레이)
│       ├── pages/            # Play / Watch / Train / Brain
│       └── components/
├── tests/fixtures/judge/     # 엔진 공용 판정 테스트 케이스
└── checkpoints/              # (git 제외)
```

## 10. 로컬 실행 흐름 (목표)

```bash
git clone https://github.com/<owner>/fly-beat.git
cd fly-beat
./scripts/download_data.sh          # 약 1.1GB 다운로드 + 검증
cd server && python3 -m venv .venv && .venv/bin/pip install -r requirements.lock -e .
.venv/bin/python -m flybeat.brain.preprocess
.venv/bin/python -m flybeat.api   # 127.0.0.1:8000
# 다른 터미널
cd web && npm ci && npm run dev     # http://localhost:5173
```

- 선택 옵션: `docker compose up`으로 한 번에 실행.
- 최소 사양(목표): RAM 16GB. GPU는 선택이다(CUDA 또는 Apple Silicon MPS).

## 11. 단계별 진행 계획

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| **Phase 0** 데이터 조사 | feather 스키마 확인, 뉴런 타입 조사, 입력·출력·DAN·KC·MBON 후보 목록 작성 | `docs/data-notes.md` 작성, 매핑 후보 확정 |
| **Phase 1** 게임 | Python 엔진 + Gymnasium Env, 채보 포맷과 랜덤 생성기, TS 엔진, 판정 골든 테스트 | 사람이 웹에서 플레이할 수 있고, 두 엔진의 판정이 일치 |
| **Phase 2** 뇌 시뮬레이터 | 전처리 파이프라인, 희소 rate 모델, 서브네트워크 추출, 성능 벤치마크 | 입력 자극에 출력 뉴런이 반응하는 것 확인, 처리량 측정 |
| **Phase 3** 모드 A | PPO 베이스라인 학습 | 쉬운 채보에서 정확도 90% 이상 |
| **Phase 4** 모드 B | 고정 커넥톰 + 입출력 매핑 학습 | 무작위 정책보다 유의미하게 높은 정확도 |
| **Phase 5** 모드 C | 도파민 3요소 학습 규칙, KC→MBON 가소성 | 학습할수록 정확도가 오르는 곡선 확인 |
| **Phase 6** 서버·UI 통합 | API, WebSocket, Watch/Train/Brain 화면 | UI에서 학습 시작부터 관전까지 가능 |
| **Phase 7** 공개 준비 | README, 데이터 출처 표기, 라이선스, CI(lint/test/audit), 데모 GIF | clean clone에서 문서대로 실행 성공 |

## 12. 열린 질문 / 결정 필요

- 코드 라이선스(MIT / Apache-2.0 등)
- 전체 뇌와 서브네트워크 중 무엇을 기본값으로 할지 (Phase 2 벤치마크 후 결정)
- 시각 입력 뉴런을 어느 단계에서 넣을지 (광수용체 / 라미나 / 메둘라)
- 모드 C에서 버섯체 밖 시냅스까지 가소성을 허용할지

## 13. 참고

- MaleCNS v1.0: https://male-cns.janelia.org/
- neuPrint: https://neuprint.janelia.org/
- FlyWire: https://flywire.ai/
- 관련 프로젝트: The Driving Fly (Mark Unthank), Fly64 (Jessica Parkett)

## 14. 진행 현황 (2026-11-23)

Phase 0~6을 마쳤다. 결과는 `docs/results.md`, 설계에서 바뀐 부분의 근거는 `docs/brain-notes.md`에 있다.

| 단계 | 상태 | 계획과 달라진 점 |
|---|---|---|
| Phase 0 | 완료 | `body-stats` 파일은 쓰지 않는다. 시각 입력은 광수용체 대신 hex 좌표가 있는 L1/L2를 쓴다 |
| Phase 1 | 완료 | 음원 대신 WebAudio로 박자를 합성한다 |
| Phase 2 | 완료 | 희소 행렬 곱은 torch 대신 scipy를 쓴다(약 9배 빠름). 기본 프로필은 compact(21,251 뉴런) |
| Phase 3 | 완료 | 관측에 "레인별 다음 노트까지 남은 시간" 채널을 추가했다 |
| Phase 4 | 완료 | 리드아웃 대상을 다리 운동 뉴런에서 운동 + 하행 뉴런으로 넓혔다. 감각 매핑 ES 탐색은 옵션(`es_iters`, 기본 0) |
| Phase 5 | 완료 | KC→KC 제거, KC k-WTA 추가. 도파민 신호는 DAN 자극량(보상 예측 오차)을 DAN→MBON 구획 연결로 전달한다. 리드아웃은 가소성 시냅스의 MBON 입력 전류 |
| Phase 6 | 완료 | Host·Origin 검사를 추가했다. 이후 Play와 Watch를 하나의 대결 화면(Play)으로 합쳤다 |
| Phase 7 | 남음 | 코드 라이선스 결정, CI, 데모 GIF |
