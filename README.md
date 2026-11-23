# fly-beat

실제 초파리 커넥톰(**MaleCNS v1.0**)으로 만든 뇌 모델에게, 직접 만든 4레인 건반형 리듬게임을 강화학습으로 가르치는 프로젝트입니다.

같은 게임을 세 가지 방식으로 학습시키고 비교합니다.

| 모드 | 방식 | 학습되는 것 |
|---|---|---|
| **A · 일반 딥러닝** | 게임 화면 → MLP 정책망, Stable-Baselines3 PPO | 신경망 전체 (비교 기준선) |
| **B · 감각-행동 매핑** | 게임 화면 → 초파리 시각 뉴런(L1/L2) → **고정된 커넥톰** → 다리 운동·하행 뉴런 → 선형 리드아웃, PPO | 리드아웃만 |
| **C · 도파민 가소성** | 보상 → 도파민 뉴런(PAM/PPL1) 자극 → **3요소 학습 규칙**으로 커넥톰의 KC→MBON 시냅스를 직접 수정 (역전파 없음) | KC→MBON 시냅스 약 5만 개 |

서버와 웹 UI는 **로컬에서 실행**하도록 만들었습니다(배포하지 않음).

- **Play**: 사람이 직접 플레이 (D F J K)
- **Watch**: 학습된 초파리의 플레이를 관전하면서 뇌 활동(시각 입력, 다리 운동 뉴런, 버섯체·도파민)을 함께 봄
- **Train**: 모드를 골라 학습을 시작하고 학습 곡선을 봄
- **Brain**: 사용 중인 커넥톰 정보와, 모드 C가 바꾼 시냅스 분석

## 결과 요약

학습에 쓰지 않은 랜덤 채보 4곡 기준 정확도입니다. 자세한 내용은 [docs/results.md](docs/results.md)에 있습니다.

| 모드 | 검증 정확도 | 학습량 / 시간 (Apple M4, CPU) |
|---|---|---|
| A | 94.6% | 30만 스텝 / 약 14초 |
| B | 76.0% (학습 전 49.5%) | 100만 스텝 / 약 16분 |
| C | 69.6% (학습 전 49.8%) | 96곡 / 약 9분 |

## 빠른 시작

필요한 것: Python 3.11 이상, Node.js 20 이상, 디스크 약 2GB, RAM 16GB 권장.

```bash
git clone https://github.com/anseonghyeon/fly-beat.git
cd fly-beat

# 1) 커넥톰 데이터 다운로드 (약 1.1GB, SHA-256 검증)
./scripts/download_data.sh

# 2) 서버 설치 + 커넥톰 전처리 (약 10초, 메모리 약 4.5GB 사용)
cd server
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock -e ".[dev]"
.venv/bin/python -m flybeat.brain.preprocess --profile compact

# 3) 서버 실행 (127.0.0.1:8000)
.venv/bin/python -m flybeat.api
```

다른 터미널에서 웹 UI를 띄웁니다.

```bash
cd web
npm ci
npm run dev          # http://127.0.0.1:5173
```

서버 없이 학습만 할 수도 있습니다.

```bash
cd server
.venv/bin/python -m flybeat.train --mode A --steps 300000
.venv/bin/python -m flybeat.train --mode B --steps 1000000
.venv/bin/python -m flybeat.train --mode C --episodes 96
```

체크포인트는 `checkpoints/<id>/`에 `model.safetensors`와 `meta.json`으로 저장됩니다.

## 구조

```
fly-beat/
├── scripts/download_data.sh     데이터 다운로드 + 체크섬 검증
├── charts/                      기본 채보 (음원 없이 WebAudio 로 박자 합성)
├── server/flybeat/
│   ├── game/                    판정 엔진(기준 구현), 채보, 랜덤 채보 생성기, Gymnasium 환경
│   ├── brain/                   커넥톰 전처리, rate 모델 시뮬레이터
│   ├── agents/                  모드 A(deep_rl) · B(mapping) · C(dopamine), 체크포인트
│   ├── api/                     FastAPI, 학습 작업 관리, 관전 WebSocket
│   └── train.py                 학습 CLI
├── web/src/
│   ├── game/                    TS 판정 엔진(사람 플레이용), 캔버스 렌더러, 사운드
│   └── pages/                   Play · Watch · Train · Brain
├── tests/fixtures/judge/        Python·TS 판정 엔진 공용 골든 테스트
└── docs/                        데이터 조사, 뇌 모델, 학습 결과 노트
```

- 판정 엔진은 Python(학습·서버)과 TypeScript(브라우저) 두 벌입니다. 같은 골든 테스트로 결과가 같은지 검증합니다. 관전 화면에서도 서버 판정과 브라우저 판정이 일치하는지 표시합니다.
- 뇌 모델 설명은 [docs/brain-notes.md](docs/brain-notes.md), 데이터 조사 내용은 [docs/data-notes.md](docs/data-notes.md)에 있습니다.

## 개발

```bash
# 서버
cd server && .venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pytest
# 웹
cd web && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
```

판정 규칙을 바꿀 때는 `server/flybeat/game/engine.py`와 `web/src/game/engine.ts`를 함께 고칩니다. 그다음 `python -m flybeat.game.fixtures`로 골든 테스트를 다시 만듭니다.

## 보안 메모

로컬 실행용이지만 기본값을 안전하게 두었습니다.

- 서버는 기본적으로 `127.0.0.1`에만 바인딩합니다. 다른 주소로 바인딩하면 경고를 출력합니다. 인증이 없으므로 외부에 노출하지 마세요.
- Host 헤더를 제한합니다(DNS 리바인딩 방지).
- POST 요청과 WebSocket은 허용된 Origin에서만 받습니다. POST는 JSON만 받습니다.
- 학습 요청 값(스텝 수 등)은 범위를 제한하고, 동시 학습 작업 수도 제한합니다.
- 채보와 체크포인트는 경로가 아니라 ID로만 지정합니다(경로 탈출 차단).
- 체크포인트는 **safetensors + JSON**으로만 저장합니다. SB3 기본 `.zip`(pickle 포함)은 쓰지 않습니다. npz는 `allow_pickle=False`로 읽습니다.
- 다른 사람이 만든 체크포인트도 열 수는 있지만, 데이터 파일만 믿고 쓰세요.

## 데이터 출처와 라이선스

- 커넥톰 데이터: **MaleCNS v1.0**, Google Research · HHMI Janelia Research Campus. *Cell* (2026). https://male-cns.janelia.org/ · 라이선스 **CC-BY 4.0**
- 이 저장소에는 원본 데이터가 들어 있지 않습니다. `scripts/download_data.sh`로 공식 주소에서 받습니다.
- 코드 라이선스: 아직 정하지 않았습니다.
- 기본 채보와 사운드는 이 프로젝트에서 직접 생성했습니다. 특정 상용 리듬게임의 곡·에셋·디자인은 쓰지 않았습니다.
