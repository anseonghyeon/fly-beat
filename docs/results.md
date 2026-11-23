# 학습 결과 기록

측정 환경: Apple M4 (10코어), 16GB RAM, CPU만 사용.
정확도는 판정 점수의 평균이다(PERFECT 100%, GREAT 70%, GOOD 30%, MISS 0%). 검증에는 학습에 쓰지 않은 고정 seed 랜덤 채보 4개(`eval_charts()`)를 쓴다.

## 모드 A: MLP + SB3 PPO

```bash
python -m flybeat.train --mode A --steps 300000
```

| 항목 | 값 |
|---|---|
| 정책망 | MLP [128, 128], ReLU |
| 학습 | 300,000 스텝 (환경 8개), 약 14초 |
| 학습 중 정확도 (최근 20판) | 50% → 96.6% |
| **검증 정확도** | **94.6%** |

참고로 무작위로 누르는 정책은 60% 미만이다(`test_env_oracle_scores_high_and_random_scores_low`). 정답 정책(오라클)은 95%를 넘는다.
