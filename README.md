# Group_Aware_Movie_Recommender
Group movie recommendation

## Project Goal
2~5명의 사용자 취향을 고려한 그룹 영화 추천 시스템

## Recommendation Method
1. Model-based Collaborative Filtering (MF)
2. Content-based Filtering (TF-IDF)

## Group Recommendation
- Average
- Least Misery

## Dataset
- MovieLens 1M
- TMDB 5000
- Amazon Reviews 2023 (외부 테스트)

## Collaborative Filtering: Matrix Factorization (B)

SGD로 학습하는 bias MF를 NumPy로 직접 구현했습니다. (라이브러리 호출 없음)

```
r̂(u, i) = μ + b_u + b_i + p_u · q_i
```

- `μ`: 학습 데이터 평균 평점, `b_u`/`b_i`: 사용자/영화 편향, `p_u`/`q_i`: 잠재요인 벡터
- 평점 1건마다 오차 `e = r − r̂`로 `b_u, b_i, p_u, q_i`를 갱신하고, L2 정규화를 적용
- 예측값은 1~5로 clip, 모르는 사용자/영화는 평균과 아는 쪽 편향으로 대체

### Files
| 파일 | 설명 |
|---|---|
| `mf.py` | MF 모델 (SGD 학습, 예측, RMSE/MAE, 전체 점수 생성) |
| `run_mf.py` | 튜닝 / 최종 평가 / 점수 파일 생성 |
| `results/mf_tuning.csv` | 튜닝 12개 조합 비교 (검증 RMSE/MAE) |
| `results/mf_final.csv`, `mf_curve.csv` | 최종 성능, 에폭별 학습·평가 곡선 |
| `results/mf_best_params.json` | 선택된 하이퍼파라미터 |
| `presentation/` | MF 슬라이드, 전반부 발표 대본 |

### Usage
```
python run_mf.py tune     # 학습 데이터 10%를 검증용으로 분리해 그리드 서치
python run_mf.py final    # 최적 파라미터로 전체 학습 후 ratings_test_eval.csv 평가
python run_mf.py export   # 전체 사용자 x 영화 점수 -> results/mf_scores.csv
```

- 튜닝에는 `ratings_test_eval.csv`를 사용하지 않았습니다. (데이터 누출 방지)
- numba가 설치돼 있으면 자동으로 학습이 빨라집니다. (선택 사항, 없어도 동작)

### Results
튜닝 그리드: k ∈ {20, 50}, lr ∈ {0.005, 0.01}, λ ∈ {0.02, 0.05, 0.1}, 20 epochs. 최적은 **k=50, lr=0.01, λ=0.05**.

평가 데이터(`ratings_test_eval.csv`, 199,984건) 기준:

| 모델 | RMSE | MAE |
|---|---|---|
| 전체 평균 기준선 (학습 평균 3.58점) | 1.1178 | 0.9344 |
| **MF (k=50)** | **0.8553** | **0.6723** |
| 개선율 | −23.5% | −28.0% |

- 학습 RMSE 0.7395 → 평가 0.8553: 약한 과적합이 있으나 평가 오차는 20 epoch까지 계속 감소
- Python 3.8(numpy 1.24, pandas 2.0)과 Python 3.11(`requirements.txt` 고정 버전)에서 결과 동일

### Output for Group Recommendation (D)
`results/mf_scores.csv`: `userId, movieId, score` (21,427,007행, 약 373MB)

- 6,040명 × 3,680편 전체 후보의 예상 평점(1~5). 학습 데이터에서 이미 평가한 조합은 제외
- 용량 문제로 git에는 올리지 않음 (`.gitignore`), 직접 전달 또는 `python run_mf.py export`로 재생성
- MF 예상 평점과 TF-IDF 유사도는 단위가 달라 직접 비교하지 않고, 그룹 점수(Average, Least Misery)는 MF 점수끼리 따로 집계
