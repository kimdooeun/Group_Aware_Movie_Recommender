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

### Preprocessed Data
`data_preprocess.py`가 생성한 전처리 완료 파일이며, 모든 모델이 `results/`의 파일을 공통으로 사용합니다.

| 파일 | 설명 |
|---|---|
| `results/ratings_train.csv` | 학습용 평점 (800,193건) |
| `results/ratings_test.csv` | 사용자별 20%를 무작위 분할한 테스트 평점 전체 (200,016건) |
| `results/ratings_test_eval.csv` | **평가 정답.** `ratings_test.csv`에서 학습 데이터에 없는 영화 33건을 제외한 199,983건 |
| `results/movies_processed.csv` | 영화 정보 3,883편 (장르, 연도, TMDB 줄거리·키워드 결합) |

- 모델 평가(RMSE/MAE 등)는 `ratings_test_eval.csv`를 정답으로 사용합니다.

## Collaborative Filtering: Matrix Factorization

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

평가 데이터(`ratings_test_eval.csv`, 199,983건) 기준:

| 모델 | RMSE | MAE |
|---|---|---|
| 전체 평균 기준선 (학습 평균 3.58점) | 1.1178 | 0.9344 |
| **MF (k=50)** | **0.8553** | **0.6723** |
| 개선율 | −23.5% | −28.0% |

- 학습 RMSE 0.7395 → 평가 0.8553: 약한 과적합이 있으나 평가 오차는 20 epoch까지 계속 감소
- Python 3.8(numpy 1.24, pandas 2.0)과 Python 3.11(`requirements.txt` 고정 버전)에서 결과 동일

### Output for Group Recommendation
`results/mf_scores.csv`: `userId, movieId, score` (21,427,007행, 약 373MB)

- 6,040명 × 3,680편 전체 후보의 예상 평점(1~5). 학습 데이터에서 이미 평가한 조합은 제외
- 용량 문제로 git에는 올리지 않음 (`.gitignore`), 직접 전달 또는 `python run_mf.py export`로 재생성
- MF 예상 평점과 TF-IDF 유사도는 단위가 달라 직접 비교하지 않고, 그룹 점수(Average, Least Misery)는 MF 점수끼리 따로 집계

## Content-Based Filtering

영화의 장르(Genres), 키워드(Keywords), 줄거리(Overview)를 활용하여 사용자의 취향에 맞는 영화를 추천하는 콘텐츠 기반 추천 모델입니다.

- **TF-IDF**: 영화의 텍스트 특징을 벡터화
- **사용자 취향 벡터**: 사용자의 영화 평점에서 3점을 뺀 값을 가중치로 적용하여 생성
- **Cosine Similarity**: 사용자 취향 벡터와 영화 벡터의 유사도를 계산
- **Top-N 추천**: 유사도가 높은 순서대로 추천하며, 이미 평가한 영화는 기본적으로 제외

### Files
| 파일 | 설명 |
|---|---|
| `content_based.py` | 콘텐츠 기반 모델 (입력: `results/movies_processed.csv`, `results/ratings_train.csv`) |
| `results/content_user1_top5.csv` | 사용자 1번의 Top-5 추천 |
| `results/content_group_scores.csv` | 사용자 1, 2, 3번의 전체 영화 추천 점수 |

### Usage
```
python content_based.py
```

실행 시 사용자 1번의 Top-5 추천 결과와 사용자 1, 2, 3번의 영화별 추천 점수를 CSV로 저장합니다.

```python
from content_based import score_user

# 사용자 1번의 Top-5 추천
recommendations = score_user(user_id=1, top_n=5)

# 특정 영화에 대한 추천 점수
scores = score_user(
    user_id=1,
    movie_ids=[1, 2, 3, 4, 5],
    exclude_seen=False
)
```

- 반환 형식: `userId, movieId, score`
- `score`는 예상 평점이 아닌 코사인 유사도 점수(-1~1)입니다.

## Group Evaluation

두 모델(MF, 콘텐츠 기반)을 **수정하지 않고** 무작위 가상 그룹(2~5명) 4,000개로 그룹 단위 추천을 평가했습니다. 구성원별 점수를 Average 또는 Least Misery로 합쳐 그룹 Top-10을 만들고, 각 구성원이 테스트에서 4점 이상 준 영화로 채점합니다.

**📄 평가 보고서: [`results/group_eval/group_eval_report.md`](results/group_eval/group_eval_report.md)**
평가 방법, 결과, 점수 해석(왜 낮은지, 그래도 의미 있는 이유), 예상 질문, 한계를 정리한 문서입니다. 그림은 `results/group_eval/figures/`에 있습니다.

### Summary
| 목록 (그룹 P@10, Average) | 값 |
|---|---|
| 무작위 | 0.0044 |
| 콘텐츠 기반 | 0.0077 |
| **MF** | **0.0332** |
| 인기도순 (평점 개수 순, 개인화 없음) | 0.1073 |

- 이 평가에서 MF가 콘텐츠 기반보다 일관되게 앞섭니다. 같은 그룹끼리 비교하면 MF가 이긴 그룹 54%, 진 그룹 8%입니다.
- 절대 성능은 낮고 두 모델 모두 인기도순보다 낮습니다. MF 목록은 영화 편향 같은 비개인화 성분에 크게 의존합니다.
- Least Misery는 Average보다 나은 점이 없었습니다. 자세한 해석과 주장할 수 없는 것은 보고서를 참고하세요.

### Files
| 파일 | 설명 |
|---|---|
| `eval_group.py` | 그룹 평가 (그룹 생성, 그룹 점수, 구성원별 채점) |
| `eval_group_report.py` | 기준점(무작위, 인기도, 상한), 개인 추천 비교, 응집도 분석, 그림 |
| `eval_group_diagnostics.py` | 목록 다양성, 인기도를 맞춘 비교, seed 반복 |
| `eval_group_diag_mf.py` | MF 점수 분해(편향/잠재요인). MF를 같은 설정으로 재학습 |
| `results/group_eval/` | 보고서, 분석 CSV, 그림 |

### Usage
```
python eval_group.py --mf-scores /path/to/mf_scores.csv
python eval_group_report.py --mf-scores /path/to/mf_scores.csv
python eval_group_diagnostics.py --mf-scores /path/to/mf_scores.csv
python eval_group_diag_mf.py --mf-scores /path/to/mf_scores.csv    # 재학습으로 수 분 걸립니다
```

- 위 순서로 실행하면 `results/group_eval/`의 결과가 다시 만들어집니다. scikit-learn이 설치된 환경에서 실행하세요.
- `mf_scores.csv`는 용량 때문에 git에 없습니다. `python run_mf.py export`로 만들거나 직접 전달받으세요.
