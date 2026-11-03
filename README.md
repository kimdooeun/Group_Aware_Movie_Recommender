Content-Based Filtering (콘텐츠 기반 추천)

1. 개요

영화의 장르(Genres), 키워드(Keywords), 줄거리(Overview)를 활용하여 사용자의 취향에 맞는 영화를 추천하는 콘텐츠 기반 추천 모델입니다.

2. 구현 방법

TF-IDF: 영화의 텍스트 특징을 벡터화합니다.

사용자 취향 벡터: 사용자의 영화 평점에서 3점을 뺀 값을 가중치로 적용하여 생성합니다.

Cosine Similarity: 사용자 취향 벡터와 영화 벡터의 유사도를 계산합니다.

Top-N 추천: 유사도가 높은 순서대로 영화를 추천하며, 이미 평가한 영화는 기본적으로 제외합니다.

3. 사용 데이터

data/movies_processed.csv: 영화 정보

data/ratings_train.csv: 사용자 평점 학습 데이터

4. 실행 방법

python content_based.py

실행 시 사용자 1번의 Top-5 추천 결과와 사용자 1, 2, 3번의 영화별 추천 점수를 CSV로 저장합니다.

5. 함수 사용법

from content_based import score_user

# 사용자 1번의 Top-5 추천
recommendations = score_user(user_id=1, top_n=5)

# 특정 영화에 대한 추천 점수
scores = score_user(
    user_id=1,
    movie_ids=[1, 2, 3, 4, 5],
    exclude_seen=False
)

반환 형식: userId, movieId, score

score는 예상 평점이 아닌 코사인 유사도 점수(-1~1)입니다.

6. 출력 파일

results/content_user1_top5.csv: 사용자 1번의 Top-5 추천

results/content_group_scores.csv: 사용자 1, 2, 3번의 전체 영화 추천 점수
