from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#----------------------------데이터 불러오기-----------------------------------

#현재 Python 파일이 있는 프로젝트 폴더
BASE_DIR = Path(__file__).resolve().parent

#데이터 파일 경로
MOVIES_PATH = BASE_DIR / "results" / "movies_processed.csv"
RATINGS_PATH = BASE_DIR / "results" / "ratings_train.csv"

#CSV 파일 불러오기
movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)



#------------------------------데이터 확인------------------------------------

"""

#데이터 크기 확인
print("영화 데이터 크기 : ", movies.shape)
print("평점 데이터 크기 : ", ratings.shape)

#컬럼 및 데이터 확인
print("\n[영화 데이터 컬럼]")
print(movies.columns.tolist())

print("\n[영화 데이터 예시]")
print(movies[["movieId", "title", "genres"]].head())

print("\n[평점 데이터 예시]")
print(ratings.head())

"""

#------------------------------TF-IDF Vectorizer------------------------------------

#결측값 처리
#줄거리와 키워드가 없는 영화는 빈 문자열로 처리
movies["genres"] = movies["genres"].fillna("")
movies["keywords"] = movies["keywords"].fillna("")
movies["overview"] = movies["overview"].fillna("")

#장르 구분자 처리
# 예시 : Action|Adventure|Comedy -> Action Adventure Comedy
movies["genres_text"] = movies["genres"].str.replace(
  "|", " ", regex=False
)

#영화 특징 텍스트 생성
############## 장르 정보가 중요하므로 3번 반복 (이 수치는 후에 변경할 수 있음!) #################
movies["combined_features"] = (
  (movies["genres_text"] + " ") * 3
  + movies["keywords"] + " "
  + movies["overview"]
)

#TF-IDF 벡터화
tfidf = TfidfVectorizer(
  stop_words = "english",
  max_features = 10000
)

tfidf_matrix = tfidf.fit_transform(
  movies["combined_features"]
)

"""
#결과 확인
print("\n[TF-IDF 벡터화 결과]")
print("행렬 크기 : ", tfidf_matrix.shape)

print("\n[첫 번째 영화]")
print(movies.iloc[0]["title"])

print("\n[영화 특징 텍스트]")
print(movies.iloc[0]["combined_features"][:300])
"""

#-------------------------------평점과 연결-------------------------------------

#movieId를 TF-IDF 행렬의 행 번호와 연결
movie_id_to_index = pd.Series(
  np.arange(len(movies)),
  index = movies["movieId"]
)

# 사용자별 영화 추천 점수를 계산하는 함수
def score_user(user_id, movie_ids=None, exclude_seen=True, top_n=None):

    # 사용자의 학습 평점 기록
    history = ratings[ratings["userId"] == user_id].copy()
    history = history[
        history["movieId"].isin(movie_id_to_index.index)
    ]

    if history.empty:
        raise ValueError(f"사용자 {user_id}의 학습 데이터가 없습니다.")

    # 사용자 취향 벡터 생성
    indices = movie_id_to_index.loc[history["movieId"]].to_numpy()
    weights = history["rating"].to_numpy(dtype=float) - 3.0

    profile = np.asarray(
        tfidf_matrix[indices].multiply(
            weights[:, np.newaxis]
        ).sum(axis=0)
    ).ravel()

    # 추천 후보 영화 목록
    candidates = movies[["movieId"]].copy()

    # 특정 후보 영화만 평가하고 싶을 때
    if movie_ids is not None:
        candidates = candidates[
            candidates["movieId"].isin(movie_ids)
        ]

    # 이미 평가한 영화 제외
    if exclude_seen:
        candidates = candidates[
            ~candidates["movieId"].isin(history["movieId"])
        ]

    if candidates.empty:
        return pd.DataFrame(columns=["userId", "movieId", "score"])

    # Cosine Similarity 계산
    candidate_indices = movie_id_to_index.loc[
        candidates["movieId"]
    ].to_numpy()

    candidates["score"] = cosine_similarity(
        tfidf_matrix[candidate_indices],
        profile.reshape(1, -1)
    ).ravel()

    candidates.insert(0, "userId", user_id)

    # 점수 내림차순 정렬
    candidates = candidates.sort_values(
        ["score", "movieId"],
        ascending=[False, True]
    ).reset_index(drop=True)

    if top_n is not None:
        candidates = candidates.head(top_n)

    return candidates



# ==========================================

if __name__ == "__main__":

    # 1. 사용자 1번의 Top-5 추천
    result = score_user(
        user_id=1,
        top_n=5
    )

    # 영화 제목 추가
    result_with_titles = result.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    print("\n[사용자 1번 추천 TOP 5]")
    print(result_with_titles.to_string(index=False))

    # 2. 결과 저장 폴더 생성
    OUTPUT_DIR = BASE_DIR / "results"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 개인 추천 결과 저장
    result.to_csv(
        OUTPUT_DIR / "content_user1_top5.csv",
        index=False
    )

    # 그룹 추천 점수 생성
    group_users = [1, 2, 3]

    group_scores = pd.concat(
        [
            score_user(
                user_id=user_id,
                movie_ids=None,
                exclude_seen=False,
                top_n=None
            )
            for user_id in group_users
        ],
        ignore_index=True
    )

    # 4. 그룹 점수 저장
    output_path = OUTPUT_DIR / "content_group_scores.csv"

    group_scores.to_csv(
        output_path,
        index=False
    )

    print("\n[CSV 생성 완료]")
    print("총 데이터 수:", len(group_scores))
    print(group_scores.head())