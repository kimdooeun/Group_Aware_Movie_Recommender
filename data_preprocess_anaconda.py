# -*- coding: utf-8 -*-
"""Group-Aware Movie Recommender - A 담당 데이터 전처리/EDA.

Anaconda/일반 Python에서 실행: python data_process_anaconda.py
선택: python data_process_anaconda.py --no-show  (그래프 창 표시 없이 PNG만 저장)

준비할 파일 (이 스크립트와 같은 폴더):
  - tmdb_5000_movies.csv 또는 이 파일이 들어 있는 ZIP 파일
  - ml-1m/ratings.dat, ml-1m/movies.dat (없으면 ml-1m.zip에서 추출하거나 공식 사이트에서 다운로드)

결과 파일은 같은 폴더의 results/에 저장됩니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.request import urlretrieve
import zipfile

import matplotlib.pyplot as plt
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CSV_NAMES = (
    "ratings_train.csv",
    "ratings_test.csv",
    "ratings_test_eval.csv",
    "movies_processed.csv",
)


def find_movielens_files(folder: Path) -> tuple[Path, Path]:
    data_dir = folder / "ml-1m"
    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"
    if ratings_path.exists() and movies_path.exists():
        return ratings_path, movies_path

    zip_path = folder / "ml-1m.zip"
    if not zip_path.exists():
        print("MovieLens 파일이 없으므로 공식 사이트에서 ml-1m.zip을 다운로드합니다...")
        try:
            urlretrieve(MOVIELENS_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                "MovieLens 자동 다운로드에 실패했습니다.\n"
                "https://grouplens.org/datasets/movielens/1m/ 에서 ml-1m.zip을 "
                "다운로드해 이 스크립트와 같은 폴더에 넣어주세요."
            ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in ("ratings.dat", "movies.dat"):
            member = f"ml-1m/{name}"
            with archive.open(member) as source, (data_dir / name).open("wb") as target:
                shutil.copyfileobj(source, target)
    return ratings_path, movies_path


def find_tmdb_csv(folder: Path) -> Path:
    csv_path = folder / "tmdb_5000_movies.csv"
    if csv_path.exists():
        return csv_path

    # Kaggle/브라우저에서 받은 ZIP을 압축 해제하지 않아도 사용 가능
    for zip_path in sorted(folder.glob("*.zip")):
        if zip_path.name == "ml-1m.zip":
            continue
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for item in archive.namelist():
                    if Path(item).name.lower() == "tmdb_5000_movies.csv":
                        with archive.open(item) as source, csv_path.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        print(f"TMDB CSV 압축 해제 완료: {zip_path.name}")
                        return csv_path
        except zipfile.BadZipFile:
            continue

    raise FileNotFoundError(
        "TMDB 파일을 찾지 못했습니다. 'tmdb_5000_movies.csv' 또는 "
        "이를 포함한 ZIP 파일을 이 스크립트와 같은 폴더에 넣어주세요.\n"
        "다운로드: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"
    )


def normalize_title(title: str) -> str:
    title = str(title).lower().strip()
    # MovieLens의 'Matrix, The' -> 'the Matrix' 형태 보정
    match = re.match(r"^(.*),\s*(the|a|an)$", title)
    if match:
        title = match.group(2) + " " + match.group(1)
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def extract_keywords(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        items = json.loads(value)
        return " ".join(item["name"] for item in items if "name" in item)
    except (ValueError, TypeError, KeyError):
        return ""


def save_plot(fig, output_dir: Path, name: str, show: bool) -> None:
    save_path = output_dir / name
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"그래프 저장: {save_path}")


def main(show_plots: bool = True) -> None:
    folder = Path(__file__).resolve().parent
    output_dir = folder / "results"
    output_dir.mkdir(exist_ok=True)

    print("[1/5] 원본 데이터 준비 및 불러오기")
    ratings_path, movies_path = find_movielens_files(folder)
    tmdb_path = find_tmdb_csv(folder)

    ratings = pd.read_csv(
        ratings_path,
        sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::", engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    tmdb = pd.read_csv(tmdb_path)

    print("평점 개수:", len(ratings))
    print("사용자 수:", ratings["userId"].nunique())
    print("MovieLens 영화 수:", len(movies))
    print("TMDB 영화 수:", len(tmdb))
    print("\n[평점 데이터 앞 5행]\n", ratings.head().to_string(index=False))
    print("\n[영화 데이터 앞 5행]\n", movies.head().to_string(index=False))

    print("\n[2/5] MovieLens - TMDB 영화 매칭")
    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)$")[0], errors="coerce"
    )
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )
    tmdb["year"] = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year
    movies["match_title"] = movies["clean_title"].apply(normalize_title)
    tmdb["match_title"] = tmdb["title"].apply(normalize_title)
    tmdb["keywords_text"] = tmdb["keywords"].apply(extract_keywords)

    tmdb_info = tmdb[
        ["id", "match_title", "year", "overview", "keywords_text"]
    ].rename(columns={"id": "tmdbId", "keywords_text": "keywords"})
    # 원본 Colab 코드와 동일하게 제목+연도를 기준으로 매칭합니다.
    tmdb_info = tmdb_info.dropna(subset=["year"]).drop_duplicates(
        subset=["match_title", "year"]
    )
    movies_processed = movies.merge(
        tmdb_info, on=["match_title", "year"], how="left", validate="many_to_one"
    )
    movies_processed["overview"] = movies_processed["overview"].fillna("")
    movies_processed["keywords"] = movies_processed["keywords"].fillna("")
    movies_processed["matched_tmdb"] = movies_processed["tmdbId"].notna()
    print("전체 영화:", len(movies_processed))
    print("TMDB 매칭 성공:", int(movies_processed["matched_tmdb"].sum()))
    print("TMDB 매칭 실패:", int((~movies_processed["matched_tmdb"]).sum()))
    print("매칭 비율: {:.2f}%".format(movies_processed["matched_tmdb"].mean() * 100))

    print("\n[3/5] 사용자별 평점 데이터 분리")
    test = ratings.groupby("userId").sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)
    train_movie_ids = set(train["movieId"])
    # 학습에 등장하지 않은 영화는 공통 평가용에서만 제외 (원본 test는 보관)
    test_eval = test[test["movieId"].isin(train_movie_ids)].copy()
    print("학습 평점 수:", len(train))
    print("원본 평가 평점 수:", len(test))
    print("공통 평가 평점 수:", len(test_eval))
    print("공통 평가에서 제외된 평점 수:", len(test) - len(test_eval))
    print("평가에만 존재하는 영화 수:", len(set(test["movieId"]) - train_movie_ids))

    for name, dataframe in (
        ("ratings_train.csv", train),
        ("ratings_test.csv", test),
        ("ratings_test_eval.csv", test_eval),
        ("movies_processed.csv", movies_processed),
    ):
        dataframe.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        print("CSV 저장:", output_dir / name)

    print("\n[4/5] EDA 그래프 생성")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    save_plot(fig, output_dir, "rating_distribution.png", show_plots)

    genres = movies["genres"].str.split("|").explode()
    fig, ax = plt.subplots(figsize=(8, 5))
    genres.value_counts().head(10).plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Movie Genres")
    ax.set_xlabel("Movie Count")
    save_plot(fig, output_dir, "genre_distribution.png", show_plots)

    fig, ax = plt.subplots(figsize=(8, 5))
    ratings.groupby("userId").size().plot(kind="hist", bins=30, ax=ax)
    ax.set_title("Ratings per User")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    save_plot(fig, output_dir, "ratings_per_user.png", show_plots)

    print("\n[5/5] 전처리 결과 검증 및 압축")
    print("전체 평점 합계:", len(train) + len(test))
    print("중복 평점 수:", int(ratings.duplicated(subset=["userId", "movieId"]).sum()))
    print("영화 ID 중복:", int(movies_processed["movieId"].duplicated().sum()))
    print("학습 사용자 수:", train["userId"].nunique())
    print("평가 사용자 수:", test["userId"].nunique())
    print("공통 평가에만 존재하는 영화 수:", len(set(test_eval["movieId"]) - train_movie_ids))

    zip_path = output_dir / "A_preprocessed_data.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in CSV_NAMES:
            archive.write(output_dir / name, arcname=name)
    print("CSV 압축 저장:", zip_path)
    print("\n완료! CSV 4개와 EDA 그래프 3개는 results 폴더에서 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens/TMDB 전처리 및 EDA")
    parser.add_argument(
        "--no-show", action="store_true", help="그래프 창을 열지 않고 PNG만 저장"
    )
    options = parser.parse_args()
    main(show_plots=not options.no_show)
