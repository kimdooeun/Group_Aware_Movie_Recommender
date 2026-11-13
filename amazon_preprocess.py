# -*- coding: utf-8 -*-
"""Amazon Reviews 2023 Movies & TV (5-core, rating_only) preprocessing.

Usage (Anaconda Prompt, from the folder containing the dataset):
    python amazon_preprocess.py --input Movies_and_TV.csv
    python amazon_preprocess.py --input Movies_and_TV.csv --user-fraction 1.0

Default uses a REPRODUCIBLE 10% sample of users (all reviews for those users)
to keep memory/time manageable. Specify --user-fraction 1.0 for all users.

Input  : user_id, parent_asin, rating, timestamp
Output : results/amazon/{amazon_train.csv, amazon_test.csv,
         amazon_test_eval.csv, amazon_user_map.csv, amazon_item_map.csv,
         amazon_summary.json}

Important:
- 'movieId' in these output files is an AMAZON product ID, not a MovieLens ID.
- These files cannot directly test a MovieLens-trained Matrix Factorization model.
  A separate Amazon-specific fit is required to evaluate the algorithm on Amazon.
- The rating-only file has NO title/genre/plot text. Content-based testing needs
  the separate Amazon item metadata and an explicit item-type filtering policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
OUTPUT_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def locate_input(path_str):
    """Allow an ordinary CSV, .gz, or an extracted folder named .csv."""
    path = Path(path_str).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.rglob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"폴더 안에 CSV 파일이 없습니다: {path}")
        raise ValueError(f"CSV가 여러 개입니다. --input에 파일 경로를 직접 입력하세요: {candidates[:5]}")
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path.resolve()}")


def load_amazon(path, user_fraction, chunk_size):
    """Read a large CSV in chunks, sampling USERS (not individual rows)."""
    selected = []
    total_rows = 0
    retained = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=chunk_size,
            usecols=lambda c: c.strip() in REQUIRED_COLUMNS,
            dtype={"user_id": "string", "parent_asin": "string",
                   "rating": "string", "timestamp": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        chunk.columns = chunk.columns.str.strip()
        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f"입력 데이터 열이 부족합니다: {sorted(missing)}. 실제 열: {list(chunk.columns)}")
        total_rows += len(chunk)
        chunk["user_id"] = chunk["user_id"].str.strip()
        chunk["parent_asin"] = chunk["parent_asin"].str.strip()
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=REQUIRED_COLUMNS)
        chunk = chunk[(chunk["user_id"] != "") & (chunk["parent_asin"] != "")]
        chunk = chunk[chunk["rating"].between(1.0, 5.0)]
        if user_fraction < 1.0:
            # The same user always makes the same inclusion decision across chunks.
            hashes = pd.util.hash_pandas_object(chunk["user_id"], index=False)
            selected_users = (hashes % 1000000) < int(user_fraction * 1000000)
            chunk = chunk.loc[selected_users]
        if len(chunk):
            selected.append(chunk[REQUIRED_COLUMNS].copy())
            retained += len(chunk)
        if chunk_number % 5 == 0:
            print(f"  읽은 행: {total_rows:,} / 선택한 행: {retained:,}")

    if not selected:
        raise ValueError("선택된 유효 평점이 없습니다. --user-fraction을 늘려 다시 시도하세요.")
    df = pd.concat(selected, ignore_index=True)
    print(f"읽기 완료: 원본 {total_rows:,}행, 유효·표본 {len(df):,}행")
    return df, total_rows


def preprocess(df, test_fraction, seed):
    # Ensure only one review per (user, Amazon item), retaining the earliest.
    before = len(df)
    df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(["user_id", "parent_asin"], keep="first").reset_index(drop=True)
    duplicates_removed = before - len(df)

    # Integer ID conversion for direct compatibility with the team's mf.py.
    # This is a NEW namespace and must not be confused with MovieLens movieId.
    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["parent_asin"].unique().tolist())
    user_lookup = pd.Series(np.arange(1, len(users) + 1, dtype=np.int64), index=users)
    item_lookup = pd.Series(np.arange(1, len(items) + 1, dtype=np.int64), index=items)
    df["userId"] = df["user_id"].map(user_lookup).astype(np.int64)
    df["movieId"] = df["parent_asin"].map(item_lookup).astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Split interactions WITHIN each user, so a test user appears in training.
    rng = np.random.default_rng(seed)
    df["_random"] = rng.random(len(df))
    df = df.sort_values(["userId", "_random"], kind="mergesort")
    user_counts = df.groupby("userId")["movieId"].transform("size").to_numpy()
    position = df.groupby("userId").cumcount().to_numpy()
    # Preserve >=1 train interaction for every user.
    test_counts = np.minimum(
        user_counts - 1,
        np.maximum(1, np.floor(user_counts * test_fraction).astype(np.int64)),
    )
    is_test = (user_counts >= 2) & (position < test_counts)
    train = df.loc[~is_test, OUTPUT_COLUMNS].sort_values(["userId", "movieId"]).reset_index(drop=True)
    test = df.loc[is_test, OUTPUT_COLUMNS].sort_values(["userId", "movieId"]).reset_index(drop=True)

    # Warm-item evaluation: exclude items absent from train ONLY in test_eval.
    warm_items = set(train["movieId"].unique())
    test_eval = test.loc[test["movieId"].isin(warm_items), OUTPUT_COLUMNS].copy().reset_index(drop=True)

    assert len(train) + len(test) == len(df)
    assert not pd.MultiIndex.from_frame(train[["userId", "movieId"]]).intersection(
        pd.MultiIndex.from_frame(test[["userId", "movieId"]])
    ).size
    assert set(test_eval["movieId"]).issubset(warm_items)

    user_map = pd.DataFrame({"userId": np.arange(1, len(users)+1), "amazon_user_id": users})
    item_map = pd.DataFrame({"movieId": np.arange(1, len(items)+1), "parent_asin": items})
    summary = {
        "input_dataset": "Amazon Reviews 2023 - Movies_and_TV (5-core rating_only)",
        "id_note": "movieId is Amazon parent_asin converted to integer; NOT MovieLens movieId",
        "duplicate_user_item_pairs_removed": int(duplicates_removed),
        "unique_users": len(users),
        "unique_amazon_items": len(items),
        "total_unique_ratings": len(df),
        "train_ratings": len(train),
        "test_ratings": len(test),
        "test_eval_ratings": len(test_eval),
        "test_ratings_excluded_for_cold_items": len(test) - len(test_eval),
        "cold_item_count": len(set(test["movieId"]) - warm_items),
        "train_user_count": int(train["userId"].nunique()),
        "test_user_count": int(test["userId"].nunique()),
        "rating_scale": [1.0, 5.0],
        "note": "This preprocessing alone is NOT an external evaluation of a MovieLens-trained model.",
    }
    return train, test, test_eval, user_map, item_map, summary


def main():
    parser = argparse.ArgumentParser(description="Preprocess Amazon 2023 Movies & TV rating-only dataset")
    parser.add_argument("--input", default="Movies_and_TV.csv", help="CSV / .csv.gz file or directory")
    parser.add_argument("--output", default="results/amazon", help="Output directory")
    parser.add_argument("--user-fraction", type=float, default=0.10,
                        help="Reproducible fraction of users to use (default 0.10; 1.0 = full dataset)")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Per-user test proportion (default 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=200000)
    args = parser.parse_args()
    if not (0 < args.user_fraction <= 1):
        parser.error("--user-fraction 값은 0 초과 1 이하이어야 합니다.")
    if not (0 < args.test_fraction < 1):
        parser.error("--test-fraction 값은 0 초과 1 미만이어야 합니다.")

    path = locate_input(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"입력: {path.resolve()}")
    print(f"사용자 표본 비율: {args.user_fraction:.0%} (동일 사용자의 모든 평점 유지)")
    df, total_rows = load_amazon(path, args.user_fraction, args.chunk_size)
    train, test, test_eval, user_map, item_map, summary = preprocess(
        df, args.test_fraction, args.seed
    )
    summary["input_rows_read"] = total_rows
    summary["user_fraction_requested"] = args.user_fraction
    summary["test_fraction_requested"] = args.test_fraction
    summary["random_seed"] = args.seed
    train.to_csv(out / "amazon_train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out / "amazon_test.csv", index=False, encoding="utf-8-sig")
    test_eval.to_csv(out / "amazon_test_eval.csv", index=False, encoding="utf-8-sig")
    user_map.to_csv(out / "amazon_user_map.csv", index=False, encoding="utf-8-sig")
    item_map.to_csv(out / "amazon_item_map.csv", index=False, encoding="utf-8-sig")
    with (out / "amazon_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== Amazon 데이터 전처리 결과 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n저장 완료: {out.resolve()}")


if __name__ == "__main__":
    main()
