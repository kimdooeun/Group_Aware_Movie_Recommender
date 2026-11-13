# -*- coding: utf-8 -*-
"""Amazon Reviews 2023 Movies & TV: attach product metadata to existing ratings IDs.

Usage, from the project ML directory:
    python amazon_preprocess.py --input Movies_and_TV.csv
    python amazon_metadata_join.py --meta meta_Movies_and_TV.jsonl

Requires the item map made by amazon_preprocess.py:
    results/amazon/amazon_item_map.csv (movieId, parent_asin)

Creates:
    results/amazon/amazon_movies_processed.csv
    results/amazon/amazon_metadata_summary.json

This script reads the large .jsonl file LINE BY LINE (no 1.29 GB in-memory load).
It also supports .jsonl.gz files. Product categories are NOT necessarily movie genres.
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def as_text(value):
    """Convert JSON lists, dicts and text into a space-separated feature string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(filter(None, (as_text(v) for v in value)))
    if isinstance(value, dict):
        return " ".join(filter(None, (as_text(v) for v in value.values())))
    return ""


def build(meta_path, map_path, out_path):
    if not meta_path.is_file():
        raise FileNotFoundError(f"메타데이터 파일이 없습니다: {meta_path.resolve()}")
    if not map_path.is_file():
        raise FileNotFoundError(
            f"Amazon 상품 ID 매핑 파일이 없습니다: {map_path.resolve()}\n"
            "먼저 다음 명령어를 실행하세요:\n"
            "  python amazon_preprocess.py --input Movies_and_TV.csv"
        )

    item_map = pd.read_csv(map_path, dtype={"parent_asin": "string"}, encoding="utf-8-sig")
    required = {"movieId", "parent_asin"}
    if not required.issubset(item_map.columns):
        raise ValueError(f"상품 매핑 파일에 {required} 컬럼이 필요합니다: {item_map.columns.tolist()}")
    if item_map["movieId"].duplicated().any() or item_map["parent_asin"].duplicated().any():
        raise ValueError("상품 매핑 파일에 중복된 movieId 또는 parent_asin이 있습니다.")

    item_map["parent_asin"] = item_map["parent_asin"].str.strip()
    target = set(item_map["parent_asin"].dropna().tolist())
    found = {}
    read_count = 0
    matched_lines = 0
    errors = 0

    opener = gzip.open if meta_path.suffix.lower() == ".gz" else open
    print(f"메타데이터 읽기 시작: {meta_path.name}")
    print(f"평점 데이터에서 사용한 Amazon 상품: {len(target):,}개")

    with opener(meta_path, "rt", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, 1):
            read_count += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            asin = str(record.get("parent_asin", "")).strip()
            if asin not in target:
                continue
            matched_lines += 1
            entry = {
                "parent_asin": asin,
                "title": as_text(record.get("title")),
                "genres": as_text(record.get("categories")),
                "keywords": as_text(record.get("features")),
                "overview": as_text(record.get("description")),
            }
            # Prefer the richer metadata row when the same parent_asin occurs twice.
            new_len = sum(len(entry[k]) for k in ("title", "genres", "keywords", "overview"))
            old = found.get(asin)
            old_len = sum(len(old[k]) for k in ("title", "genres", "keywords", "overview")) if old else -1
            if new_len > old_len:
                found[asin] = entry
            if line_number % 100000 == 0:
                print(f"  {line_number:,}줄 처리 / 매칭된 고유 상품 {len(found):,}개")

    meta = pd.DataFrame(list(found.values()), columns=["parent_asin", "title", "genres", "keywords", "overview"])
    combined = item_map.merge(meta, on="parent_asin", how="left", validate="one_to_one")
    fields = ["title", "genres", "keywords", "overview"]
    combined[fields] = combined[fields].fillna("")
    combined["metadata_found"] = combined["parent_asin"].isin(found)
    combined["has_content_text"] = combined[["genres", "keywords", "overview"]].ne("").any(axis=1)
    combined = combined[["movieId", "parent_asin", "title", "genres", "keywords", "overview", "metadata_found", "has_content_text"]]
    combined = combined.sort_values("movieId")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "metadata_file": str(meta_path),
        "rated_items": len(item_map),
        "matched_items": len(found),
        "unmatched_items": len(item_map) - len(found),
        "items_with_text_features": int(combined["has_content_text"].sum()),
        "match_rate": round(len(found) / max(len(item_map), 1), 4),
        "jsonl_lines_read": read_count,
        "matching_jsonl_lines": matched_lines,
        "invalid_json_lines": errors,
        "note": "'genres' means Amazon product categories, not guaranteed film genres. No rating/average_rating used as features.",
    }
    summary_path = out_path.parent / "amazon_metadata_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 처리 결과 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n파일 저장: {out_path.resolve()}")
    print(f"통계 저장: {summary_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Join Amazon 2023 Movies & TV product metadata to numeric item IDs")
    parser.add_argument("--meta", default="meta_Movies_and_TV.jsonl", help="Metadata .jsonl or .jsonl.gz file")
    parser.add_argument("--item-map", default="results/amazon/amazon_item_map.csv")
    parser.add_argument("--output", default="results/amazon/amazon_movies_processed.csv")
    args = parser.parse_args()
    build(Path(args.meta), Path(args.item_map), Path(args.output))


if __name__ == "__main__":
    main()
