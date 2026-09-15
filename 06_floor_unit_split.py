# -*- coding: utf-8 -*-
"""
6단계: 층정보를 정규화하고, 검증완료된 행은 총층수 기준으로 층/호를 분할한다.

- 검증완료(건축물대장 매칭됨) + 층정보가 3자리 이상 숫자(예: '106', '1204')
  -> split_floor_unit()으로 총층수 기준 동적 분할 (예: 9층 건물의 '106' -> 층1, 호06)
- 그 외 모든 경우 -> normalize_floor_token()으로 표기만 통일
  ('01'->'1', 'B1'/'지'->'지하1', '반'/'반지층'->'반지하')
- 검증불가(매칭 안 됨) 행은 원본 층정보를 정규화만 해서 그대로 둔다
  (임의로 채우거나 지우지 않음 -> 이후 분석에서 층정보_신뢰가능여부로 필터링해서 쓸지 결정)

실행:
    python 06_floor_unit_split.py

출력:
    intermediate/df_all_floor_split.parquet (층_최종, 호_분할 컬럼 추가)
"""

import os
import re
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")


def main():
    df_all = pd.read_parquet(config.out_path("df_all_flagged.parquet"))
    building_agg = pd.read_parquet(config.out_path("building_agg.parquet"))

    total_floors_map = building_agg["총층수"]
    df_all["총층수"] = df_all["PNU19"].map(total_floors_map)

    is_multi_digit = df_all["층정보"].fillna("").str.fullmatch(r"\d{3,}")
    split_target = df_all["건물정보_매칭여부"] & is_multi_digit

    print(f"층/호 동적 분할 대상 행수: {split_target.sum():,}")

    split_result = df_all.loc[split_target].apply(
        lambda r: utils.split_floor_unit(r["층정보"], r["총층수"]), axis=1
    )

    df_all["층_최종"] = df_all["층정보"].apply(utils.normalize_floor_token)
    df_all["호_분할"] = None

    if len(split_result) > 0:
        floors = split_result.apply(lambda t: t[0])
        units = split_result.apply(lambda t: t[1])
        df_all.loc[split_target, "층_최종"] = floors.apply(utils.normalize_floor_token)
        df_all.loc[split_target, "호_분할"] = units

    # 이상치 점검: 분할된 층이 총층수를 초과하는 경우
    numeric_floor = pd.to_numeric(df_all.loc[split_target, "층_최종"], errors="coerce")
    total_floors_for_target = df_all.loc[split_target, "총층수"]
    outlier = (numeric_floor > total_floors_for_target).sum()
    print(f"분할된 층이 총층수를 초과하는 이상치: {outlier:,} / {split_target.sum():,}")

    out = config.out_path("df_all_floor_split.parquet")
    df_all.to_parquet(out, index=False)
    print(f"\n저장 완료: {out}")

    print("\n표본 30건:")
    sample_cols = ["건물관리번호", "층정보", "총층수", "층_최종", "호_분할", "층정보_신뢰가능여부"]
    print(df_all.loc[split_target, sample_cols].sample(min(30, split_target.sum()), random_state=42))


if __name__ == "__main__":
    main()
