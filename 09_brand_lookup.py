# -*- coding: utf-8 -*-
"""
9단계: 상가업소번호 기준 brand_lookup 테이블을 만든다.

검증 결과 98.7%는 상호명이 하나로 고정되어 있어서, 상호명이 유지되는 구간별로
시작/종료 분기를 압축하는 방식으로 만든다 (상호명이 안 바뀐 98.7%는 자동으로 1행이 됨).

실행:
    python 09_brand_lookup.py

출력:
    intermediate/brand_lookup.parquet
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")


def main():
    t0 = utils.start_timer()

    utils.log("shop_period_raw.parquet 로딩 시작...")
    shop_period_raw = pd.read_parquet(
        config.out_path("shop_period_raw.parquet"),
        columns=["상가업소번호", "상호명", "지점명", "period"],
    )
    utils.log(f"로딩 완료 ({len(shop_period_raw):,}행)", t0)

    utils.log("상가업소번호별 상호명 변경 여부 집계 시작...")
    brand_check = shop_period_raw.groupby("상가업소번호")["상호명"].nunique()
    changed = brand_check[brand_check > 1]
    utils.log("집계 완료", t0)
    print(
        f"상호명이 바뀐 상가업소번호 수: {len(changed):,} / "
        f"전체 {shop_period_raw['상가업소번호'].nunique():,} "
        f"({len(changed) / shop_period_raw['상가업소번호'].nunique() * 100:.2f}%)"
    )

    utils.log("brand_lookup 테이블 생성 시작...")
    brand_lookup = (
        shop_period_raw.sort_values(["상가업소번호", "period"])
        .groupby(["상가업소번호", "상호명", "지점명"], dropna=False)
        .agg(시작분기=("period", "min"), 종료분기=("period", "max"))
        .reset_index()
    )
    brand_lookup["상호명_변경여부"] = brand_lookup["상가업소번호"].isin(changed.index)
    utils.log("생성 완료", t0)

    print(f"\nbrand_lookup 행수: {len(brand_lookup):,}")

    utils.log("brand_lookup.parquet 저장 시작...")
    out = config.out_path("brand_lookup.parquet")
    brand_lookup.to_parquet(out, index=False)
    utils.log(f"저장 완료: {out}", t0)


if __name__ == "__main__":
    main()
