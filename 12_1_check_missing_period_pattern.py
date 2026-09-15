# -*- coding: utf-8 -*-
"""
12-1단계(진단): 12단계에서 찾은 '구멍난 episode'들의 실제 빠진 분기가
특정 시점에 몰려있는지 확인한다.

배경: 12단계 결과에서 "7분기 빠짐"이 중앙값이자 75%에 해당할 만큼 반복되는데,
이게 무작위라면 이렇게 특정 숫자에 몰릴 수가 없다. 특정 분기(들)이
많은 위치에서 동시에 빠졌을 가능성(=데이터 수집 자체의 구조적 공백)을
확인한다.

방법: episode_gap_check.csv의 구멍난 episode들에 대해, 실제로 어떤
분기가 빠졌는지(시작~종료 사이의 전역 그리드 분기 중 관측 안 된 것)를
역산해서, 그 "빠진 분기" 값 자체의 빈도를 센다.

실행:
    python 12_1_check_missing_period_pattern.py

출력:
    콘솔에 "가장 자주 빠지는 분기" 순위 출력
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

    utils.log("episode_gap_check.csv 로딩...")
    gap_ep = pd.read_csv(config.out_path("episode_gap_check.csv"))
    print(f"구멍 있는 episode 수: {len(gap_ep):,}")

    utils.log("전역 분기 그리드 및 원본 관측 로딩 시작...")
    raw = pd.read_parquet(
        config.out_path("shop_period_raw.parquet"),
        columns=["위치ID", "상가업소번호", "period", "층정보_신뢰가능여부"],
    )
    raw = raw[raw["층정보_신뢰가능여부"] == "검증완료"]
    period_grid = sorted(raw["period"].unique())
    utils.log(f"로딩 완료 (전역 분기 {len(period_grid)}개)", t0)

    # 구멍난 episode들에 대해서만 실제 관측 분기 집합을 다시 조회
    key_cols = ["위치ID", "상가업소번호"]
    gap_keys = gap_ep[key_cols].drop_duplicates()
    raw_sub = raw.merge(gap_keys, on=key_cols, how="inner")

    utils.log("episode별 실제 관측 분기 집합 구성 시작...")
    observed = raw_sub.groupby(key_cols)["period"].apply(set)
    utils.log(f"구성 완료 ({len(observed):,}개 조합)", t0)

    missing_period_counter = {}
    gap_ep_indexed = gap_ep.set_index(key_cols)

    for key, obs_periods in observed.items():
        row = gap_ep_indexed.loc[key]
        # 여러 episode가 같은 (위치ID,상가업소번호)에 있을 수 있어 첫 행만 사용
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        start, end = row["시작분기"], row["종료분기"]
        expected = [p for p in period_grid if start <= p <= end]
        for p in expected:
            if p not in obs_periods:
                missing_period_counter[p] = missing_period_counter.get(p, 0) + 1

    missing_series = pd.Series(missing_period_counter).sort_values(ascending=False)
    print("\n[가장 자주 빠지는 분기 TOP 15]")
    print(missing_series.head(15))

    utils.log("완료", t0)


if __name__ == "__main__":
    main()
