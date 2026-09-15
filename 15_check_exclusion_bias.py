# -*- coding: utf-8 -*-
"""
15단계(검증): 배제된 44.62%가 특정 업종/건물유형에 쏠려있는지 확인한다.

배경: 층정보 신뢰도 판정으로 전체의 44.62%가 분석에서 빠졌는데(검증불가_층정보없음
33.46%, 검증불가_동시과다 5.04%, 검증불가 3.79%, 검증불가_1층쏠림 2.32%), 이게
무작위로 빠진 게 아니라 특정 업종에 집중돼서 빠졌다면, 그 업종에 대한 분석
결과(교체율/생존율)는 표본이 왜곡된 채로 나온 것일 수 있다.

방법: 업종대분류(상권업종대분류명)별로 "배제율"(그 업종 전체 중 검증완료가
아닌 비율)을 계산해서, 전체 평균 배제율(44.62%)과 비교한다. 특정 업종만
유난히 높거나 낮으면 편향 의심.

실행:
    python 15_check_exclusion_bias.py

출력:
    intermediate/exclusion_bias_by_industry.csv
    콘솔에 업종별 배제율 순위 출력
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
    df = pd.read_parquet(
        config.out_path("shop_period_raw.parquet"),
        columns=["상권업종대분류명", "상권업종중분류명", "층정보_신뢰가능여부"],
    )
    utils.log(f"로딩 완료 ({len(df):,}행)", t0)

    df["배제여부"] = df["층정보_신뢰가능여부"] != "검증완료"
    overall_rate = df["배제여부"].mean()
    print(f"\n전체 평균 배제율: {overall_rate*100:.2f}%")

    # --- 업종대분류별 배제율 ---
    utils.log("업종대분류별 배제율 집계 시작...")
    by_major = (
        df.groupby("상권업종대분류명")
        .agg(전체건수=("배제여부", "size"), 배제건수=("배제여부", "sum"))
        .assign(배제율=lambda d: (d["배제건수"] / d["전체건수"] * 100).round(2))
        .sort_values("배제율", ascending=False)
    )
    utils.log("집계 완료", t0)

    print("\n[업종대분류별 배제율]")
    print(by_major.to_string())

    print(f"\n편차 확인: 전체 평균({overall_rate*100:.2f}%) 대비")
    dev = by_major["배제율"] - overall_rate * 100
    print("가장 많이 배제된 업종(평균보다 높은 정도):")
    print(dev.sort_values(ascending=False).head(5))
    print("가장 적게 배제된 업종(평균보다 낮은 정도):")
    print(dev.sort_values().head(5))

    # --- 배제 사유별 x 업종별 교차표 (어떤 이유로 어떤 업종이 빠지는지) ---
    utils.log("배제 사유 x 업종 교차표 생성 시작...")
    cross = pd.crosstab(df["상권업종대분류명"], df["층정보_신뢰가능여부"], normalize="index") * 100
    cross = cross.round(2)
    utils.log("생성 완료", t0)

    print("\n[업종별 신뢰가능여부 구성비 (%, 행 기준)]")
    print(cross.to_string())

    out = config.out_path("exclusion_bias_by_industry.csv")
    by_major.to_csv(out, encoding="utf-8-sig")
    cross.to_csv(config.out_path("exclusion_bias_by_industry_detail.csv"), encoding="utf-8-sig")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
