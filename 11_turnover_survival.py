# -*- coding: utf-8 -*-
"""
11단계: 정식 위치ID(shop_period_raw.parquet, 8~9단계 결과) 기준으로
위치별 교체율과 생존율을 계산한다.

10_turnover_check.py는 근사 위치키(건물관리번호+원본층정보)로 만든 임시
확인용이었다. 이제 정식 위치ID(PNU19+층_최종+슬롯번호)가 나왔으니
이 스크립트로 다시 계산한다.

핵심 전제: '층정보_신뢰가능여부' == '검증완료'인 행만 분석에 포함한다.
(전체의 44.62%는 원본 데이터 한계 또는 확인된 이상치라 배제하기로 확정함)

정의:
- episode(입점 이력): 같은 위치ID에서, 같은 상가업소번호가 연속된 분기 동안
  유지되는 구간. 상가업소번호가 바뀌면 새 episode 시작.
- 생존분기수: 그 episode가 몇 개 분기 동안 유지됐는지
- 생존여부: 마지막 관측 분기가 데이터의 최종 분기와 같으면 "생존중", 아니면 "폐업"
- 교체횟수: 그 위치ID에서 episode가 몇 번 바뀌었는지 (2번째 episode부터 +1)
- 교체율 = 교체횟수 / (관측분기수 - 1)  (관측분기 1개면 계산 불가 -> NaN)

실행:
    python 11_turnover_survival.py

출력:
    intermediate/tenancy_episode.parquet   (위치ID x 상가업소번호 단위 이력)
    intermediate/location_metrics.csv      (위치ID 단위 교체율/평균생존기간)
    intermediate/location_metrics_top_frequent.csv (자주 바뀌는 위치 상위 목록)
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")

TOP_N = 50


def main():
    t0 = utils.start_timer()

    utils.log("shop_period_raw.parquet 로딩 시작...")
    df = pd.read_parquet(
        config.out_path("shop_period_raw.parquet"),
        columns=["위치ID", "period", "상가업소번호", "층정보_신뢰가능여부"],
    )
    utils.log(f"로딩 완료 ({len(df):,}행)", t0)

    before = len(df)
    df = df[df["층정보_신뢰가능여부"] == "검증완료"].copy()
    print(f"검증완료 필터링: {before:,}행 -> {len(df):,}행 ({len(df)/before*100:.1f}%)")

    last_period = df["period"].max()
    print(f"데이터의 마지막 관측 분기: {last_period}")

    utils.log("위치ID별 정렬 및 episode(입점 구간) 탐지 시작...")
    df = df.sort_values(["위치ID", "period"])

    # 같은 위치ID 안에서, 직전 분기와 상가업소번호가 달라지면 새 episode 시작
    df["직전상가"] = df.groupby("위치ID")["상가업소번호"].shift(1)
    df["새episode시작"] = df["직전상가"].isna() | (df["상가업소번호"] != df["직전상가"])
    df["episode_id"] = df.groupby("위치ID")["새episode시작"].cumsum()
    utils.log("episode 탐지 완료", t0)

    utils.log("episode 단위 집계 시작...")
    tenancy_episode = (
        df.groupby(["위치ID", "episode_id"])
        .agg(
            상가업소번호=("상가업소번호", "first"),
            시작분기=("period", "min"),
            종료분기=("period", "max"),
            생존분기수=("period", "nunique"),
        )
        .reset_index()
    )
    tenancy_episode["생존여부"] = tenancy_episode["종료분기"].eq(last_period).map(
        {True: "생존중", False: "폐업"}
    )
    utils.log(f"episode 집계 완료 ({len(tenancy_episode):,}개)", t0)

    utils.log("tenancy_episode.parquet 저장 시작...")
    out_episode = config.out_path("tenancy_episode.parquet")
    tenancy_episode.to_parquet(out_episode, index=False)
    utils.log(f"저장 완료: {out_episode}", t0)

    # --- 위치ID 단위 최종 지표 ---
    utils.log("위치ID 단위 교체율/생존기간 집계 시작...")
    loc_stat = (
        tenancy_episode.groupby("위치ID")
        .agg(
            관측분기수=("생존분기수", "sum"),
            교체횟수=("episode_id", "nunique"),
            평균생존분기수=("생존분기수", "mean"),
            최장생존분기수=("생존분기수", "max"),
        )
        .reset_index()
    )
    # 교체횟수는 episode 개수이므로, "교체가 일어난 횟수"는 episode개수-1
    loc_stat["교체횟수"] = loc_stat["교체횟수"] - 1
    loc_stat["교체율"] = loc_stat["교체횟수"] / (loc_stat["관측분기수"] - 1).clip(lower=1)
    loc_stat.loc[loc_stat["관측분기수"] <= 1, "교체율"] = pd.NA

    loc_stat = loc_stat.sort_values(["교체율", "교체횟수"], ascending=False)
    utils.log("집계 완료", t0)

    print("\n교체율 분포 요약:")
    print(loc_stat["교체율"].describe())
    print("\n평균생존분기수 분포 요약:")
    print(loc_stat["평균생존분기수"].describe())

    utils.log("결과 저장 시작...")
    out_metrics = config.out_path("location_metrics.csv")
    loc_stat.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    print(f"\n위치별 지표 전체 저장 완료: {out_metrics}")

    top = loc_stat.head(TOP_N)
    out_top = config.out_path("location_metrics_top_frequent.csv")
    top.to_csv(out_top, index=False, encoding="utf-8-sig")
    print(f"자주 바뀌는 위치 TOP {TOP_N} 저장 완료: {out_top}")
    utils.log("저장 완료", t0)

    print("\n[자주 바뀌는 위치 TOP 10]")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
