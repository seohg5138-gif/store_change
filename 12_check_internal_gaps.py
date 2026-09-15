# -*- coding: utf-8 -*-
"""
12단계(검증): 11단계에서 만든 episode 안에, "같은 상가업소번호가 유지되긴 했는데
중간에 관측 자체가 통째로 빠진 분기"가 있는지 확인한다.

배경: 11단계의 episode 판정은 "직전 관측 행"끼리만 비교해서 상가업소번호가
바뀌었는지 보기 때문에, 중간에 몇 분기가 데이터에서 통째로 빠져도(그 위치
자체가 그 분기 스냅샷에 안 잡혔거나) 앞뒤 상가업소번호가 같으면 하나의
episode로 이어붙여진다. 이러면 그 episode의 실제 존재 기간(시작~종료)과
"생존분기수"(관측된 분기 개수)가 어긋나게 되고, 생존기간이 실제보다
과소평가될 수 있다.

방법: 전체 데이터에 실제로 존재하는 분기 목록(전역 그리드)을 만들고,
각 episode마다 "시작분기~종료분기 사이에 있어야 할 분기 수"와
"실제로 관측된 생존분기수"를 비교한다. 둘이 다르면 그 episode는
내부에 구멍이 있다는 뜻.

실행:
    python 12_check_internal_gaps.py

출력:
    intermediate/episode_gap_check.csv (구멍 있는 episode 목록)
    콘솔에 요약 통계 출력
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

    utils.log("shop_period_raw.parquet에서 전역 분기 목록 로딩 시작...")
    all_periods = pd.read_parquet(config.out_path("shop_period_raw.parquet"), columns=["period"])
    period_grid = sorted(all_periods["period"].unique())
    period_index = {p: i for i, p in enumerate(period_grid)}
    print(f"전역 분기 그리드: {len(period_grid)}개 ({period_grid[0]} ~ {period_grid[-1]})")
    utils.log("전역 분기 목록 준비 완료", t0)

    utils.log("tenancy_episode.parquet 로딩 시작...")
    ep = pd.read_parquet(config.out_path("tenancy_episode.parquet"))
    utils.log(f"로딩 완료 ({len(ep):,}개 episode)", t0)

    # 시작분기~종료분기 사이에 "있어야 할" 분기 개수 (전역 그리드 기준, 양끝 포함)
    ep["시작idx"] = ep["시작분기"].map(period_index)
    ep["종료idx"] = ep["종료분기"].map(period_index)
    ep["기대분기수"] = ep["종료idx"] - ep["시작idx"] + 1

    ep["구멍있음"] = ep["기대분기수"] > ep["생존분기수"]
    ep["구멍난분기수"] = ep["기대분기수"] - ep["생존분기수"]

    n_gap = ep["구멍있음"].sum()
    print(f"\n내부에 구멍이 있는 episode 수: {n_gap:,} / {len(ep):,} ({n_gap/len(ep)*100:.2f}%)")
    print("\n구멍난 분기수 분포 (구멍 있는 episode만):")
    print(ep.loc[ep["구멍있음"], "구멍난분기수"].describe())

    print("\n표본 20건 (구멍 많은 순):")
    cols = ["위치ID", "상가업소번호", "시작분기", "종료분기", "생존분기수", "기대분기수", "구멍난분기수"]
    print(ep[ep["구멍있음"]].sort_values("구멍난분기수", ascending=False)[cols].head(20).to_string(index=False))

    utils.log("결과 저장 시작...")
    out = config.out_path("episode_gap_check.csv")
    ep[ep["구멍있음"]][cols].to_csv(out, index=False, encoding="utf-8-sig")
    utils.log(f"저장 완료: {out}", t0)


if __name__ == "__main__":
    main()
