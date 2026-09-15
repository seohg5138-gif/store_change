# -*- coding: utf-8 -*-
"""
13단계: episode의 구멍(데이터 공백)을 보정한다.

배경: 12/12-1단계에서 확인됨 - 202303~202409 구간(7개 분기)이 다른 분기보다
압도적으로 많은 위치에서 동시에 관측 누락됨 (2만2천~2만5천건 vs 나머지는
수백~수천건). 이건 무작위 노이즈가 아니라 그 기간의 데이터 수집 자체가
구조적으로 비어있었다는 뜻. 이 구간 때문에:
  1) 실제로는 안 바뀐 가게인데 생존분기수가 과소평가됨
  2) 이 구간에만 등장하는 다른 코드 때문에, 실제론 계속 있던 가게가
     "폐업->재입점"으로 잘못 쪼개짐 (교체횟수 과대평가)
  3) 이런 구멍이 한 위치에서 여러 번 발생할 수도 있음 (A->B->A->C->A 등)

해결:
  1) 구조적 공백 분기를 데이터 기반으로 자동 판별 (최댓값 대비 비율 기준,
     하드코딩 안 함 -> 나중에 데이터가 갱신돼도 재사용 가능)
  2) 위치ID별로 episode를 순서대로 보면서, "중간 episode의 전체 구간이
     공백분기 안에만 있고, 그 앞뒤 episode의 상가업소번호가 같으면" 그
     중간 episode를 노이즈로 보고 앞뒤를 하나로 병합. 더 이상 합칠 게
     없을 때까지 반복(while)해서 여러 번 구멍난 경우도 다 처리.
  3) 병합된(또는 원래 하나였던) episode의 생존분기수를, 공백분기는
     "있었다"고 보정해서 다시 계산.

실행:
    python 13_episode_gap_correction.py

    출력:
        intermediate/tenancy_episode_corrected.parquet
        intermediate/location_metrics_corrected.csv
        intermediate/bad_periods.csv (구조적 공백으로 판정된 분기 목록, 참고용)
    """

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")

# 최댓값의 이 비율 이상인 분기만 "구조적 공백"으로 인정.
# (실제 데이터: 최댓값 25,400 -> 20% 컷오프 5,080 -> 7개 분기가 걸러짐,
#  그 아래(2,300~2,500대)는 애매한 수준이라 보수적으로 제외됨)
BAD_PERIOD_RATIO = 0.2


def detect_bad_periods(ep, raw, period_grid):
    ep = ep.copy()
    period_index = {p: i for i, p in enumerate(period_grid)}
    ep["기대분기수"] = ep["종료분기"].map(period_index) - ep["시작분기"].map(period_index) + 1
    gap_ep = ep[ep["기대분기수"] > ep["생존분기수"]]

    key_cols = ["위치ID", "상가업소번호"]
    gap_keys = gap_ep[key_cols].drop_duplicates()
    raw_sub = raw.merge(gap_keys, on=key_cols, how="inner")
    observed = raw_sub.groupby(key_cols)["period"].apply(set)
    gap_idx = gap_ep.set_index(key_cols)

    missing_counter = {}
    for key, obs in observed.items():
        row = gap_idx.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        start, end = row["시작분기"], row["종료분기"]
        for p in period_grid:
            if start <= p <= end and p not in obs:
                missing_counter[p] = missing_counter.get(p, 0) + 1

    missing_series = pd.Series(missing_counter).sort_values(ascending=False)
    if len(missing_series) == 0:
        return set(), missing_series
    cutoff = missing_series.max() * BAD_PERIOD_RATIO
    bad_periods = set(missing_series[missing_series >= cutoff].index)
    return bad_periods, missing_series


def merge_episodes_for_location(episodes, bad_periods, period_grid):
    """
    episodes: 한 위치ID의 episode 딕셔너리 리스트 (시작분기 순 정렬됨).
    반환: (병합된 episodes, 이 위치에서 발생한 병합 횟수, 병합으로 지워진 분기 목록)
    """
    merge_count = 0
    swallowed_periods = []
    changed = True
    while changed:
        changed = False
        for i in range(1, len(episodes) - 1):
            mid, prev, nxt = episodes[i], episodes[i - 1], episodes[i + 1]
            mid_fully_bad = all(
                p in bad_periods
                for p in period_grid
                if mid["시작분기"] <= p <= mid["종료분기"]
            )
            if mid_fully_bad and prev["상가업소번호"] == nxt["상가업소번호"]:
                prev["종료분기"] = nxt["종료분기"]
                swallowed_periods.extend(
                    p for p in period_grid if mid["시작분기"] <= p <= mid["종료분기"]
                )
                del episodes[i + 1]
                del episodes[i]
                merge_count += 1
                changed = True
                break
    return episodes, merge_count, swallowed_periods


def main():
    t0 = utils.start_timer()

    utils.log("tenancy_episode.parquet, shop_period_raw.parquet 로딩 시작...")
    ep = pd.read_parquet(config.out_path("tenancy_episode.parquet"))
    raw = pd.read_parquet(
        config.out_path("shop_period_raw.parquet"),
        columns=["위치ID", "상가업소번호", "period", "층정보_신뢰가능여부"],
    )
    raw = raw[raw["층정보_신뢰가능여부"] == "검증완료"]
    period_grid = sorted(raw["period"].unique())
    utils.log(f"로딩 완료 (episode {len(ep):,}개, 전역분기 {len(period_grid)}개)", t0)

    utils.log("구조적 공백 분기 자동 판별 시작...")
    bad_periods, missing_series = detect_bad_periods(ep, raw, period_grid)
    print(f"\n구조적 공백으로 판정된 분기 ({len(bad_periods)}개): {sorted(bad_periods)}")
    print(f"(기준: 최댓값 {missing_series.max():,}건의 {BAD_PERIOD_RATIO*100:.0f}% 이상)")
    pd.Series(sorted(bad_periods), name="bad_period").to_csv(
        config.out_path("bad_periods.csv"), index=False
    )
    utils.log("판별 완료", t0)

    utils.log("위치ID별 episode 병합 시작 (여러 번의 구멍도 반복 처리)...")
    ep_sorted = ep.sort_values(["위치ID", "episode_id"])
    merged_records = []
    n_before = len(ep_sorted)

    merge_count_per_location = []  # 위치별 몇 번 병합됐는지
    all_swallowed_periods = []     # 병합으로 지워진 분기들 (어느 공백기에서 주로 일어났는지)

    for loc_id, group in ep_sorted.groupby("위치ID", sort=False):
        episodes = group.to_dict("records")
        episodes, merge_count, swallowed = merge_episodes_for_location(
            episodes, bad_periods, period_grid
        )
        merge_count_per_location.append(merge_count)
        all_swallowed_periods.extend(swallowed)
        for e in episodes:
            e["위치ID"] = loc_id
            merged_records.append(e)

    ep_merged = pd.DataFrame(merged_records)
    utils.log(f"병합 완료 (episode {n_before:,}개 -> {len(ep_merged):,}개)", t0)

    # --- 위치별로 몇 번씩 병합됐는지 케이스별 집계 ---
    merge_count_series = pd.Series(merge_count_per_location)
    print("\n[위치별 병합 횟수 분포]")
    dist = merge_count_series.value_counts().sort_index()
    for n_merge, n_loc in dist.items():
        label = "구멍 없음(병합 안 함)" if n_merge == 0 else f"구멍 {n_merge}번 병합됨"
        print(f"  {label}: {n_loc:,}개 위치")
    print(f"  ---")
    print(f"  전체 위치 수: {len(merge_count_series):,}")
    print(f"  병합이 1번이라도 일어난 위치 수: {(merge_count_series > 0).sum():,} "
          f"({(merge_count_series > 0).mean()*100:.2f}%)")
    print(f"  총 병합 횟수(= 허위 교체로 정정된 건수): {merge_count_series.sum():,}")

    # --- 어느 공백 분기에서 주로 병합이 일어났는지 ---
    if all_swallowed_periods:
        swallowed_counts = pd.Series(all_swallowed_periods).value_counts().sort_index()
        print("\n[병합으로 지워진 분기별 건수 (어느 공백기에서 주로 발생했는지)]")
        print(swallowed_counts.to_string())

    utils.log("최종 생존분기수 재계산 시작 (공백 구간은 '있었다'고 보정)...")
    raw_pairs = raw.groupby(["위치ID", "상가업소번호"])["period"].apply(set)

    def recompute_survival(row):
        obs = raw_pairs.get((row["위치ID"], row["상가업소번호"]), set())
        span = [p for p in period_grid if row["시작분기"] <= p <= row["종료분기"]]
        return sum(1 for p in span if (p in obs) or (p in bad_periods))

    ep_merged["생존분기수_보정"] = ep_merged.apply(recompute_survival, axis=1)
    ep_merged["생존여부"] = ep_merged["종료분기"].eq(period_grid[-1]).map(
        {True: "생존중", False: "폐업"}
    )
    utils.log("재계산 완료", t0)

    out_ep = config.out_path("tenancy_episode_corrected.parquet")
    ep_merged.to_parquet(out_ep, index=False)
    print(f"저장 완료: {out_ep}")

    # --- 위치ID 단위 최종 지표 재계산 ---
    utils.log("위치ID 단위 교체율/생존기간 재집계 시작...")
    loc_stat = (
        ep_merged.groupby("위치ID")
        .agg(
            관측분기수=("생존분기수_보정", "sum"),
            에피소드수=("위치ID", "size"),
            평균생존분기수=("생존분기수_보정", "mean"),
            최장생존분기수=("생존분기수_보정", "max"),
        )
        .reset_index()
    )
    loc_stat["교체횟수"] = loc_stat["에피소드수"] - 1
    loc_stat["교체율"] = loc_stat["교체횟수"] / (loc_stat["관측분기수"] - 1).clip(lower=1)
    loc_stat.loc[loc_stat["관측분기수"] <= 1, "교체율"] = pd.NA
    loc_stat = loc_stat.drop(columns="에피소드수").sort_values(
        ["교체율", "교체횟수"], ascending=False
    )
    utils.log("재집계 완료", t0)

    print("\n[보정 전후 비교]")
    print(f"episode 개수: {n_before:,} -> {len(ep_merged):,} "
          f"({n_before - len(ep_merged):,}개 병합되어 사라짐, 즉 허위 교체로 판정된 건수)")

    out_metrics = config.out_path("location_metrics_corrected.csv")
    loc_stat.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_metrics}")

    print("\n교체율 분포 요약 (보정 후):")
    print(loc_stat["교체율"].describe())


if __name__ == "__main__":
    main()
