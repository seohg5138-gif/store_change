# -*- coding: utf-8 -*-
"""
5단계: 층정보 신뢰가능여부를 확정한다.

확정된 규칙:
- PNU19가 건축물대장에 매칭 안 됨 -> '검증불가'
  (임의로 값을 채우거나 지우지 않고, 플래그만 남긴 채 원본 층정보는 그대로 둔다)
- 층정보 원본 자체가 결측(NaN) -> '검증불가_층정보없음' (매칭 성공 여부와 무관하게 적용)
  (이걸 놓치면, 층정보 없는 행들이 전부 '층=nan'이라는 같은 값으로 뭉쳐져서
  7단계 슬롯 배정에서 서로 다른 실제 층에 있던 상가들이 한 자리로 오인되고
  슬롯번호가 수백 개까지 폭발하는 문제가 생김 - 실제로 발생해서 확인함)
- PNU19가 매칭됨 + 건물(PNU19) 단위로 봤을 때 "총층수 5층 이상인데 그 건물
  상가의 80% 이상이 층='1'"인 경우 -> '검증불가_1층쏠림'
  (4-1단계 검증 결과: 이런 건물이 4단계 카테고리 평균으로는 안 보였지만,
  건물 단위 분포를 보면 90~100% 구간에 8,443개가 몰려있는 이상 패턴이 확인됨.
  주용도별로 "이건 정상일 수 있다"고 봐주는 것도 결국 근거 없는 추측이라,
  주용도 구분 없이 이 조건에 해당하면 전부 검증불가로 통일한다.
  -> 12층 건물에서 '1층'이라고 적힌 게 진짜 1층인지, 결측이 1로 채워진 건지는
  우리가 가진 정보로는 원리적으로 구분 불가능하기 때문에, 억지로 판단하지 않고
  포기하는 게 맞다는 결론)
- 그 외(매칭됨 + 층정보 있음 + 쏠림 없음) -> '검증완료'
  (6단계에서 총층수 기준 층/호 분할 예정)

실행:
    python 05_floor_reliability_flag.py

출력:
    intermediate/df_all_flagged.parquet
    intermediate/suspicious_buildings.csv (쏠림으로 강등된 건물 목록, 참고용)
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")

MIN_TOTAL_FLOORS = 5
SUSPICIOUS_THRESHOLD = 0.8


def main():
    t0 = utils.start_timer()

    utils.log("df_all_pnu.parquet, building_agg.parquet 로딩 시작...")
    df_all = pd.read_parquet(config.out_path("df_all_pnu.parquet"))
    building_agg = pd.read_parquet(config.out_path("building_agg.parquet"))
    utils.log(f"로딩 완료 ({len(df_all):,}행)", t0)

    # 기본 플래그: 매칭 여부만으로 우선 결정
    df_all["층정보_신뢰가능여부"] = df_all["건물정보_매칭여부"].map(
        {True: "검증완료", False: "검증불가"}
    )
    utils.log("매칭 여부 기준 기본 플래그 부여 완료", t0)

    # 층정보 원본 자체가 결측인 경우는 매칭 성공 여부와 무관하게 검증불가로 처리.
    # (건물이 매칭돼도, 애초에 층정보가 없으면 검증할 대상 자체가 없음.
    #  이걸 놓치면 층='nan'인 행들이 같은 층으로 잘못 묶여서 7단계 슬롯 배정에서
    #  슬롯번호가 수백 개까지 폭발하는 문제가 생긴다.)
    is_nan_floor = df_all["층정보"].isna()
    df_all.loc[is_nan_floor & df_all["층정보_신뢰가능여부"].eq("검증완료"), "층정보_신뢰가능여부"] = "검증불가_층정보없음"
    utils.log(f"층정보 결측 플래그 부여 완료 (결측 {is_nan_floor.sum():,}행)", t0)

    # --- 건물(PNU19) 단위 1층 쏠림 계산 (매칭된 행만 대상) ---
    utils.log("건물 단위 1층 쏠림 집계 시작... (그룹 수가 많으면 몇 분 걸릴 수 있음)")
    matched = df_all[df_all["건물정보_매칭여부"]]
    building_stat = (
        matched.groupby("PNU19")
        .agg(표본수=("층정보", "size"), 층1_비율=("층정보", lambda s: (s == "1").mean()))
    )
    building_stat["총층수"] = building_agg["총층수"]
    utils.log(f"건물 단위 집계 완료 ({len(building_stat):,}개 건물)", t0)

    suspicious_pnu = building_stat[
        (building_stat["총층수"] >= MIN_TOTAL_FLOORS)
        & (building_stat["층1_비율"] >= SUSPICIOUS_THRESHOLD)
    ]
    print(
        f"1층 쏠림으로 강등되는 건물 수: {len(suspicious_pnu):,} "
        f"(총층수>={MIN_TOTAL_FLOORS} & 층1_비율>={SUSPICIOUS_THRESHOLD*100:.0f}%)"
    )

    downgrade_mask = (
        df_all["층정보_신뢰가능여부"].eq("검증완료")
        & df_all["PNU19"].isin(suspicious_pnu.index)
    )
    df_all.loc[downgrade_mask, "층정보_신뢰가능여부"] = "검증불가_1층쏠림"
    print(f"1층 쏠림으로 강등된 행 수: {downgrade_mask.sum():,}")
    utils.log("1층 쏠림 강등 처리 완료", t0)

    print("\n최종 분포:")
    print(df_all["층정보_신뢰가능여부"].value_counts())
    print(
        "\n비율:\n",
        (df_all["층정보_신뢰가능여부"].value_counts(normalize=True) * 100).round(2),
    )

    suspicious_out = config.out_path("suspicious_buildings.csv")
    suspicious_pnu.reset_index().to_csv(suspicious_out, index=False, encoding="utf-8-sig")
    print(f"\n의심 건물 목록 저장: {suspicious_out}")

    utils.log("df_all_flagged.parquet 저장 시작...")
    out = config.out_path("df_all_flagged.parquet")
    df_all.to_parquet(out, index=False)
    utils.log(f"저장 완료: {out}", t0)


if __name__ == "__main__":
    main()
