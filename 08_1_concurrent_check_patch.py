# -*- coding: utf-8 -*-
"""
8-1단계(패치): 슬롯번호 자체를 이용해서, 05단계의 '1층 쏠림' 판정이 놓친
동시존재과다 위치를 추가로 걸러낸다.

문제: 05단계의 1층 쏠림 판정은 건물의 "전체 기간 평균" 층='1' 비율로 계산했는데,
이러면 특정 분기 하나에만 국소적으로 몰린 이상치가 평균에 희석되어 안 걸러진다.
실제로 (PNU19=1165010800113370006, 층_최종='1')처럼 슬롯번호 최댓값이 350까지
나온 그룹이 '검증완료'로 남아있는 게 확인됨.

해결: "이 (PNU19, 층_최종) 조합에서 어느 한 시점에라도 동시에 존재했던
상가업소번호 개수의 최댓값"(=슬롯번호 최댓값)을 직접 기준으로 삼는다.
이게 건물 전체 평균보다 훨씬 직접적이고 희석되지 않는 신호다.

기준값(CONCURRENT_THRESHOLD)은 임의의 기준이라 조정 가능하게 열어뒀다.
너무 낮게 잡으면 진짜 대형 시장(남대문시장 지하 같은 곳)까지 걸러질 수 있고,
너무 높게 잡으면 이상치를 못 거른다. 일단 30으로 시작하고,
suspicious_concurrent_locations.csv를 열어서 표본을 눈으로 확인한 뒤
필요하면 이 값을 조정해서 다시 돌리세요.

실행:
    python 08_1_concurrent_check_patch.py

출력:
    intermediate/shop_period_raw.parquet (덮어씀, 신뢰가능여부만 수정)
    intermediate/suspicious_concurrent_locations.csv (걸러진 위치 목록, 참고용)
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")

CONCURRENT_THRESHOLD = 30  # 필요하면 조정하세요


def main():
    t0 = utils.start_timer()

    utils.log("shop_period_raw.parquet 로딩 시작...")
    df = pd.read_parquet(config.out_path("shop_period_raw.parquet"))
    utils.log(f"로딩 완료 ({len(df):,}행)", t0)

    utils.log("(PNU19, 층_최종)별 최대 슬롯번호 계산 시작...")
    max_slot = df.groupby(["PNU19", "층_최종"])["슬롯번호"].max()
    utils.log(f"계산 완료 ({len(max_slot):,}개 조합)", t0)

    suspicious = max_slot[max_slot >= CONCURRENT_THRESHOLD]
    print(
        f"동시존재과다로 걸러지는 (PNU19,층) 조합 수: {len(suspicious):,} "
        f"(최대 동시존재 >= {CONCURRENT_THRESHOLD})"
    )

    suspicious_keys = pd.MultiIndex.from_tuples(suspicious.index, names=["PNU19", "층_최종"])
    df_keys = pd.MultiIndex.from_frame(df[["PNU19", "층_최종"]])
    is_suspicious = df_keys.isin(suspicious_keys)

    downgrade = is_suspicious & df["층정보_신뢰가능여부"].eq("검증완료")
    df.loc[downgrade, "층정보_신뢰가능여부"] = "검증불가_동시과다"
    print(f"동시존재과다로 강등된 행 수: {downgrade.sum():,}")
    utils.log("플래그 수정 완료", t0)

    print("\n최종 분포:")
    print(df["층정보_신뢰가능여부"].value_counts())
    print(
        "\n비율:\n",
        (df["층정보_신뢰가능여부"].value_counts(normalize=True) * 100).round(2),
    )

    out_csv = config.out_path("suspicious_concurrent_locations.csv")
    suspicious.reset_index(name="최대동시존재수").to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n걸러진 위치 목록 저장: {out_csv}")

    utils.log("shop_period_raw.parquet 덮어쓰기 저장 시작...")
    out = config.out_path("shop_period_raw.parquet")
    df.to_parquet(out, index=False)
    utils.log(f"저장 완료: {out}", t0)


if __name__ == "__main__":
    main()
