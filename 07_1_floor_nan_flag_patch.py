# -*- coding: utf-8 -*-
"""
7-1단계(패치, 74분짜리 07단계를 다시 돌리지 않기 위한 후처리):

문제: 05단계의 '검증완료' 판정은 "건축물대장에 건물이 매칭됐는지"만 봤고,
"층정보 원본 자체가 결측인지"는 전혀 확인하지 않았다. 그 결과 층정보가
아예 없는 상가 638만 건(전체의 33.5%)이 '검증완료'로 잘못 분류되어 있었고,
이게 07단계 슬롯 배정에서 "같은 nan층"으로 뭉쳐지며 슬롯번호가 500개 이상
폭발하는 원인이 됐다.

해결: df_all_slotted.parquet은 원본 층정보 컬럼을 그대로 갖고 있으므로,
07단계(슬롯 배정, 74분 소요)를 다시 돌릴 필요 없이 플래그만 후처리로 고친다.
슬롯번호 자체는 이미 계산된 값을 그대로 두되(nan층 그룹에서 나온 값이라
의미 없는 값이지만, 어차피 이후 단계에서 이 플래그로 걸러낼 것이므로 무해함),
'층정보_신뢰가능여부'만 새 카테고리로 강등한다.

※ 앞으로 처음부터(1단계부터) 다시 돌릴 일이 있다면, 이 로직은
05_floor_reliability_flag.py에 이미 반영해뒀으니 07-1을 따로 실행할 필요 없다.

실행:
    python 07_1_floor_nan_flag_patch.py

출력:
    intermediate/df_all_slotted.parquet (덮어씀, 신뢰가능여부만 수정)
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

    utils.log("df_all_slotted.parquet 로딩 시작... (용량이 커서 시간이 좀 걸릴 수 있음)")
    path = config.out_path("df_all_slotted.parquet")
    df = pd.read_parquet(path)
    utils.log(f"로딩 완료 ({len(df):,}행)", t0)

    is_nan_floor = df["층정보"].isna()
    was_verified = df["층정보_신뢰가능여부"].eq("검증완료")
    downgrade = is_nan_floor & was_verified

    print(f"층정보 원본 결측 행 수: {is_nan_floor.sum():,} / {len(df):,} ({is_nan_floor.mean()*100:.2f}%)")
    print(f"그 중 잘못 '검증완료'였던 행 수: {downgrade.sum():,}")

    df.loc[downgrade, "층정보_신뢰가능여부"] = "검증불가_층정보없음"
    utils.log("플래그 수정 완료", t0)

    print("\n수정 후 최종 분포:")
    print(df["층정보_신뢰가능여부"].value_counts())
    print(
        "\n비율:\n",
        (df["층정보_신뢰가능여부"].value_counts(normalize=True) * 100).round(2),
    )

    utils.log("df_all_slotted.parquet 덮어쓰기 저장 시작...")
    df.to_parquet(path, index=False)
    utils.log(f"저장 완료: {path}", t0)


if __name__ == "__main__":
    main()
