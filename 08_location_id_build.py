# -*- coding: utf-8 -*-
"""
8단계: 위치ID = PNU19 + 층_최종 + 슬롯번호 로 최종 조립하고,
분석용 최소 컬럼만 담은 shop_period_raw 테이블을 만든다.

실행:
    python 08_location_id_build.py   (7단계에서 SAMPLE_FRAC=1.0으로 전체 실행한 뒤 돌리세요)

출력:
    intermediate/shop_period_raw.parquet
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

    utils.log("df_all_slotted.parquet 로딩 시작...")
    df_all = pd.read_parquet(config.out_path("df_all_slotted.parquet"))
    utils.log(f"로딩 완료 ({len(df_all):,}행)", t0)

    # 슬롯번호가 groupby.apply 과정에서 float으로 바뀌는 경우가 있어(예: 1.0),
    # 위치ID에 ".0"이 붙는 걸 막기 위해 정수로 고정한다.
    df_all["슬롯번호"] = df_all["슬롯번호"].astype("Int64")

    df_all["위치ID"] = (
        df_all["PNU19"].astype(str) + "_"
        + df_all["층_최종"].astype(str) + "_"
        + df_all["슬롯번호"].astype(str)
    )
    utils.log("위치ID 조립 완료", t0)

    print(f"고유 위치ID 개수: {df_all['위치ID'].nunique():,}")

    keep_cols = [
        "위치ID", "period", "상가업소번호", "상호명", "지점명",
        "상권업종대분류명", "상권업종중분류명", "상권업종소분류명",
        "층정보_신뢰가능여부", "PNU19", "층_최종", "슬롯번호",
        "경도", "위도", "시군구명", "법정동명", "도로명주소",
    ]
    keep_cols = [c for c in keep_cols if c in df_all.columns]

    shop_period_raw = df_all[keep_cols].copy()

    utils.log("shop_period_raw.parquet 저장 시작...")
    out = config.out_path("shop_period_raw.parquet")
    shop_period_raw.to_parquet(out, index=False)
    utils.log(f"저장 완료: {out} ({len(shop_period_raw):,}행)", t0)


if __name__ == "__main__":
    main()
