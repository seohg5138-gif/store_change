# -*- coding: utf-8 -*-
"""
3단계: PNU19(법정동코드10+대지구분1+본번4+부번4)를 조인 키로 만들어
df_all과 건축물대장을 연결한다.

- 건축물대장에는 법정동코드가 텍스트(법정동코드명)로만 있어서,
  df_all의 (시군구명, 법정동명) -> 법정동코드 매핑표로 역매핑한다.
- PNU19 생성 시 dtype을 반드시 int64로 강제한 뒤 문자열로 바꾼다.
  (법정동코드 컬럼에 결측치가 하나라도 있으면 float으로 자동 승격되면서
   "1121510100.0" 같은 값이 생겨 매칭이 통째로 깨지는 문제가 있었음 -> 여기서 방지)
- 필지(PNU19)에 건물(동)이 여러 개 딸린 fan-out 케이스는 총층수=최댓값,
  주용도=최빈값으로 집계한다.

실행:
    python 03_pnu_join.py

출력:
    intermediate/building_agg.parquet   (PNU19별 총층수/동개수/대표주용도)
    intermediate/df_all_pnu.parquet     (df_all + PNU19 + matched 플래그)
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")
utils = import_module("utils")


def mode_or_none(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    return s.value_counts().idxmax()


def main():
    df_all = pd.read_parquet(config.out_path("df_all.parquet"))
    df_building = pd.read_parquet(config.out_path("building_raw.parquet"))

    # --- 법정동코드명(텍스트) -> 법정동코드(숫자) 역매핑 ---
    code_map = (
        df_all[["시군구명", "법정동명", "법정동코드"]]
        .dropna()
        .drop_duplicates()
        .assign(법정동코드=lambda d: pd.to_numeric(d["법정동코드"], errors="coerce"))
        .dropna(subset=["법정동코드"])
        .set_index(["시군구명", "법정동명"])["법정동코드"]
        .astype("int64")
        .to_dict()
    )

    df_building["시군구명_정리"] = (
        df_building["시군구코드명"].astype(str).str.replace("서울특별시 ", "", regex=False).str.strip()
    )
    df_building["법정동코드_추정"] = df_building.apply(
        lambda r: code_map.get((r["시군구명_정리"], r["법정동코드명"])), axis=1
    )
    map_rate = df_building["법정동코드_추정"].notna().mean() * 100
    print(f"건축물대장 -> 법정동코드 역매핑 성공률: {map_rate:.2f}%")

    # --- 대지구분코드명(텍스트) -> 숫자 ---
    jibun_map = {"대지": 1, "산": 2}
    df_building["대지구분코드_추정"] = df_building["대지구분코드명"].map(jibun_map).fillna(1)

    # --- 건축물대장 쪽 PNU19 ---
    df_building["PNU19"] = utils.build_pnu19_vectorized(
        df_building["법정동코드_추정"],
        df_building["대지구분코드_추정"],
        df_building["주지번"],
        df_building["부지번"],
    )
    df_building.loc[df_building["법정동코드_추정"].isna(), "PNU19"] = None

    valid_pnu_rate = df_building["PNU19"].notna().mean() * 100
    print(f"건축물대장 PNU19 생성 성공: {valid_pnu_rate:.2f}%")

    # --- fan-out 점검 및 필지 단위 집계 ---
    grp = df_building.dropna(subset=["PNU19"]).groupby("PNU19")
    dup_pnu = grp.size()
    print(
        f"고유 PNU19: {len(dup_pnu):,}개, 동이 2개 이상인 PNU19: "
        f"{(dup_pnu > 1).sum():,}개 ({(dup_pnu > 1).mean() * 100:.2f}%)"
    )

    floor_nunique = grp["지상층수"].nunique()
    print(
        f"그 중 지상층수 값이 서로 다른 PNU19: {(floor_nunique > 1).sum():,}개 "
        f"({(floor_nunique > 1).mean() * 100:.2f}%)"
    )

    building_agg = grp.agg(
        총층수=("지상층수", "max"),
        동개수=("지상층수", "size"),
    )
    building_agg["주용도_대표"] = grp["주용도코드명"].agg(mode_or_none)

    building_agg.to_parquet(config.out_path("building_agg.parquet"))
    print(f"저장 완료: {config.out_path('building_agg.parquet')}")

    # --- df_all 쪽 PNU19 ---
    df_all["PNU19"] = utils.build_pnu19_vectorized(
        df_all["법정동코드"], df_all["대지구분코드"],
        df_all["지번본번지"], df_all["지번부번지"],
    )

    matched = df_all["PNU19"].isin(building_agg.index)
    df_all["건물정보_매칭여부"] = matched
    match_rate = matched.mean() * 100
    print(f"\ndf_all PNU19가 건축물대장에 매칭되는 비율: {match_rate:.2f}%")

    out = config.out_path("df_all_pnu.parquet")
    df_all.to_parquet(out, index=False)
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
