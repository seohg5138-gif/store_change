# -*- coding: utf-8 -*-
"""
4-1단계: 카테고리(주용도) 단위 평균 비교 대신, "건물(PNU19) 단위"로 1층 쏠림을
직접 확인한다.

배경: 4단계에서 주거계열 카테고리 평균(46.7%)이 근린생활시설(31.5%)보다 높게
나왔지만, 이건 "원래 1층에 상가가 몰리는 게 정상인 저층 단독주택"과 "총층수가
높은데도 비정상적으로 1층에 쏠린 건물(예: 20층 오피스텔에 350개가 전부 1층)"이
섞여있어서 구분이 안 됨. 그래서 카테고리 평균이 아니라, "총층수 5층 이상인
건물 하나하나가 1층에 얼마나 쏠려있는지" 분포를 직접 본다.

실행:
    python 04_1_floor1_concentration_check.py

출력:
    intermediate/building_floor1_ratio.csv  (건물별 1층 비율 전체)
    콘솔에 분포(히스토그램 구간별 건물 수) 출력
"""

import os
import sys

import duckdb
import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")

MIN_TOTAL_FLOORS = 5       # 이 총층수 이상인 건물만 대상으로 함
SUSPICIOUS_THRESHOLD = 0.8  # 1층 비율이 이 이상이면 의심 건물로 표시


def main():
    df_all_path = config.out_path("df_all_pnu.parquet")
    building_agg_path = config.out_path("building_agg.parquet")

    con = duckdb.connect()
    query = f"""
        SELECT
            a.PNU19,
            b.총층수,
            b.주용도_대표,
            COUNT(*) AS 표본수,
            AVG(CASE WHEN a.층정보 = '1' THEN 1.0 ELSE 0.0 END) AS 층1_비율
        FROM read_parquet('{df_all_path}') a
        JOIN read_parquet('{building_agg_path}') b
          ON a.PNU19 = b.PNU19
        WHERE a.건물정보_매칭여부 = true
          AND b.총층수 >= {MIN_TOTAL_FLOORS}
        GROUP BY a.PNU19, b.총층수, b.주용도_대표
    """
    building_ratio = con.execute(query).df()

    print(f"총층수 {MIN_TOTAL_FLOORS} 이상 건물 수: {len(building_ratio):,}")

    # 분포를 10% 구간으로 나눠서 건물 수 카운트
    bins = [i / 10 for i in range(11)]
    building_ratio["구간"] = pd.cut(building_ratio["층1_비율"], bins=bins, include_lowest=True)
    dist = building_ratio["구간"].value_counts().sort_index()
    print("\n[1층 비율 분포 (건물 수 기준)]")
    print(dist.to_string())

    suspicious = building_ratio[building_ratio["층1_비율"] >= SUSPICIOUS_THRESHOLD]
    print(
        f"\n1층 비율 {SUSPICIOUS_THRESHOLD*100:.0f}% 이상인 '의심 건물' 수: "
        f"{len(suspicious):,} / {len(building_ratio):,} "
        f"({len(suspicious) / len(building_ratio) * 100:.2f}%)"
    )
    print(f"의심 건물에 속한 상가 행 수(표본수 합): {suspicious['표본수'].sum():,}")

    print("\n의심 건물의 주용도 분포 (상위 15개):")
    print(suspicious["주용도_대표"].value_counts().head(15).to_string())

    print("\n의심 건물 표본 20개 (총층수, 표본수, 층1_비율):")
    print(
        suspicious.sort_values("표본수", ascending=False)
        .head(20)[["PNU19", "총층수", "주용도_대표", "표본수", "층1_비율"]]
        .to_string(index=False)
    )

    out = config.out_path("building_floor1_ratio.csv")
    building_ratio.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
