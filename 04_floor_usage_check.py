# -*- coding: utf-8 -*-
"""
4단계: 주용도(주거계열 vs 근린생활시설 등)별로 층정보='1' 비율을 확인한다.

이전에 pandas groupby/assign에서 MemoryError가 났던 부분을,
duckdb로 parquet 파일을 직접 쿼리해서 해결한다 (전체를 메모리에 안 올림).

실행:
    python 04_floor_usage_check.py

출력:
    intermediate/floor_usage_ratio.csv
    콘솔에 주거계열 vs 근린생활시설 비교 결과 출력
"""

import os
import sys

import duckdb
import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")


def main():
    con = duckdb.connect()

    df_all_path = config.out_path("df_all_pnu.parquet")
    building_agg_path = config.out_path("building_agg.parquet")

    query = f"""
        SELECT
            b.주용도_대표 AS 주용도,
            AVG(CASE WHEN a.층정보 = '1' THEN 1.0 ELSE 0.0 END) * 100 AS 층1_비율,
            COUNT(*) AS 표본수
        FROM read_parquet('{df_all_path}') a
        JOIN read_parquet('{building_agg_path}') b
          ON a.PNU19 = b.PNU19
        WHERE a.건물정보_매칭여부 = true
        GROUP BY b.주용도_대표
        ORDER BY 표본수 DESC
    """
    ratio = con.execute(query).df()
    print(ratio.head(25).to_string(index=False))

    out = config.out_path("floor_usage_ratio.csv")
    ratio.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out}")

    residential = ["단독주택", "공동주택"]
    commercial = ["제1종근린생활시설", "제2종근린생활시설", "업무시설"]

    res_ratio = ratio.loc[ratio["주용도"].isin(residential), "층1_비율"].mean()
    com_ratio = ratio.loc[ratio["주용도"].isin(commercial), "층1_비율"].mean()
    print(f"\n주거계열(단독+공동주택) 평균 층1 비율: {res_ratio:.2f}%")
    print(f"근린생활+업무시설 평균 층1 비율: {com_ratio:.2f}%")
    print(
        "\n※ 이 결과에서 주거계열이 근린생활시설보다 뚜렷이 높게 나오면, "
        "building_master의 층정보_신뢰가능여부 규칙에 '주거계열 건물은 불가'를 "
        "추가할지 검토하세요. (05번 스크립트에서 RESIDENTIAL_UNRELIABLE 플래그로 반영 가능)"
    )


if __name__ == "__main__":
    main()
