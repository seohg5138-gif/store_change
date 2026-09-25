# -*- coding: utf-8 -*-
"""
50번: 창업 지도(03 페이지) 데이터를 만든다. (발표자료 20~27p)

지도 동작: 상권 경계 지도 -> 상권 클릭 -> 그 상권 위치 마커(위험도 색) -> 마커 클릭 -> 상세
그래서 파일을 두 층으로 나눈다 (30만 개 위치를 한 번에 올리면 브라우저가 버티지 못함):
  map_data/districts.geojson        상권 폴리곤 + 상권별 위치수/등급별 위치수/평균 위험도
  map_data/loc/{TRDAR_CD}.json      상권 클릭 시 불러올 위치 목록 (상세정보 포함)

위치 상세(마커 클릭)에 들어가는 것:
  - 도로명주소, 법정동, 위험도(0~100) + 위험등급
  - 물리 정보: 층구분 / 상권유형 / 주용도 / 연면적 / 지상층수 / 지하철거리(최근접역)
  - 10년 업종 이력: episode(입점 구간)별 시작~종료 분기, 버틴 분기수, 업종, 상호명, 영업중 여부
  - 추천 업종: "비슷한 조건"(층구분 x 연면적구간 x 주용도) 위치들에서 2년 이상 버틴 비율이 높은 업종 상위 3개
      * 2년 생존율은 "관측 기간 안에 8분기를 채울 수 있었던" episode만으로 계산 (최근 입점은 제외)
        + 좌측절단 episode 제외 -> 영업 중인 가게 때문에 생기는 편향을 줄임
      * 해당 조합에 표본(업종별 episode MIN_REC_EPISODES개 이상)이 없으면 층구분만으로 대체
  - 같은 상권·같은 층구분에서 위험도가 가장 낮은 위치 3곳 (비교 기능용)

위험등급 (발표자료 26p): 매우위험 >65% / 위험 50~65 / 보통 35~50 / 안전 20~35 / 매우안전 <=20
  * 위험도는 48번 out-of-fold 예측값 (학습에 쓰인 위치를 자기 자신으로 예측한 과장된 값이 아님)

입력:  45_location_risk_base.csv, 47_location_features.csv, 48_oof_predictions.csv,
       tenancy_episode_corrected.parquet, shop_period_raw.parquet, 상권 shp (risk_common.SHP_PATH)
출력:  map_data/districts.geojson, map_data/loc/*.json, map_data/meta.json,
       50_recommend_industry.csv
"""

import json
import os
from importlib import import_module

import numpy as np
import pandas as pd

import risk_common as rc

rp = import_module("49_risk_profile")  # 연면적 구간 정의를 49번과 공유

MIN_REC_EPISODES = 30
TOP_REC = 3
TOP_SIMILAR = 3
OUT_DIR = "map_data"


def r(x, nd=1):
    return None if pd.isna(x) else round(float(x), nd)


def main():
    t0 = rc.start_timer()
    os.makedirs(os.path.join(OUT_DIR, "loc"), exist_ok=True)

    base = pd.read_csv("45_location_risk_base.csv", dtype={"TRDAR_CD": str}, low_memory=False)
    feat = pd.read_csv("47_location_features.csv", low_memory=False)
    pred = pd.read_csv("48_oof_predictions.csv", usecols=["위치ID", "위험도", "위험등급"])
    df = base.merge(feat, on="위치ID", how="inner").merge(pred, on="위치ID", how="inner")
    df["TRDAR_CD"] = rc.normalize_trdar(df["TRDAR_CD"])
    df["연면적구간"] = rp.binned(df, "연면적")
    rc.log(f"지도 대상 위치 {len(df):,}개", t0)

    # ------------------------------------------------------------------ episode + 업종/상호
    ep = pd.read_parquet(
        "tenancy_episode_corrected.parquet",
        columns=["위치ID", "상가업소번호", "시작분기", "종료분기", "생존분기수_보정", "생존여부"],
    )
    ep = ep[ep["위치ID"].isin(set(df["위치ID"]))].copy()
    brand = (
        pd.read_parquet("shop_period_raw.parquet",
                        columns=["상가업소번호", "상호명", "상권업종대분류명", "상권업종중분류명"])
        .drop_duplicates(subset="상가업소번호", keep="last")
    )
    ep = ep.merge(brand, on="상가업소번호", how="left")
    dong = (
        pd.read_parquet("shop_period_raw.parquet", columns=["위치ID", "법정동명"])
        .drop_duplicates(subset="위치ID", keep="last")
    )
    df = df.merge(dong, on="위치ID", how="left")
    rc.log(f"episode {len(ep):,}개에 업종/상호 결합", t0)

    # ------------------------------------------------------------------ 추천 업종
    grid = sorted(set(ep["시작분기"]) | set(ep["종료분기"]))
    pos = {p: i for i, p in enumerate(grid)}
    ep["좌측절단"] = ep["시작분기"] == grid[0]
    ep["2년관측가능"] = ep["시작분기"].map(pos) <= len(grid) - rc.SURVIVAL_2Y_QUARTERS
    ep["2년생존"] = (ep["생존분기수_보정"] >= rc.SURVIVAL_2Y_QUARTERS).astype(int)
    rec_ep = ep[(~ep["좌측절단"]) & ep["2년관측가능"]].merge(
        df[["위치ID", "층구분", "연면적구간", "주용도"]], on="위치ID", how="inner")

    def top_industries(keys):
        g = (rec_ep.groupby(keys + ["상권업종중분류명"])["2년생존"]
             .agg(episode수="size", 생존율="mean").reset_index())
        g = g[g["episode수"] >= MIN_REC_EPISODES]
        g = g.sort_values(keys + ["생존율", "episode수"], ascending=[True] * len(keys) + [False, False])
        return g.groupby(keys).head(TOP_REC)

    rec_full = top_industries(["층구분", "연면적구간", "주용도"])
    rec_floor = top_industries(["층구분"])
    rc.save_csv(rec_full, "50_recommend_industry.csv")

    rec_full_map = {k: v[["상권업종중분류명", "생존율", "episode수"]].values.tolist()
                    for k, v in rec_full.groupby(["층구분", "연면적구간", "주용도"])}
    rec_floor_map = {k if isinstance(k, str) else k[0]: v[["상권업종중분류명", "생존율", "episode수"]].values.tolist()
                     for k, v in rec_floor.groupby("층구분")}

    # ------------------------------------------------------------------ 이력
    ep = ep.sort_values(["위치ID", "시작분기"])
    hist_map = {}
    for loc_id, g in ep.groupby("위치ID", sort=False):
        hist_map[loc_id] = [
            dict(시작=int(a), 종료=int(b), 분기수=int(c), 대분류=d if pd.notna(d) else None,
                 중분류=e if pd.notna(e) else None, 상호=f if pd.notna(f) else None, 영업중=(h == "생존중"))
            for a, b, c, d, e, f, h in zip(g["시작분기"], g["종료분기"], g["생존분기수_보정"],
                                            g["상권업종대분류명"], g["상권업종중분류명"], g["상호명"], g["생존여부"])
        ]
    rc.log("위치별 10년 이력 생성 완료", t0)

    # ------------------------------------------------------------------ 상권별 위치 JSON
    n_files = 0
    for cd, g in df.groupby("TRDAR_CD"):
        g = g.sort_values("위험도")
        safest = {fl: sub.head(TOP_SIMILAR + 1)[["위치ID", "도로명주소", "위험도"]].values.tolist()
                  for fl, sub in g.groupby("층구분")}
        records = []
        for row in g.itertuples(index=False):
            key = (row.층구분, row.연면적구간, row.주용도)
            rec = rec_full_map.get(key)
            rec_basis = "층구분x연면적x주용도"
            if not rec:
                rec, rec_basis = rec_floor_map.get(row.층구분, []), "층구분"
            similar = [dict(위치ID=a, 주소=b, 위험도=r(c * 100))
                       for a, b, c in safest.get(row.층구분, []) if a != row.위치ID][:TOP_SIMILAR]
            records.append(dict(
                id=row.위치ID, lat=r(row.위도, 6), lon=r(row.경도, 6),
                주소=row.도로명주소, 법정동=getattr(row, "법정동명", None), 시군구=row.시군구명,
                위험도=r(row.위험도 * 100), 위험등급=row.위험등급, 실제위험여부=int(row.위험여부),
                층구분=row.층구분, 층=row.층_최종, 상권유형=row.상권유형, 주용도=row.주용도_원본,
                연면적=r(row.연면적, 0), 지상층수=r(row.건물지상층수, 0), 건물내위치수=r(row.건물내위치수, 0),
                지하철거리=r(row.지하철거리_m, 0), 최근접역=row.최근접역,
                교체횟수=int(row.교체횟수), 평균생존분기=r(row.평균생존분기수, 1), 관측분기=int(row.관측분기수),
                이력=hist_map.get(row.위치ID, []),
                추천업종=[dict(업종=a, 생존율2년=r(b * 100), 표본=int(c)) for a, b, c in rec],
                추천기준=rec_basis,
                비교후보=similar,
            ))
        with open(os.path.join(OUT_DIR, "loc", f"{cd}.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
        n_files += 1
    rc.log(f"상권별 위치 JSON {n_files:,}개 저장", t0)

    # ------------------------------------------------------------------ 상권 GeoJSON
    import geopandas as gpd

    dist = df.groupby("TRDAR_CD").agg(
        상권명=("TRDAR_CD_N", "first"), 상권유형=("TRDAR_SE_1", "first"),
        위치수=("위치ID", "size"), 평균위험도=("위험도", "mean"),
        상권평균교체율=("상권평균교체율", "first"),
    )
    grade_cnt = pd.crosstab(df["TRDAR_CD"], df["위험등급"]).reindex(columns=rc.GRADE_LABELS, fill_value=0)
    dist = dist.join(grade_cnt)
    dist["위험위치수"] = dist["매우위험"] + dist["위험"]
    dist["안전위치수"] = dist["안전"] + dist["매우안전"]

    poly = gpd.read_file(rc.SHP_PATH, encoding="utf-8")[["TRDAR_CD", "geometry"]]
    poly["TRDAR_CD"] = rc.normalize_trdar(poly["TRDAR_CD"])
    poly = poly.merge(dist.reset_index(), on="TRDAR_CD", how="inner")
    poly = poly.to_crs(epsg=5181)
    poly["geometry"] = poly.geometry.simplify(5)          # 5m 단순화 (파일 크기 절감)
    poly = poly.to_crs(epsg=4326)
    for c in ["평균위험도"]:
        poly[c] = (poly[c] * 100).round(1)
    poly["상권평균교체율"] = poly["상권평균교체율"].round(4)
    poly.to_file(os.path.join(OUT_DIR, "districts.geojson"), driver="GeoJSON")
    print(f"[완료] 저장: {OUT_DIR}/districts.geojson ({len(poly):,}개 상권)")

    meta = dict(
        위험등급기준={lab: [rc.GRADE_BINS[i] * 100, rc.GRADE_BINS[i + 1] * 100] for i, lab in enumerate(rc.GRADE_LABELS)},
        위치수=int(len(df)), 상권수=int(len(poly)),
        위험지표=f"상권대비{rc.RISK_BASIS} 상위 {int(round((1 - rc.RISK_QUANTILE) * 100))}%",
        주의="위험도는 물리적 입지 조건만 반영한 참고 지표. 임대료·유동인구·업종 선택은 포함되지 않음.",
    )
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[완료] 저장: {OUT_DIR}/meta.json")
    rc.log("50번 완료", t0)


if __name__ == "__main__":
    main()
