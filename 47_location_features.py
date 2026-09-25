# -*- coding: utf-8 -*-
"""
47번: 위치 단위 물리적 특성 변수를 만든다. (발표자료 12p의 8개 변수)

| 변수          | 만드는 방법 (새 전처리 기준)                                                   |
|---------------|---------------------------------------------------------------------------------|
| 층구분        | 위치ID(PNU19_층_최종_슬롯) 파싱 -> 1층 / 2층이상 / 지하                          |
| 연면적        | 대표 건물의 연면적 (부속건축물 제외, 이상값 결측 처리)                         |
|               |   * 02번 building_raw에는 연면적 컬럼이 빠져 있어서 원본을 다시 읽는다          |
| 건물지상층수  | 대표 건물(필지 내 연면적 최대 주건축물)의 지상층수                               |
| 주용도        | 대표 건물의 주용도 (없으면 03번 주용도_대표), 희소 범주는 '기타'로 묶음        |
| 건물내위치수  | 같은 PNU19에 있는 위치ID 수 (24번 전체 위치 기준)                                |
| 주변상가수    | 반경 200m 안 위치 수 (24번 전체 위치 기준, 자기 자신 제외, BallTree haversine) |
| 지하철거리    | 가장 가까운 지하철역까지 직선거리(m)                                             |
| 상권유형      | TRDAR_SE_1 (골목/발달/전통시장/관광특구)                                         |

  * 이전 버전은 도로명주소로 건축물대장을 매칭했지만(96.8%), 새 전처리는 위치ID 자체가 PNU19를
    포함하고 '검증완료' 위치는 03번에서 이미 건축물대장 매칭이 확인된 것들이라 PNU19로 바로 붙인다.
  * 보조변수(모델 기본값에는 안 넣음, risk_common.EXTRA_FEATURES로 추가 가능):
    층번호, 건물지하층수, 건물연령, 동개수, 지하철역수_500m

입력:
  45_location_risk_base.csv, 24_full_location_with_district.csv,
  building_agg.parquet (03번), shop_period_raw.parquet (08번, 법정동코드 역매핑용),
  건축물대장 표제부 원본 CSV (00_config.BUILDING_PATH), 지하철역 좌표 CSV (risk_common.SUBWAY_PATH)
출력:
  47_location_features.csv, 47_feature_log.csv, 47_building_area.parquet
"""

import numpy as np
import pandas as pd

import risk_common as rc

RARE_CATEGORY_SHARE = 0.005   # 주용도 중 비중 0.5% 미만은 '기타'로 묶음
DENSITY_RADIUS_M = 200
SUBWAY_COUNT_RADIUS_M = 500


# ---------------------------------------------------------------------------
# 건축물대장: 연면적 / 지하층수 / 사용승인연도를 PNU19 단위로
# ---------------------------------------------------------------------------
def build_pnu19(code10, gb, bon, bu):
    """03번과 같은 PNU19 = 법정동코드10 + 대지구분1 + 본번4 + 부번4.
    법정동코드가 결측이면 float 승격으로 '.0'이 붙는 문제가 있었으므로(03번 주석) Int64로 고정."""
    c = pd.to_numeric(code10, errors="coerce").astype("Int64")
    g = pd.to_numeric(gb, errors="coerce").fillna(1).astype("Int64")
    b1 = pd.to_numeric(bon, errors="coerce").fillna(0).astype("Int64")
    b2 = pd.to_numeric(bu, errors="coerce").fillna(0).astype("Int64")
    out = (c.astype(str).str.zfill(10) + g.astype(str)
           + b1.astype(str).str.zfill(4) + b2.astype(str).str.zfill(4))
    out[c.isna()] = np.nan
    return out


def load_building_area(t0):
    """필지(PNU19)마다 '대표 건물' 한 동을 골라 그 동의 연면적/층수/연령/주용도를 쓴다.

    규칙:
      1) 부속건축물(창고, 경비실 등)은 제외하고 주건축물만 사용 (주부속구분코드명)
      2) 연면적 이상값 정리
         - 100만㎡ 초과 -> 결측 (서울 최대 단일 건물이 약 80만㎡)
         - 연면적 > 건축면적 x (지상+지하층수) x 3 -> 결측
           (한 층 바닥이 건축면적의 3배를 넘는 건 물리적으로 불가능 -> 입력 오류로 판단)
      3) 필지 안 주건축물 중 연면적이 가장 큰 동 = 대표 건물
         연면적 / 지상층수 / 지하층수 / 사용승인연도 / 주용도를 모두 이 한 줄에서 가져옴
         (이전 버전은 연면적=최대 동, 연령=중앙값, 층수=03번 최댓값처럼 서로 다른 동에서 가져와서 어긋났음)
      4) 필지 전체 연면적 합계와 주건축물 동수는 보조 변수로 남김
    """
    rc.log("법정동코드 역매핑표 생성 (shop_period_raw의 PNU19 앞 10자리 사용)...")
    raw_codes = pd.read_parquet("shop_period_raw.parquet", columns=["PNU19", "시군구명", "법정동명"])
    raw_codes = raw_codes.dropna().drop_duplicates()
    raw_codes["법정동코드"] = raw_codes["PNU19"].astype(str).str[:10]
    code_map = (
        raw_codes.groupby(["시군구명", "법정동명"])["법정동코드"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    rc.log(f"역매핑표 {len(code_map):,}개 (시군구, 법정동)", t0)

    rc.log("건축물대장 원본 로딩...")
    head = rc.read_csv_any(rc.BUILDING_PATH, nrows=3)
    want = ["대지위치", "시군구코드명", "법정동코드명", "대지구분코드명", "주지번", "부지번",
            "대장구분코드명", "주부속구분코드명", "주용도코드명", "건축면적", "연면적",
            "지상층수", "지하층수", "사용승인일자"]
    use = [c for c in want if c in head.columns]
    missing = sorted(set(want) - set(use))
    if missing:
        print(f"  [주의] 건축물대장에 없는 컬럼: {missing}")
    b = rc.read_csv_any(rc.BUILDING_PATH, usecols=use, dtype={"사용승인일자": str})
    n_all = len(b)
    rc.log(f"건축물대장 {n_all:,}행 로딩", t0)

    # --- PNU19 ---
    b["시군구명_정리"] = b["시군구코드명"].astype(str).str.replace("서울특별시 ", "", regex=False).str.strip()
    keys = pd.MultiIndex.from_arrays([b["시군구명_정리"], b["법정동코드명"]])
    b["법정동코드_추정"] = code_map.reindex(keys).values
    b["대지구분"] = b["대지구분코드명"].map({"대지": 1, "산": 2}).fillna(1)
    b["PNU19"] = build_pnu19(b["법정동코드_추정"], b["대지구분"], b["주지번"], b["부지번"])
    print(f"  PNU19 생성 성공: {b['PNU19'].notna().mean()*100:.2f}%")
    b = b.dropna(subset=["PNU19"])

    # --- 1) 주건축물만 ---
    if "주부속구분코드명" in b.columns:
        print("  [주부속구분 분포] " + b["주부속구분코드명"].value_counts(dropna=False).to_dict().__str__())
        b = b[b["주부속구분코드명"].fillna("주건축물").str.contains("주")].copy()
        print(f"  주건축물만 남김: {len(b):,}행")

    # --- 숫자 변환 ---
    for c in ["연면적", "건축면적", "지상층수", "지하층수"]:
        b[c] = pd.to_numeric(b[c], errors="coerce") if c in b.columns else np.nan
    b.loc[b["연면적"] <= 0, "연면적"] = np.nan
    yr = pd.to_numeric(b["사용승인일자"].astype(str).str.replace(r"\D", "", regex=True).str[:4],
                       errors="coerce") if "사용승인일자" in b.columns else pd.Series(np.nan, index=b.index)
    b["사용승인연도"] = yr.where(yr.between(1900, 2026))
    print(f"  사용승인연도 추출 성공: {b['사용승인연도'].notna().mean()*100:.2f}%")

    # --- 2) 연면적 이상값 ---
    floors = (b["지상층수"].fillna(0) + b["지하층수"].fillna(0)).clip(lower=1)
    cap = b["건축면적"] * floors * 3
    too_big = b["연면적"] > 1_000_000
    impossible = (b["연면적"] > cap) & b["건축면적"].gt(0)
    print(f"  연면적 이상값: 100만㎡ 초과 {too_big.sum():,}행 / 건축면적x층수x3 초과 {impossible.sum():,}행 -> 결측 처리")
    show = [c for c in ["대지위치", "대장구분코드명", "주용도코드명", "건축면적", "지상층수", "지하층수", "연면적"]
            if c in b.columns]
    if (too_big | impossible).any():
        print("  [결측 처리된 연면적 상위 10행]")
        print(b.loc[too_big | impossible, show].nlargest(10, "연면적").to_string(index=False))
    b.loc[too_big | impossible, "연면적"] = np.nan

    # --- 3) 대표 건물 (필지 내 연면적 최대 주건축물) ---
    b["_정렬"] = b["연면적"].fillna(-1)
    rep = b.sort_values(["PNU19", "_정렬"], ascending=[True, False]).drop_duplicates("PNU19")
    agg = rep.set_index("PNU19")[["연면적", "지상층수", "지하층수", "사용승인연도", "주용도코드명"]].rename(
        columns={"지상층수": "건물지상층수", "지하층수": "건물지하층수", "주용도코드명": "주용도_대표건물"})

    # --- 4) 보조: 필지 합계 / 주건축물 동수 ---
    extra = b.groupby("PNU19").agg(연면적_필지합계=("연면적", lambda s: s.sum(min_count=1)),
                                   주건축물동수=("연면적", "size"))
    agg = agg.join(extra)

    agg.to_parquet("47_building_area.parquet")
    print(f"[완료] 저장: 47_building_area.parquet ({len(agg):,}개 필지)")
    print("  [대표 건물 연면적 분포] " + agg["연면적"].describe(percentiles=[.5, .99, .999]).round(1).to_dict().__str__())

    ba = pd.read_parquet("building_agg.parquet")
    overlap = ba.index.astype(str).isin(agg.index).mean()
    print(f"  03번 building_agg PNU19 중 이번 집계에서도 찾아지는 비율: {overlap*100:.2f}%")
    if overlap < 0.9:
        print("[경고] 90% 미만입니다. PNU19 조립 규칙 또는 주건축물 필터를 확인하세요.")
    return agg


# ---------------------------------------------------------------------------
# 지하철역
# ---------------------------------------------------------------------------
def load_subway():
    st = rc.read_csv_any(rc.SUBWAY_PATH)
    cols = list(st.columns)

    print(f"  지하철역 CSV 컬럼: {cols}")

    def pick(cands):
        # 후보 순서가 우선순위: '위도'가 있으면 'Y좌표'보다 먼저 잡는다
        for k in cands:
            for c in cols:
                if k in c.lower():
                    return c
        return None

    lat_c = rc.SUBWAY_LAT_COL or pick(["위도", "lat", "y좌표", "ycoord"])
    lon_c = rc.SUBWAY_LON_COL or pick(["경도", "lon", "lng", "x좌표", "xcoord"])
    name_c = rc.SUBWAY_NAME_COL or pick(["역사명", "역명", "역이름", "역한글명칭", "명칭", "station", "name"])
    if lat_c is None or lon_c is None:
        raise ValueError(f"지하철역 CSV에서 위도/경도 컬럼을 못 찾았습니다: {cols}")
    st = st.rename(columns={lat_c: "위도", lon_c: "경도"})
    st["위도"] = pd.to_numeric(st["위도"], errors="coerce")
    st["경도"] = pd.to_numeric(st["경도"], errors="coerce")
    # 위도/경도가 뒤바뀐 파일 대비
    if st["위도"].median() > 90:
        st[["위도", "경도"]] = st[["경도", "위도"]].values
    st["역명"] = st[name_c].astype(str) if name_c else "역" + st.index.astype(str)
    ok = st["경도"].between(126.5, 127.5) & st["위도"].between(37.2, 37.9)
    st = st.loc[ok, ["역명", "위도", "경도"]].reset_index(drop=True)
    # 환승역은 호선마다 행이 따로 있고 좌표가 같거나 거의 같음 -> 역명이 없어도 좌표(약 10m 격자)로 한 역으로 묶음
    st["_key"] = st["위도"].round(4).astype(str) + "_" + st["경도"].round(4).astype(str)
    n_unique = st["_key"].nunique()
    print(f"  사용 컬럼: 위도={lat_c}, 경도={lon_c}, 역명={name_c}")
    print(f"  서울 인근 좌표 {len(st):,}행 -> 좌표 기준 고유 역 {n_unique:,}개 (환승역 중복 제거)")
    if name_c is None:
        print("[경고] 역명 컬럼을 못 찾았습니다. 거리 계산에는 문제없지만, 지도에 최근접역 이름을 띄우려면 "
              "risk_common.SUBWAY_NAME_COL에 컬럼명을 적어주세요.")
    return st


# ---------------------------------------------------------------------------
def main():
    t0 = rc.start_timer()
    base = pd.read_csv("45_location_risk_base.csv", dtype={"TRDAR_CD": str}, low_memory=False)
    universe = pd.read_csv("24_full_location_with_district.csv",
                           usecols=["위치ID", "경도", "위도"], low_memory=False)
    universe["PNU19"] = rc.parse_location_id(universe["위치ID"])["PNU19"]
    rc.log(f"분석대상 {len(base):,}개 / 전체 위치(밀도 계산용) {len(universe):,}개", t0)

    feat = base[["위치ID", "PNU19", "층_최종", "경도", "위도", "TRDAR_SE_1"]].copy()
    feat["PNU19"] = feat["PNU19"].astype(str)
    log_rows = []

    # 1) 층구분
    feat["층구분"] = feat["층_최종"].apply(rc.floor_category)
    feat["층번호"] = feat["층_최종"].apply(rc.floor_number)
    feat.loc[(feat["층번호"] < -10) | (feat["층번호"] > 130), "층번호"] = np.nan
    print("\n[층구분 분포]")
    print(feat["층구분"].value_counts().to_string())

    # 2~4) 건물 정보: 47번 대표 건물 기준 (연면적/층수/연령/주용도를 같은 동에서)
    ba = pd.read_parquet("building_agg.parquet")
    ba.index = ba.index.astype(str)
    area = load_building_area(t0)
    feat["연면적"] = feat["PNU19"].map(area["연면적"])
    feat["연면적_필지합계"] = feat["PNU19"].map(area["연면적_필지합계"])
    feat["건물지상층수"] = feat["PNU19"].map(area["건물지상층수"])
    feat["건물지하층수"] = feat["PNU19"].map(area["건물지하층수"])
    feat["건물연령"] = (2025 - feat["PNU19"].map(area["사용승인연도"])).clip(lower=0)
    feat["동개수"] = feat["PNU19"].map(area["주건축물동수"])
    feat["주용도_원본"] = feat["PNU19"].map(area["주용도_대표건물"])
    # 대표 건물 정보가 비면 03번 값으로 보완
    feat["건물지상층수"] = feat["건물지상층수"].fillna(feat["PNU19"].map(ba["총층수"]))
    feat["주용도_원본"] = feat["주용도_원본"].fillna(feat["PNU19"].map(ba["주용도_대표"]))
    feat.loc[feat["건물지상층수"] <= 0, "건물지상층수"] = np.nan

    share = feat["주용도_원본"].value_counts(normalize=True)
    keep_uses = share[share >= RARE_CATEGORY_SHARE].index
    feat["주용도"] = np.where(feat["주용도_원본"].isin(keep_uses), feat["주용도_원본"],
                           np.where(feat["주용도_원본"].isna(), "정보없음", "기타"))
    print(f"\n[주용도] 원본 {feat['주용도_원본'].nunique()}개 범주 -> {feat['주용도'].nunique()}개 "
          f"(비중 {RARE_CATEGORY_SHARE*100:.1f}% 미만은 '기타')")

    # 5) 건물내위치수 (24번 전체 위치 기준)
    per_building = universe.groupby("PNU19")["위치ID"].size()
    feat["건물내위치수"] = feat["PNU19"].map(per_building)

    # 6) 주변상가수 200m (자기 자신 제외)
    uni = universe.dropna(subset=["경도", "위도"])
    has_xy = feat["경도"].notna() & feat["위도"].notna()
    rc.log("BallTree 생성 (주변상가수)...", t0)
    tree = rc.build_balltree(uni["위도"].values, uni["경도"].values)
    cnt = rc.count_within(tree, feat.loc[has_xy, "위도"].values, feat.loc[has_xy, "경도"].values,
                          DENSITY_RADIUS_M)
    feat.loc[has_xy, "주변상가수_200m"] = cnt - 1
    rc.log("주변상가수 계산 완료", t0)

    # 7) 지하철
    st = load_subway()
    st_tree = rc.build_balltree(st["위도"].values, st["경도"].values)
    d, idx = rc.nearest_distance_m(st_tree, feat.loc[has_xy, "위도"].values, feat.loc[has_xy, "경도"].values)
    feat.loc[has_xy, "지하철거리_m"] = d
    feat.loc[has_xy, "최근접역"] = st["역명"].values[idx]
    st_unique = st.drop_duplicates("_key")
    st_tree_u = rc.build_balltree(st_unique["위도"].values, st_unique["경도"].values)
    feat.loc[has_xy, "지하철역수_500m"] = rc.count_within(
        st_tree_u, feat.loc[has_xy, "위도"].values, feat.loc[has_xy, "경도"].values, SUBWAY_COUNT_RADIUS_M)
    rc.log("지하철 거리 계산 완료", t0)

    # 8) 상권유형
    feat["상권유형"] = feat["TRDAR_SE_1"].fillna("정보없음")

    # --- 결측/매칭 로그 ---
    all_vars = rc.CAT_FEATURES + rc.NUM_FEATURES + ["연면적_필지합계", "층번호", "건물지하층수", "건물연령", "동개수", "지하철역수_500m"]
    for v in all_vars:
        s = feat[v]
        missing = s.isna() | s.isin(["정보없음", "층정보없음"])
        row = dict(변수=v, 결측수=int(missing.sum()), 채움률=1 - missing.mean())
        if pd.api.types.is_numeric_dtype(s):
            row.update(최소=s.min(), 중앙값=s.median(), 최대=s.max())
        else:
            row.update(범주수=s.nunique())
        log_rows.append(row)
    flog = pd.DataFrame(log_rows)
    rc.print_table(flog, "변수별 채움률")

    out_cols = ["위치ID", "층구분", "층번호", "주용도", "주용도_원본", "연면적", "연면적_필지합계", "건물지상층수", "건물지하층수",
                "건물연령", "동개수", "건물내위치수", "주변상가수_200m", "지하철거리_m", "최근접역",
                "지하철역수_500m", "상권유형"]
    rc.save_csv(feat[out_cols], "47_location_features.csv")
    rc.save_csv(flog, "47_feature_log.csv")
    rc.log("47번 완료", t0)


if __name__ == "__main__":
    main()
