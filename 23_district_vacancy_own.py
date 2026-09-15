import pandas as pd
import geopandas as gpd

# =========================================================================
# 23번: 22번(위치별 공실)을 상권/구 단위로 집계하고, 20번(자주 바뀌는 위치)과
# 합쳐서 "경쟁심화형" vs "침체형"을 분류한다.
# =========================================================================
# 주의: '상권 공실률'은 자주 바뀌는 위치들만으로 계산하면 순환논리가 되므로,
# 반드시 '전체 신뢰가능 위치'를 대상으로 상권 공실률 기준선을 먼저 만든 뒤,
# 그 기준선을 자주 바뀌는 위치 목록에 갖다 붙이는 순서로 진행한다.
#
# 좌표는 18번(+19번)이 만든 location_coords.csv를 그대로 읽는다. 좌표 복구
# 로직은 여기 없다 - 18번 하나에만 있고, 여기는 그 결과만 갖다 쓴다.
# 실행 순서: 18번(+19번), 20번, 22번을 먼저 돌린 뒤 이 스크립트를 돌릴 것.

vac = pd.read_csv("22_location_vacancy.csv")
location_coords = pd.read_csv("location_coords.csv")

vac_geo = vac.merge(location_coords, on="위치ID", how="left")
missing_coord = vac_geo["경도"].isna().sum()
if missing_coord:
    print(f"⚠️ 좌표 없는 위치 {missing_coord}개 - 상권 결합에서 제외됨")
vac_geo = vac_geo.dropna(subset=["경도", "위도"])

gdf_points = gpd.GeoDataFrame(
    vac_geo,
    geometry=gpd.points_from_xy(vac_geo["경도"], vac_geo["위도"]),
    crs="EPSG:4326",
).to_crs(epsg=5181)

poly = gpd.read_file(
    r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\area\서울시 상권분석서비스(영역-상권).shp",
    encoding="utf-8",
).to_crs(epsg=5181)
poly = poly[["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "geometry"]]
poly["TRDAR_CD"] = poly["TRDAR_CD"].astype(str)

joined_all = gpd.sjoin(gdf_points, poly, how="left", predicate="within")
joined_all = joined_all.drop_duplicates(subset="위치ID", keep="first")

print(f"전체 위치 상권 매칭률: {joined_all['TRDAR_CD'].notna().mean()*100:.2f}%")

# --- 상권(TRDAR_CD) 단위 공실률 기준선 (전체 위치 기준) ---
district_baseline = (
    joined_all.dropna(subset=["TRDAR_CD"])
    .groupby(["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])
    .agg(전체위치수=("위치ID", "size"), 상권평균공실률=("공실률", "mean"))
    .reset_index()
)
district_baseline.to_csv("23_district_vacancy_baseline.csv", index=False, encoding="utf-8-sig")
print(f"저장 완료: 23_district_vacancy_baseline.csv ({len(district_baseline):,}개 상권)")

# --- 20번(자주 바뀌는 위치) 불러와서 상권 공실률 기준선 붙이기 ---
frequent = pd.read_csv("20_frequent_change_with_district.csv", dtype={"TRDAR_CD": str})

merged = frequent.merge(
    district_baseline[["TRDAR_CD", "전체위치수", "상권평균공실률"]],
    on="TRDAR_CD", how="left",
)

median_vacancy = district_baseline["상권평균공실률"].median()
merged["유형추정"] = merged["상권평균공실률"].apply(
    lambda x: "침체 의심(공실 높음)" if pd.notna(x) and x >= median_vacancy
    else ("경쟁심화 의심(공실 낮음)" if pd.notna(x) else "미분류")
)

print(f"\n공실률 중앙값(분류 기준선): {median_vacancy:.4f}")

summary = (
    merged.groupby(["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "유형추정"])
    .agg(자주바뀌는위치수=("위치ID", "size"), 상권평균공실률=("상권평균공실률", "first"))
    .reset_index()
    .sort_values("자주바뀌는위치수", ascending=False)
)

print("\n=== 자주 바뀌는 위치가 많은 상권 TOP 20 + 유형 추정 ===")
print(summary.head(20)[
    ["TRDAR_CD_N", "TRDAR_SE_1", "자주바뀌는위치수", "상권평균공실률", "유형추정"]
].to_string(index=False))

print("\n=== 유형별 상권-위치 개수 ===")
print(merged["유형추정"].value_counts())

merged.to_csv("23_frequent_with_vacancy_type.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 23_frequent_with_vacancy_type.csv")

# --- 구 단위 요약도 같이 ---
gu_summary = (
    merged.dropna(subset=["상권평균공실률"])
    .groupby("시군구명")
    .agg(자주바뀌는위치수=("위치ID", "size"), 평균공실률=("상권평균공실률", "mean"))
    .sort_values("평균공실률", ascending=False)
)
print("\n=== 구별 평균 공실률 (자주 바뀌는 위치가 속한 상권 기준) ===")
print(gu_summary.round(4))
gu_summary.to_csv("23_gu_vacancy_summary.csv", encoding="utf-8-sig")
print("저장 완료: 23_gu_vacancy_summary.csv")
