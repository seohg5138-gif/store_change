import pandas as pd
import geopandas as gpd

# =========================================================================
# 20번: 상권 폴리곤(TRDAR_CD) 결합
# =========================================================================
# 17번(자주 바뀌는 위치)에, 18번(+19번)이 만든 location_coords.csv의 좌표를
# 붙이고, 서울시 상권분석서비스 폴리곤과 point-in-polygon 조인한다.
# (주의: 폴리곤 CRS가 EPSG:5181이라 좌표를 반드시 재투영해야 함 - 이전에 검증한 그대로)
#
# 실행 순서: 반드시 18번(+19번) 다음에 돌릴 것. location_coords.csv가
# 없으면 여기서 바로 에러가 나므로, 순서를 건너뛰면 바로 알 수 있다.

frequent = pd.read_csv("17_frequent_change_locations.csv")
location_coords = pd.read_csv("location_coords.csv")

frequent_geo = frequent.merge(location_coords, on="위치ID", how="left")

missing_coord = frequent_geo["경도"].isna().sum()
if missing_coord:
    print(f"⚠️ 좌표 없는 위치 {missing_coord}개 - 상권 결합에서 제외됨")

gdf_points = gpd.GeoDataFrame(
    frequent_geo,
    geometry=gpd.points_from_xy(frequent_geo["경도"], frequent_geo["위도"]),
    crs="EPSG:4326",
).to_crs(epsg=5181)

poly = gpd.read_file(
    r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\area\서울시 상권분석서비스(영역-상권).shp",
    encoding="utf-8",
).to_crs(epsg=5181)
poly = poly[["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "geometry"]]

joined = gpd.sjoin(gdf_points, poly, how="left", predicate="within")
# 관광특구 등으로 폴리곤이 겹쳐 중복 매칭되는 경우가 있었으므로(이전 검증 결과),
# 위치ID당 첫 매칭만 남긴다.
joined = joined.drop_duplicates(subset="위치ID", keep="first")

matched_rate = joined["TRDAR_CD"].notna().mean()
print(f"상권 매칭률: {matched_rate*100:.2f}%")

joined.drop(columns="geometry").to_csv(
    "20_frequent_change_with_district.csv", index=False, encoding="utf-8-sig"
)
print("저장 완료: 20_frequent_change_with_district.csv")
