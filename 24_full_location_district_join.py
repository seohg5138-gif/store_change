import pandas as pd
import geopandas as gpd

# =========================================================================
# 24번: 전체 위치(442,528개, 관측분기수 제한 없음)에 좌표 + 상권코드를 결합.
# =========================================================================
# 지금까지(17~23번)는 "고교체 위치"만 골라서 상권과 결합했음.
# 여기서는 그 필터를 걷어내고 location_metrics_corrected.csv 전체를 대상으로
# 상권 결합을 다시 수행한다. 좌표는 여전히 18/19번이 만든
# location_coords.csv를 그대로 읽는다 (복구 로직 재작성 금지 — 단일 소스 원칙 유지).
#
# 주의: 여기서 만드는 산출물은 "고교체 여부"와 무관하게 전체 위치를 담으므로,
# 이후 25~27번이 상권/업종 기준선을 계산할 때 순환논리 없이 쓸 수 있는
# 비교군(comparison population)이 된다.

df_loc = pd.read_csv("location_metrics_corrected.csv")
location_coords = pd.read_csv("location_coords.csv")
# 22번 산출물 컬럼: 위치ID, 첫분기, 끝분기, 관측span분기수, 공실분기수, 공실률
# -> 25번이 "관측기간" 집계에 쓸 첫분기/끝분기까지 전부 가져온다 (공실률만
#    가져오면 25번에서 KeyError).
vac = pd.read_csv("22_location_vacancy.csv")[["위치ID", "첫분기", "끝분기", "공실률"]]

# --- 위치ID별 업종 정보 산출: "대표업종"(참고/표시용) + "업종비중"(통계용) ---
# location_metrics_corrected.csv에는 업종 컬럼이 없음(위치ID, 관측분기수,
# 교체횟수, 평균생존분기수, 최장생존분기수, 교체율뿐). 업종은 상가업소번호
# 단위로 shop_period_raw.parquet에만 있음.
#
# 중요: 한 위치가 여러 업종을 거쳐갔을 수 있음(예: 숙박 2분기 -> 음식 2분기
# -> 교육 2분기). 이걸 "가장 오래 차지한 업종 1개"로만 뽑으면(대표업종),
# 동률이거나 고르게 섞인 위치는 임의로 하나에 쏠려버려서 26번의 업종별
# 집계에서 위치수/고교체위치수 총합이 전체 총합과 안 맞는 문제가 생김.
# 그래서 두 가지를 다 만든다:
#   - 대표업종: 최장 생존기간 업종 1개 (25번의 "업종구성_TOP3" 같은 설명용에만 씀)
#   - 업종비중(24_location_industry_shares.csv): 위치ID별로 각 업종이 차지한
#     생존분기수 비중(합계 1). 26번의 실제 통계 집계는 이걸로 가중해서
#     계산해야 위치수/고교체위치수 총합이 항상 정확하게 보존됨.
ep = pd.read_parquet(
    "tenancy_episode_corrected.parquet",
    columns=["위치ID", "상가업소번호", "생존분기수_보정"],
)
brand_industry = (
    pd.read_parquet("shop_period_raw.parquet", columns=["상가업소번호", "상권업종대분류명"])
    .drop_duplicates(subset="상가업소번호", keep="first")
)
ep_industry = ep.merge(brand_industry, on="상가업소번호", how="left")

industry_duration = (
    ep_industry.groupby(["위치ID", "상권업종대분류명"])["생존분기수_보정"]
    .sum()
    .reset_index()
    .rename(columns={"상권업종대분류명": "업종"})
)

# 업종비중 = 위치ID 안에서 이 업종의 생존분기수 / 위치ID 전체 생존분기수
location_total_duration = industry_duration.groupby("위치ID")["생존분기수_보정"].transform("sum")
industry_duration["업종비중"] = industry_duration["생존분기수_보정"] / location_total_duration
industry_shares = industry_duration[["위치ID", "업종", "업종비중"]]
industry_shares.to_csv("24_location_industry_shares.csv", index=False, encoding="utf-8-sig")
print(f"위치별 업종비중 저장 완료: 24_location_industry_shares.csv "
      f"({industry_shares['위치ID'].nunique():,}개 위치, {len(industry_shares):,}행)")

dominant_industry = (
    industry_duration.sort_values("생존분기수_보정", ascending=False)
    .drop_duplicates(subset="위치ID", keep="first")[["위치ID", "업종"]]
    .rename(columns={"업종": "대표업종"})
)
missing_industry = df_loc["위치ID"].isin(dominant_industry["위치ID"]).eq(False).sum()
if missing_industry:
    print(f"⚠️ 대표업종 매칭 안 된 위치 {missing_industry}개 (대표업종=NaN으로 남음)")

# --- 좌표 결합 ---
full = df_loc.merge(location_coords, on="위치ID", how="left")
missing_coord = full["경도"].isna().sum()
if missing_coord:
    print(f"⚠️ 좌표 없는 위치 {missing_coord}개 - 상권 결합에서 제외됨")

# --- 공실률(+첫분기/끝분기) 결합 (22번 산출물, 전체 위치 기준이라 순환논리 없음) ---
full = full.merge(vac, on="위치ID", how="left")

# --- 대표업종 결합 ---
full = full.merge(dominant_industry, on="위치ID", how="left")

full_geo = full.dropna(subset=["경도", "위도"]).copy()

gdf_points = gpd.GeoDataFrame(
    full_geo,
    geometry=gpd.points_from_xy(full_geo["경도"], full_geo["위도"]),
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
joined_all = joined_all.drop(columns=["geometry", "index_right"])

match_rate = joined_all["TRDAR_CD"].notna().mean() * 100
print(f"전체 위치 상권 매칭률: {match_rate:.2f}% (대상 {len(joined_all):,}개)")

# 상권배정여부: TRDAR_CD가 없는 위치는 "오류 데이터"가 아니라 상권 폴리곤이
# 커버하지 않는 구역에 있는 정상 위치임(코드북 23번 항목에서 이미 확인됨:
# 매칭/미매칭 간 교체율 평균 차이 거의 없음 0.1217 vs 0.1222). 그래서 지우지
# 않고 플래그만 남겨서, 25~27번이 "상권외"도 하나의 비교군으로 다룰 수 있게 함.
joined_all["상권배정여부"] = joined_all["TRDAR_CD"].notna()
unassigned_n = (~joined_all["상권배정여부"]).sum()
print(f"  상권 미배정(커버리지 공백, 오류 아님): {unassigned_n:,}개 "
      f"-> 25~27번에서 '상권외' 비교군으로 유지됨")

# location_metrics_corrected.csv를 직접 읽는 구조라 17번의 자주바뀜여부
# 플래그는 여기 없음 -> 17번과 동일한 정의(관측분기수 15+ & 평균생존분기수 <
# 8분기)로 항상 새로 계산한다.
joined_all["자주바뀜여부"] = (
    (joined_all["관측분기수"] >= 15) & (joined_all["평균생존분기수"] < 8)
).astype(int)

joined_all.to_csv("24_full_location_with_district.csv", index=False, encoding="utf-8-sig")
print("저장 완료: 24_full_location_with_district.csv")
print(f"  전체 위치 수: {len(joined_all):,}")
print(f"  상권 매칭 성공: {joined_all['TRDAR_CD'].notna().sum():,}")
