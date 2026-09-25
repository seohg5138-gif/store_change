import pandas as pd

# 1) 원본에서 모든 위치ID의 모든 분기 좌표를 가져옴 (전체 대상, 5291개에 한정 안 함 -
#    나중에 다른 이상치 발견 시 재사용할 수 있게 전체 위치 기준으로 짜둠)
raw_full = pd.read_parquet(
    "shop_period_raw.parquet", columns=["위치ID", "period", "경도", "위도"]
)

raw_full["정상범위"] = raw_full["경도"].between(126.7, 127.2) & raw_full["위도"].between(37.4, 37.7)

# 2) 위치ID별로 '정상범위'인 좌표 중 하나를 대표 좌표로 채택 (가장 최신 분기 우선)
valid_coords = (
    raw_full[raw_full["정상범위"]]
    .sort_values("period", ascending=False)
    .drop_duplicates(subset="위치ID", keep="first")
    [["위치ID", "경도", "위도"]]
    .rename(columns={"경도": "경도_복구", "위도": "위도_복구"})
)

print(f"복구 가능한 위치 수: {len(valid_coords):,}")

# 3) 원래 쓰던 raw_geo(위치ID별 대표 좌표)를 이걸로 교체
raw_geo = pd.read_parquet(
    "shop_period_raw.parquet", columns=["위치ID", "경도", "위도", "시군구명", "법정동명", "도로명주소"]
).drop_duplicates(subset="위치ID", keep="first")

raw_geo = raw_geo.merge(valid_coords, on="위치ID", how="left")

# 정상 좌표가 확보된 위치는 그걸로 덮어쓰고, 없는 위치(774개)는 원래 값 그대로 둠
raw_geo["경도"] = raw_geo["경도_복구"].combine_first(raw_geo["경도"])
raw_geo["위도"] = raw_geo["위도_복구"].combine_first(raw_geo["위도"])
raw_geo = raw_geo.drop(columns=["경도_복구", "위도_복구"])

# 4) 여전히 범위를 벗어나는 것들(=진짜 복구 불가능한 774개 근처)만 최종 제외
still_bad = ~(raw_geo["경도"].between(126.7, 127.2) & raw_geo["위도"].between(37.4, 37.7))
print(f"복구 후에도 좌표 이상인 위치 수: {still_bad.sum():,}")

raw_geo.loc[still_bad, ["경도", "위도"]] = None  # 억지로 안 채우고 결측 처리

raw_geo.to_csv("22_raw_geo_fixed.csv", index=False, encoding="utf-8-sig")
print("저장 완료: 22_raw_geo_fixed.csv")