import pandas as pd

# =========================================================================
# 18번: 위치ID별 대표 좌표를 복구한다.
# =========================================================================
# 배경: 원본 좌표(경도/위도)가 특정 분기에서만 서울 범위를 크게 벗어난 값으로
# 잘못 기록된 경우가 있었음(약 80만 건). 위치ID 첫 분기 좌표만 쓰면 하필
# 오류난 분기의 좌표를 그대로 쓰게 되므로, 같은 위치ID의 다른 분기 좌표 중
# 서울 정상범위(경도 126.7~127.2, 위도 37.4~37.7)에 드는 값이 있으면 그걸
# 대표 좌표로 채택한다.
#
# 이 단계가 만드는 'location_coords.csv'는 이후 모든 단계(20번 상권결합,
# 23번 공실 상권결합)가 공통으로 읽는 좌표의 단일 소스다. 19번(지오코딩)이
# 이 파일을 이어받아 결측을 추가로 채운다.
#
# 실행 순서: 반드시 17번 다음, 19번보다 먼저 돌릴 것.

raw_full = pd.read_parquet(
    "shop_period_raw.parquet",
    columns=["위치ID", "period", "경도", "위도", "시군구명", "도로명주소"],
)
raw_full["정상범위"] = raw_full["경도"].between(126.7, 127.2) & raw_full["위도"].between(37.4, 37.7)

valid_coords = (
    raw_full[raw_full["정상범위"]]
    .sort_values("period", ascending=False)
    .drop_duplicates(subset="위치ID", keep="first")
    [["위치ID", "경도", "위도"]]
)
print(f"정상범위 좌표로 복구 가능한 위치 수: {len(valid_coords):,}")

location_coords = raw_full.drop_duplicates(subset="위치ID", keep="first")[
    ["위치ID", "경도", "위도", "시군구명", "도로명주소"]
]
location_coords = location_coords.merge(
    valid_coords, on="위치ID", how="left", suffixes=("_원본", "_복구")
)
location_coords["경도"] = location_coords["경도_복구"].combine_first(location_coords["경도_원본"])
location_coords["위도"] = location_coords["위도_복구"].combine_first(location_coords["위도_원본"])
location_coords = location_coords[["위치ID", "경도", "위도", "시군구명", "도로명주소"]]

# 복구 후에도 여전히 서울 범위를 벗어나면 좌표 이상으로 보고 결측 처리 (억지로 안 채움)
still_bad = ~(location_coords["경도"].between(126.7, 127.2) & location_coords["위도"].between(37.4, 37.7))
print(f"복구 후에도 좌표 이상인 위치 수: {still_bad.sum():,}")
location_coords.loc[still_bad, ["경도", "위도"]] = None

print(f"좌표 결측(19번 지오코딩 대상): {location_coords['경도'].isna().sum():,}")

location_coords.to_csv("location_coords.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: location_coords.csv")
