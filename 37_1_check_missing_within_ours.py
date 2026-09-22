import pandas as pd

# =========================================================================
# 37-1번: 37번의 결측치가 "우리가 실제로 쓰는 1,614개 상권" 안에서 생긴
# 건지, 아니면 "API에만 있고 우리 쪽엔 없는 36개" 안에서 생긴 건지 구분.
# =========================================================================

profile = pd.read_csv("37_clustering_ready_features.csv", dtype={"TRDAR_CD": str})
our = pd.read_csv("25_district_full_baseline.csv", dtype={"TRDAR_CD": str})
our_codes = set(our.loc[our["실제상권여부"], "TRDAR_CD"])

profile["우리상권여부"] = profile["TRDAR_CD"].isin(our_codes)

print(f"전체 profile 행수: {len(profile):,}")
print(f"이 중 우리 1,614개 안에 있는 행: {profile['우리상권여부'].sum():,}")
print(f"이 중 API 전용(우리 쪽엔 없는) 행: {(~profile['우리상권여부']).sum():,}")

check_cols = [
    "분기총매출", "매출변동성_CV", "총상주인구", "총직장인구", "업무형비율",
]

print("\n=== 우리 1,614개 안에서만 본 결측치 ===")
ours_only = profile[profile["우리상권여부"]]
missing_ours = ours_only[check_cols].isna().sum()
print(missing_ours)
print(f"(분모: {len(ours_only):,}개)")

print("\n=== API 전용(우리 쪽엔 없는 36개) 안에서 본 결측치 ===")
api_only = profile[~profile["우리상권여부"]]
missing_api_only = api_only[check_cols].isna().sum()
print(missing_api_only)
print(f"(분모: {len(api_only):,}개)")
