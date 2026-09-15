import pandas as pd

# =========================================================================
# 21번: 지역/상권유형별 EDA
# =========================================================================
# 20번 산출물(자주 바뀌는 위치 + 상권정보)을 갖고 분포를 본다.
# "전체 신뢰가능 위치 대비 비율"을 계산하려면 자주 바뀌는 위치뿐 아니라
# 전체 신뢰가능 위치의 시군구명도 필요해서, 이 부분은 17번 결과에 의존하지 않고
# location_metrics_corrected.csv에서 직접 다시 만든다 (스크립트 간 독립성 유지).

joined = pd.read_csv("20_frequent_change_with_district.csv")

print("=== 자주 바뀌는 위치의 상권유형(TRDAR_SE_1)별 분포 ===")
print(joined["TRDAR_SE_1"].value_counts())

print("\n=== 자주 바뀌는 위치의 시군구별 분포 (상위 15개) ===")
print(joined["시군구명"].value_counts().head(15))

# --- 전체 신뢰가능 위치(비교 기준) 다시 만들기 ---
df_loc = pd.read_csv("location_metrics_corrected.csv")
reliable = df_loc[df_loc["관측분기수"] >= 15].copy()

raw_geo = pd.read_parquet(
    "shop_period_raw.parquet", columns=["위치ID", "시군구명"]
).drop_duplicates(subset="위치ID", keep="first")

all_geo = reliable.merge(raw_geo, on="위치ID", how="left")

print("\n=== 시군구별 '자주 바뀌는 위치' 비율 (전체 신뢰가능 위치 대비) ===")
total_by_gu = all_geo["시군구명"].value_counts()
freq_by_gu = joined["시군구명"].value_counts()
ratio_by_gu = (freq_by_gu / total_by_gu * 100).dropna().sort_values(ascending=False)
print(ratio_by_gu.round(2))

print("\n=== 자주 바뀌는 위치가 몰린 상권 TOP 20 (상권명 기준) ===")
print(joined["TRDAR_CD_N"].value_counts().head(20))

# 종합 요약 저장
summary = pd.DataFrame({
    "구분": ["전체 신뢰가능 위치", "자주 바뀌는 위치"],
    "개수": [len(reliable), len(joined)],
})
summary.to_csv("21_eda_summary.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 21_eda_summary.csv")
