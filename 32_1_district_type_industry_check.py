import pandas as pd

# =========================================================================
# 32-1번: 상권유형(TRDAR_SE_1) 안에서 업종별 고교체비율이 얼마나 갈리는지 확인.
# =========================================================================
# 32번에서 "상권유형 간 차이는 2%p밖에 안 난다"는 게 나왔는데, 그게 상권유형
# '안'에서도 업종에 따라 크게 갈리는지, 아니면 상권유형 안에서도 업종별로
# 고르게 나오는지를 확인한다. 26번이 이미 만든 상권유형×업종 기준선
# (26_industry_baseline_by_district_type.csv)을 그대로 쓴다.


def print_table(df, title=None):
    if title:
        print(f"\n=== {title} ===")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except Exception:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))


ti = pd.read_csv("26_industry_baseline_by_district_type.csv")

for t in sorted(ti["TRDAR_SE_1"].unique()):
    sub = ti[ti["TRDAR_SE_1"] == t].sort_values("고교체비율", ascending=False)
    print_table(
        sub[["업종", "위치수", "고교체비율", "중앙교체율", "평균공실률"]],
        title=f"{t} 안에서 업종별 고교체비율",
    )

# --- 상권유형별로 "업종 간 차이(최댓값-최솟값)"를 한눈에 비교 ---
range_by_type = ti.groupby("TRDAR_SE_1")["고교체비율"].agg(
    최소="min", 최대="max"
)
range_by_type["범위(최대-최소)"] = range_by_type["최대"] - range_by_type["최소"]
range_by_type = range_by_type.sort_values("범위(최대-최소)", ascending=False).reset_index()

print_table(range_by_type, title="상권유형별: 업종 간 고교체비율 범위 (상권유형 간 범위 2%p와 비교용)")

print("\n참고: 32번에서 나온 상권유형 '간' 범위는 2%p였음. 위 표의 '범위(최대-최소)'가")
print("그보다 훨씬 크면 -> 업종 효과가 상권유형 효과보다 훨씬 크다는 뜻.")
