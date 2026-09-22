import pandas as pd

# =========================================================================
# 33번: 고회전(고교체) 위치를 상권유형(TRDAR_SE_1) 단위로 본다.
# =========================================================================

UNASSIGNED_CODE = "UNASSIGNED"
UNASSIGNED_TYPE = "상권외"


def print_table(df, title=None):
    if title:
        print(f"\n=== {title} ===")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except Exception:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))


full = pd.read_csv("24_full_location_with_district.csv", dtype={"TRDAR_CD": str})
reliable = full[full["관측분기수"] >= 15].copy()
reliable["TRDAR_CD"] = reliable["TRDAR_CD"].fillna(UNASSIGNED_CODE)
reliable["TRDAR_SE_1"] = reliable["TRDAR_SE_1"].fillna(UNASSIGNED_TYPE)

# --- 1) 상권유형 단위 집계 ---
by_type = reliable.groupby("TRDAR_SE_1").agg(
    유형내_전체위치수=("위치ID", "size"),
    유형내_고교체위치수=("자주바뀜여부", "sum"),
    유형내_상권수=("TRDAR_CD", "nunique"),
).reset_index()
by_type["유형내_고교체비율"] = by_type["유형내_고교체위치수"] / by_type["유형내_전체위치수"]
by_type = by_type.sort_values("유형내_고교체비율", ascending=False)
by_type.to_csv("33_high_turnover_by_district_type.csv", index=False, encoding="utf-8-sig")

print_table(by_type, title="상권유형별 고교체 현황")

# --- 2) 구 x 상권유형 단위 집계 ---
by_gu_type = reliable.groupby(["시군구명", "TRDAR_SE_1"]).agg(
    전체위치수=("위치ID", "size"),
    고교체위치수=("자주바뀜여부", "sum"),
).reset_index()
by_gu_type["고교체비율"] = by_gu_type["고교체위치수"] / by_gu_type["전체위치수"]
by_gu_type = by_gu_type.sort_values(["시군구명", "고교체위치수"], ascending=[True, False])
by_gu_type.to_csv("33_high_turnover_by_gu_district_type.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: 33_high_turnover_by_district_type.csv ({len(by_type):,}개 유형), "
      f"33_high_turnover_by_gu_district_type.csv ({len(by_gu_type):,}개 구x유형)")
