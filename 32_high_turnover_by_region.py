import pandas as pd

# =========================================================================
# 32번: 고회전(고교체) 위치를 구 -> 상권 단위로 드릴다운해서 본다.
# =========================================================================
# "구"는 상권 매칭(TRDAR_CD) 여부와 무관하게 location_coords.csv에서 온
# 시군구명 컬럼을 그대로 쓴다 (좌표 기반이라 상권 미배정 위치도 구는 있음).
# 위치수는 관측분기수 15+(신뢰가능) 기준으로만 센다 - 고교체 여부 자체가
# 이 기준 위에서 정의된 것이라 분모를 다르게 하면 비율이 왜곡됨.

UNASSIGNED_CODE = "UNASSIGNED"
UNASSIGNED_NAME = "상권외(미배정)"


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
reliable["TRDAR_CD_N"] = reliable["TRDAR_CD_N"].fillna(UNASSIGNED_NAME)

# --- 0) 서울 전체 ---
seoul_total = len(reliable)
seoul_high = int(reliable["자주바뀜여부"].sum())
print(f"서울 전체 신뢰가능 위치 수: {seoul_total:,}")
print(f"서울 전체 고교체 위치 수: {seoul_high:,} ({seoul_high / seoul_total:.4f})")

# --- 1) 구 단위 집계 ---
by_gu = reliable.groupby("시군구명").agg(
    구내_전체위치수=("위치ID", "size"),
    구내_고교체위치수=("자주바뀜여부", "sum"),
    구내_상권수=("TRDAR_CD", "nunique"),
).reset_index()
by_gu["구내_고교체비율"] = by_gu["구내_고교체위치수"] / by_gu["구내_전체위치수"]
by_gu = by_gu.sort_values("구내_고교체위치수", ascending=False)
by_gu.to_csv("32_high_turnover_by_gu.csv", index=False, encoding="utf-8-sig")

print_table(by_gu.head(10), title="구별 고교체 위치수 상위 10")

# --- 2) 구 x 상권 단위 집계 (상권 내 점포 수 포함) ---
by_gu_district = reliable.groupby(["시군구명", "TRDAR_CD", "TRDAR_CD_N"]).agg(
    상권내_전체위치수=("위치ID", "size"),
    상권내_고교체위치수=("자주바뀜여부", "sum"),
).reset_index()
by_gu_district["상권내_고교체비율"] = (
    by_gu_district["상권내_고교체위치수"] / by_gu_district["상권내_전체위치수"]
)
by_gu_district = by_gu_district.sort_values(
    ["시군구명", "상권내_고교체위치수"], ascending=[True, False]
)
by_gu_district.to_csv("32_high_turnover_by_gu_district.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: 32_high_turnover_by_gu.csv ({len(by_gu):,}개 구), "
      f"32_high_turnover_by_gu_district.csv ({len(by_gu_district):,}개 구x상권)")

print_table(
    by_gu_district.sort_values("상권내_고교체위치수", ascending=False).head(10)[
        ["시군구명", "TRDAR_CD_N", "상권내_전체위치수", "상권내_고교체위치수", "상권내_고교체비율"]
    ],
    title="상권별 고교체 위치수 상위 10 (구 표시 포함)",
)

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