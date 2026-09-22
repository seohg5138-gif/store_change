import pandas as pd
import numpy as np


def print_table(df, title=None):
    """표 형태로 콘솔 출력. tabulate가 있으면 grid 표로, 없으면(또는 표
    렌더링 중 문제가 생기면) 기존 to_string 방식으로 그대로 돌아간다."""
    if title:
        print(f"\n=== {title} ===")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except Exception:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))

# =========================================================================
# 27번: 위치마다 "상권 평균 대비" / "동일 상권×업종 평균 대비" 상대지표를 만든다.
# =========================================================================
# 문서의 수식:
#   상대교체도_i        = 위치교체율_i - 상권평균교체율
#   상대생존도_i        = 위치평균생존기간_i - 상권평균생존기간
#   업종조정상대교체도_i = 위치교체율_i - 동일상권·동일업종평균교체율
#   업종조정상대생존도_i = 위치생존기간_i - 동일상권·동일업종평균생존기간
#
# 상권×업종 표본이 10개 미만이면 그 셀 평균은 불안정하므로, 26번에서 만든
# 상권유형×업종 평균으로 대체(fallback)한다. 어느 기준을 썼는지는
# `업종조정기준` 컬럼에 남겨서 나중에 걸러낼 수 있게 한다.

MIN_LOCATIONS_PER_CELL = 10
UNASSIGNED_CODE = "UNASSIGNED"

full = pd.read_csv("24_full_location_with_district.csv", dtype={"TRDAR_CD": str})
reliable_matched = full[full["관측분기수"] >= 15].copy()
# 25/26번과 동일하게 상권 미배정 위치를 "상권외" 그룹으로 채운다. 이렇게 하면
# 아래 merge들이 25/26번이 이미 만들어둔 "상권외" 행과 자동으로 결합되어,
# 상권 미배정 위치도 (전체 상권외 평균 대비) 상대지표를 얻게 된다. 다만 이
# "상권외 평균"은 실제 이웃 상권 평균이 아니라 도시 전역 미배정 위치의
# 평균이므로, 진짜 "옆 자리 대비"가 아니라 "도시 평균적인 미배정 위치 대비"
# 라는 걸 구분해서 봐야 함 -> 업종조정기준/실제상권여부 컬럼에 남김.
reliable_matched["TRDAR_CD"] = reliable_matched["TRDAR_CD"].fillna(UNASSIGNED_CODE)
reliable_matched["TRDAR_CD_N"] = reliable_matched["TRDAR_CD_N"].fillna("상권외(미배정)")
reliable_matched["TRDAR_SE_1"] = reliable_matched["TRDAR_SE_1"].fillna("상권외")
reliable_matched["실제상권여부"] = reliable_matched["TRDAR_CD"] != UNASSIGNED_CODE

district_baseline = pd.read_csv("25_district_full_baseline.csv", dtype={"TRDAR_CD": str})
district_industry = pd.read_csv("26_industry_baseline_by_district.csv", dtype={"TRDAR_CD": str})
type_industry = pd.read_csv("26_industry_baseline_by_district_type.csv")

# --- 1) 상권 평균 대비 ---
merged = reliable_matched.merge(
    district_baseline[["TRDAR_CD", "평균교체율", "평균생존분기수"]]
    .rename(columns={"평균교체율": "상권평균교체율", "평균생존분기수": "상권평균생존분기수"}),
    on="TRDAR_CD", how="left",
)
merged["상대교체도"] = merged["교체율"] - merged["상권평균교체율"]
merged["상대생존도"] = merged["평균생존분기수"] - merged["상권평균생존분기수"]

# --- 2) 동일 상권×업종 평균 대비 (표본충분한 셀만 우선 사용) ---
# reliable_matched(24번 산출물)의 위치별 업종 표시는 "대표업종"(참고용,
# 최다점유업종 1개)이고, district_industry(26번, 업종비중 가중집계)의 업종
# 카테고리는 "업종" 컬럼임 - 이름이 달라 left_on/right_on으로 매칭한다.
di_sufficient = district_industry[district_industry["표본충분"]][
    ["TRDAR_CD", "업종", "평균교체율", "평균생존분기수"]
].rename(columns={"평균교체율": "업종조정_상권업종교체율", "평균생존분기수": "업종조정_상권업종생존분기수"})

merged = merged.merge(
    di_sufficient, left_on=["TRDAR_CD", "대표업종"], right_on=["TRDAR_CD", "업종"], how="left"
)
merged["업종조정기준"] = np.where(merged["업종조정_상권업종교체율"].notna(), "상권×업종", None)

# --- fallback: 상권×업종 표본 부족한 행은 상권유형×업종 평균으로 대체 ---
type_industry_renamed = type_industry.rename(
    columns={"평균교체율": "유형업종교체율", "평균생존분기수": "유형업종생존분기수", "업종": "업종_유형매칭용"}
)[["TRDAR_SE_1", "업종_유형매칭용", "유형업종교체율", "유형업종생존분기수"]]

merged = merged.merge(
    type_industry_renamed, left_on=["TRDAR_SE_1", "대표업종"], right_on=["TRDAR_SE_1", "업종_유형매칭용"],
    how="left",
)
merged = merged.drop(columns=["업종_유형매칭용"])

need_fallback = merged["업종조정_상권업종교체율"].isna()
merged.loc[need_fallback, "업종조정_상권업종교체율"] = merged.loc[need_fallback, "유형업종교체율"]
merged.loc[need_fallback, "업종조정_상권업종생존분기수"] = merged.loc[need_fallback, "유형업종생존분기수"]
merged.loc[need_fallback & merged["유형업종교체율"].notna(), "업종조정기준"] = "상권유형×업종(대체)"
merged["업종조정기준"] = merged["업종조정기준"].fillna("없음(표본부족)")
# "상권외" 위치는 di_sufficient/type_industry 매칭에 성공해도 실제 이웃
# 상권 평균이 아니라 도시 전역 미배정 위치 평균일 뿐이므로 기준 라벨을
# 구분해서 남긴다 (해석 시 실제 상권과 혼동하지 않도록).
merged.loc[~merged["실제상권여부"], "업종조정기준"] = (
    "상권외 평균 대비(" + merged.loc[~merged["실제상권여부"], "업종조정기준"] + ")"
)

merged["업종조정상대교체도"] = merged["교체율"] - merged["업종조정_상권업종교체율"]
merged["업종조정상대생존도"] = merged["평균생존분기수"] - merged["업종조정_상권업종생존분기수"]

merged = merged.drop(columns=["유형업종교체율", "유형업종생존분기수", "업종"])

# --- 4분류 라벨링 (상권×업종 조정 기준, 부호로 분류) ---
def classify(row):
    if pd.isna(row["업종조정상대교체도"]) or pd.isna(row["업종조정상대생존도"]):
        return "미분류"
    high_change = row["업종조정상대교체도"] > 0
    high_survive = row["업종조정상대생존도"] > 0
    if high_change and not high_survive:
        return "상대교체높음·상대생존낮음"
    if high_change and high_survive:
        return "상대교체높음·상대생존높음"
    if not high_change and not high_survive:
        return "상대교체낮음·상대생존낮음"
    return "상대교체낮음·상대생존높음"

merged["상대분류"] = merged.apply(classify, axis=1)

merged.to_csv("27_location_relative_metrics.csv", index=False, encoding="utf-8-sig")

print(f"위치 수: {len(merged):,}")
print("\n=== 업종조정기준 분포 ===")
print(merged["업종조정기준"].value_counts())
print("\n=== 상대분류 분포 ===")
print(merged["상대분류"].value_counts())

print("\n=== 상권은 안정적인데 유독 해당 위치만 생존기간이 짧은 경우 (예시 10개, 실제 상권만) ===")
# 상권평균교체율이 낮은(하위 25%, 실제 상권 기준) 상권 소속인데
# 업종조정상대교체도가 높은 위치. "상권외"는 실제 이웃이 아니므로 제외.
real = merged[merged["실제상권여부"]]
q25 = real["상권평균교체율"].quantile(0.25)
candidates = real[
    (real["상권평균교체율"] <= q25) & (real["업종조정상대교체도"] > real["업종조정상대교체도"].quantile(0.9))
]
print_table(candidates[
    ["위치ID", "TRDAR_CD_N", "대표업종", "교체율", "상권평균교체율", "업종조정상대교체도", "업종조정기준"]
].head(10))

print(f"\n저장 완료: 27_location_relative_metrics.csv")

print("\n=== 상권배정 vs 상권외(미배정) 상대분류 비교 ===")
print(merged.groupby("실제상권여부")["상대분류"].value_counts(normalize=True).round(3))
print("\n=== 상권배정 vs 상권외(미배정) 원지표 비교 ===")
print(merged.groupby("실제상권여부")[["교체율", "평균생존분기수"]].mean().round(4))
