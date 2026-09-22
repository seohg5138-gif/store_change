import numpy as np
import pandas as pd


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


def weighted_mean(values, weights):
    w = weights.sum()
    return np.nan if w == 0 else (values * weights).sum() / w


def weighted_median(values, weights):
    """가중 중앙값. weights 누적합이 전체 weight의 절반을 넘는 지점의 값."""
    mask = values.notna() & weights.notna()
    values, weights = values[mask].to_numpy(), weights[mask].to_numpy()
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(values) - 1)
    return values[idx]


# =========================================================================
# 26번: 업종 기준선 3단계 구축 (업종비중 가중 버전).
# =========================================================================
# 한 위치가 여러 업종을 거쳐갔을 수 있어서(예: 숙박 2분기 -> 음식 2분기 ->
# 교육 2분기), 24번이 "가장 오래 차지한 업종 1개"로만 집계하면 위치수/
# 고교체위치수 총합이 실제 총합(위치 311,428개 / 고교체 17,022개)과 안 맞게
# 된다. 그래서 24번이 만든 위치별 업종비중(24_location_industry_shares.csv,
# 위치ID당 여러 업종에 걸쳐 비중 합계=1)으로 위치를 "펼쳐서" 가중집계한다.
# 이렇게 하면 각 업종별 위치수(가중)를 다 더하면 항상 정확히 전체 위치수와
# 같아지고, 고교체위치수(가중)를 다 더하면 항상 정확히 17,022가 된다
# (위치별 비중 합이 1이기 때문에 수학적으로 보장됨).

MIN_LOCATIONS_PER_CELL = 10
UNASSIGNED_CODE = "UNASSIGNED"
UNASSIGNED_NAME = "상권외(미배정)"
UNASSIGNED_TYPE = "상권외"

full = pd.read_csv("24_full_location_with_district.csv", dtype={"TRDAR_CD": str})
reliable = full[full["관측분기수"] >= 15].copy()
reliable["TRDAR_CD"] = reliable["TRDAR_CD"].fillna(UNASSIGNED_CODE)
reliable["TRDAR_CD_N"] = reliable["TRDAR_CD_N"].fillna(UNASSIGNED_NAME)
reliable["TRDAR_SE_1"] = reliable["TRDAR_SE_1"].fillna(UNASSIGNED_TYPE)

shares = pd.read_csv("24_location_industry_shares.csv")

# --- 위치를 업종별로 "펼치기": 위치ID 하나가 자기가 거쳐간 업종 수만큼
# 여러 행으로 늘어나고, 각 행은 그 업종의 비중(업종비중)을 갖는다. ---
expanded = reliable.merge(shares, on="위치ID", how="left")
unmatched = expanded["업종"].isna().sum()
if unmatched:
    print(f"⚠️ 업종비중 매칭 안 된 행 {unmatched}개 (집계에서 자동 제외됨)")
expanded = expanded.dropna(subset=["업종"])


def weighted_agg(df, group_cols):
    """업종비중(w)으로 가중집계. 위치수/고교체위치수는 비중의 합(=가중 개수),
    나머지 지표는 가중평균/가중중앙값."""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        w = g["업종비중"]
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["위치수"] = w.sum()
        row["고교체위치수"] = (w * g["자주바뀜여부"]).sum()
        row["평균교체율"] = weighted_mean(g["교체율"], w)
        row["중앙교체율"] = weighted_median(g["교체율"], w)
        row["평균생존분기수"] = weighted_mean(g["평균생존분기수"], w)
        row["중앙생존분기수"] = weighted_median(g["평균생존분기수"], w)
        row["평균공실률"] = weighted_mean(g["공실률"], w)
        rows.append(row)
    return pd.DataFrame(rows)


# --- 1) 서울 전체 업종 기준 ---
industry_seoul = weighted_agg(expanded, ["업종"])
industry_seoul["고교체비율"] = industry_seoul["고교체위치수"] / industry_seoul["위치수"]

# 상권 간 편차: 업종별로 "상권별 가중중앙교체율"의 표준편차
per_district = weighted_agg(expanded, ["업종", "TRDAR_CD"])
district_variation = (
    per_district.groupby("업종")["중앙교체율"].std().rename("상권간교체율표준편차")
)
industry_seoul = industry_seoul.merge(district_variation, on="업종", how="left")
industry_seoul.to_csv("26_industry_baseline_seoul.csv", index=False, encoding="utf-8-sig")

print(f"서울 전체 업종 기준선: {len(industry_seoul)}개 업종")
print(f"  검증: 위치수(가중) 총합 = {industry_seoul['위치수'].sum():,.1f} "
      f"(신뢰가능 위치 {len(reliable):,}개와 일치해야 함)")
print(f"  검증: 고교체위치수(가중) 총합 = {industry_seoul['고교체위치수'].sum():,.1f} "
      f"(전체 자주바뀜여부 합 {reliable['자주바뀜여부'].sum():,}개와 일치해야 함)")

# --- 2) 상권×업종 기준 ---
district_industry = weighted_agg(expanded, ["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "업종"])
district_industry["고교체비율"] = district_industry["고교체위치수"] / district_industry["위치수"]
district_industry["표본충분"] = district_industry["위치수"] >= MIN_LOCATIONS_PER_CELL
district_industry["실제상권여부"] = district_industry["TRDAR_CD"] != UNASSIGNED_CODE
district_industry.to_csv("26_industry_baseline_by_district.csv", index=False, encoding="utf-8-sig")
print(f"상권×업종 기준선: {len(district_industry):,}개 셀 "
      f"(실제상권 표본충분 "
      f"{(district_industry['실제상권여부'] & district_industry['표본충분']).sum():,}개, "
      f"상권외 셀 {(~district_industry['실제상권여부']).sum():,}개)")

# --- 3) 상권유형×업종 기준 (상권×업종 표본이 부족할 때 대체용) ---
type_industry = weighted_agg(expanded, ["TRDAR_SE_1", "업종"])
type_industry["고교체비율"] = type_industry["고교체위치수"] / type_industry["위치수"]
type_industry.to_csv("26_industry_baseline_by_district_type.csv", index=False, encoding="utf-8-sig")
print(f"상권유형×업종 기준선: {len(type_industry)}개 셀")

print_table(
    industry_seoul.sort_values("고교체비율", ascending=False).head(10)[
        ["업종", "위치수", "고교체비율", "중앙교체율", "중앙생존분기수", "평균공실률", "상권간교체율표준편차"]
    ],
    title="서울 전체 고교체비율 상위 업종 (업종비중 가중)",
)

print("\n저장 완료: 26_industry_baseline_seoul.csv, "
      "26_industry_baseline_by_district.csv, "
      "26_industry_baseline_by_district_type.csv")
