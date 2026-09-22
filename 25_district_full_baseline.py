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
# 25번: 24번 산출물(전체 위치+상권)에서 "관측분기수 15+"(신뢰가능) 위치만
# 대상으로 상권별 전체 구조 기준선을 만든다.
# =========================================================================
# 여기서 만드는 표가 문서에서 말한 "상권별 전체 구조" 테이블이다:
#   전체 신뢰가능 위치 수 / 고교체 위치 수 / 고교체 비율 /
#   평균·중앙 교체율 / 평균·중앙 생존기간 / 평균 공실률 / 공실 경험 위치 비율 /
#   업종 구성 / 관측기간
#
# 주의(표본 크기): 위치 10개 중 3개가 고교체면 30%지만 신뢰할 수 없으므로,
# 최소 표본 기준(상권 전체 30개 이상)을 만족하지 않는 상권은 최종 해석에서
# "참고값"으로 별도 표시한다. 여기서 걷어내지는 않고 플래그만 남긴다
# (걷어내는 판단은 해석 단계에서 하도록 원자료를 보존).

MIN_LOCATIONS_PER_DISTRICT = 30
UNASSIGNED_CODE = "UNASSIGNED"
UNASSIGNED_NAME = "상권외(미배정)"
UNASSIGNED_TYPE = "상권외"

full = pd.read_csv("24_full_location_with_district.csv", dtype={"TRDAR_CD": str})
reliable = full[full["관측분기수"] >= 15].copy()

# 상권 미배정 위치를 지우지 않고 "상권외"라는 별도 그룹으로 채운다.
# 이유: 코드북에서 이미 확인됐듯 미매칭은 좌표/폴리곤 오류가 아니라 상권
# 폴리곤의 커버리지 공백일 뿐이라, 오류 데이터로 취급해 빼면 안 됨. 이렇게
# 하면 groupby가 자동으로 "상권외" 행을 하나 만들어서 다른 상권들과 나란히
# 비교할 수 있게 됨.
reliable["TRDAR_CD"] = reliable["TRDAR_CD"].fillna(UNASSIGNED_CODE)
reliable["TRDAR_CD_N"] = reliable["TRDAR_CD_N"].fillna(UNASSIGNED_NAME)
reliable["TRDAR_SE_1"] = reliable["TRDAR_SE_1"].fillna(UNASSIGNED_TYPE)
reliable_matched = reliable  # 이름은 유지하되 이제 아무것도 걷어내지 않음

print(f"신뢰가능 위치(관측15+): {len(reliable):,}")
print(f"  이 중 상권 배정됨: {(reliable['TRDAR_CD'] != UNASSIGNED_CODE).sum():,}")
print(f"  이 중 상권외(미배정): {(reliable['TRDAR_CD'] == UNASSIGNED_CODE).sum():,}")

# --- 업종 구성: 상권별 최다 업종 3개를 문자열로 요약 (상세 분포는 26번에서) ---
# groupby().apply(..., include_groups=False)는 pandas 2.2+에서만 지원되므로
# (이전 스크립트들의 pandas 버전이 확인되지 않아 위험) 업종 컬럼만 잘라서
# apply하는 방식으로 버전 의존성을 없앤다.
def top_industries(s, n=3):
    counts = s.value_counts()
    return ", ".join(f"{idx}({cnt})" for idx, cnt in counts.head(n).items())

industry_summary = (
    reliable_matched.groupby(["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])["대표업종"]
    .apply(top_industries)
    .rename("업종구성_TOP3")
)

# --- 관측기간: 상권 내 위치들의 첫분기~끝분기 범위 ---
period_range = (
    reliable_matched.groupby(["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])
    .agg(관측시작=("첫분기", "min"), 관측종료=("끝분기", "max"))
)

# --- 핵심 집계 ---
baseline = (
    reliable_matched.groupby(["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])
    .agg(
        전체위치수=("위치ID", "size"),
        고교체위치수=("자주바뀜여부", "sum"),
        평균교체율=("교체율", "mean"),
        중앙교체율=("교체율", "median"),
        평균생존분기수=("평균생존분기수", "mean"),
        중앙생존분기수=("평균생존분기수", "median"),
        평균공실률=("공실률", "mean"),
        # NaN > 0은 pandas에서 자동으로 False가 되어 "공실 없음"으로 잘못
        # 집계될 수 있으므로, 공실률이 결측인 위치는 분자·분모 양쪽에서
        # 제외하고 계산한다 (22번의 "알 수 없음" 처리 원칙과 일관되게).
        공실경험위치수=("공실률", lambda s: (s.dropna() > 0).sum()),
        공실률관측위치수=("공실률", lambda s: s.notna().sum()),
    )
    .reset_index()
)

baseline["고교체비율"] = baseline["고교체위치수"] / baseline["전체위치수"]
baseline["공실경험비율"] = baseline["공실경험위치수"] / baseline["공실률관측위치수"].replace(0, np.nan)
baseline["표본충분"] = baseline["전체위치수"] >= MIN_LOCATIONS_PER_DISTRICT
# "상권외"는 특정 지리적 상권이 아니라 도시 전역의 미배정 위치를 모아놓은
# 인위적 집단이라 표본은 항상 충분하지만 다른 상권들과 순위 비교(어느 상권이
# 제일 특이한가)에는 넣으면 안 됨. 단, 존재 자체는 지우지 않고 별도 플래그로
# 표시해 "상권 안 vs 상권 밖" 비교에는 계속 쓸 수 있게 한다.
baseline["실제상권여부"] = baseline["TRDAR_CD"] != UNASSIGNED_CODE

baseline = baseline.merge(industry_summary, on=["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])
baseline = baseline.merge(period_range, on=["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1"])

baseline = baseline.sort_values("전체위치수", ascending=False)

baseline.to_csv("25_district_full_baseline.csv", index=False, encoding="utf-8-sig")

print(f"\n상권 수(상권외 포함): {len(baseline):,} "
      f"(실제 상권 {baseline['실제상권여부'].sum():,} / 상권외 1개)")
print(f"  실제 상권 중 표본충분 {baseline.loc[baseline['실제상권여부'], '표본충분'].sum():,} / "
      f"참고값 {(baseline['실제상권여부'] & ~baseline['표본충분']).sum():,}")

print("\n=== 상권 안 vs 상권 밖(미배정) 비교 ===")
print_table(baseline[~baseline["실제상권여부"]][
    ["TRDAR_CD_N", "전체위치수", "고교체위치수", "고교체비율", "중앙교체율", "중앙생존분기수", "평균공실률"]
])

print_table(
    baseline[baseline["실제상권여부"]].head(10)[
        ["TRDAR_CD_N", "전체위치수", "고교체위치수", "고교체비율", "중앙교체율", "중앙생존분기수", "평균공실률"]
    ],
    title="전체 위치수 상위 상권 (상권외 제외)",
)
print_table(
    baseline[baseline["실제상권여부"] & baseline["표본충분"]]
    .sort_values("고교체비율", ascending=False).head(10)[
        ["TRDAR_CD_N", "전체위치수", "고교체위치수", "고교체비율", "중앙교체율", "중앙생존분기수", "평균공실률"]
    ],
    title="고교체비율 상위 상권 (표본충분·상권외 제외)",
)
print("\n저장 완료: 25_district_full_baseline.csv (상권외 행 포함)")
