import numpy as np
import pandas as pd

# =========================================================================
# 37번: 상권 프로파일(클러스터링용) 변수 생성.
# =========================================================================
# 겹치는 구간(202103~202512)만 남기고, TRDAR_CD별로 평균 낸 뒤, 문서에
# 있던 파생변수(로그변환, 성비, 연령대 비율, 엔트로피 등)를 계산한다.
#
# 주의: 상주인구 API에 "1인 가구 수" 컬럼이 없어서(TOT_HSHLD_CO만 있고
# 1인가구 세분류가 안 보임) "1인 가구 비율"은 계산 못 함 - 필요하면 다른
# 컬럼이 실제로 더 있는지 API 응답 전체를 다시 확인해야 함.

PERIOD_MIN, PERIOD_MAX = 202103, 202512


def entropy(probs):
    """섀넌 엔트로피. probs는 합이 1인 비율들의 리스트(0은 자동 무시)."""
    probs = np.array([p for p in probs if p > 0])
    if len(probs) == 0:
        return np.nan
    return -(probs * np.log(probs)).sum()


def filter_period(df):
    return df[(df["분기코드_통일"] >= PERIOD_MIN) & (df["분기코드_통일"] <= PERIOD_MAX)].copy()


# =========================================================================
# 1) 길단위인구 (생활인구)
# =========================================================================
flpop = filter_period(pd.read_csv("34_길단위인구_raw.csv"))

flpop_avg = flpop.groupby("TRDAR_CD").mean(numeric_only=True).reset_index()

flpop_feat = pd.DataFrame({"TRDAR_CD": flpop_avg["TRDAR_CD"]})
flpop_feat["총생활인구"] = flpop_avg["TOT_FLPOP_CO"]
flpop_feat["총생활인구_로그"] = np.log1p(flpop_avg["TOT_FLPOP_CO"])
flpop_feat["여성비율"] = flpop_avg["FML_FLPOP_CO"] / flpop_avg["TOT_FLPOP_CO"]

age_cols = ["AGRDE_10_FLPOP_CO", "AGRDE_20_FLPOP_CO", "AGRDE_30_FLPOP_CO",
            "AGRDE_40_FLPOP_CO", "AGRDE_50_FLPOP_CO", "AGRDE_60_ABOVE_FLPOP_CO"]
age_labels = ["10대", "20대", "30대", "40대", "50대", "60대이상"]
for col, label in zip(age_cols, age_labels):
    flpop_feat[f"연령비율_{label}"] = flpop_avg[col] / flpop_avg["TOT_FLPOP_CO"]

weekday_cols = ["MON_FLPOP_CO", "TUES_FLPOP_CO", "WED_FLPOP_CO", "THUR_FLPOP_CO", "FRI_FLPOP_CO"]
weekend_cols = ["SAT_FLPOP_CO", "SUN_FLPOP_CO"]
flpop_feat["주말인구비율"] = (
    flpop_avg[weekend_cols].sum(axis=1) / flpop_avg[weekday_cols + weekend_cols].sum(axis=1)
)

time_cols = ["TMZON_00_06_FLPOP_CO", "TMZON_06_11_FLPOP_CO", "TMZON_11_14_FLPOP_CO",
             "TMZON_14_17_FLPOP_CO", "TMZON_17_21_FLPOP_CO", "TMZON_21_24_FLPOP_CO"]
time_labels = ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]
for col, label in zip(time_cols, time_labels):
    flpop_feat[f"시간대비율_{label}"] = flpop_avg[col] / flpop_avg["TOT_FLPOP_CO"]

# 방문자 다양성(엔트로피): 연령대 비율, 시간대 비율 각각
age_ratio_cols = [f"연령비율_{l}" for l in age_labels]
time_ratio_cols = [f"시간대비율_{l}" for l in time_labels]
flpop_feat["연령다양성_엔트로피"] = flpop_feat[age_ratio_cols].apply(entropy, axis=1)
flpop_feat["시간대다양성_엔트로피"] = flpop_feat[time_ratio_cols].apply(entropy, axis=1)

print(f"길단위인구 features: {len(flpop_feat):,}개 상권")

# =========================================================================
# 2) 직장인구
# =========================================================================
wrc = filter_period(pd.read_csv("34_직장인구_raw.csv"))
wrc_avg = wrc.groupby("TRDAR_CD").mean(numeric_only=True).reset_index()

wrc_feat = pd.DataFrame({"TRDAR_CD": wrc_avg["TRDAR_CD"]})
wrc_feat["총직장인구"] = wrc_avg["TOT_WRC_POPLTN_CO"]
wrc_feat["총직장인구_로그"] = np.log1p(wrc_avg["TOT_WRC_POPLTN_CO"])
wrc_feat["직장인구_여성비율"] = wrc_avg["FML_WRC_POPLTN_CO"] / wrc_avg["TOT_WRC_POPLTN_CO"]

print(f"직장인구 features: {len(wrc_feat):,}개 상권")

# =========================================================================
# 3) 상주인구
# =========================================================================
repop = filter_period(pd.read_csv("34_상주인구_raw.csv"))
repop_avg = repop.groupby("TRDAR_CD").mean(numeric_only=True).reset_index()

repop_feat = pd.DataFrame({"TRDAR_CD": repop_avg["TRDAR_CD"]})
repop_feat["총상주인구"] = repop_avg["TOT_REPOP_CO"]
repop_feat["총상주인구_로그"] = np.log1p(repop_avg["TOT_REPOP_CO"])
repop_feat["상주인구_여성비율"] = repop_avg["FML_REPOP_CO"] / repop_avg["TOT_REPOP_CO"]
repop_feat["총가구수"] = repop_avg["TOT_HSHLD_CO"]
repop_feat["아파트가구비율"] = repop_avg["APT_HSHLD_CO"] / repop_avg["TOT_HSHLD_CO"]
# 1인가구비율: API에 해당 컬럼 없어서 계산 불가 (컬럼 있으면 여기 추가)

print(f"상주인구 features: {len(repop_feat):,}개 상권")

# =========================================================================
# 4) 추정매출 (상권x업종x분기 -> 먼저 업종 합산해서 상권x분기로 접은 뒤 평균)
# =========================================================================
sales = filter_period(pd.read_csv("34_추정매출_raw.csv"))

sales_amt_cols = ["THSMON_SELNG_AMT", "MDWK_SELNG_AMT", "WKEND_SELNG_AMT",
                   "TMZON_21_24_SELNG_AMT"]
sales_co_cols = ["THSMON_SELNG_CO"]
sales_district_q = (
    sales.groupby(["TRDAR_CD", "분기코드_통일"])[sales_amt_cols + sales_co_cols]
    .sum()
    .reset_index()
)
sales_avg = sales_district_q.groupby("TRDAR_CD").mean(numeric_only=True).reset_index()

sales_feat = pd.DataFrame({"TRDAR_CD": sales_avg["TRDAR_CD"]})
sales_feat["분기총매출"] = sales_avg["THSMON_SELNG_AMT"]
sales_feat["분기총매출_로그"] = np.log1p(sales_avg["THSMON_SELNG_AMT"])
sales_feat["건당매출액"] = sales_avg["THSMON_SELNG_AMT"] / sales_avg["THSMON_SELNG_CO"].replace(0, np.nan)
sales_feat["주말매출비율"] = sales_avg["WKEND_SELNG_AMT"] / sales_avg["THSMON_SELNG_AMT"]
sales_feat["야간매출비율"] = sales_avg["TMZON_21_24_SELNG_AMT"] / sales_avg["THSMON_SELNG_AMT"]

# 매출 성장률·변동성: 분기별 시계열이 있으니 그대로 계산 가능 (이건 상권의
# "평소 수준"이 아니라 "추세" 정보라 평균과는 별개로 의미 있음 -> 같이 넣음)
sales_district_q_sorted = sales_district_q.sort_values(["TRDAR_CD", "분기코드_통일"])
growth = sales_district_q_sorted.groupby("TRDAR_CD")["THSMON_SELNG_AMT"].agg(
    매출_시작값="first", 매출_끝값="last", 매출_표준편차="std", 매출_평균="mean"
).reset_index()
growth["매출성장률"] = (growth["매출_끝값"] / growth["매출_시작값"].replace(0, np.nan)) - 1
growth["매출변동성_CV"] = growth["매출_표준편차"] / growth["매출_평균"].replace(0, np.nan)
sales_feat = sales_feat.merge(growth[["TRDAR_CD", "매출성장률", "매출변동성_CV"]], on="TRDAR_CD", how="left")

print(f"추정매출 features: {len(sales_feat):,}개 상권")

# =========================================================================
# 5) 병합 + 교차비율
# =========================================================================
profile = flpop_feat.merge(wrc_feat, on="TRDAR_CD", how="outer")
profile = profile.merge(repop_feat, on="TRDAR_CD", how="outer")
profile = profile.merge(sales_feat, on="TRDAR_CD", how="outer")

profile["생활인구_상주인구_비율"] = profile["총생활인구"] / profile["총상주인구"].replace(0, np.nan)
profile["직장인구_상주인구_비율"] = profile["총직장인구"] / profile["총상주인구"].replace(0, np.nan)
profile["생활인구당매출"] = profile["분기총매출"] / profile["총생활인구"].replace(0, np.nan)

profile.to_csv("36_district_profile_features.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: 36_district_profile_features.csv ({len(profile):,}개 상권, {len(profile.columns)}개 컬럼)")
missing = profile.isna().sum()
print("\n결측치가 있는 컬럼 (4개 API 중 일부만 있는 상권들):")
print(missing[missing > 0].sort_values(ascending=False))

# --- 우리 파이프라인 TRDAR_CD와 얼마나 겹치는지 확인 ---
our = pd.read_csv("25_district_full_baseline.csv", dtype={"TRDAR_CD": str})
our_codes = set(our.loc[our["실제상권여부"], "TRDAR_CD"])
profile["TRDAR_CD"] = profile["TRDAR_CD"].astype(str)
api_codes = set(profile["TRDAR_CD"])
print(f"\n우리 상권 수: {len(our_codes):,}, API 상권 수: {len(api_codes):,}")
print(f"양쪽 다 있는 상권: {len(our_codes & api_codes):,}")
print(f"우리만 있고 API엔 없는 상권: {len(our_codes - api_codes):,}")
print(f"API만 있고 우리 쪽엔 없는 상권: {len(api_codes - our_codes):,}")
