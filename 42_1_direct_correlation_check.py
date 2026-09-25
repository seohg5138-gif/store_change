# -*- coding: utf-8 -*-
"""
42_1_direct_correlation_check.py

42번에서 클러스터링(4개 유형)을 거친 뒤의 연관성을 봤는데 전부 Cramér's V
0.04~0.05로 사실상 무관했다. 이게 "4개로 나누는 방식이 신호를 못 잡는 것"인지
"애초에 상권 변수 자체에 신호가 없는 것"인지 구분하기 위해, 클러스터링을
거치지 않고 39개 변수 각각을 상권별 고교체비율과 직접 비교한다.

  1. 위치 단위 타겟을 상권(TRDAR_CD) 단위로 집계 (상권별 고교체비율, 평균 교체율)
  2. 39개 변수 각각과 스피어만 상관계수 (순위 기반이라 분포 왜곡에 덜 민감)
  3. 39개 변수를 전부 넣은 랜덤포레스트 회귀로 R^2 하나 계산
     -> 이게 "상권 변수로 설명 가능한 이론적 상한선"에 가깝다.
        이것도 낮으면 클러스터링 방식 문제가 아니라 상권 변수 자체의 한계로
        해석하는 게 맞다.

입력:
  39_screened_variables.csv
  24_full_location_with_district.csv

출력:
  42_1_direct_correlation_results.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

SCREENED_DATA_PATH = "39_screened_variables.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"
MIN_OBS_QUARTERS = 15
RANDOM_SEED = 42


def normalize_trdar(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


# ----------------------------------------------------------------------
# 1. 위치 타겟을 상권 단위로 집계
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
df[ID_COL] = normalize_trdar(df[ID_COL])

loc = pd.read_csv(LOCATION_PATH, dtype={"TRDAR_CD": str})
loc = loc[loc["관측분기수"] >= MIN_OBS_QUARTERS].copy()
loc[ID_COL] = normalize_trdar(loc[ID_COL])
loc = loc[loc[ID_COL].isin(df[ID_COL])]

district_target = loc.groupby(ID_COL).agg(
    상권내_위치수=("위치ID", "size"),
    상권내_고교체비율=("자주바뀜여부", "mean"),
    상권내_평균교체율=("교체율", "mean"),
).reset_index()

merged = df.merge(district_target, on=ID_COL, how="inner")
print(f"위치->상권 집계 후 매칭된 상권: {len(merged):,}개 / 전체 {len(df):,}개")
print(f"상권내_고교체비율 분포: 평균 {merged['상권내_고교체비율'].mean():.4f}, "
      f"표준편차 {merged['상권내_고교체비율'].std():.4f}")

# ----------------------------------------------------------------------
# 2. 변수별 스피어만 상관 (구성비 변수 포함 전부)
# ----------------------------------------------------------------------
feature_cols = [c for c in df.columns if c != ID_COL]

corr_rows = []
for c in feature_cols:
    x = merged[c]
    y = merged["상권내_고교체비율"]
    valid = x.notna() & y.notna()
    if valid.sum() < 10:
        continue
    rho, p = spearmanr(x[valid], y[valid])
    corr_rows.append({"변수": c, "스피어만_rho": round(rho, 4), "p값": round(p, 4)})

corr_df = pd.DataFrame(corr_rows).sort_values("스피어만_rho", key=lambda s: s.abs(), ascending=False)
corr_df.to_csv("42_1_direct_correlation_results.csv", index=False, encoding="utf-8-sig")

print("\n=== 상권내_고교체비율과 상관 |rho| 상위 15개 변수 ===")
print(corr_df.head(15).to_string(index=False))

# ----------------------------------------------------------------------
# 3. 39개 변수 전체로 랜덤포레스트 R^2 (이론적 상한선 참고용)
# ----------------------------------------------------------------------
X = merged[feature_cols].fillna(merged[feature_cols].median())
y = merged["상권내_고교체비율"]

rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print(f"\n39개 변수 전체 랜덤포레스트 5-fold 교차검증 R^2: "
      f"평균 {scores.mean():.4f} (개별: {[round(s,4) for s in scores]})")

if scores.mean() < 0.05:
    print("\n-> R^2도 사실상 0에 가까움: 상권 단위 변수 자체가 고교체비율을 "
          "설명하는 힘이 거의 없다는 뜻. 클러스터링 방식(4개 자르기) 문제가 "
          "아니라 상권 변수 자체의 한계로 보는 게 타당함.")
else:
    print("\n-> R^2가 어느 정도 나옴: 상권 변수에 신호가 있는데 4개로 나누는 "
          "클러스터링이 그 신호를 못 살리고 있을 가능성. 클러스터 개수나 "
          "방식을 재검토할 필요 있음.")
