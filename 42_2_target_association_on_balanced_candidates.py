# -*- coding: utf-8 -*-
"""
42_2_target_association_on_balanced_candidates.py

40번 그리드서치 상위 결과 중, 방금 실측으로 확인한 "균형 통과"(최소군집비율
>=3%) 후보 8개(알고리즘까지 포함하면 16개 조합)에 대해 타겟(자주바뀜여부/
교체율) 연관성을 카이제곱(Cramér's V)·크루스칼-왈리스(epsilon^2)로 검정한다.

이 8개는 전부 39_candidate_variable_sets.csv(업로드본) 기준으로 실제 변수를
조회하고, 로컬에서 k=4로 재피팅해서 최소군집비율을 직접 확인한 것들이다.

입력:
  39_screened_variables.csv
  24_full_location_with_district.csv

출력:
  42_2_balanced_candidates_association.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

SCREENED_DATA_PATH = "39_screened_variables.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"
MIN_OBS_QUARTERS = 15
RANDOM_SEED = 42

# (set_id, 변수목록, 균형 통과한 알고리즘 리스트)
BALANCED_CANDIDATES = [
    ("S0970", ["연령비율_40대", "야간매출비율", "매출성장률", "매출변동성_CV"], ["Ward", "KMeans"]),
    ("S1561", ["시간대비율_06_11", "시간대비율_14_17", "업종비중_소매", "야간매출비율"], ["Ward", "KMeans"]),
    ("S0604", ["건당매출액", "주말매출비율", "매출성장률", "매출변동성_CV"], ["Ward", "KMeans"]),
    ("S1306", ["업종비중_수리·개인", "업종비중_숙박", "업종비중_시설관리·임대", "매출성장률"], ["KMeans"]),
    ("S1609", ["시간대비율_21_24", "업종비중_교육", "건당매출액", "주말매출비율"], ["KMeans"]),
    ("S1492", ["전체점포수_로그", "업종비중_보건의료", "업종비중_부동산", "주말매출비율",
               "야간매출비율", "매출변동성_CV"], ["Ward", "KMeans"]),
    ("S1379", ["업종비중_과학·기술", "업종비중_수리·개인", "업종비중_시설관리·임대",
               "업종비중_예술·스포츠", "매출변동성_CV"], ["Ward", "KMeans"]),
    ("S1495", ["전체점포수_로그", "업종비중_과학·기술", "업종비중_보건의료",
               "업종비중_시설관리·임대", "주말매출비율", "매출변동성_CV"], ["Ward", "KMeans"]),
    ("S1332", ["전체점포수_로그", "업종비중_시설관리·임대", "업종비중_예술·스포츠",
               "주말매출비율", "매출성장률"], ["KMeans"]),
    ("S0726", ["총생활인구_로그", "주말인구비율", "연령비율_30대", "연령비율_50대",
               "업종집중도_HHI"], ["Ward", "KMeans"]),
]


def normalize_trdar(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def fit_labels(X, algo):
    if algo == "KMeans":
        return KMeans(n_clusters=4, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    if algo == "Ward":
        return AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
    raise ValueError(algo)


def cramers_v(chi2, n, shape):
    r, c = shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def epsilon_squared(h_stat, n, k_groups):
    return max(0.0, (h_stat - k_groups + 1) / (n - k_groups))


# ----------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
df[ID_COL] = normalize_trdar(df[ID_COL])

loc = pd.read_csv(LOCATION_PATH, dtype={"TRDAR_CD": str})
loc = loc[loc["관측분기수"] >= MIN_OBS_QUARTERS].copy()
loc[ID_COL] = normalize_trdar(loc[ID_COL])

trdar_to_idx = {t: i for i, t in enumerate(df[ID_COL])}
loc_idx = loc[ID_COL].map(trdar_to_idx)
valid_mask = loc_idx.notna()
loc = loc[valid_mask]
loc_idx_valid = loc_idx[valid_mask].astype(int).to_numpy()
loc_binary_valid = loc["자주바뀜여부"].to_numpy()
loc_cont_valid = loc["교체율"].to_numpy()
print(f"타겟 위치: {len(loc):,}개\n")

# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
results = []
for set_id, var_list, algos in BALANCED_CANDIDATES:
    X = StandardScaler().fit_transform(df[var_list].to_numpy())
    for algo in algos:
        labels = fit_labels(X, algo)
        sizes = pd.Series(labels).value_counts()
        min_frac = sizes.min() / len(labels)

        cluster_per_loc = labels[loc_idx_valid]
        uniq = np.unique(cluster_per_loc)

        table = pd.crosstab(cluster_per_loc, loc_binary_valid)
        chi2, p_chi, _, _ = chi2_contingency(table)
        v = cramers_v(chi2, table.values.sum(), table.shape)

        groups = [loc_cont_valid[cluster_per_loc == g] for g in uniq]
        h_stat, p_kw = kruskal(*groups)
        eps2 = epsilon_squared(h_stat, len(cluster_per_loc), len(uniq))

        rates = [loc_binary_valid[cluster_per_loc == g].mean() for g in uniq]
        rate_range = max(rates) - min(rates)

        results.append({
            "set_id": set_id, "변수목록": ";".join(var_list), "변수개수": len(var_list),
            "알고리즘": algo, "최소군집비율": round(min_frac, 4),
            "CramersV": round(v, 4), "카이제곱_p": p_chi,
            "epsilon_squared": round(eps2, 4), "KW_p": p_kw,
            "군집별_고교체비율_범위": round(rate_range, 4),
        })
        print(f"[{set_id}/{algo}] CramersV={v:.4f}  epsilon2={eps2:.4f}  "
              f"고교체비율범위={rate_range:.4f}")

result_df = pd.DataFrame(results).sort_values("CramersV", ascending=False)
result_df.to_csv("42_2_balanced_candidates_association.csv", index=False, encoding="utf-8-sig")

print("\n=== Cramér's V 순위 ===")
print(result_df[["set_id", "변수목록", "알고리즘", "CramersV", "epsilon_squared",
                  "군집별_고교체비율_범위"]].to_string(index=False))
print("\n저장: 42_2_balanced_candidates_association.csv")
