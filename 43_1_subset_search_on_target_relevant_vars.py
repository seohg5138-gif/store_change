# -*- coding: utf-8 -*-
"""
43_1_subset_search_on_target_relevant_vars.py

42_1번에서 찾은 15개 타겟관련 변수를, 원래 하던 대로 A(유동인구)/B(산업)/
C(매출) 카테고리로 쪼갠 뒤 각 카테고리(와 그 조합) 안에서만 클러스터링한다.
15개를 아무렇게나 섞은 조합이 아니라 "매출로만", "유동인구로만", "산업으로만"
나눠보고, 필요하면 그 조합(AB/AC/BC/ABC)도 본다 - 처음 콘셉트(매출/유동인구/
산업특성으로 각각 클러스터링)를 타겟관련 변수 15개로 좁혀서 그대로 반복하는 것.

카테고리별 pool 크기가 다르다 (A=9, B=4, C=2). pool이 4개 미만이면 조합을 뽑을
필요 없이 그 pool 전체를 통째로 하나의 "세트"로 쓴다. pool이 충분히 크면
(A, AB, AC, ABC 등) 4~6개짜리 부분집합을 샘플링한다.

이번에도 순위는 실루엣이 아니라 Cramér's V로 매긴다 - 목표가 "통계적으로
예쁜 군집"이 아니라 "타겟과 실제로 관련되면서 해석 가능한 군집"이기 때문.

입력:
  39_screened_variables.csv
  24_full_location_with_district.csv

출력:
  43_1_subset_search_results.csv
"""

import itertools
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

SCREENED_DATA_PATH = "39_screened_variables.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"
MIN_OBS_QUARTERS = 15
RANDOM_SEED = 42

TARGET_RELEVANT_VARS = [
    "업종비중_과학·기술", "시간대비율_06_11", "전체점포수_로그", "업종비중_예술·스포츠",
    "시간대비율_21_24", "매출변동성_CV", "주말인구비율", "연령비율_40대",
    "건당매출액", "총가구수", "총상주인구_로그", "시간대비율_11_14",
    "업종비중_수리·개인", "연령비율_30대", "총생활인구_로그",
]

# 39번과 같은 분류 규칙 - 15개 타겟관련 변수를 A(유동인구)/B(산업)/C(매출)로 나눔
CATEGORY_KEYWORDS = {
    "A_유동인구": ["생활인구", "여성비율", "주말인구비율", "연령다양성", "시간대다양성",
               "직장인구", "상주인구", "총가구수", "연령비율_", "시간대비율_"],
    "B_산업": ["업종다양성", "업종집중도", "상위3업종비중", "전체점포수", "업종비중_"],
    "C_매출": ["매출", "업무형비율"],
}


def assign_category(var):
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in var for kw in keywords):
            return cat
    return "기타"


cat_to_vars = {}
for v in TARGET_RELEVANT_VARS:
    cat_to_vars.setdefault(assign_category(v), []).append(v)

print("=== 타겟관련 15개 변수를 카테고리로 분리 ===")
for cat, vs in cat_to_vars.items():
    print(f"  {cat} ({len(vs)}개): {vs}")

A = cat_to_vars.get("A_유동인구", [])
B = cat_to_vars.get("B_산업", [])
C = cat_to_vars.get("C_매출", [])

GROUPINGS = {
    "A_유동인구만": A,
    "B_산업만": B,
    "C_매출만": C,
    "AB_유동인구산업": A + B,
    "AC_유동인구매출": A + C,
    "BC_산업매출": B + C,
    "ABC_전체": A + B + C,
}

SUBSET_SIZES = [4, 5, 6]
MAX_COMBOS_PER_GROUP_SIZE = 100
K_RANGE = range(3, 7)
ALGOS = ["KMeans", "Ward"]
MIN_CLUSTER_FRACTION = 0.03
BOOT_ITER = 15
BOOT_SAMPLE_FRAC = 0.8
N_JOBS = -1  # 40번과 동일하게 전체 코어 사용 (43_1번 이전 버전엔 이게 빠져있었음)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def normalize_trdar(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


# ----------------------------------------------------------------------
# 데이터 로드 (위치 타겟은 여기서 미리 numpy 배열로 변환해둔다 - 조합마다
# 25만 행짜리 데이터프레임을 복사(loc.copy())하던 게 병목이었음)
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
print(f"\n타겟 위치: {len(loc):,}개")

# ----------------------------------------------------------------------
# 그룹별 세트 생성: pool이 작으면(<4) 통째로 하나, 크면 4~6개 부분집합 샘플링
# ----------------------------------------------------------------------
combos = []  # (그룹이름, 변수튜플)
for group_name, pool in GROUPINGS.items():
    if len(pool) == 0:
        continue
    if len(pool) < SUBSET_SIZES[0]:
        combos.append((group_name, tuple(pool)))  # 통째로
        continue
    for size in SUBSET_SIZES:
        if len(pool) < size:
            continue
        all_combos = list(itertools.combinations(pool, size))
        if len(all_combos) <= MAX_COMBOS_PER_GROUP_SIZE:
            picked = all_combos
        else:
            picked = [all_combos[i] for i in
                      random.sample(range(len(all_combos)), MAX_COMBOS_PER_GROUP_SIZE)]
        combos.extend([(group_name, c) for c in picked])

print(f"세트 {len(combos)}개 생성 (그룹별 분포):")
print(pd.Series([g for g, _ in combos]).value_counts())


def fit_labels(X, algo, k):
    if algo == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    if algo == "Ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    raise ValueError(algo)


def bootstrap_stability(X, algo, k):
    n = X.shape[0]
    base = fit_labels(X, algo, k)
    rng = np.random.RandomState(RANDOM_SEED)
    aris = []
    for _ in range(BOOT_ITER):
        idx = rng.choice(n, size=int(n * BOOT_SAMPLE_FRAC), replace=False)
        sub = fit_labels(X[idx], algo, k)
        aris.append(adjusted_rand_score(base[idx], sub))
    return float(np.mean(aris)), float(np.std(aris))


def cramers_v(chi2, n, shape):
    r, c = shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def epsilon_squared(h_stat, n, k_groups):
    return max(0.0, (h_stat - k_groups + 1) / (n - k_groups))


# ----------------------------------------------------------------------
# 조합 하나 처리 (병렬 워커에서 호출됨)
# ----------------------------------------------------------------------
def evaluate_one(group_name, var_list, algo, k, df, loc_idx_valid, loc_binary_valid, loc_cont_valid):
    X_raw = df[var_list].to_numpy()
    X = StandardScaler().fit_transform(X_raw)

    labels = fit_labels(X, algo, k)
    sizes = np.bincount(labels)
    min_frac = sizes.min() / len(labels)
    if min_frac < MIN_CLUSTER_FRACTION:
        return None

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ari_mean, ari_std = bootstrap_stability(X, algo, k)

    # dataframe copy/map/groupby 대신 numpy 배열 인덱싱으로 한 번에 join
    cluster_per_loc = labels[loc_idx_valid]
    uniq = np.unique(cluster_per_loc)
    if len(uniq) < k:
        return None

    table = pd.crosstab(cluster_per_loc, loc_binary_valid)
    chi2, p_chi, _, _ = chi2_contingency(table)
    v = cramers_v(chi2, table.values.sum(), table.shape)

    groups = [loc_cont_valid[cluster_per_loc == g] for g in uniq]
    h_stat, p_kw = kruskal(*groups)
    eps2 = epsilon_squared(h_stat, len(cluster_per_loc), len(uniq))

    rates = [loc_binary_valid[cluster_per_loc == g].mean() for g in uniq]
    rate_range = max(rates) - min(rates)

    return {
        "그룹": group_name,
        "변수목록": ";".join(var_list), "변수개수": len(var_list),
        "알고리즘": algo, "k": k,
        "실루엣": round(sil, 4), "CH": round(ch, 2), "DB": round(db, 4),
        "안정성ARI_평균": round(ari_mean, 4),
        "최소군집비율": round(min_frac, 4),
        "CramersV": round(v, 4), "카이제곱_p": p_chi,
        "epsilon_squared": round(eps2, 4), "KW_p": p_kw,
        "군집별_고교체비율_범위": round(rate_range, 4),
    }


# ----------------------------------------------------------------------
# 메인: 병렬 실행 (40번과 동일하게 joblib 사용 - 이전 버전엔 이게 빠져있었음)
# ----------------------------------------------------------------------
jobs = []
for group_name, combo in combos:
    var_list = list(combo)
    for algo in ALGOS:
        for k in K_RANGE:
            jobs.append(delayed(evaluate_one)(
                group_name, var_list, algo, k, df, loc_idx_valid, loc_binary_valid, loc_cont_valid
            ))

print(f"\n총 {len(jobs):,}개 작업을 병렬 실행 (N_JOBS={N_JOBS})...")
raw_results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=5)(jobs)
results = [r for r in raw_results if r is not None]

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(["CramersV", "안정성ARI_평균"], ascending=False)
result_df.to_csv("43_1_subset_search_results.csv", index=False, encoding="utf-8-sig")

print(f"\n균형 통과 조합: {len(result_df):,}개 -> 43_1_subset_search_results.csv")
print("\n=== Cramér's V 상위 20 ===")
cols = ["그룹", "변수목록", "변수개수", "알고리즘", "k", "CramersV", "epsilon_squared",
        "군집별_고교체비율_범위", "실루엣", "안정성ARI_평균"]
print(result_df[cols].head(20).to_string(index=False))
