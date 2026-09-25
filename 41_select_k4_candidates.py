# -*- coding: utf-8 -*-
"""
41_select_k4_candidates.py

40번 결과(40_grid_search_results.csv)에서 k=4만 남기고, 실루엣+안정성ARI로
정렬한 뒤, 상위 후보들만 실제로 다시 피팅해서 "군집 크기가 한쪽으로 쏠리지
않았는지"(최소 군집 비율)를 검사한다. 이 균형 체크가 없으면 이상치 하나가
만드는 가짜 군집을 걸러내지 못한다 (39번에서 생활인구당매출/점포당매출을
윈저라이징해놨지만, 다른 변수에서 비슷한 문제가 또 없으리라는 보장은 없음).

출력:
  41_top_k4_candidates.csv : 상위 후보 + 군집별 크기 + 최소군집비율
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCREENED_DATA_PATH = "39_screened_variables.csv"
RESULT_PATH = "40_grid_search_results.csv"

TOP_N_TO_CHECK = 50          # 실루엣+안정성ARI 상위 몇 개를 실제로 재피팅해서 검증할지
MIN_CLUSTER_FRACTION = 0.03  # 가장 작은 군집이 전체의 이 비율보다 작으면 "쏠림"으로 간주
RANDOM_SEED = 42

full = pd.read_csv(RESULT_PATH)

# k=4만
k4 = full[full["군집수"] == 4].copy()
print(f"k=4 조합: {len(k4)}개")

k4_sorted = k4.sort_values(["실루엣", "안정성ARI_평균"], ascending=False)
top = k4_sorted.head(TOP_N_TO_CHECK).copy()


# --- 상위 후보 실제 재피팅해서 군집 균형 체크 ---
def refit_and_check_balance(row, df_lookup):
    var_list = row["변수목록"].split(";")
    X_raw = df_lookup[var_list].to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    if row["변형"].startswith("PCA"):
        n_comp = int(row["변형"].replace("PCA", "").replace("차원", ""))
        X = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(X_std)
    else:
        X = X_std

    algo = row["알고리즘"]
    if algo == "KMeans":
        labels = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    elif algo == "Ward":
        labels = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
    else:
        return None

    sizes = pd.Series(labels).value_counts().sort_index()
    min_fraction = sizes.min() / len(labels)
    return sizes.to_dict(), min_fraction


df_main = pd.read_csv(SCREENED_DATA_PATH)

results = []
for _, row in top.iterrows():
    out = refit_and_check_balance(row, df_main)
    if out is None:
        continue
    sizes, min_fraction = out
    results.append({
        **row.to_dict(),
        "군집별크기": sizes,
        "최소군집비율": round(min_fraction, 4),
        "쏠림의심": min_fraction < MIN_CLUSTER_FRACTION,
    })

result_df = pd.DataFrame(results)
result_df.to_csv("41_top_k4_candidates.csv", index=False, encoding="utf-8-sig")

n_skewed = result_df["쏠림의심"].sum()
print(f"\n상위 {len(result_df)}개 중 쏠림 의심(최소군집비율 < {MIN_CLUSTER_FRACTION}): {n_skewed}개")
print("\n=== 쏠림 없는 상위 15개 ===")
clean = result_df[~result_df["쏠림의심"]].sort_values(["실루엣", "안정성ARI_평균"], ascending=False)
print(clean[["set_id", "범주그룹", "변수목록", "변형", "알고리즘", "실루엣",
             "안정성ARI_평균", "최소군집비율"]].head(15).to_string(index=False))

print(f"\n저장: 41_top_k4_candidates.csv")

