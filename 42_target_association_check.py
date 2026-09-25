# -*- coding: utf-8 -*-
"""
42_target_association_check.py

지금까지(39~41번)는 클러스터링 자체의 통계적 품질(실루엣·안정성·군집균형)만
봤다. 이 스크립트는 "그 군집이 실제로 고교체율을 설명하는가"를 본다.

방법: 위치(위치ID) 단위 타겟(자주바뀜여부, 교체율)을 그 위치가 속한 상권
(TRDAR_CD)의 군집 라벨에 연결한 뒤:
  - 카이제곱검정: 군집(4개 범주) x 자주바뀜여부(이진) 독립성 검정
  - 크루스칼-왈리스: 군집별 교체율(연속) 분포 차이 검정
표본이 30만 개 이상이라 p-value는 거의 항상 유의하게 나온다(그래서 p-value로
순위를 매기지 않는다). 대신 효과크기(Cramér's V, epsilon-squared)로 줄을
세운다 - "통계적으로 유의한가"가 아니라 "실제로 차이가 큰가"를 보기 위함.

40번 k=4 결과 전부(수천 개)를 이 저비용 검정으로 훑은 뒤, 상위 N개만
다음 단계(43번, 지도학습 비교)로 넘긴다. 클러스터링 재피팅 자체는
초 단위라 전수(k=4 전체)를 다 돌려도 부담이 크지 않다.

주의: 24_full_location_with_district.csv의 TRDAR_CD는 매칭 안 되는 위치가
about 19% 있다(상권 폴리곤 커버리지 공백, 프로젝트에서 이미 편향 아님을
확인한 부분). 여기서도 그 비매칭 위치는 자동으로 빠진다.

입력:
  39_screened_variables.csv          - 상권(TRDAR_CD) 단위 피처
  40_grid_search_results.csv          - 클러스터링 그리드서치 결과
  24_full_location_with_district.csv  - 위치 단위 타겟 + TRDAR_CD

출력:
  42_target_association_results.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCREENED_DATA_PATH = "39_screened_variables.csv"
GRID_RESULTS_PATH = "40_grid_search_results.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"

MIN_CLUSTER_FRACTION = 0.03   # 41번과 동일한 쏠림 기준 - 여기서도 걸러낸다
MIN_OBS_QUARTERS = 15         # 신뢰가능 위치 기준 (프로젝트 관례와 동일)
RANDOM_SEED = 42

OUTPUT_PATH = "42_target_association_results.csv"


def normalize_trdar(series):
    """TRDAR_CD를 정수 문자열로 통일 (한쪽은 float, 한쪽은 str로 읽히는 문제 방지)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


# ----------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
df[ID_COL] = normalize_trdar(df[ID_COL])

grid = pd.read_csv(GRID_RESULTS_PATH)
k4 = grid[grid["군집수"] == 4].copy()

# 40_grid_search_results.csv가 이전 버전의 39_screened_variables.csv 기준으로
# 생성된 채 남아있으면(예: 39번 재실행으로 컬럼이 바뀐 뒤 40번을 안 돌린 경우),
# 지금은 없는 변수를 참조하는 조합이 섞여있을 수 있다. 크래시 대신 건너뛴다.
screened_cols = set(df.columns)
has_all_vars = k4["변수목록"].apply(
    lambda s: all(v in screened_cols for v in s.split(";"))
)
n_stale = (~has_all_vars).sum()
if n_stale:
    stale_vars = sorted({
        v for s in k4.loc[~has_all_vars, "변수목록"] for v in s.split(";")
        if v not in screened_cols
    })
    print(f"[경고] {SCREENED_DATA_PATH}에 없는 변수를 참조하는 조합 {n_stale}개 건너뜀 "
          f"({GRID_RESULTS_PATH}가 이전 버전의 {SCREENED_DATA_PATH} 기준으로 생성된 듯) "
          f"- 없는 변수: {stale_vars}")
    k4 = k4[has_all_vars].copy()

print(f"k=4 조합: {len(k4):,}개 (39_variable_screening.py + 40_clustering_grid_search.py 결과)")

loc = pd.read_csv(LOCATION_PATH, dtype={"TRDAR_CD": str})
loc = loc[loc["관측분기수"] >= MIN_OBS_QUARTERS].copy()
loc[ID_COL] = normalize_trdar(loc[ID_COL])
loc = loc[loc[ID_COL].isin(df[ID_COL])]  # 우리 클러스터링 universe(1,550개 상권)로 제한
print(f"타겟 위치(신뢰가능 & 우리 상권 universe 내): {len(loc):,}개")
print(f"  자주바뀜 비율(전체 평균): {loc['자주바뀜여부'].mean():.4f}")


# ----------------------------------------------------------------------
# 후보 하나 처리: 재피팅 -> 균형체크 -> 위치에 라벨 join -> 카이제곱/크루스칼-왈리스
# ----------------------------------------------------------------------
def refit_labels(row, df_lookup):
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
        return KMeans(n_clusters=4, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    if algo == "Ward":
        return AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
    raise ValueError(algo)


def cramers_v(chi2, n, table_shape):
    r, c = table_shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def epsilon_squared(h_stat, n, k_groups):
    # Kruskal-Wallis 효과크기 (0~1, ANOVA의 eta-squared에 대응)
    return max(0.0, (h_stat - k_groups + 1) / (n - k_groups))


results = []
for i, (_, row) in enumerate(k4.iterrows()):
    labels = refit_labels(row, df)
    sizes = pd.Series(labels).value_counts()
    if sizes.min() / len(labels) < MIN_CLUSTER_FRACTION:
        continue  # 쏠린 군집은 스킵 (41번과 같은 기준)

    label_map = dict(zip(df[ID_COL], labels))
    loc_c = loc.copy()
    loc_c["군집"] = loc_c[ID_COL].map(label_map)
    loc_c = loc_c.dropna(subset=["군집"])
    if loc_c["군집"].nunique() < 4:
        continue  # 위치 단위로 봤을 때 4개 군집이 다 살아있지 않으면 스킵

    # --- 카이제곱: 군집 x 자주바뀜여부 ---
    table = pd.crosstab(loc_c["군집"], loc_c["자주바뀜여부"])
    chi2, p_chi, _, _ = chi2_contingency(table)
    v = cramers_v(chi2, table.values.sum(), table.shape)

    # --- 크루스칼-왈리스: 군집별 교체율 ---
    groups = [g["교체율"].values for _, g in loc_c.groupby("군집")]
    h_stat, p_kw = kruskal(*groups)
    eps2 = epsilon_squared(h_stat, len(loc_c), loc_c["군집"].nunique())

    by_cluster = loc_c.groupby("군집")["자주바뀜여부"].mean()

    results.append({
        "set_id": row["set_id"],
        "범주그룹": row["범주그룹"],
        "변수목록": row["변수목록"],
        "변형": row["변형"],
        "알고리즘": row["알고리즘"],
        "실루엣": row["실루엣"],
        "안정성ARI_평균": row["안정성ARI_평균"],
        "위치수_매칭": len(loc_c),
        "카이제곱_p": p_chi,
        "CramersV": round(v, 4),
        "KW_p": p_kw,
        "epsilon_squared": round(eps2, 4),
        "군집별_고교체비율_최소": round(by_cluster.min(), 4),
        "군집별_고교체비율_최대": round(by_cluster.max(), 4),
        "군집별_고교체비율_범위": round(by_cluster.max() - by_cluster.min(), 4),
    })

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(k4)}개 처리...")

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(["CramersV", "epsilon_squared"], ascending=False)
result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n균형 통과 & 4개 군집 다 살아있는 조합: {len(result_df):,}개 -> {OUTPUT_PATH}")
print("\n=== Cramers V(고교체여부 연관성) 상위 20 ===")
cols = ["set_id", "범주그룹", "변수목록", "변형", "알고리즘",
        "CramersV", "epsilon_squared", "군집별_고교체비율_범위", "실루엣", "안정성ARI_평균"]
print(result_df[cols].head(20).to_string(index=False))
