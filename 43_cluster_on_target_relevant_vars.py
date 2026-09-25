# -*- coding: utf-8 -*-
"""
43_cluster_on_target_relevant_vars.py

42_1번에서 "상권내_고교체비율"과 스피어만 상관이 높았던 15개 변수만으로
클러스터링한다. 무작정 900~1,822개 후보를 돌리는 대신, 처음부터 타겟과
관련된 변수로 군집을 나누니 해석("이 군집은 고교체율이 높은/낮은 이유가
설명되는 유형이다")이 자연스럽게 붙는다.

1. 15개 변수로 미니 그리드서치 (표준화/PCA x KMeans/Ward x k=3~6)
   - 조합이 16개뿐이라 부트스트랩 반복을 39/40번보다 넉넉히 준다
2. 균형(최소군집비율) 통과한 것 중 실루엣+안정성ARI 최고를 선택
3. 선택된 군집으로:
   - 39개 변수 전체 평균을 군집별로 비교 (표 + Kruskal-Wallis p-value)
   - 군집별 고교체비율/평균교체율도 같이 계산 (이번엔 진짜 관련 있을 것으로 기대)
   - 히트맵(군집 x 변수, z-score) 이미지 저장

입력:
  39_screened_variables.csv
  24_full_location_with_district.csv

출력:
  43_cluster_profile.csv       : 군집별 39개 변수 평균 + Kruskal-Wallis p-value
  43_cluster_target_summary.csv: 군집별 위치수/고교체비율/평균교체율
  43_cluster_heatmap.png       : 군집 x 변수 z-score 히트맵
  43_cluster_assignment.csv    : 상권(TRDAR_CD)별 군집 라벨
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCREENED_DATA_PATH = "39_screened_variables.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"
MIN_OBS_QUARTERS = 15
RANDOM_SEED = 42
MIN_CLUSTER_FRACTION = 0.03
K_RANGE = range(3, 7)
BOOT_ITER = 30          # 변수 하나로 고정이라 넉넉히 (39/40번은 조합이 많아서 줄였었음)
BOOT_SAMPLE_FRAC = 0.8

# 42_1번 스피어만 상관 상위 15개 (|rho| 기준)
TARGET_RELEVANT_VARS = [
    "업종비중_과학·기술", "시간대비율_06_11", "전체점포수_로그", "업종비중_예술·스포츠",
    "시간대비율_21_24", "매출변동성_CV", "주말인구비율", "연령비율_40대",
    "건당매출액", "총가구수", "총상주인구_로그", "시간대비율_11_14",
    "업종비중_수리·개인", "연령비율_30대", "총생활인구_로그",
]

for name in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    try:
        fm.findfont(name, fallback_to_default=False)
        plt.rcParams["font.family"] = name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def normalize_trdar(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


# ----------------------------------------------------------------------
# 1. 데이터 로드 + 15개 변수로 미니 그리드서치
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
df[ID_COL] = normalize_trdar(df[ID_COL])

missing = [v for v in TARGET_RELEVANT_VARS if v not in df.columns]
if missing:
    raise ValueError(f"39_screened_variables.csv에 없는 변수: {missing}")

X_raw = df[TARGET_RELEVANT_VARS].to_numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)
pca = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_std)
print(f"PCA(90%): {len(TARGET_RELEVANT_VARS)}차원 -> {X_pca.shape[1]}차원")

variants = {"표준화": X_std, f"PCA{X_pca.shape[1]}차원": X_pca}


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


rows = []
for variant_name, X in variants.items():
    for algo in ["KMeans", "Ward"]:
        for k in K_RANGE:
            labels = fit_labels(X, algo, k)
            sizes = pd.Series(labels).value_counts()
            min_frac = sizes.min() / len(labels)
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ari_mean, ari_std = bootstrap_stability(X, algo, k)
            rows.append({
                "변형": variant_name, "알고리즘": algo, "k": k,
                "실루엣": round(sil, 4), "CH": round(ch, 2), "DB": round(db, 4),
                "안정성ARI_평균": round(ari_mean, 4), "안정성ARI_표준편차": round(ari_std, 4),
                "최소군집비율": round(min_frac, 4), "쏠림의심": min_frac < MIN_CLUSTER_FRACTION,
                "labels": labels,
            })

grid = pd.DataFrame(rows)
print("\n=== 15개 변수 미니 그리드서치 결과 ===")
print(grid.drop(columns="labels").sort_values(["실루엣", "안정성ARI_평균"], ascending=False).to_string(index=False))

# ----------------------------------------------------------------------
# 2. 균형 통과 & k=4 우선으로 최종 선택 (프로젝트 목표가 4개 유형이므로)
# ----------------------------------------------------------------------
candidates_k4 = grid[(~grid["쏠림의심"]) & (grid["k"] == 4)]
if len(candidates_k4):
    best = candidates_k4.sort_values(["실루엣", "안정성ARI_평균"], ascending=False).iloc[0]
    print(f"\nk=4 & 균형통과 후보 중 최선 선택: {best['변형']} / {best['알고리즘']}")
else:
    clean = grid[~grid["쏠림의심"]]
    if len(clean) == 0:
        raise RuntimeError("균형 통과하는 조합이 하나도 없음 - MIN_CLUSTER_FRACTION을 낮추거나 변수셋을 재검토할 것")
    best = clean.sort_values(["실루엣", "안정성ARI_평균"], ascending=False).iloc[0]
    print(f"\nk=4는 전부 쏠림 -> 균형 통과한 것 중 최선 선택: "
          f"{best['변형']} / {best['알고리즘']} / k={best['k']}")

df["군집"] = best["labels"]
df[[ID_COL, "군집"]].to_csv("43_cluster_assignment.csv", index=False, encoding="utf-8-sig")
print(f"군집 크기: {pd.Series(best['labels']).value_counts().sort_index().to_dict()}")

# ----------------------------------------------------------------------
# 3. 39개 변수 전체로 군집 프로파일링 (Kruskal-Wallis)
# ----------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in (ID_COL, "군집")]
profile_rows = []
for c in feature_cols:
    group_means = df.groupby("군집")[c].mean()
    groups = [g[c].values for _, g in df.groupby("군집")]
    try:
        h_stat, p = kruskal(*groups)
    except ValueError:
        h_stat, p = np.nan, np.nan
    row = {"변수": c, "타겟관련변수": c in TARGET_RELEVANT_VARS, "KW_p": p}
    for cl, m in group_means.items():
        row[f"군집{cl}_평균"] = round(m, 4)
    profile_rows.append(row)

profile_df = pd.DataFrame(profile_rows).sort_values("KW_p")
profile_df.to_csv("43_cluster_profile.csv", index=False, encoding="utf-8-sig")

print("\n=== 군집별로 가장 뚜렷하게 갈리는 변수 상위 15개 (KW p-value 기준) ===")
print(profile_df.head(15).to_string(index=False))

# ----------------------------------------------------------------------
# 4. 위치 단위 타겟과 군집 관계 (이번엔 관련 있을 것으로 기대)
# ----------------------------------------------------------------------
loc = pd.read_csv(LOCATION_PATH, dtype={"TRDAR_CD": str})
loc = loc[loc["관측분기수"] >= MIN_OBS_QUARTERS].copy()
loc[ID_COL] = normalize_trdar(loc[ID_COL])

label_map = dict(zip(df[ID_COL], df["군집"]))
loc["군집"] = loc[ID_COL].map(label_map)
loc_c = loc.dropna(subset=["군집"])

target_summary = loc_c.groupby("군집").agg(
    위치수=("위치ID", "size"),
    고교체비율=("자주바뀜여부", "mean"),
    평균교체율=("교체율", "mean"),
).reset_index()
target_summary.to_csv("43_cluster_target_summary.csv", index=False, encoding="utf-8-sig")
print("\n=== 군집별 타겟 요약 ===")
print(target_summary.to_string(index=False))

table = pd.crosstab(loc_c["군집"], loc_c["자주바뀜여부"])
from scipy.stats import chi2_contingency
chi2, p_chi, _, _ = chi2_contingency(table)
n = table.values.sum()
v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
print(f"\n카이제곱(군집 x 자주바뀜여부): p={p_chi:.2e}, Cramér's V={v:.4f}")

# ----------------------------------------------------------------------
# 5. 히트맵
# ----------------------------------------------------------------------
z_table = df.groupby("군집")[feature_cols].mean()
z_table = (z_table - z_table.mean()) / z_table.std()

fig, ax = plt.subplots(figsize=(10, max(8, len(feature_cols) * 0.28)))
im = ax.imshow(z_table.T.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(z_table.index)))
ax.set_xticklabels([f"군집{c}" for c in z_table.index])
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels(feature_cols, fontsize=8)
ax.set_title("군집별 변수 프로파일 (z-score)")
fig.colorbar(im, ax=ax, label="z-score")
fig.tight_layout()
fig.savefig("43_cluster_heatmap.png", dpi=150)
plt.close(fig)

print("\n저장: 43_cluster_profile.csv, 43_cluster_target_summary.csv, "
      "43_cluster_heatmap.png, 43_cluster_assignment.csv")
