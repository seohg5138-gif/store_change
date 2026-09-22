import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
#vs코드 한글 반영
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)

# =========================================================================
# 40번: 39번에서 못 다룬 경우의 수를 마저 채운다.
#   - 알고리즘: GMM(가우시안 혼합), 계층적(Ward) 추가 (39번은 KMeans만)
#   - DBSCAN: k를 정하지 않는 밀도 기반 방식이라 별도 섹션으로 분리
# 39번과 같은 피처셋(A/B/AB) x 정규화(표준화/PCA90%) 틀을 그대로 써서
# 나중에 39번 결과와 합쳐서 비교할 수 있게 함.
# =========================================================================

K_RANGE = range(2, 11)
N_BOOTSTRAP = 20
RANDOM_STATE = 42

df = pd.read_csv("38_clustering_ready_final.csv", dtype={"TRDAR_CD": str})
print(f"입력: {len(df):,}개 상권")

# --- 39번과 동일한 피처셋 구성 (재사용을 위해 그대로 복붙) ---
industry_share_cols = [c for c in df.columns if c.startswith("업종비중_")]
age_share_cols = [c for c in df.columns if c.startswith("연령비율_")]
time_share_cols = [c for c in df.columns if c.startswith("시간대비율_")]

a_cols = [
    "총생활인구_로그", "여성비율", "주말인구비율",
    "연령다양성_엔트로피", "시간대다양성_엔트로피",
    "총직장인구_로그", "직장인구_여성비율",
    "총상주인구_로그", "상주인구_여성비율",
    "업무형비율",
    "분기총매출_로그", "건당매출액", "주말매출비율", "야간매출비율",
    "매출성장률", "매출변동성_CV", "생활인구당매출",
] + age_share_cols + time_share_cols
b_cols = ["전체점포수_로그", "업종다양성_엔트로피", "업종집중도_HHI", "상위3업종비중"] + industry_share_cols


def drop_reference_category(cols, df):
    if not cols:
        return cols, None
    ref = df[cols].mean().idxmax()
    return [c for c in cols if c != ref], ref


age_share_cols_trim, age_ref = drop_reference_category(age_share_cols, df)
time_share_cols_trim, time_ref = drop_reference_category(time_share_cols, df)
industry_share_cols_trim, industry_ref = drop_reference_category(industry_share_cols, df)

a_cols_trim = [c for c in a_cols if c not in age_share_cols + time_share_cols] + age_share_cols_trim + time_share_cols_trim
b_cols_trim = [c for c in b_cols if c not in industry_share_cols] + industry_share_cols_trim

FEATURE_SETS = {
    "A_방문자만": a_cols_trim,
    "B_산업만": b_cols_trim,
    "AB_전체": a_cols_trim + b_cols_trim,
}

ALGORITHMS = {
    "GMM": lambda k, seed: GaussianMixture(n_components=k, covariance_type="full", random_state=seed),
    "Ward": lambda k, seed: AgglomerativeClustering(n_clusters=k, linkage="ward"),
}


def fit_predict(model_factory, X, k, seed):
    """GMM은 fit+predict, Ward는 fit_predict만 지원 -> 인터페이스 통일."""
    model = model_factory(k, seed)
    if hasattr(model, "predict") and hasattr(model, "fit"):
        try:
            model.fit(X)
            return model.predict(X)
        except AttributeError:
            pass
    return model.fit_predict(X)


def bootstrap_stability(model_factory, X, k, n_boot=N_BOOTSTRAP, sample_frac=0.8, seed=RANDOM_STATE):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    base_labels = fit_predict(model_factory, X, k, seed)

    aris = []
    for i in range(n_boot):
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        try:
            sub_labels = fit_predict(model_factory, X[idx], k, seed + i + 1)
            aris.append(adjusted_rand_score(base_labels[idx], sub_labels))
        except Exception:
            continue
    if not aris:
        return np.nan, np.nan
    return np.mean(aris), np.std(aris)


# =========================================================================
# 1) GMM / Ward : 39번과 같은 틀(피처셋 x 정규화 x k)로 반복
# =========================================================================
results = []
for set_name, cols in FEATURE_SETS.items():
    X_raw = df[cols].to_numpy()
    X = StandardScaler().fit_transform(X_raw)

    variants = {"표준화": X}
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    variants[f"PCA({X_pca.shape[1]}차원)"] = X_pca

    for variant_name, X_v in variants.items():
        for algo_name, model_factory in ALGORITHMS.items():
            print(f"\n--- {set_name} / {variant_name} / {algo_name} ---")
            for k in K_RANGE:
                try:
                    labels = fit_predict(model_factory, X_v, k, RANDOM_STATE)
                    if len(set(labels)) < 2:
                        continue
                    sil = silhouette_score(X_v, labels)
                    ch = calinski_harabasz_score(X_v, labels)
                    db = davies_bouldin_score(X_v, labels)
                    ari_mean, ari_std = bootstrap_stability(model_factory, X_v, k)
                except Exception as e:
                    print(f"  k={k}: 실패 ({e})")
                    continue

                results.append({
                    "알고리즘": algo_name, "피처셋": set_name, "변형": variant_name, "k": k,
                    "실루엣": sil, "calinski_harabasz": ch, "davies_bouldin": db,
                    "안정성_ARI_평균": ari_mean, "안정성_ARI_표준편차": ari_std,
                })
                print(f"  k={k}: 실루엣={sil:.3f} CH={ch:.1f} DB={db:.3f} "
                      f"안정성ARI={ari_mean:.3f}(±{ari_std:.3f})")

results_df = pd.DataFrame(results)
results_df.to_csv("40_gmm_ward_results.csv", index=False, encoding="utf-8-sig")
print(f"\n저장 완료: 40_gmm_ward_results.csv ({len(results_df)}개 조합)")

# =========================================================================
# 2) DBSCAN : k 없이 eps/min_samples 그리드. eps는 k-거리 분포에서 데이터
# 기반으로 후보를 뽑는다 (임의로 찍지 않음).
# =========================================================================
dbscan_results = []
for set_name, cols in FEATURE_SETS.items():
    X_raw = df[cols].to_numpy()
    X = StandardScaler().fit_transform(X_raw)
    variants = {"표준화": X}
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    variants[f"PCA(90%)"] = pca.fit_transform(X)

    for variant_name, X_v in variants.items():
        for min_samples in [5, 10, 15]:
            # min_samples번째 최근접이웃까지의 거리 분포에서 25/50/75/90 백분위를 eps 후보로
            nn = NearestNeighbors(n_neighbors=min_samples).fit(X_v)
            dists, _ = nn.kneighbors(X_v)
            kth_dist = np.sort(dists[:, -1])
            eps_candidates = np.percentile(kth_dist, [25, 50,75, 90])

            for eps in eps_candidates:
                db_model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db_model.fit_predict(X_v)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = (labels == -1).mean()

                if n_clusters < 2:
                    dbscan_results.append({
                        "피처셋": set_name, "변형": variant_name,
                        "min_samples": min_samples, "eps": eps,
                        "군집수": n_clusters, "노이즈비율": noise_ratio,
                        "실루엣": np.nan,
                    })
                    continue

                mask = labels != -1
                sil = silhouette_score(X_v[mask], labels[mask]) if mask.sum() > 1 else np.nan
                dbscan_results.append({
                    "피처셋": set_name, "변형": variant_name,
                    "min_samples": min_samples, "eps": eps,
                    "군집수": n_clusters, "노이즈비율": noise_ratio,
                    "실루엣": sil,
                })

dbscan_df = pd.DataFrame(dbscan_results)
dbscan_df.to_csv("40_dbscan_results.csv", index=False, encoding="utf-8-sig")
print(f"저장 완료: 40_dbscan_results.csv ({len(dbscan_df)}개 조합)")
print("\n=== DBSCAN: 군집 2~6개 사이이면서 노이즈비율 30% 미만인 후보 ===")
candidates = dbscan_df[(dbscan_df["군집수"].between(2, 6)) & (dbscan_df["노이즈비율"] < 0.3)]
print(candidates.sort_values("실루엣", ascending=False).head(10).to_string(index=False))
