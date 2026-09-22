import matplotlib
matplotlib.use("Agg")  # 디스플레이 없는 환경에서도 파일 저장이 막히지 않게 고정
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score,
)
#한글 반영
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 39번: KMeans 군집 개수(k)와 피처셋 조합을 체계적으로 비교.
# =========================================================================
# "k=4가 맞다"를 전제하지 않고, k=2~10 x 피처셋 4종류를 전부 돌려서
# 4개 내부평가지표 + 부트스트랩 안정성(ARI)으로 어떤 조합이 가장
# 설명력 있는지 비교한다.

K_RANGE = range(2, 11)
N_BOOTSTRAP = 20
RANDOM_STATE = 42

df = pd.read_csv("38_clustering_ready_final.csv", dtype={"TRDAR_CD": str})
print(f"입력: {len(df):,}개 상권")

# --- A/B 피처 컬럼 분리 ---
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

# --- 구성비(합=1) 그룹에서 기준범주 하나씩 제거 (가장 흔한 값 기준) ---
def drop_reference_category(cols, df):
    if not cols:
        return cols, None
    ref = df[cols].mean().idxmax()  # 평균적으로 가장 큰 카테고리를 기준으로 제외
    return [c for c in cols if c != ref], ref

age_share_cols_trim, age_ref = drop_reference_category(age_share_cols, df)
time_share_cols_trim, time_ref = drop_reference_category(time_share_cols, df)
industry_share_cols_trim, industry_ref = drop_reference_category(industry_share_cols, df)
print(f"기준범주 제외: 연령={age_ref}, 시간대={time_ref}, 업종={industry_ref}")

a_cols_trim = [c for c in a_cols if c not in age_share_cols + time_share_cols] + age_share_cols_trim + time_share_cols_trim
b_cols_trim = [c for c in b_cols if c not in industry_share_cols] + industry_share_cols_trim

FEATURE_SETS = {
    "A_방문자만": a_cols_trim,
    "B_산업만": b_cols_trim,
    "AB_전체": a_cols_trim + b_cols_trim,
}


def bootstrap_stability(X, k, n_boot=N_BOOTSTRAP, sample_frac=0.8, seed=RANDOM_STATE):
    """전체 데이터로 기준 클러스터링을 만들고, 부분표집으로 재클러스터링해서
    ARI(Adjusted Rand Index)를 여러 번 재서 평균/표준편차를 낸다.
    ARI가 1에 가까울수록 표본이 바뀌어도 같은 구조가 나온다는 뜻(안정적)."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    base_km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    base_labels = base_km.labels_

    aris = []
    for i in range(n_boot):
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + i + 1).fit(X[idx])
        aris.append(adjusted_rand_score(base_labels[idx], km.labels_))
    return np.mean(aris), np.std(aris)


results = []
for set_name, cols in FEATURE_SETS.items():
    X_raw = df[cols].to_numpy()
    X = StandardScaler().fit_transform(X_raw)

    variants = {"표준화": X}
    # PCA 변형: 분산 90% 설명하는 선까지 축소
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    variants[f"PCA({X_pca.shape[1]}차원)"] = X_pca

    for variant_name, X_v in variants.items():
        print(f"\n--- {set_name} / {variant_name} (변수 {X_v.shape[1]}개) ---")
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_v)
            labels = km.labels_
            sil = silhouette_score(X_v, labels)
            ch = calinski_harabasz_score(X_v, labels)
            db = davies_bouldin_score(X_v, labels)
            ari_mean, ari_std = bootstrap_stability(X_v, k)

            results.append({
                "피처셋": set_name, "변형": variant_name, "k": k,
                "inertia": km.inertia_, "실루엣": sil,
                "calinski_harabasz": ch, "davies_bouldin": db,
                "안정성_ARI_평균": ari_mean, "안정성_ARI_표준편차": ari_std,
            })
            print(f"  k={k}: 실루엣={sil:.3f} CH={ch:.1f} DB={db:.3f} "
                  f"안정성ARI={ari_mean:.3f}(±{ari_std:.3f})")

results_df = pd.DataFrame(results)
results_df.to_csv("39_kmeans_model_selection_results.csv", index=False, encoding="utf-8-sig")
print(f"\n저장 완료: 39_kmeans_model_selection_results.csv ({len(results_df)}개 조합)")

# --- 조합별로 그래프 저장 (실루엣 + 안정성을 같이 보는 게 핵심) ---
for (set_name, variant_name), g in results_df.groupby(["피처셋", "변형"]):
    g = g.sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(g["k"], g["inertia"], marker="o")
    axes[0].set_title("Elbow (inertia)")
    axes[0].set_xlabel("k")

    axes[1].plot(g["k"], g["실루엣"], marker="o", color="tab:orange")
    axes[1].set_title("Silhouette (높을수록 좋음)")
    axes[1].set_xlabel("k")

    axes[2].errorbar(g["k"], g["안정성_ARI_평균"], yerr=g["안정성_ARI_표준편차"],
                      marker="o", color="tab:green", capsize=3)
    axes[2].set_title("Bootstrap 안정성 ARI (높을수록 좋음)")
    axes[2].set_xlabel("k")

    fig.suptitle(f"{set_name} / {variant_name}")
    fig.tight_layout()
    fname = f"39_plot_{set_name}_{variant_name}.png".replace("(", "").replace(")", "").replace(" ", "")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"저장: {fname}")

# --- 각 (피처셋, 변형) 조합에서 "실루엣 + 안정성"을 같이 고려한 추천 k ---
print("\n=== 조합별 추천 k (실루엣 상위 3개 중 안정성ARI 최고) ===")
for (set_name, variant_name), g in results_df.groupby(["피처셋", "변형"]):
    top3 = g.nlargest(3, "실루엣")
    best = top3.nlargest(1, "안정성_ARI_평균")
    print(f"{set_name}/{variant_name}: k={int(best['k'].iloc[0])} "
          f"(실루엣={best['실루엣'].iloc[0]:.3f}, 안정성={best['안정성_ARI_평균'].iloc[0]:.3f})")
