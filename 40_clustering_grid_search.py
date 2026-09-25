# -*- coding: utf-8 -*-
"""
40_clustering_grid_search.py

39_candidate_variable_sets.csv에 있는 후보 변수셋 전부에 대해
표준화/PCA 두 변형 x KMeans/Ward 두 알고리즘을 그리드서치한다. k=3~6.
각 조합마다 실루엣/CH/DB + 부트스트랩 안정성ARI를 계산한다.

원래는 GMM/DBSCAN까지 4개 알고리즘, k=2~10을 다 돌렸는데:
  - k=2는 이상치 하나가 "그 점 vs 나머지"로 쪼개지며 실루엣을 인위적으로
    부풀리는 사고가 실제로 있었음 (TRDAR_CD 3120117, 생활인구당매출/점포당매출)
  - k=7~10은 예전 그리드서치 결과에서 안정성ARI가 급격히 떨어지는 구간으로
    이미 확인됨 (k=6 0.76대 -> k=9 0.45대)
  - GMM은 같은 예전 결과에서 실루엣·안정성 둘 다 KMeans/Ward보다 낮고
    ConvergenceWarning도 반복 발생
  - DBSCAN은 군집 수를 알고리즘이 정하므로 "4개 유형"이라는 목표와 맞지 않음
근거가 뚜렷한 것만 남겨서 k=3~6 x KMeans/Ward 두 개로 줄였다. 그래도
1,822개 세트를 다 도니 계산량은 여전히 크다 (병렬 실행 전제).

무지성으로 다 해보고 나중에 상위권을 골라서 해석하는 전략이므로, 이 스크립트는
"점수만" 계산한다. 해석(변수 프로파일링, ANOVA)은 다음 단계(41번)에서 한다.

출력:
  40_grid_search_results.csv   : 조합별 실루엣/CH/DB/안정성ARI (append, resumable)
  40_grid_search_summary.csv   : 위 결과를 세트별 최고 조합으로 요약 + 상위 N개
"""

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
SCREENED_DATA_PATH = "39_screened_variables.csv"
CANDIDATE_SETS_PATH = "39_candidate_variable_sets.csv"
ID_COL = "TRDAR_CD"

RESULT_PATH = "40_grid_search_results.csv"
SUMMARY_PATH = "40_grid_search_summary.csv"

K_RANGE = range(3, 7)           # KMeans/Ward: k=3~6 (근거는 위 docstring 참고)
PCA_VARIANCE = 0.90             # PCA 변형 기준 설명분산

BOOT_ITER = 15          # 부트스트랩 안정성 반복 횟수 (조합이 워낙 많아서 39/40번의
                         # 30회보다 줄임 - 늘리고 싶으면 여기만 바꾸면 됨)
BOOT_SAMPLE_FRAC = 0.8  # 재표본 비율
RANDOM_SEED = 42

N_JOBS = -1  # 병렬 실행 코어 수. -1이면 전체 코어 사용. 메모리 부족하면 줄일 것.

np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
candidates = pd.read_csv(CANDIDATE_SETS_PATH)
print(f"상권 {len(df):,}개, 후보 변수셋 {len(candidates):,}개 로드")

# 이미 계산된 조합은 건너뛰기 (재실행/중단 대비)
done_keys = set()
if os.path.exists(RESULT_PATH):
    prev = pd.read_csv(RESULT_PATH)
    done_keys = set(
        zip(prev["set_id"], prev["변형"], prev["알고리즘"], prev["파라미터"].astype(str))
    )
    print(f"기존 결과 {len(prev):,}행 발견, 이어서 진행 (건너뛸 조합 {len(done_keys):,}개)")


# ----------------------------------------------------------------------
# 변형(표준화 / PCA) 생성
# ----------------------------------------------------------------------
def build_variants(X_raw):
    variants = {}
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    variants["표준화"] = X_std

    if X_std.shape[1] >= 2:
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_std)
        if X_pca.shape[1] >= 2:
            variants[f"PCA{X_pca.shape[1]}차원"] = X_pca
    return variants


# ----------------------------------------------------------------------
# 알고리즘별 라벨 생성
# ----------------------------------------------------------------------
def fit_labels(X, algo, params):
    if algo == "KMeans":
        model = KMeans(n_clusters=params["k"], n_init=10, random_state=RANDOM_SEED)
        return model.fit_predict(X)
    if algo == "Ward":
        model = AgglomerativeClustering(n_clusters=params["k"], linkage="ward")
        return model.fit_predict(X)
    raise ValueError(algo)


# ----------------------------------------------------------------------
# 지표 계산
# ----------------------------------------------------------------------
def compute_metrics(X, labels):
    mask = labels != -1  # DBSCAN 노이즈 제외
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() < n_clusters + 1:
        return None
    try:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
    except Exception:
        return None
    noise_ratio = 1 - mask.mean()
    return sil, ch, db, n_clusters, noise_ratio


def bootstrap_stability(X, algo, params):
    n = X.shape[0]
    base_labels = fit_labels(X, algo, params)
    aris = []
    rng = np.random.RandomState(RANDOM_SEED)
    for _ in range(BOOT_ITER):
        idx = rng.choice(n, size=int(n * BOOT_SAMPLE_FRAC), replace=False)
        try:
            sub_labels = fit_labels(X[idx], algo, params)
            aris.append(adjusted_rand_score(base_labels[idx], sub_labels))
        except Exception:
            continue
    if not aris:
        return np.nan, np.nan
    return float(np.mean(aris)), float(np.std(aris))


# ----------------------------------------------------------------------
# 후보 하나(= set_id, 변형, 알고리즘, 파라미터 하나) 처리
# ----------------------------------------------------------------------
def evaluate_one(set_id, category, var_list_str, variant_name, X_variant, algo, params):
    key = (set_id, variant_name, algo, str(params))
    if key in done_keys:
        return None

    labels = fit_labels(X_variant, algo, params)
    metrics = compute_metrics(X_variant, labels)
    if metrics is None:
        return None
    sil, ch, db, n_clusters, noise_ratio = metrics
    ari_mean, ari_std = bootstrap_stability(X_variant, algo, params)

    return {
        "set_id": set_id,
        "범주그룹": category,
        "변수목록": var_list_str,
        "변형": variant_name,
        "알고리즘": algo,
        "파라미터": str(params),
        "군집수": n_clusters,
        "노이즈비율": round(noise_ratio, 4),
        "실루엣": round(sil, 4),
        "CH": round(ch, 2),
        "DB": round(db, 4),
        "안정성ARI_평균": round(ari_mean, 4) if not np.isnan(ari_mean) else np.nan,
        "안정성ARI_표준편차": round(ari_std, 4) if not np.isnan(ari_std) else np.nan,
    }


def param_grid_for(algo, X_variant):
    if algo in ("KMeans", "Ward"):
        return [{"k": k} for k in K_RANGE]
    raise ValueError(algo)


# ----------------------------------------------------------------------
# 메인 루프: 후보 변수셋 x 변형 -> 조합 리스트 생성 -> 병렬 실행 -> 즉시 저장
# ----------------------------------------------------------------------
ALGOS = ["KMeans", "Ward"]

write_header = not os.path.exists(RESULT_PATH)
start_time = time.time()
total_sets = len(candidates)

for i, row in candidates.iterrows():
    set_id = row["set_id"]
    category = row["범주그룹"]
    var_list_str = row["변수목록"]
    var_list = var_list_str.split(";")

    X_raw = df[var_list].to_numpy()
    variants = build_variants(X_raw)

    jobs = []
    for variant_name, X_variant in variants.items():
        for algo in ALGOS:
            for params in param_grid_for(algo, X_variant):
                jobs.append(
                    delayed(evaluate_one)(
                        set_id, category, var_list_str, variant_name, X_variant, algo, params
                    )
                )

    results = Parallel(n_jobs=N_JOBS, prefer="processes")(jobs)
    results = [r for r in results if r is not None]

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(RESULT_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        write_header = False

    elapsed = time.time() - start_time
    print(f"[{i+1}/{total_sets}] {set_id} ({category}, {var_list_str[:40]}...) "
          f"완료 - 조합 {len(results)}개 저장, 누적 경과 {elapsed/60:.1f}분")

print(f"\n전체 완료. 결과: {RESULT_PATH}")

# ----------------------------------------------------------------------
# 요약: 각 (set_id, 변형, 알고리즘) 중 K 최적값, 그리고 전체 상위 30개
# ----------------------------------------------------------------------
full = pd.read_csv(RESULT_PATH)

# 실루엣 상위 3개 중 안정성ARI가 가장 높은 조합을 각 (set_id, 변형, 알고리즘)의 대표로 선정
# (groupby+apply로 Series를 반환하면 pandas 버전에 따라 그룹 키 컬럼이 빠지거나
#  중복 행이 생기는 경우가 있어, idxmax 기반 벡터 연산으로 처리한다)
full["_실루엣순위"] = full.groupby(["set_id", "변형", "알고리즘"])["실루엣"] \
    .rank(method="first", ascending=False)
top3 = full[full["_실루엣순위"] <= 3].copy()
rep_idx = top3.groupby(["set_id", "변형", "알고리즘"])["안정성ARI_평균"].idxmax()
rep = full.loc[rep_idx].drop(columns=["_실루엣순위"]).reset_index(drop=True)

rep_sorted = rep.sort_values(["실루엣", "안정성ARI_평균"], ascending=False)
rep_sorted.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

print(f"요약 저장: {SUMMARY_PATH} ({len(rep_sorted):,}개 대표 조합)")
print("\n=== 상위 30개 (실루엣 기준) ===")
print(rep_sorted.head(30).to_string(index=False))
