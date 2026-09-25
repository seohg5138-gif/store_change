# -*- coding: utf-8 -*-
"""
44_supervised_learning_comparison.py

41번에서 균형 통과했던 두 후보(S0075, S1028)를 KMeans(k=4)로 피팅해서 군집
라벨을 만들고, 이걸 "군집 라벨(범주형)로 넣었을 때"와 "그 군집을 만드는 데
쓴 원본 변수(연속형)를 그대로 넣었을 때" 지도학습 성능을 나란히 비교한다.

지금까지 결론(군집화가 연속적 신호를 뭉갠다)을 실루엣/Cramér's V가 아니라
실제 예측 성능(R²/AUC)으로 확인하는 단계.

같은 상권(TRDAR_CD)의 위치들이 train/test에 같이 섞이면 누수가 생기므로,
GroupKFold(groups=TRDAR_CD)로 상권 단위 분리를 강제한다.

비교하는 피처셋:
  1. S0075_군집   : S0075 변수로 만든 KMeans 군집 라벨 (원핫, 4개 범주)
  2. S0075_연속   : S0075의 원본 연속변수 그대로 (4개)
  3. S1028_군집   : S1028 변수로 만든 KMeans 군집 라벨 (원핫, 4개 범주)
  4. S1028_연속   : S1028의 원본 연속변수 그대로 (5개)
  5. 전체39개_연속 : 39개 변수 전체 (42_1번과 비교할 상한선 참고용)

모델:
  - 연속 타겟(교체율)   : Ridge, Lasso, RandomForestRegressor
  - 이진 타겟(자주바뀜여부): LogisticRegression

입력:
  39_screened_variables.csv
  24_full_location_with_district.csv

출력:
  44_supervised_comparison_results.csv
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score

SCREENED_DATA_PATH = "39_screened_variables.csv"
LOCATION_PATH = "24_full_location_with_district.csv"
ID_COL = "TRDAR_CD"
MIN_OBS_QUARTERS = 15
RANDOM_SEED = 42
N_SPLITS = 5

S0075_VARS = ["연령다양성_엔트로피", "총직장인구", "상주인구_여성비율", "시간대비율_21_24"]
S1028_VARS = ["주말인구비율", "연령다양성_엔트로피", "총직장인구", "상주인구_여성비율", "연령비율_20대"]


def normalize_trdar(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


# ----------------------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(SCREENED_DATA_PATH)
df[ID_COL] = normalize_trdar(df[ID_COL])

loc = pd.read_csv(LOCATION_PATH, dtype={"TRDAR_CD": str})
loc = loc[loc["관측분기수"] >= MIN_OBS_QUARTERS].copy()
loc[ID_COL] = normalize_trdar(loc[ID_COL])
loc = loc[loc[ID_COL].isin(df[ID_COL])].reset_index(drop=True)
print(f"위치(신뢰가능 & universe 내): {len(loc):,}개, 상권: {loc[ID_COL].nunique():,}개")
print(f"자주바뀜여부 비율: {loc['자주바뀜여부'].mean():.4f}")

# ----------------------------------------------------------------------
# 2. S0075/S1028 KMeans(k=4) 군집 라벨 생성 (상권 단위)
# ----------------------------------------------------------------------
def make_cluster_labels(var_list):
    X = StandardScaler().fit_transform(df[var_list].to_numpy())
    labels = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_SEED).fit_predict(X)
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"  {var_list} -> 군집 크기 {sizes.to_dict()}")
    return labels


df["S0075_군집"] = make_cluster_labels(S0075_VARS)
df["S1028_군집"] = make_cluster_labels(S1028_VARS)

# ----------------------------------------------------------------------
# 3. 상권 단위 값을 위치 단위로 전개 (merge)
# ----------------------------------------------------------------------
merge_cols = [ID_COL, "S0075_군집", "S1028_군집"] + S0075_VARS + \
             [c for c in S1028_VARS if c not in S0075_VARS]
loc_full = loc.merge(df[merge_cols].drop_duplicates(subset=ID_COL), on=ID_COL, how="left")
loc_full = loc_full.dropna(subset=merge_cols[1:])
groups = loc_full[ID_COL].to_numpy()
y_reg = loc_full["교체율"].to_numpy()
y_clf = loc_full["자주바뀜여부"].to_numpy()

print(f"최종 모델링 대상 위치: {len(loc_full):,}개")


# ----------------------------------------------------------------------
# 4. 피처셋 구성
# ----------------------------------------------------------------------
def onehot(col):
    enc = OneHotEncoder(sparse_output=False, drop="first")
    return enc.fit_transform(loc_full[[col]])


FEATURE_SETS = {
    "S0075_군집": onehot("S0075_군집"),
    "S0075_연속": loc_full[S0075_VARS].to_numpy(),
    "S1028_군집": onehot("S1028_군집"),
    "S1028_연속": loc_full[S1028_VARS].to_numpy(),
    "전체39개_연속": loc_full[[c for c in df.columns if c != ID_COL]].to_numpy()
        if set(df.columns) - {ID_COL} <= set(loc_full.columns) else None,
}
# 39개 전체가 loc_full에 다 merge 안 돼있을 수 있어서 별도 처리
all_vars = [c for c in df.columns if c != ID_COL]
loc_full_all = loc.merge(df[[ID_COL] + all_vars], on=ID_COL, how="left").dropna(subset=all_vars)
FEATURE_SETS["전체39개_연속"] = loc_full_all[all_vars].to_numpy()


# ----------------------------------------------------------------------
# 5. 모델 평가 (GroupKFold, 상권 단위 분리)
# ----------------------------------------------------------------------
def eval_regression(X, y, groups, model_builder):
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    return np.mean(scores), np.std(scores)


def eval_classification(X, y, groups, model_builder):
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], proba))
    return np.mean(scores), np.std(scores)


results = []
for name, X in FEATURE_SETS.items():
    if X is None:
        continue
    y_reg_use, groups_use = (y_reg, groups) if name != "전체39개_연속" else \
        (loc_full_all["교체율"].to_numpy(), loc_full_all[ID_COL].to_numpy())
    y_clf_use = y_clf if name != "전체39개_연속" else loc_full_all["자주바뀜여부"].to_numpy()

    for model_name, builder, kind in [
        ("Ridge", lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED), "reg"),
        ("Lasso", lambda: Lasso(alpha=0.01, random_state=RANDOM_SEED, max_iter=5000), "reg"),
        ("RandomForest", lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1), "reg"),
    ]:
        mean_r2, std_r2 = eval_regression(X, y_reg_use, groups_use, builder)
        results.append({"피처셋": name, "모델": model_name, "타겟": "교체율",
                         "지표": "R2", "평균": round(mean_r2, 4), "표준편차": round(std_r2, 4)})
        print(f"[{name}] {model_name} (교체율, R²): {mean_r2:.4f} (±{std_r2:.4f})")

    mean_auc, std_auc = eval_classification(
        X, y_clf_use, groups_use,
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    results.append({"피처셋": name, "모델": "LogisticRegression", "타겟": "자주바뀜여부",
                     "지표": "AUC", "평균": round(mean_auc, 4), "표준편차": round(std_auc, 4)})
    print(f"[{name}] LogisticRegression (자주바뀜여부, AUC): {mean_auc:.4f} (±{std_auc:.4f})")

result_df = pd.DataFrame(results)
result_df.to_csv("44_supervised_comparison_results.csv", index=False, encoding="utf-8-sig")

print("\n=== 전체 결과 ===")
print(result_df.to_string(index=False))
print("\n저장: 44_supervised_comparison_results.csv")
