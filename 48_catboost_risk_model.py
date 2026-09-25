# -*- coding: utf-8 -*-
"""
48번: CatBoost로 위험 위치(상권대비 교체 상위 10%)를 분류한다. (발표자료 13~16p)

설계:
  - 입력변수: risk_common.CAT_FEATURES + NUM_FEATURES (+ EXTRA_FEATURES)
      범주형(층구분/주용도/상권유형)은 CatBoost에 그대로 넣음
  - 검증: StratifiedGroupKFold(5, groups=TRDAR_CD)
      같은 상권 위치가 train/test에 섞이면 이웃 위치 정보가 새어 들어가서 성능이 부풀려짐
      (44번과 같은 원칙). 위험 비율(10%)은 fold마다 비슷하게 유지.
  - 불균형(위험 10%): auto_class_weights="Balanced"
      -> 출력 확률은 "보정된 실제 확률"이 아니라 "위험 자리와 닮은 정도" 점수로 해석해야 함
         (0.5를 넘어도 실제 위험 확률이 50%라는 뜻은 아님)
  - 위험도(지도에 표시할 값) = out-of-fold 예측확률
      학습에 쓴 위치를 같은 모델로 예측하면 값이 과장되므로, 각 위치는 자기가 test였던 fold의 예측을 씀
  - 임계값 표: 고정값(발표자료 0.40/0.60/0.65/0.75) + 기준별 자동 선택
      F1최고 / 균형(Youden J 최대) / Recall중시(F2 최대) / Precision중시(F0.5 최대)
  - 변수 중요도: 두 가지를 같이 냄
      순열중요도(기본, 발표용) = test fold에서 그 변수 값을 섞었을 때 ROC-AUC가 얼마나 떨어지나
                                 (학습에 안 쓴 데이터 기준이라 "예측에 실제로 기여하는 정도")
      PredictionValuesChange   = 학습 중 분할에 얼마나 쓰였나. 연속형 변수가 잘게 쪼개지며 부풀려지는 경향
      * 1차 실행에서 지하철거리가 PredictionValuesChange 1위(15.9%)였지만 49번 위험비율은 거의 평평했음
        -> 발표에는 순열중요도를 씀
  - 변수셋 비교(47-1번 PCA 결과가 있으면): 같은 fold로 아래 세 가지를 돌려 성능을 비교한 뒤,
    FINAL_FEATURE_SET으로 최종 모델/위험도/중요도를 만든다.
      원변수      : 층구분·주용도·상권유형 + 수치형 원변수 (발표자료 방식)
      PCA         : 층구분·주용도·상권유형 + 수치형 대신 주성분 점수
      원변수+PCA  : 둘 다
    * 범주형은 세 경우 모두 그대로 넣음 (PCA 대상은 수치형만)
    * PCA 셋의 중요도는 "PC1, PC2..." 단위라 발표자료처럼 "연면적이 몇 %"로는 말할 수 없음

입력:  45_location_risk_base.csv, 47_location_features.csv, (선택) 47_1_pca_scores.csv
출력:  48_oof_predictions.csv, 48_model_metrics.csv, 48_threshold_table.csv,
       48_feature_importance.csv, 48_feature_importance.png, 48_catboost_final.cbm,
       48_feature_set_comparison.csv (비교 실행 시)
"""

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import risk_common as rc

N_SPLITS = 5
COMPARE_FEATURE_SETS = False         # True면 COMPARE_SETS끼리 같은 fold로 성능 비교 (업종비중 누수 크기를 보고 싶을 때)
COMPARE_SETS = ["원변수", "원변수+업종비중"]   # PCA 셋은 1차 실행에서 비교 완료
FINAL_FEATURE_SET = "원변수"          # 위치 모델은 입지 변수만. 업종 효과는 가게 단위(51~52번)에서 봄

# [주의] 업종비중을 모델에 넣으면 정답이 샌다 (가짜 데이터로 확인):
#   업종비중은 "이 자리를 거쳐간 가게들의 업종 구성"이라, 한 업종 100%면 가게가 거의 안 바뀐 자리,
#   여러 업종이 섞여 있으면 가게가 여러 번 바뀐 자리라는 뜻이 됨. 즉 업종 정보가 아니라 "교체 횟수"를 알려줌.
#   업종을 완전히 무작위로 만든 가짜 데이터에서도 업종비중을 넣으면 AUC가 0.73 -> 0.83으로 뛰었고,
#   "최대 업종비중" 하나만 넣어도 0.80이 됨. 반면 대표업종(범주 1개)은 0.73 그대로 (업종이 무작위니 당연).
#   -> 위치 모델에는 업종 변수를 넣지 않음. 업종비중 셋은 누수 크기 확인용 비교에만 둠.

# 업종 가중치 (24번 24_location_industry_shares.csv)
#   업종비중_<업종> : 이 위치를 10년 동안 각 업종이 차지한 기간 비율 (위치마다 합 = 1)
#   업종가중_기대교체율 = sum(업종비중 x 그 업종의 평균 교체율)  (26번 업종비중 가중 기준선과 같은 계산)
#     -> "이 자리를 거쳐간 업종 구성으로 보면 교체율이 이 정도일 것"이라는 업종 기대값
#   * 업종별 평균 교체율은 fold마다 train 위치로만 다시 계산해서 test 정답이 새지 않게 함
#   * 업종비중은 target과 같은 10년 동안의 기록이라 "계약 전에 알 수 있는 정보"는 아님 -> 해석 시 주의
INDUSTRY_SHARE_PATH = "24_location_industry_shares.csv"
FIXED_THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.75]
CB_PARAMS = dict(
    iterations=800,
    learning_rate=0.08,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=rc.RANDOM_SEED,
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)


def threshold_metrics(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan

    def fbeta(b):
        if not prec or not rec or np.isnan(prec) or np.isnan(rec):
            return 0.0
        return (1 + b * b) * prec * rec / (b * b * prec + rec)

    return dict(임계값=round(thr, 2), Precision=prec, Recall=rec, F1=fbeta(1), F2=fbeta(2), F05=fbeta(0.5),
                NPV=npv, Specificity=spec, Youden=(rec + spec - 1) if not np.isnan(spec) else np.nan,
                위험예측수=tp + fp)


PERM_REPEATS = 3


def permutation_auc_drop(model, X_te, y_te, cat_cols, base_auc, rng, perm_groups=None):
    """perm_groups: {표시이름: [컬럼들]} - 묶인 컬럼은 행 단위로 함께 섞음(업종비중 10개를 '업종' 하나로 평가)"""
    groups = dict(perm_groups or {})
    grouped = {c for cols in groups.values() for c in cols}
    for col in X_te.columns:
        if col not in grouped:
            groups[col] = [col]
    drops = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X_te.columns]
        if not cols:
            continue
        vals = []
        for _ in range(PERM_REPEATS):
            Xp = X_te.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[idx]
            p = model.predict_proba(Pool(Xp, cat_features=cat_cols))[:, 1]
            vals.append(base_auc - roc_auc_score(y_te, p))
        drops[name] = float(np.mean(vals))
    return pd.Series(drops)


FOLD_FEATURES = {}   # {컬럼명: {fold번호: 값 배열}} - fold마다 다시 계산해야 하는 컬럼
PERM_GROUPS = {}
CAT_SET = set(rc.CAT_FEATURES) | {"대표업종"}


def run_cv(X, y, splits, cat_cols, t0, label, with_perm=False):
    oof = np.zeros(len(X))
    fold_rows, imp_list, perm_list = [], [], []
    rng = np.random.default_rng(rc.RANDOM_SEED)
    for k, (tr, te) in enumerate(splits, start=1):
        for col, per_fold in FOLD_FEATURES.items():
            if col in X.columns:
                X = X.copy()
                X[col] = per_fold[k - 1]
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_cols))
        p = model.predict_proba(Pool(X.iloc[te], cat_features=cat_cols))[:, 1]
        oof[te] = p
        fold_rows.append(dict(fold=k, test위치수=len(te), test위험비율=y[te].mean(),
                              ROC_AUC=roc_auc_score(y[te], p), PR_AUC=average_precision_score(y[te], p)))
        imp_list.append(pd.Series(model.get_feature_importance(type="PredictionValuesChange"),
                                  index=X.columns))
        if with_perm:
            perm_list.append(permutation_auc_drop(model, X.iloc[te], y[te], cat_cols,
                                                  fold_rows[-1]["ROC_AUC"], rng, PERM_GROUPS))
        rc.log(f"[{label}] fold {k}/{len(splits)} AUC={fold_rows[-1]['ROC_AUC']:.4f}", t0)
    if with_perm:
        return oof, pd.DataFrame(fold_rows), (imp_list, perm_list)
    return oof, pd.DataFrame(fold_rows), imp_list


def main():
    t0 = rc.start_timer()
    plt = rc.set_korean_font()

    base = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "TRDAR_CD", "위험여부"],
                       dtype={"TRDAR_CD": str})
    feat = pd.read_csv("47_location_features.csv", low_memory=False)
    df = base.merge(feat, on="위치ID", how="inner")

    raw_features = rc.CAT_FEATURES + rc.NUM_FEATURES + list(rc.EXTRA_FEATURES)
    cat_cols = [c for c in raw_features if c in rc.CAT_FEATURES]
    for c in cat_cols:
        df[c] = df[c].fillna("정보없음").astype(str)

    feature_sets = {"원변수": raw_features}
    share_cols = []
    if os.path.exists(INDUSTRY_SHARE_PATH):
        sh = pd.read_csv(INDUSTRY_SHARE_PATH)
        wide = sh.pivot_table(index="위치ID", columns="업종", values="업종비중", aggfunc="sum", fill_value=0)
        wide.columns = [f"업종비중_{c}" for c in wide.columns]
        share_cols = list(wide.columns)
        df = df.merge(wide.reset_index(), on="위치ID", how="left")
        df[share_cols] = df[share_cols].fillna(0)
        rates = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "교체율"])
        df = df.merge(rates, on="위치ID", how="left")
        df["업종가중_기대교체율"] = np.nan
        feature_sets["원변수+업종비중"] = raw_features + share_cols + ["업종가중_기대교체율"]
        print(f"업종 가중치 로드: {len(share_cols)}개 업종 {[c.replace('업종비중_', '') for c in share_cols]}")
        print(f"  업종비중 없는 위치: {(df[share_cols].sum(axis=1) == 0).sum():,}개")
    elif FINAL_FEATURE_SET == "원변수+업종비중":
        raise FileNotFoundError(f"{INDUSTRY_SHARE_PATH}가 없습니다 (24번 산출물).")
    # 대표업종: 45번 결과(24번에서 계산된 '가장 오래 차지한 업종')
    rep = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "대표업종"])
    df = df.merge(rep, on="위치ID", how="left")
    df["대표업종"] = df["대표업종"].fillna("정보없음").astype(str)
    feature_sets["원변수+대표업종"] = raw_features + ["대표업종"]
    if os.path.exists("47_1_pca_scores.csv"):
        pcs = pd.read_csv("47_1_pca_scores.csv")
        pc_cols = [c for c in pcs.columns if c.startswith("PC")]
        df = df.merge(pcs, on="위치ID", how="left")
        feature_sets["PCA"] = cat_cols + pc_cols
        feature_sets["원변수+PCA"] = raw_features + pc_cols
        print(f"47-1번 주성분 {len(pc_cols)}개 로드: {pc_cols}")
    elif FINAL_FEATURE_SET != "원변수":
        raise FileNotFoundError("FINAL_FEATURE_SET이 PCA 계열인데 47_1_pca_scores.csv가 없습니다. 47-1번을 먼저 돌리세요.")

    y, groups = df["위험여부"].values, df["TRDAR_CD"].values
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rc.RANDOM_SEED)
    splits = list(skf.split(df, y, groups))   # 모든 변수셋이 같은 fold를 쓰도록 고정
    print(f"학습 데이터: {len(df):,}개 위치, 위험 비율 {y.mean()*100:.2f}%")

    # 업종가중_기대교체율: fold마다 train 위치로 업종별 평균 교체율 계산 -> test 위치에 적용 (정답 누수 방지)
    # train 위치 자신의 값은 전체 train 기준(자기 자신 포함)이지만 업종당 수만 개 위치 평균이라 영향 무시 가능
    fold_feature = {}
    if share_cols:
        S = df[share_cols].values
        r = df["교체율"].values
        for k, (tr, te) in enumerate(splits):
            w = S[tr]
            ind_rate = (w * r[tr, None]).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-9)
            fold_feature[k] = (S @ ind_rate)   # 이 fold 기준으로 전체 위치 값 (train/test 모두)
        full_rate = (S * r[:, None]).sum(axis=0) / np.maximum(S.sum(axis=0), 1e-9)
        df["업종가중_기대교체율"] = S @ full_rate      # 최종 모델/표시용 (전체 기준)
        print("  업종별 평균 교체율(전체): " + ", ".join(
            f"{c.replace('업종비중_', '')} {v:.3f}" for c, v in sorted(zip(share_cols, full_rate), key=lambda x: -x[1])))

    # ------------------------------------------------------------------ 변수셋 비교
    global FOLD_FEATURES, PERM_GROUPS
    if fold_feature:
        FOLD_FEATURES = {"업종가중_기대교체율": fold_feature}
        PERM_GROUPS = {"업종(업종비중+기대교체율)": share_cols + ["업종가중_기대교체율"]}

    if COMPARE_FEATURE_SETS and len(feature_sets) > 1:
        comp = []
        for name, cols in feature_sets.items():
            if name not in COMPARE_SETS:
                continue
            if name == FINAL_FEATURE_SET:
                continue  # 최종 셋은 아래 본 CV에서 계산하고 합침
            oof_c, folds_c, _ = run_cv(df[cols], y, splits, [c for c in cols if c in CAT_SET], t0, name)
            comp.append(dict(변수셋=name, 변수수=len(cols), ROC_AUC=roc_auc_score(y, oof_c),
                             PR_AUC=average_precision_score(y, oof_c),
                             ROC_AUC_fold표준편차=folds_c["ROC_AUC"].std()))
    else:
        comp = []

    # ------------------------------------------------------------------ 최종 변수셋 CV
    features = feature_sets[FINAL_FEATURE_SET]
    cat_cols = [c for c in features if c in CAT_SET]
    X = df[features]
    print(f"\n[최종 변수셋: {FINAL_FEATURE_SET}] 변수 {len(features)}개")
    print(f"  범주형: {cat_cols}")
    print(f"  수치형: {[c for c in features if c not in cat_cols]}")
    oof, folds, (imp_list, perm_list) = run_cv(X, y, splits, cat_cols, t0, FINAL_FEATURE_SET, with_perm=True)
    fold_id = np.zeros(len(df), dtype=int)
    for k, (_, te) in enumerate(splits, start=1):
        fold_id[te] = k

    if comp:
        comp.append(dict(변수셋=FINAL_FEATURE_SET, 변수수=len(features), ROC_AUC=roc_auc_score(y, oof),
                         PR_AUC=average_precision_score(y, oof), ROC_AUC_fold표준편차=folds["ROC_AUC"].std()))
        comp = pd.DataFrame(comp).sort_values("ROC_AUC", ascending=False)
        rc.print_table(comp, "변수셋 비교 (같은 fold, OOF)")
        a = comp.set_index("변수셋")["ROC_AUC"]
        if "원변수+업종비중" in a and "원변수+대표업종" in a:
            print(f"  업종비중 셋이 대표업종 셋보다 AUC {a['원변수+업종비중'] - a['원변수+대표업종']:+.4f} "
                  "-> 이 차이는 업종 효과가 아니라 '교체 횟수' 누수로 봐야 함 (상단 주석 참고)")
        print("  * 차이가 0.005 안팎이면 사실상 같은 성능 -> 해석 가능한 원변수를 쓰는 게 낫다")
        rc.save_csv(comp, "48_feature_set_comparison.csv")

    rc.print_table(folds, "fold별 성능 (상권 단위 분리)")

    auc, pr_auc = roc_auc_score(y, oof), average_precision_score(y, oof)
    print(f"\n[OOF 전체] ROC-AUC {auc:.4f} / PR-AUC {pr_auc:.4f} (무작위 기준 PR-AUC = {y.mean():.3f})")

    # ------------------------------------------------------------------ 임계값
    grid = [threshold_metrics(y, oof, t) for t in np.round(np.arange(0.05, 0.96, 0.01), 2)]
    grid = pd.DataFrame(grid)
    picks = [
        ("F1최고", grid.loc[grid["F1"].idxmax()]),
        ("균형(Youden)", grid.loc[grid["Youden"].idxmax()]),
        ("Recall중시(F2)", grid.loc[grid["F2"].idxmax()]),
        ("Precision중시(F0.5)", grid.loc[grid["F05"].idxmax()]),
    ]
    table = pd.DataFrame([dict(목적=name, **row.to_dict()) for name, row in picks])
    fixed = pd.DataFrame([dict(목적="고정값", **threshold_metrics(y, oof, t)) for t in FIXED_THRESHOLDS])
    table = pd.concat([table, fixed], ignore_index=True)
    show = ["목적", "임계값", "Precision", "Recall", "F1", "NPV", "Specificity", "위험예측수"]
    rc.print_table(table[show], "임계값별 성능 (OOF)")

    rec_row = table[table["목적"] == "F1최고"].iloc[0]
    print(f"\n[참고] 대표 임계값 {rec_row['임계값']:.2f}: Recall {rec_row['Recall']*100:.1f}%, "
          f"Precision {rec_row['Precision']*100:.1f}%, 위험 판정 비율 {rec_row['위험예측수']/len(y)*100:.1f}%")
    print("  * Recall만 인용하지 말 것: 임계값을 낮추면 Recall은 얼마든지 올라가지만 위험 판정이 전체의 대부분이 됨")

    metrics = pd.DataFrame([dict(
        학습위치수=len(df), 위험비율=y.mean(), ROC_AUC=auc, PR_AUC=pr_auc,
        ROC_AUC_fold평균=folds["ROC_AUC"].mean(), ROC_AUC_fold표준편차=folds["ROC_AUC"].std(),
        대표임계값=rec_row["임계값"], Recall=rec_row["Recall"], Precision=rec_row["Precision"],
        NPV=rec_row["NPV"], F1=rec_row["F1"],
    )])

    # ------------------------------------------------------------------ 중요도
    imp = pd.concat(imp_list, axis=1)
    perm = pd.concat(perm_list, axis=1)
    pvc = imp.mean(axis=1)
    pvc_grouped = {name: pvc[[c for c in cols if c in pvc.index]].sum() for name, cols in PERM_GROUPS.items()}
    imp_df = pd.DataFrame({
        "변수": perm.index,
        "AUC감소_평균": perm.mean(axis=1).values,
        "AUC감소_표준편차": perm.std(axis=1).values,
        "PVC_평균": [pvc_grouped.get(v, pvc.get(v, np.nan)) for v in perm.index],
    })
    pos = imp_df["AUC감소_평균"].clip(lower=0)
    imp_df["중요도(%)"] = pos / pos.sum() * 100 if pos.sum() > 0 else 0.0
    imp_df["PVC(%)"] = imp_df["PVC_평균"] / imp_df["PVC_평균"].sum() * 100
    imp_df = imp_df.sort_values("중요도(%)", ascending=False)
    rc.print_table(imp_df, "변수 중요도 (중요도(%) = 순열중요도 기준, PVC = CatBoost 기본 중요도 참고용)")
    print("  * AUC감소가 0 근처거나 음수면 그 변수는 test 데이터 예측에 사실상 기여하지 않는다는 뜻")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(imp_df) + 1))
    ax.barh(imp_df["변수"][::-1], imp_df["중요도(%)"][::-1], color="#8e6cc0")
    for i, v in enumerate(imp_df["중요도(%)"][::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("중요도 (%) - 순열중요도(섞었을 때 AUC 감소) 기준")
    ax.set_title(f"변수 중요도 - CatBoost (전체 {len(df):,}개 위치)")
    fig.tight_layout()
    fig.savefig("48_feature_importance.png", dpi=150)

    # ------------------------------------------------------------------ 최종 모델 (전체 학습, 새 위치 점수용)
    final = CatBoostClassifier(**CB_PARAMS)
    final.fit(Pool(X, y, cat_features=cat_cols))
    final.save_model("48_catboost_final.cbm")
    print("[완료] 저장: 48_catboost_final.cbm (새 위치 점수 계산용. 지도의 위험도는 OOF 값을 사용)")

    # ------------------------------------------------------------------ 저장
    pred = pd.DataFrame({"위치ID": df["위치ID"], "TRDAR_CD": df["TRDAR_CD"], "위험여부": y,
                         "위험도": oof, "fold": fold_id})
    pred["위험등급"] = rc.assign_grade(pred["위험도"])
    g = pred.groupby("위험등급").agg(위치수=("위치ID", "size"), 실제위험비율=("위험여부", "mean"))
    g = g.reindex(rc.GRADE_LABELS)
    g["lift"] = g["실제위험비율"] / y.mean()
    print(f"\n[위험등급별 실제 위험비율 (등급 방식: {rc.GRADE_METHOD})]")
    print(g.to_string())
    rc.save_csv(g.reset_index(), "48_grade_lift.csv")

    rc.save_csv(pred, "48_oof_predictions.csv")
    rc.save_csv(metrics, "48_model_metrics.csv")
    rc.save_csv(pd.concat([table, grid.assign(목적="전체스캔")], ignore_index=True), "48_threshold_table.csv")
    rc.save_csv(imp_df, "48_feature_importance.csv")
    rc.save_csv(folds, "48_fold_metrics.csv")
    rc.log("48번 완료", t0)


if __name__ == "__main__":
    main()
