# -*- coding: utf-8 -*-
"""
52번: 점포(가게) 단위로 "2년 안에 폐업하나"를 CatBoost로 예측하고,
      입지 -> +업종 -> +프랜차이즈 -> +자리이력 순서로 변수를 더해가며 성능이 얼마나 오르는지 본다.

  질문: "위치 탓인가, 업종 탓인가, 프랜차이즈면 다른가?"
    - 입지만                    : 47번 물리 변수 (48번과 같은 9개) + 입점연도
    - 입지+업종                 : + 업종대분류, 업종중분류
    - 입지+업종+프랜차이즈      : + 프랜차이즈 여부                    <- 최종(중요도/효과 계산 기준)
    - 입지+업종+프랜차이즈+자리이력 : + 이 가게 전에 이 자리를 거쳐간 가게 수 (계약 전에 알 수 있는 정보)
    * 입점연도는 모든 셋에 넣음: 코로나 시기 입점처럼 "시기" 효과가 업종/입지 효과로 잘못 잡히는 것을 막기 위한 통제변수

  검증: 48번과 같이 StratifiedGroupKFold(5, groups=TRDAR_CD), 모든 셋이 같은 fold
  중요도: 최종 셋의 순열중요도 (test fold에서 섞었을 때 AUC 감소)
  프랜차이즈 보정 효과: 전체 데이터로 학습한 최종 모델에서, 같은 가게들의 프랜차이즈 값만 0/1로 바꿔
                        예측 폐업확률 평균을 비교 (입지·업종·시기가 같다면 프랜차이즈 여부만으로 얼마나 다른가)
    * 여전히 인과는 아님: 본사의 입지 선정 능력, 자본력 같은 관측 안 된 차이가 섞여 있음

입력:  51_store_episodes.parquet
출력:  52_feature_set_comparison.csv, 52_feature_importance.csv, 52_franchise_effect.csv,
       52_oof_predictions.parquet, 52_feature_importance.png
"""

from importlib import import_module

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import risk_common as rc

m48 = import_module("48_catboost_risk_model")

N_SPLITS = 5
CB_PARAMS = dict(m48.CB_PARAMS, iterations=500)   # 셋 4개 x 5fold라 반복 수를 줄임
FINAL_SET = "입지+업종+프랜차이즈"
LOCATION_VARS = rc.CAT_FEATURES + rc.NUM_FEATURES + list(rc.EXTRA_FEATURES) + ["입점연도"]
STORE_CATS = ["업종대분류", "업종중분류"]
FEATURE_SETS = {
    "입지만": LOCATION_VARS,
    "입지+업종": LOCATION_VARS + STORE_CATS,
    "입지+업종+프랜차이즈": LOCATION_VARS + STORE_CATS + ["프랜차이즈"],
    "입지+업종+프랜차이즈+자리이력": LOCATION_VARS + STORE_CATS + ["프랜차이즈", "이전입점수"],
}
CAT_ALL = set(rc.CAT_FEATURES) | set(STORE_CATS)


def main():
    t0 = rc.start_timer()
    plt = rc.set_korean_font()

    ep = pd.read_parquet("51_store_episodes.parquet")
    df = ep[ep["분석대상"] & ep["TRDAR_CD"].notna()].reset_index(drop=True)
    for c in CAT_ALL:
        df[c] = df[c].fillna("정보없음").astype(str)
    y, groups = df["2년내폐업"].values, df["TRDAR_CD"].values
    print(f"학습 데이터: 가게 {len(df):,}곳, 2년 내 폐업률 {y.mean()*100:.1f}%, "
          f"프랜차이즈 {df['프랜차이즈'].mean()*100:.1f}%")

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rc.RANDOM_SEED)
    splits = list(skf.split(df, y, groups))

    # 48번 run_cv를 그대로 쓰되 반복 수만 줄임
    m48.CB_PARAMS = CB_PARAMS

    comp, final_out = [], None
    for name, cols in FEATURE_SETS.items():
        cat_cols = [c for c in cols if c in CAT_ALL]
        is_final = name == FINAL_SET
        res = m48.run_cv(df[cols], y, splits, cat_cols, t0, name, with_perm=is_final)
        oof, folds = res[0], res[1]
        comp.append(dict(변수셋=name, 변수수=len(cols), ROC_AUC=roc_auc_score(y, oof),
                         PR_AUC=average_precision_score(y, oof), fold표준편차=folds["ROC_AUC"].std()))
        if is_final:
            final_out = (cols, cat_cols, oof, res[2][1])

    comp = pd.DataFrame(comp)
    comp["AUC증가"] = comp["ROC_AUC"].diff()
    rc.print_table(comp, f"변수셋 비교 (가게 단위, 2년 내 폐업, 기준 폐업률 {y.mean():.3f})")
    rc.save_csv(comp, "52_feature_set_comparison.csv")

    # ------------------------------------------------------------------ 중요도 (최종 셋)
    cols, cat_cols, oof, perm_list = final_out
    perm = pd.concat(perm_list, axis=1)
    imp = pd.DataFrame({"변수": cols, "AUC감소_평균": perm.mean(axis=1).values,
                        "AUC감소_표준편차": perm.std(axis=1).values})
    pos = imp["AUC감소_평균"].clip(lower=0)
    imp["중요도(%)"] = pos / pos.sum() * 100 if pos.sum() > 0 else 0.0
    imp = imp.sort_values("중요도(%)", ascending=False)
    rc.print_table(imp, f"변수 중요도 - {FINAL_SET} (순열중요도)")
    rc.save_csv(imp, "52_feature_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(imp) + 1))
    colors = ["#c0392b" if v in STORE_CATS + ["프랜차이즈"] else "#8e6cc0" if v != "입점연도" else "#999"
              for v in imp["변수"][::-1]]
    ax.barh(imp["변수"][::-1], imp["중요도(%)"][::-1], color=colors)
    ax.set_xlabel("중요도 (%) - 순열중요도")
    ax.set_title("가게 2년 내 폐업 요인 (빨강 = 가게 속성, 보라 = 입지, 회색 = 시기)")
    fig.tight_layout()
    fig.savefig("52_feature_importance.png", dpi=150)

    # ------------------------------------------------------------------ 프랜차이즈 보정 효과
    rc.log("최종 모델 전체 학습 (프랜차이즈 보정 효과 계산용)...", t0)
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(Pool(df[cols], y, cat_features=cat_cols))

    def avg_pred(sub, fr_value):
        X = sub[cols].copy()
        X["프랜차이즈"] = fr_value
        return model.predict_proba(Pool(X, cat_features=cat_cols))[:, 1].mean()

    rows = []
    for label, sub in [("전체", df)] + [(k, g) for k, g in df.groupby("업종대분류")]:
        if len(sub) < 1000 or sub["프랜차이즈"].sum() < 100:
            continue
        p0, p1 = avg_pred(sub, 0), avg_pred(sub, 1)
        raw0 = sub.loc[sub["프랜차이즈"] == 0, "2년내폐업"].mean()
        raw1 = sub.loc[sub["프랜차이즈"] == 1, "2년내폐업"].mean()
        rows.append(dict(업종대분류=label, 가게수=len(sub), 프랜차이즈수=int(sub["프랜차이즈"].sum()),
                         단순_개인=raw0, 단순_프랜차이즈=raw1, 단순차이=raw1 - raw0,
                         보정_개인=p0, 보정_프랜차이즈=p1, 보정차이=p1 - p0))
    eff = pd.DataFrame(rows)
    rc.print_table(eff, "프랜차이즈 효과: 단순 비교 vs 입지·업종·시기 보정 (예측 폐업확률, Balanced 가중이라 수준보다 차이를 볼 것)")
    print("  * 단순차이와 보정차이가 크게 다르면, 프랜차이즈의 낮은 폐업률 일부는 '좋은 자리·업종 선택' 덕분이라는 뜻")
    rc.save_csv(eff, "52_franchise_effect.csv")

    out = df[["위치ID", "상가업소번호", "TRDAR_CD", "업종대분류", "업종중분류", "프랜차이즈", "2년내폐업"]].copy()
    out["폐업위험도"] = oof
    out.to_parquet("52_oof_predictions.parquet", index=False)
    print("[완료] 저장: 52_oof_predictions.parquet")
    rc.log("52번 완료", t0)


if __name__ == "__main__":
    main()
