# -*- coding: utf-8 -*-
"""
47-1번: 위치 물리 변수 PCA (48번 모델 전에 변수 구조를 확인하는 단계)

무엇을 보나:
  1) 수치형 변수 간 상관행렬 + VIF  -> 서로 겹치는 정보(다중공선성)가 얼마나 있는지
  2) PCA 설명분산 -> 몇 개 주성분이면 정보의 90%가 담기는지
  3) 적재량(loading) -> 각 주성분이 무엇을 뜻하는지 이름 붙이기 (예: "건물 규모", "상권 밀집")
  4) 주성분 점수 구간별 위험비율 -> 주성분이 위험과 관련 있는지
  5) 주성분 점수 저장 -> 48번에서 "원변수 vs PCA vs 원변수+PCA" 성능 비교에 사용

전처리:
  - 대상: 수치형만. 범주형(층구분/주용도/상권유형)은 PCA에 넣지 않는다
    (원-핫을 PCA에 넣으면 분산 크기가 범주 비율에 좌우돼서 해석이 왜곡됨. 모델에는 범주형 그대로 넣음)
  - 치우친 변수(연면적, 건물내위치수, 주변상가수, 지하철거리 등)는 log1p 변환
    (안 하면 극단값 몇 개가 PC1을 통째로 차지함)
  - 결측은 중앙값 대체 -> 표준화 -> PCA

입력:  45_location_risk_base.csv, 47_location_features.csv
출력:  47_1_pca_scores.csv, 47_1_pca_summary.csv, 47_1_pca_loadings.csv, 47_1_correlation.csv,
       47_1_pca_risk_by_decile.csv, 47_1_pca_scree.png, 47_1_pca_loadings.png, 47_1_pca_scatter.png
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import risk_common as rc

PCA_VARS = [
    "연면적", "건물지상층수", "건물지하층수", "건물연령", "건물내위치수",
    "주변상가수_200m", "지하철거리_m", "지하철역수_500m", "층번호",
]
LOG_VARS = ["연면적", "건물내위치수", "주변상가수_200m", "지하철거리_m"]
VAR_THRESHOLD = 0.90


def vif_from_corr(corr):
    """VIF_i = (R^-1)_ii. statsmodels 없이 상관행렬 역행렬 대각으로 계산."""
    inv = np.linalg.pinv(corr.values)
    return pd.Series(np.diag(inv), index=corr.index)


def main():
    t0 = rc.start_timer()
    plt = rc.set_korean_font()

    base = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "위험여부"])
    feat = pd.read_csv("47_location_features.csv", low_memory=False)
    df = base.merge(feat, on="위치ID", how="inner")
    vars_ = [v for v in PCA_VARS if v in df.columns]
    rc.log(f"{len(df):,}개 위치, PCA 변수 {len(vars_)}개: {vars_}", t0)

    X = df[vars_].astype(float).copy()
    for v in LOG_VARS:
        if v in X:
            X[v] = np.log1p(X[v].clip(lower=0))
    miss = X.isna().mean()
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------------ 1) 상관 + VIF
    corr = pd.DataFrame(Z, columns=vars_).corr()
    vif = vif_from_corr(corr)
    rc.save_csv(corr.round(3).reset_index().rename(columns={"index": "변수"}), "47_1_correlation.csv")
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .rename("상관").reset_index().rename(columns={"level_0": "변수1", "level_1": "변수2"}))
    pairs = pairs.reindex(pairs["상관"].abs().sort_values(ascending=False).index)
    rc.print_table(pairs.head(8), "상관이 큰 변수 쌍 (log 변환 후)")
    rc.print_table(pd.DataFrame({"변수": vif.index, "VIF": vif.values, "결측대체비율": miss[vif.index].values}),
                   "VIF (10 넘으면 다중공선성 심함)")

    # ------------------------------------------------------------------ 2) PCA
    pca = PCA(random_state=rc.RANDOM_SEED).fit(Z)
    ev = pca.explained_variance_ratio_
    summary = pd.DataFrame({
        "주성분": [f"PC{i+1}" for i in range(len(ev))],
        "고유값": pca.explained_variance_,
        "설명분산": ev,
        "누적설명분산": ev.cumsum(),
    })
    n_keep = int(np.searchsorted(ev.cumsum(), VAR_THRESHOLD) + 1)
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    rc.print_table(summary, "주성분별 설명분산")
    print(f"  누적 {VAR_THRESHOLD*100:.0f}% 도달: PC1~PC{n_keep} / 고유값>1(Kaiser): {n_kaiser}개")
    rc.save_csv(summary, "47_1_pca_summary.csv")

    # ------------------------------------------------------------------ 3) 적재량
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                            index=vars_, columns=summary["주성분"])
    show_k = max(n_keep, 3)
    rc.print_table(loadings.iloc[:, :show_k].round(3).reset_index().rename(columns={"index": "변수"}),
                   f"적재량 (PC1~PC{show_k}, |값|이 클수록 그 주성분을 대표)")
    for pc in loadings.columns[:show_k]:
        top = loadings[pc].abs().sort_values(ascending=False).head(3).index
        desc = ", ".join(f"{v}({loadings.loc[v, pc]:+.2f})" for v in top)
        print(f"  {pc}: {desc}")
    rc.save_csv(loadings.round(4).reset_index().rename(columns={"index": "변수"}), "47_1_pca_loadings.csv")

    # ------------------------------------------------------------------ 4) 주성분 구간별 위험비율
    scores = pca.transform(Z)[:, :n_keep]
    score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_keep)])
    score_df.insert(0, "위치ID", df["위치ID"].values)
    base_rate = df["위험여부"].mean()
    rows = []
    for pc in score_df.columns[1:]:
        dec = pd.qcut(score_df[pc], 10, labels=False, duplicates="drop") + 1
        g = df.groupby(dec.values)["위험여부"].mean()
        for d, rate in g.items():
            rows.append(dict(주성분=pc, 십분위=int(d), 위험비율=rate, lift=rate / base_rate))
    dec_df = pd.DataFrame(rows)
    wide = dec_df.pivot(index="십분위", columns="주성분", values="위험비율")
    print(f"\n[주성분 십분위별 위험비율 (%), 전체 {base_rate*100:.1f}%]")
    print((wide * 100).round(1).to_string())
    spread = (wide.max() - wide.min()).sort_values(ascending=False)
    print("\n  십분위 간 위험비율 차이(최대-최소, %p): "
          + ", ".join(f"{k} {v*100:.1f}" for k, v in spread.items()))
    rc.save_csv(dec_df, "47_1_pca_risk_by_decile.csv")
    rc.save_csv(score_df, "47_1_pca_scores.csv")

    # ------------------------------------------------------------------ 그림
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["주성분"], summary["설명분산"] * 100, color="#2a78d6")
    ax.plot(summary["주성분"], summary["누적설명분산"] * 100, color="#c0392b", marker="o")
    ax.axhline(VAR_THRESHOLD * 100, color="#999", ls="--", lw=1)
    ax.set_ylabel("설명분산 (%)")
    ax.set_title("PCA 설명분산 (막대) / 누적 (선)")
    fig.tight_layout()
    fig.savefig("47_1_pca_scree.png", dpi=150)

    fig, ax = plt.subplots(figsize=(1.2 * show_k + 3, 0.45 * len(vars_) + 1.5))
    L = loadings.iloc[:, :show_k]
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(show_k), L.columns)
    ax.set_yticks(range(len(vars_)), L.index)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("PCA 적재량")
    fig.tight_layout()
    fig.savefig("47_1_pca_loadings.png", dpi=150)

    samp = score_df.join(df[["위험여부"]]).sample(min(30000, len(score_df)), random_state=rc.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, color, name in [(0, "#b0c4de", "안전"), (1, "#c0392b", "위험(상위10%)")]:
        s = samp[samp["위험여부"] == lab]
        ax.scatter(s["PC1"], s["PC2"] if "PC2" in s else 0, s=3, alpha=0.35, c=color, label=name)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)" if len(ev) > 1 else "")
    ax.legend(markerscale=4)
    ax.set_title("PC1-PC2 위치 분포 (표본 3만)")
    fig.tight_layout()
    fig.savefig("47_1_pca_scatter.png", dpi=150)
    print("[완료] 저장: 47_1_pca_scree.png, 47_1_pca_loadings.png, 47_1_pca_scatter.png")
    rc.log("47-1번 완료", t0)


if __name__ == "__main__":
    main()
