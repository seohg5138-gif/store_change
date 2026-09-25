# -*- coding: utf-8 -*-
"""
46번: 가설 검정 + "2년"의 의미 (발표자료 7~8p, 10~11p)

H0: 같은 상권 안 위치들의 교체율(생존기간)은 차이가 없다.
H1: 같은 상권 안에서도 위치에 따라 유의미한 차이가 있다.

계산:
  1) 상권 내 변동계수(CV) = 표준편차 / 평균 x 100
     - 발표자료의 184.6%가 어떤 집계였는지 문서에 정의가 없어서, 세 가지를 다 계산해서 남긴다:
       상권별 CV의 평균 / 중앙값 / 위치수 가중평균  (+ 참고로 전체 위치 CV)
     - 교체율, 평균생존분기수 두 지표 모두
  2) 상권을 집단으로 하는 일원분산분석(ANOVA) + Kruskal-Wallis
     - 표본이 30만 개 수준이라 p값은 사실상 항상 0 -> 42번 관례대로 효과크기(eta^2, epsilon^2)를 같이 보고
     - eta^2 = "상권이 설명하는 분산 비율". 1 - eta^2 가 "같은 상권 안 위치 간 차이" 몫.
       이게 "상권 평균은 개별 위치를 설명하지 못한다"의 직접적인 근거가 된다.
  3) 상권대비 교체율 분포 히스토그램 (상권평균=0 / 상위10% / 상위5% 선)
  4) 점포(episode) 단위 생존 분포: 2년(8분기) 안에 폐업한 비율
     - 좌측절단(데이터 첫 분기부터 이미 영업 중이던 episode) 제외 (17번 주석과 같은 원칙)
     - (a) 폐업 episode만 본 단순 비율  (b) 영업중(우측검열)까지 반영한 Kaplan-Meier 추정
  5) 위험그룹 평균 생존기간 -> "상위 10% = 평균 2년" 주장 검증 (45번 요약 재사용)

입력:
  45_location_risk_base.csv, 45_threshold_summary.csv, tenancy_episode_corrected.parquet
출력:
  46_hypothesis_results.csv, 46_district_cv.csv, 46_survival_summary.csv,
  46_relative_turnover_hist.png, 46_episode_survival_dist.png
"""

import numpy as np
import pandas as pd
from scipy import stats

import risk_common as rc


def district_cv(df, value_col):
    g = df.groupby("TRDAR_CD")[value_col]
    out = pd.DataFrame({"위치수": g.size(), "평균": g.mean(), "표준편차": g.std(ddof=1)})
    out = out[out["평균"] > 0]
    out["CV(%)"] = out["표준편차"] / out["평균"] * 100
    return out


def anova_effects(df, value_col):
    sub = df[["TRDAR_CD", value_col]].dropna()
    groups = [g.values for _, g in sub.groupby("TRDAR_CD")[value_col]]
    f, p_f = stats.f_oneway(*groups)
    h, p_h = stats.kruskal(*groups)

    grand = sub[value_col].mean()
    grp_mean = sub.groupby("TRDAR_CD")[value_col].transform("mean")
    ss_between = ((grp_mean - grand) ** 2).sum()
    ss_total = ((sub[value_col] - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    n, k = len(sub), len(groups)
    eps2 = (h - k + 1) / (n - k)  # Kruskal-Wallis epsilon^2 (Tomczak & Tomczak 2014)
    return dict(F=f, p_ANOVA=p_f, H=h, p_Kruskal=p_h, eta2=eta2, epsilon2=eps2,
                상권내_분산비중=1 - eta2, 표본수=n, 상권수=k)


def kaplan_meier(duration, event, at_times):
    """duration: 생존분기수, event: 1=폐업 관측, 0=영업중(검열). S(t) = P(T > t)."""
    d = pd.DataFrame({"t": duration, "e": event})
    tab = d.groupby("t")["e"].agg(["sum", "size"]).sort_index()
    at_risk = tab["size"][::-1].cumsum()[::-1]
    surv = (1 - tab["sum"] / at_risk).cumprod()
    res = {}
    for t in at_times:
        s = surv[surv.index <= t]
        res[t] = float(s.iloc[-1]) if len(s) else 1.0
    return res


def main():
    t0 = rc.start_timer()
    plt = rc.set_korean_font()

    df = pd.read_csv("45_location_risk_base.csv", dtype={"TRDAR_CD": str}, low_memory=False)
    summary45 = pd.read_csv("45_threshold_summary.csv")
    rc.log(f"45번 결과 로딩 ({len(df):,}개 위치)", t0)

    # ------------------------------------------------------------------ 1) CV
    cv_rows, cv_tables = [], []
    for col in ["교체율", "평균생존분기수"]:
        cv = district_cv(df, col)
        cv_tables.append(cv.add_prefix(f"{col}_"))
        overall = df[col].std() / df[col].mean() * 100
        cv_rows.append(dict(
            지표=col,
            상권별CV_평균=cv["CV(%)"].mean(),
            상권별CV_중앙값=cv["CV(%)"].median(),
            상권별CV_위치수가중=np.average(cv["CV(%)"], weights=cv["위치수"]),
            전체위치CV=overall,
            CV계산상권수=len(cv),
        ))
    cv_summary = pd.DataFrame(cv_rows)
    rc.print_table(cv_summary, "상권 내 변동계수(CV, %)")
    rc.save_csv(pd.concat(cv_tables, axis=1).reset_index(), "46_district_cv.csv")

    # ------------------------------------------------------------------ 2) ANOVA
    test_rows = []
    for col in ["교체율", "평균생존분기수"]:
        r = anova_effects(df, col)
        r["지표"] = col
        test_rows.append(r)
    tests = pd.DataFrame(test_rows)[
        ["지표", "표본수", "상권수", "F", "p_ANOVA", "eta2", "H", "p_Kruskal", "epsilon2", "상권내_분산비중"]
    ]
    rc.print_table(tests, "상권 간 차이 검정 (p값보다 효과크기를 볼 것)")
    for _, r in tests.iterrows():
        print(f"  [{r['지표']}] 상권이 설명하는 분산 {r['eta2']*100:.1f}% / "
              f"같은 상권 안 위치 간 차이 {r['상권내_분산비중']*100:.1f}%")

    results = cv_summary.merge(tests, on="지표")
    rc.save_csv(results, "46_hypothesis_results.csv")

    # ------------------------------------------------------------------ 3) 히스토그램
    col = f"상권대비{rc.RISK_BASIS}"
    s45 = summary45[summary45["기준"] == rc.RISK_BASIS].iloc[0]
    q90, q95 = s45["상위10_임계값"], s45["상위5_임계값"]
    lo, hi = df[col].quantile([0.001, 0.999])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[col].clip(lo, hi), bins=60, color="#2a78d6", edgecolor="white")
    ax.axvline(0, color="#c0392b", lw=1.5, label="상권평균")
    ax.axvline(q90, color="#e0a800", lw=1.5, label=f"상위10% ({q90:.3f})")
    ax.axvline(q95, color="#8e44ad", lw=1.5, label=f"상위5% ({q95:.3f})")
    ax.set_title(f"{col} 분포")
    ax.set_xlabel(col)
    ax.set_ylabel("위치 수")
    ax.legend()
    fig.tight_layout()
    fig.savefig("46_relative_turnover_hist.png", dpi=150)
    print("[완료] 저장: 46_relative_turnover_hist.png")

    # ------------------------------------------------------------------ 4) episode 생존
    ep = pd.read_parquet(
        "tenancy_episode_corrected.parquet",
        columns=["위치ID", "시작분기", "종료분기", "생존분기수_보정", "생존여부"],
    )
    first_period = ep["시작분기"].min()
    ep["좌측절단"] = ep["시작분기"] == first_period
    ep["폐업"] = (ep["생존여부"] == "폐업").astype(int)
    target_ids = set(df["위치ID"])

    surv_rows = []
    for scope, sub in [("전체 검증완료 위치", ep), ("45번 분석대상 위치", ep[ep["위치ID"].isin(target_ids)])]:
        sub = sub[~sub["좌측절단"]]
        closed = sub[sub["폐업"] == 1]
        naive = (closed["생존분기수_보정"] < rc.SURVIVAL_2Y_QUARTERS).mean()
        km = kaplan_meier(sub["생존분기수_보정"].values, sub["폐업"].values,
                          [rc.SURVIVAL_2Y_QUARTERS - 1, 11, 19])
        surv_rows.append(dict(
            범위=scope,
            episode수_좌측절단제외=len(sub),
            폐업episode수=len(closed),
            폐업중_2년미만비율=naive,
            폐업episode_생존분기_중앙값=closed["생존분기수_보정"].median(),
            KM_2년내폐업확률=1 - km[rc.SURVIVAL_2Y_QUARTERS - 1],
            KM_3년내폐업확률=1 - km[11],
            KM_5년내폐업확률=1 - km[19],
        ))
    surv = pd.DataFrame(surv_rows)
    rc.print_table(surv, "점포(episode) 단위 생존 - 2년(8분기)의 의미")
    print("  * '폐업중_2년미만비율'은 폐업한 가게만 본 값(발표자료 36%와 같은 방식),")
    print("    'KM_2년내폐업확률'은 아직 영업 중인 가게까지 반영한 값 -> 보고용으로는 KM이 더 정확")
    rc.save_csv(surv, "46_survival_summary.csv")

    closed_all = ep[(~ep["좌측절단"]) & (ep["폐업"] == 1)]
    counts = closed_all["생존분기수_보정"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c0392b" if q < rc.SURVIVAL_2Y_QUARTERS else "#2a78d6" for q in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("폐업한 점포의 생존분기 분포 (좌측절단 제외, 빨강 = 2년 미만)")
    ax.set_xlabel("생존분기수")
    ax.set_ylabel("episode 수")
    fig.tight_layout()
    fig.savefig("46_episode_survival_dist.png", dpi=150)
    print("[완료] 저장: 46_episode_survival_dist.png")

    # ------------------------------------------------------------------ 5) 위험그룹 = 2년?
    print(f"\n[위험그룹(상권대비 {rc.RISK_BASIS} 상위10%) 생존기간]")
    print(f"  평균생존분기수 평균 {s45['위험그룹_평균생존분기수']:.2f}분기 "
          f"(= {s45['위험그룹_평균생존년']:.2f}년), 중앙값 {s45['위험그룹_중앙생존분기수']:.2f}분기")
    print(f"  나머지 90% 평균 {s45['나머지_평균생존분기수']:.2f}분기")
    rc.log("46번 완료", t0)


if __name__ == "__main__":
    main()
