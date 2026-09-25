# -*- coding: utf-8 -*-
"""
45번: 위치 단위 "상권대비 교체" 지표와 위험 위치 라벨을 만든다. (발표자료 9~10p)

입력:
    24_full_location_with_district.csv   (24번, 전체 위치 + 좌표 + 상권 + 공실률 + 대표업종)

정의 (발표자료 흐름 그대로, 지표만 새 전처리 결과로 교체):
    개인 교체        = 위치의 교체율 (= 교체횟수 / (관측분기수-1), 13번 보정 후 값)
    상권 평균 교체   = 같은 상권(TRDAR_CD) 안 신뢰가능 위치들의 교체율 평균
    상권대비 교체율  = 개인 교체 - 상권 평균 교체        (27번 '상대교체도'와 같은 식)
    위험 위치        = 상권대비 교체율 상위 10%  (>= 90 분위수)

    * RISK_BASIS="교체횟수"로 바꾸면 발표자료 1차본처럼 교체"횟수" 기준으로 계산한다.
      다만 관측분기수가 긴 위치일수록 횟수가 커지는 편향이 있어서 기본값은 교체율.
      두 기준을 다 계산해서 겹치는 비율을 로그로 남긴다.

대상:
    - 관측분기수 >= 15 (프로젝트 관례)
    - 실제 상권에 배정된 위치만 ("같은 상권 안에서 비교"가 가설의 전제라서 상권외는 제외)
    - 상권 안 신뢰가능 위치가 10개 미만인 상권은 평균이 불안정하므로 제외

출력:
    45_location_risk_base.csv       위치 단위 (이후 46~50번의 기준 테이블)
    45_district_baseline.csv        상권 단위 평균
    45_threshold_summary.csv        임계값/위험그룹 요약 (발표자료 10~11p 수치)
    45_turnover_count_dist.csv      위험그룹의 교체횟수 분포표 (발표자료 10p 우측 표)
"""

import numpy as np
import pandas as pd

import risk_common as rc


def main():
    t0 = rc.start_timer()
    rc.log("24_full_location_with_district.csv 로딩...")
    full = pd.read_csv("24_full_location_with_district.csv", dtype={"TRDAR_CD": str}, low_memory=False)
    rc.log(f"로딩 완료 ({len(full):,}개 위치)", t0)

    full["TRDAR_CD"] = rc.normalize_trdar(full["TRDAR_CD"])

    # --- 대상 필터 ---
    n0 = len(full)
    df = full[full["관측분기수"] >= rc.MIN_OBS_QUARTERS].copy()
    n1 = len(df)
    df = df[df["TRDAR_CD"].notna()].copy()
    n2 = len(df)
    print(f"전체 위치: {n0:,}")
    print(f"  관측분기수 >= {rc.MIN_OBS_QUARTERS}: {n1:,}")
    print(f"  그 중 실제 상권 배정: {n2:,} (상권외 {n1 - n2:,}개 제외)")

    # --- 위치ID 파싱 (PNU19, 층) ---
    parsed = rc.parse_location_id(df["위치ID"])
    df["PNU19"] = parsed["PNU19"]
    df["층_최종"] = parsed["층_최종"]

    # --- 상권 평균 ---
    district = (
        df.groupby("TRDAR_CD")
        .agg(
            상권명=("TRDAR_CD_N", "first"),
            상권유형=("TRDAR_SE_1", "first"),
            상권위치수=("위치ID", "size"),
            상권평균교체율=("교체율", "mean"),
            상권평균교체횟수=("교체횟수", "mean"),
            상권평균생존분기수=("평균생존분기수", "mean"),
            상권교체율표준편차=("교체율", "std"),
        )
        .reset_index()
    )
    small = district["상권위치수"] < rc.MIN_LOC_PER_DISTRICT
    print(f"\n상권 수: {len(district):,} (위치 {rc.MIN_LOC_PER_DISTRICT}개 미만 상권 {small.sum():,}개 제외)")
    district = district[~small].copy()

    df = df.merge(
        district[["TRDAR_CD", "상권위치수", "상권평균교체율", "상권평균교체횟수", "상권평균생존분기수"]],
        on="TRDAR_CD", how="inner",
    )
    print(f"최종 분석 대상 위치: {len(df):,}")

    df["상권대비교체율"] = df["교체율"] - df["상권평균교체율"]
    df["상권대비교체횟수"] = df["교체횟수"] - df["상권평균교체횟수"]
    df["상권대비생존분기수"] = df["평균생존분기수"] - df["상권평균생존분기수"]

    # --- 임계값 & 라벨 ---
    rows = []
    labels = {}
    for basis in ["교체율", "교체횟수"]:
        col = f"상권대비{basis}"
        q90 = df[col].quantile(rc.RISK_QUANTILE)
        q95 = df[col].quantile(rc.TOP5_QUANTILE)
        lab = (df[col] >= q90).astype(int)
        labels[basis] = lab
        risk = df[lab == 1]
        rest = df[lab == 0]
        rows.append(dict(
            기준=basis,
            사용여부="채택" if basis == rc.RISK_BASIS else "비교용",
            상위10_임계값=q90,
            상위5_임계값=q95,
            위험위치수=int(lab.sum()),
            위험비율=lab.mean(),
            위험그룹_평균생존분기수=risk["평균생존분기수"].mean(),
            위험그룹_중앙생존분기수=risk["평균생존분기수"].median(),
            위험그룹_평균생존년=risk["평균생존분기수"].mean() / 4,
            위험그룹_최소교체횟수=risk["교체횟수"].min(),
            위험그룹_중앙교체횟수=risk["교체횟수"].median(),
            나머지_평균생존분기수=rest["평균생존분기수"].mean(),
        ))
    summary = pd.DataFrame(rows)

    df["위험여부"] = labels[rc.RISK_BASIS]
    df["위험여부_교체율기준"] = labels["교체율"]
    df["위험여부_교체횟수기준"] = labels["교체횟수"]
    df["상위5여부"] = (df[f"상권대비{rc.RISK_BASIS}"] >= summary.loc[
        summary["기준"] == rc.RISK_BASIS, "상위5_임계값"].iloc[0]).astype(int)

    rc.print_table(summary, "상위 10% 임계값과 위험그룹 요약")

    # --- 두 기준 / 기존 자주바뀜여부(평균생존<8분기)와의 겹침 ---
    a, b = df["위험여부_교체율기준"], df["위험여부_교체횟수기준"]
    jacc = ((a == 1) & (b == 1)).sum() / max(((a == 1) | (b == 1)).sum(), 1)
    print(f"\n교체율 기준 vs 교체횟수 기준 위험위치 겹침(Jaccard): {jacc:.3f}")
    if "자주바뀜여부" in df.columns:
        ct = pd.crosstab(df["위험여부"], df["자주바뀜여부"], margins=True)
        print("\n[위험여부(상권대비 상위10%) x 자주바뀜여부(평균생존<8분기, 17/24번 정의)]")
        print(ct.to_string())

    # --- 위험그룹 교체횟수 분포 (발표자료 10p 우측 표) ---
    dist = (
        df.loc[df["위험여부"] == 1, "교체횟수"].value_counts().sort_index()
        .rename_axis("교체횟수").reset_index(name="위치수")
    )
    print("\n[위험그룹 교체횟수 분포 (앞부분)]")
    print(dist.head(15).to_string(index=False))

    # --- 저장 ---
    keep = [
        "위치ID", "PNU19", "층_최종", "경도", "위도", "시군구명", "도로명주소",
        "TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "대표업종",
        "관측분기수", "교체횟수", "교체율", "평균생존분기수", "최장생존분기수", "공실률",
        "첫분기", "끝분기",
        "상권위치수", "상권평균교체율", "상권평균교체횟수", "상권평균생존분기수",
        "상권대비교체율", "상권대비교체횟수", "상권대비생존분기수",
        "위험여부", "상위5여부", "위험여부_교체율기준", "위험여부_교체횟수기준", "자주바뀜여부",
    ]
    keep = [c for c in keep if c in df.columns]
    rc.save_csv(df[keep], "45_location_risk_base.csv")
    rc.save_csv(district, "45_district_baseline.csv")
    rc.save_csv(summary, "45_threshold_summary.csv")
    rc.save_csv(dist, "45_turnover_count_dist.csv")
    rc.log("45번 완료", t0)


if __name__ == "__main__":
    main()
