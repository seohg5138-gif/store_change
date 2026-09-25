# -*- coding: utf-8 -*-
"""
49번: 원인 탐색 페이지용 집계 (발표자료 14p, 16p)

  1) 변수별 "위험비율" = 그 범주/구간에 속한 위치 중 실제 위험 위치(상위 10%) 비율
     + lift = 위험비율 / 전체 위험비율(0.10)
  2) "전체 vs 위험" = 전체 위치의 분포와 위험 위치의 분포 비교 (범주별 비중)
  3) 조합 세그먼트: 층구분 x 연면적구간 x 주용도 별 위험비율
     -> "대형 상업건물 1층 판매시설이 가장 위험"이 새 데이터에서도 나오는지 확인
     (표본 MIN_SEGMENT_N 미만 조합은 제외: 작은 셀의 우연한 고비율을 결론으로 쓰지 않기 위해)
  4) 대시보드용 JSON (48번 중요도/성능이 있으면 같이 넣음)

  * 수치형은 해석이 쉬운 고정 구간을 쓰고, 구간 정의가 없는 변수는 5분위로 자른다.

입력:  45_location_risk_base.csv, 47_location_features.csv, (선택) 48_feature_importance.csv, 48_model_metrics.csv
출력:  49_variable_risk_profile.csv, 49_segment_risk.csv, 49_dashboard_profile.json
"""

import json
import os

import numpy as np
import pandas as pd

import risk_common as rc

MIN_SEGMENT_N = 300

FIXED_BINS = {
    "연면적": ([0, 500, 1000, 3000, 10000, np.inf],
             ["500㎡ 미만", "500~1천㎡", "1천~3천㎡", "3천~1만㎡", "1만㎡ 이상"]),
    "건물지상층수": ([0, 2, 4, 9, 19, np.inf], ["1~2층", "3~4층", "5~9층", "10~19층", "20층 이상"]),
    "건물내위치수": ([0, 1, 3, 9, 29, np.inf], ["1곳", "2~3곳", "4~9곳", "10~29곳", "30곳 이상"]),
    "지하철거리_m": ([-1, 250, 500, 1000, np.inf], ["250m 이내", "250~500m", "500m~1km", "1km 초과"]),
}


def binned(df, var):
    s = df[var]
    if not pd.api.types.is_numeric_dtype(s):
        return s.fillna("정보없음").astype(str)
    if var in FIXED_BINS:
        edges, labels = FIXED_BINS[var]
        out = pd.cut(s, bins=edges, labels=labels, right=True).astype(str)
    else:
        out = pd.qcut(s, q=5, duplicates="drop").astype(str)
    return out.replace({"nan": "정보없음"})


def profile(df, var, base_rate):
    b = binned(df, var)
    g = df.groupby(b, observed=True)["위험여부"].agg(위치수="size", 위험수="sum").reset_index()
    g = g.rename(columns={g.columns[0]: "구간"})
    g["위험비율"] = g["위험수"] / g["위치수"]
    g["lift"] = g["위험비율"] / base_rate
    g["전체분포비중"] = g["위치수"] / g["위치수"].sum()
    g["위험분포비중"] = g["위험수"] / g["위험수"].sum()
    g.insert(0, "변수", var)
    if var in FIXED_BINS:
        order = {lab: i for i, lab in enumerate(FIXED_BINS[var][1])}
        g["_o"] = g["구간"].map(order).fillna(99)
        g = g.sort_values("_o").drop(columns="_o")
    elif not pd.api.types.is_numeric_dtype(df[var]):
        g = g.sort_values("위치수", ascending=False)
    return g


def main():
    t0 = rc.start_timer()
    base = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "위험여부"])
    feat = pd.read_csv("47_location_features.csv", low_memory=False)
    df = base.merge(feat, on="위치ID", how="inner")
    base_rate = df["위험여부"].mean()
    print(f"분석 위치 {len(df):,}개, 전체 위험비율 {base_rate*100:.2f}%")

    variables = rc.CAT_FEATURES + rc.NUM_FEATURES + list(rc.EXTRA_FEATURES)
    prof = pd.concat([profile(df, v, base_rate) for v in variables], ignore_index=True)
    for v in variables:
        rc.print_table(prof[prof["변수"] == v].drop(columns="변수"), f"{v} - 위험비율 / 전체 vs 위험")
    rc.save_csv(prof, "49_variable_risk_profile.csv")

    # ------------------------------------------------------------------ 조합 세그먼트
    df["연면적구간"] = binned(df, "연면적")
    seg = (
        df.groupby(["층구분", "연면적구간", "주용도"], observed=True)["위험여부"]
        .agg(위치수="size", 위험수="sum").reset_index()
    )
    seg["위험비율"] = seg["위험수"] / seg["위치수"]
    seg["lift"] = seg["위험비율"] / base_rate
    seg_ok = seg[seg["위치수"] >= MIN_SEGMENT_N].sort_values("위험비율", ascending=False)
    if seg_ok.empty:
        print(f"[경고] 표본 {MIN_SEGMENT_N}개 이상인 조합이 없습니다. MIN_SEGMENT_N을 낮춰서 다시 보세요.")
    rc.print_table(seg_ok.head(15), f"위험비율 상위 조합 (표본 {MIN_SEGMENT_N}개 이상)")
    rc.print_table(seg_ok.tail(10), "위험비율 하위 조합")
    rc.save_csv(seg.sort_values("위험비율", ascending=False), "49_segment_risk.csv")

    two_way = (
        df.groupby(["층구분", "연면적구간"], observed=True)["위험여부"].mean().unstack()
    )
    order = [c for c in FIXED_BINS["연면적"][1] if c in two_way.columns]
    print("\n[층구분 x 연면적구간 위험비율]")
    print((two_way[order] * 100).round(1).to_string())

    # ------------------------------------------------------------------ 대시보드 JSON
    dash = {"전체위험비율": base_rate, "분석위치수": int(len(df)), "변수": {}}
    for v in variables:
        p = prof[prof["변수"] == v]
        dash["변수"][v] = {
            "구간": p["구간"].tolist(),
            "위험비율": p["위험비율"].round(4).tolist(),
            "전체분포비중": p["전체분포비중"].round(4).tolist(),
            "위험분포비중": p["위험분포비중"].round(4).tolist(),
            "위치수": p["위치수"].astype(int).tolist(),
        }
    if os.path.exists("48_feature_importance.csv"):
        imp = pd.read_csv("48_feature_importance.csv")
        dash["중요도"] = dict(zip(imp["변수"], imp["중요도(%)"].round(2)))
    if os.path.exists("48_model_metrics.csv"):
        dash["모델성능"] = pd.read_csv("48_model_metrics.csv").iloc[0].round(4).to_dict()
    top = seg_ok.head(10)
    dash["위험조합_상위"] = top[["층구분", "연면적구간", "주용도", "위치수", "위험비율"]].round(4).to_dict("records")

    with open("49_dashboard_profile.json", "w", encoding="utf-8") as f:
        json.dump(dash, f, ensure_ascii=False, indent=1, default=float)
    print("[완료] 저장: 49_dashboard_profile.json")
    rc.log("49번 완료", t0)


if __name__ == "__main__":
    main()
