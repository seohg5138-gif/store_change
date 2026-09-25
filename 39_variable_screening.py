# -*- coding: utf-8 -*-
"""
39_variable_screening.py

기존 39/40번(무조건 A/B/AB 통째로 넣고 알고리즘만 바꾸던 버전)은 폐기.
이 스크립트는 클러스터링에 넣기 전 변수 자체를 점검하는 단계만 담당한다.

순서:
  0. 데이터 로드 + 결측치 제거 (상관계수 계산 전에 처리)
  1. 구성비 변수(업종비중/연령비율/시간대비율) 기준범주 제거 → 이 그룹은
     정의상 합이 1이라 상관계수/VIF 점검 대상에서 제외하고 따로 보관
  2. 나머지 일반 변수 대상 저분산 변수 제거
  3. 상관계수 기준 1차 제거 (|r| > 0.8)
  4. VIF 기준 반복 제거 (VIF > 10)
  5. 의미 중복 검토용 변수 덴드로그램 (자동 제거 없음, 사람이 보고 판단)
  6. 최종 변수 목록 확정 저장
  7. 후보 변수셋(4~6개 조합) 구성 → 다음 단계(클러스터링 그리드서치, 40번)에서 사용

출력:
  39_screened_variables.csv       : 최종 살아남은 변수만 남긴 상권 데이터
  39_variable_log.csv             : 각 변수의 처리 이력(유지/제거 사유)
  39_dendrogram.png               : 의미 중복 검토용 변수 덴드로그램
  39_candidate_variable_sets.csv  : 후보 변수 조합 목록
"""

import itertools
import random

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------------------------------------------------
# 설정값 (필요하면 여기만 바꾸면 됨)
# ----------------------------------------------------------------------
INPUT_PATH = "38_clustering_ready_final.csv"
ID_COL = "TRDAR_CD"

CORR_THRESHOLD = 0.8      # 상관계수 1차 제거 기준
VIF_THRESHOLD = 10.0      # VIF 반복 제거 기준
LOW_VAR_MODE_RATIO = 0.95 # 최빈값 비율이 이 이상이면 저분산으로 간주
DENDRO_DISTANCE_CUT = 0.3 # 덴드로그램에서 "같은 묶음"으로 볼 거리 기준(참고용)

CANDIDATE_SIZES = [4, 5, 6]   # 후보 변수셋 크기
MAX_COMBOS_PER_GROUP_SIZE = 100  # (범주, 크기)별 최대 후보 개수 - 넘으면 랜덤 샘플
RANDOM_SEED = 42

# 저분산/상관계수/VIF 단계에서 무조건 살아남기는 변수 (사람이 직접 결정)
FORCE_KEEP = {"전체점포수_로그"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 한글 폰트 (없으면 그냥 넘어감 - 라벨이 깨질 수 있음)
for name in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    try:
        fm.findfont(name, fallback_to_default=False)
        plt.rcParams["font.family"] = name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

log_rows = []  # 변수 처리 이력 기록용


def log(var, status, reason):
    log_rows.append({"변수": var, "상태": status, "사유": reason})


# ----------------------------------------------------------------------
# 0. 데이터 로드 + 결측치 제거
# ----------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
n_before = len(df)
df = df.dropna()
n_after = len(df)
print(f"[0] 로드: {n_before}개 상권 -> 결측치 제거 후 {n_after}개 "
      f"(제거 {n_before - n_after}개)")

# --- 파생변수: 점포당매출 = 분기총매출 / 전체점포수 ---
# 전체점포수_로그는 np.log1p(점포수)로 만들어졌으므로(38번 스크립트 확인)
# expm1로 원래 점포수를 복원한다. 전체점포수_로그와 분기총매출_로그가
# 상관계수 0.822로 거의 같은 "상권 규모" 정보를 담고 있어서 하나가
# 제거됐었는데, 매출을 점포수로 나누면 규모 효과가 빠진 생산성 지표가
# 되어 두 변수를 같이 쓸 수 있게 된다.
df["전체점포수"] = np.expm1(df["전체점포수_로그"])
df["점포당매출"] = df["분기총매출"] / df["전체점포수"].replace(0, np.nan)

print(f"[0-2] 파생변수 추가: 점포당매출 = 분기총매출 / 전체점포수 "
      f"(상관계수 전체점포수_로그: {df['점포당매출'].corr(df['전체점포수_로그']):.3f}, "
      f"분기총매출_로그: {df['점포당매출'].corr(df['분기총매출_로그']):.3f})")

all_cols = [c for c in df.columns if c != ID_COL]

# ----------------------------------------------------------------------
# 1. 구성비 변수 그룹 분리 + 기준범주 제거
# ----------------------------------------------------------------------
COMP_GROUPS = {
    "연령비율": {"cols": [c for c in all_cols if c.startswith("연령비율_")],
              "reference": "연령비율_60대이상"},
    "시간대비율": {"cols": [c for c in all_cols if c.startswith("시간대비율_")],
                "reference": "시간대비율_00_06"},
    "업종비중": {"cols": [c for c in all_cols if c.startswith("업종비중_")],
              "reference": "업종비중_음식"},
}

comp_kept_cols = []
for group_name, info in COMP_GROUPS.items():
    for c in info["cols"]:
        if c == info["reference"]:
            log(c, "제외", f"{group_name} 기준범주(구성비 합=1 구조적 공선성)")
        else:
            comp_kept_cols.append(c)
            log(c, "유지", f"{group_name} 구성비 변수(기준범주 제외 처리 완료, "
                          f"상관계수/VIF 점검 대상 아님)")

comp_all_cols = [c for g in COMP_GROUPS.values() for c in g["cols"]]

# 전체점포수(원본 개수)는 점포당매출 계산용 중간 산출물이라 모델링 변수 풀에서
# 제외. 분기총매출(원본)은 점포당매출로 대체하므로 여기서 제외하고
# 로그 버전(분기총매출_로그)은 일반 점검 절차에 그대로 맡긴다.
EXCLUDE_INTERMEDIATE = {"전체점포수", "분기총매출"}
for c in EXCLUDE_INTERMEDIATE:
    log(c, "제외", "중간 산출물 또는 점포당매출로 대체됨")

general_cols = [c for c in all_cols if c not in comp_all_cols and c not in EXCLUDE_INTERMEDIATE]

print(f"[1] 구성비 변수: {len(comp_all_cols)}개 중 기준범주 {len(comp_all_cols)-len(comp_kept_cols)}개 제외, "
      f"{len(comp_kept_cols)}개 유지")
print(f"    점검 대상 일반 변수: {len(general_cols)}개")

work = df[general_cols].copy()

# ----------------------------------------------------------------------
# 1-1. 극단 이상치 자동 윈저라이징
#      손으로 하나씩 잡다 보면 두더지잡기가 된다 (생활인구당매출/점포당매출을
#      잡았더니 매출성장률/건당매출액/총직장인구 등에서 또 나옴). 그래서
#      전체 일반 변수를 훑어서 |z-score| > OUTLIER_Z_THRESHOLD인 변수는
#      자동으로 1~99th 백분위수로 윈저라이징한다. 이런 극단값이 있으면
#      어떤 변수 조합에 들어가든 "그 점 하나 vs 나머지"로 쪼개지면서
#      실루엣이 인위적으로 치솟는 가짜 군집을 만든다 (실제로 겪은 문제).
# ----------------------------------------------------------------------
OUTLIER_Z_THRESHOLD = 8.0
WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99

for c in work.columns:
    z = (work[c] - work[c].mean()) / work[c].std()
    max_abs_z = z.abs().max()
    if max_abs_z > OUTLIER_Z_THRESHOLD:
        lo, hi = work[c].quantile(WINSOR_LOWER), work[c].quantile(WINSOR_UPPER)
        n_capped = ((work[c] < lo) | (work[c] > hi)).sum()
        work[c] = work[c].clip(lower=lo, upper=hi)
        log(c, "윈저라이징", f"|z|={max_abs_z:.1f} > {OUTLIER_Z_THRESHOLD} -> "
                          f"{n_capped}개 상권을 [{lo:,.0f}, {hi:,.0f}]로 클리핑")
        print(f"[1-1] {c}: |z|={max_abs_z:.1f} -> {n_capped}개 윈저라이징 "
              f"(1~99th: {lo:,.0f} ~ {hi:,.0f})")

# 윈저라이징된 값을 원본 df에도 반영해서 이후 단계(덴드로그램 등)와
# 값이 어긋나지 않게 한다.
df[general_cols] = work

# ----------------------------------------------------------------------
# 2. 저분산 변수 제거
# ----------------------------------------------------------------------
low_var_cols = []
for c in work.columns:
    if c in FORCE_KEEP:
        continue
    vc = work[c].value_counts(normalize=True)
    mode_ratio = vc.iloc[0] if len(vc) else 1.0
    std = work[c].std()
    if std == 0 or mode_ratio > LOW_VAR_MODE_RATIO:
        low_var_cols.append(c)
        log(c, "제거", f"저분산 (최빈값 비율={mode_ratio:.3f}, std={std:.4g})")

work = work.drop(columns=low_var_cols)
print(f"[2] 저분산 제거: {len(low_var_cols)}개 제거 -> {low_var_cols if low_var_cols else '없음'}")
print(f"    남은 변수: {work.shape[1]}개")

# ----------------------------------------------------------------------
# 3. 상관계수 기준 1차 제거 (|r| > CORR_THRESHOLD)
#    - 원본/로그 쌍처럼 이름이 짝인 경우 로그 버전을 우선 유지
#    - 그 외에는 다른 변수들과의 평균 절대상관이 더 높은(더 중복되는) 쪽을 제거
# ----------------------------------------------------------------------
def prefer_keep(a, b):
    """a, b 중 우선적으로 유지할 변수를 고른다. 로그변환 버전을 우선."""
    a_is_log, b_is_log = a.endswith("_로그"), b.endswith("_로그")
    a_base = a.replace("_로그", "")
    b_base = b.replace("_로그", "")
    if a_base == b_base:
        if a_is_log and not b_is_log:
            return a, b  # keep a(log), drop b(raw)
        if b_is_log and not a_is_log:
            return b, a
    return None  # 이름 짝이 아니면 상관도 기준으로 별도 처리


corr_removed = []
while True:
    corr = work.corr().abs()
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 0)
    max_val = corr_vals.max()
    if max_val <= CORR_THRESHOLD or work.shape[1] <= 1:
        break
    i, j = np.unravel_index(np.argmax(corr_vals), corr_vals.shape)
    a, b = corr.index[i], corr.columns[j]

    if a in FORCE_KEEP or b in FORCE_KEEP:
        keep, drop = (a, b) if a in FORCE_KEEP else (b, a)
        reason_tag = "강제 유지 변수와 상관 - 반대쪽 제거"
    else:
        pair_pref = prefer_keep(a, b)
        if pair_pref is not None:
            keep, drop = pair_pref
            reason_tag = "원본/로그 쌍, 로그버전 우선 유지"
        else:
            corr_zero_diag = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
            mean_corr_a = corr_zero_diag[a].mean()
            mean_corr_b = corr_zero_diag[b].mean()
            drop, keep = (a, b) if mean_corr_a >= mean_corr_b else (b, a)
            reason_tag = "다른 변수들과의 평균 상관 더 높은 쪽 제거"

    work = work.drop(columns=[drop])
    corr_removed.append(drop)
    log(drop, "제거", f"상관계수 |r|={max_val:.3f} (짝: {keep}) - {reason_tag}")

print(f"[3] 상관계수 제거: {len(corr_removed)}개 제거 -> {corr_removed if corr_removed else '없음'}")
print(f"    남은 변수: {work.shape[1]}개")

# ----------------------------------------------------------------------
# 4. VIF 기준 반복 제거
# ----------------------------------------------------------------------
def calc_vif(frame):
    x = frame.copy()
    x = (x - x.mean()) / x.std()  # 표준화 후 계산 (스케일 차이로 인한 왜곡 방지)
    x.insert(0, "const", 1.0)
    vifs = []
    for idx in range(1, x.shape[1]):
        vifs.append(variance_inflation_factor(x.values, idx))
    return pd.Series(vifs, index=frame.columns)


vif_removed = []
while work.shape[1] > 1:
    vif_series = calc_vif(work)
    droppable = vif_series.drop(index=[c for c in FORCE_KEEP if c in vif_series.index],
                                 errors="ignore")
    if droppable.empty or droppable.max() <= VIF_THRESHOLD:
        break
    drop_var = droppable.idxmax()
    max_vif = droppable.max()
    work = work.drop(columns=[drop_var])
    vif_removed.append(drop_var)
    log(drop_var, "제거", f"VIF={max_vif:.2f} (기준={VIF_THRESHOLD})")

print(f"[4] VIF 제거: {len(vif_removed)}개 제거 -> {vif_removed if vif_removed else '없음'}")
print(f"    최종 일반 변수: {work.shape[1]}개 -> {list(work.columns)}")

final_general_cols = list(work.columns)
for c in final_general_cols:
    log(c, "최종유지", "저분산/상관계수/VIF 통과")

# ----------------------------------------------------------------------
# 5. 의미 중복 검토용 덴드로그램 (자동 제거 없음)
# ----------------------------------------------------------------------
final_vars_for_dendro = final_general_cols + comp_kept_cols
dendro_data = df[final_vars_for_dendro].copy()
dendro_corr = dendro_data.corr().abs()
dist_vals = 1 - dendro_corr.to_numpy(copy=True)
np.fill_diagonal(dist_vals, 0)
condensed = squareform(dist_vals, checks=False)
Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(10, max(6, len(final_vars_for_dendro) * 0.28)))
dendrogram(
    Z,
    labels=dendro_data.columns.tolist(),
    orientation="right",
    color_threshold=DENDRO_DISTANCE_CUT,
    ax=ax,
)
ax.set_xlabel("거리 (1 - |상관계수|)")
ax.set_title("변수 간 유사도 덴드로그램 (의미 중복 검토용, 자동 제거 아님)")
fig.tight_layout()
fig.savefig("39_dendrogram.png", dpi=150)
plt.close(fig)

clusters = fcluster(Z, t=DENDRO_DISTANCE_CUT, criterion="distance")
dendro_groups = pd.DataFrame({"변수": dendro_data.columns, "묶음번호": clusters})
dendro_groups = dendro_groups.sort_values("묶음번호")
grouped_more_than_1 = dendro_groups.groupby("묶음번호").filter(lambda g: len(g) > 1)
print(f"[5] 덴드로그램 저장: 39_dendrogram.png")
if len(grouped_more_than_1):
    print(f"    거리 {DENDRO_DISTANCE_CUT} 기준으로 같이 묶인 변수군 (사람이 의미 중복 판단):")
    for gid, g in grouped_more_than_1.groupby("묶음번호"):
        print(f"      묶음 {gid}: {', '.join(g['변수'])}")
else:
    print(f"    거리 {DENDRO_DISTANCE_CUT} 기준으로 묶이는 변수 없음 (변수들이 서로 충분히 구분됨)")

# ----------------------------------------------------------------------
# 6. 최종 변수 목록 저장
# ----------------------------------------------------------------------
final_cols = [ID_COL] + final_general_cols + comp_kept_cols
screened = df[final_cols]
screened.to_csv("39_screened_variables.csv", index=False, encoding="utf-8-sig")

log_df = pd.DataFrame(log_rows)
log_df.to_csv("39_variable_log.csv", index=False, encoding="utf-8-sig")

print(f"[6] 최종 변수 {len(final_cols)-1}개 (일반 {len(final_general_cols)}개 + "
      f"구성비 {len(comp_kept_cols)}개) 저장: 39_screened_variables.csv")
print(f"    변수 처리 이력 저장: 39_variable_log.csv")

# ----------------------------------------------------------------------
# 7. 후보 변수셋(4~6개 조합) 구성
#    - A(방문자류) / B(산업류) / C(매출류) 세 범주로 나누고
#      범주 내부, 그리고 A+B 조합에서 4~6개씩 뽑는다
# ----------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "A_방문자": ["생활인구", "여성비율", "주말인구비율", "연령다양성", "시간대다양성",
              "직장인구", "상주인구", "총가구수", "연령비율_", "시간대비율_"],
    "B_산업": ["업종다양성", "업종집중도", "상위3업종비중", "전체점포수", "업종비중_"],
    "C_매출": ["매출", "업무형비율"],
}


def assign_category(var):
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in var for kw in keywords):
            return cat
    return "기타"


category_map = {v: assign_category(v) for v in final_cols if v != ID_COL}
cat_to_vars = {}
for v, cat in category_map.items():
    cat_to_vars.setdefault(cat, []).append(v)

print("[7] 범주별 최종 변수 풀:")
for cat, vars_ in cat_to_vars.items():
    print(f"    {cat} ({len(vars_)}개): {vars_}")

A = cat_to_vars.get("A_방문자", [])
B = cat_to_vars.get("B_산업", [])
C = cat_to_vars.get("C_매출", [])

# A/B/C 세 범주와 그 조합(AB/AC/BC/ABC)까지 전부 만든다. (예전 버전은 C_매출을
# 여기서 빠뜨렸던 버그가 있었음 - 이번엔 처음부터 넣음)
GROUPINGS = {
    "A_방문자만": A,
    "B_산업만": B,
    "C_매출만": C,
    "AB_방문자산업": A + B,
    "AC_방문자매출": A + C,
    "BC_산업매출": B + C,
    "ABC_전체": A + B + C,
}


def sampled_combinations(pool, size, cap):
    total = list(itertools.combinations(pool, size))
    if len(total) <= cap:
        return total
    idx = random.sample(range(len(total)), cap)
    return [total[i] for i in idx]


candidate_rows = []
set_id = 0
for group_name, pool in GROUPINGS.items():
    for size in CANDIDATE_SIZES:
        if len(pool) < size:
            continue
        combos = sampled_combinations(pool, size, MAX_COMBOS_PER_GROUP_SIZE)
        for combo in combos:
            set_id += 1
            candidate_rows.append({
                "set_id": f"S{set_id:04d}",
                "범주그룹": group_name,
                "변수개수": size,
                "변수목록": ";".join(combo),
            })

candidate_df = pd.DataFrame(candidate_rows)
candidate_df.to_csv("39_candidate_variable_sets.csv", index=False, encoding="utf-8-sig")

print(f"[7] 후보 변수셋 {len(candidate_df)}개 생성 -> 39_candidate_variable_sets.csv")
print(candidate_df["범주그룹"].value_counts())
