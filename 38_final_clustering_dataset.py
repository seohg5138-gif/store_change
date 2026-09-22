import numpy as np
import pandas as pd

# =========================================================================
# 38번: 클러스터링 최종 피처셋 = A(방문자 특성, 37번) + B(산업 특성, 우리
# 25·26번 데이터로 직접 계산) 결합. 우리 1,614개 상권만 남기고, 최종
# 피처 컬럼 기준으로 결측치가 있는 행을 dropna해서 정확한 손실 행수를
# 확인한다. (결측치를 채우지 않고 그대로 드롭 - 모델로 미리 채우는 건
# 순환논리이므로 하지 않음)


def entropy(probs):
    probs = np.array([p for p in probs if p > 0])
    if len(probs) == 0:
        return np.nan
    return -(probs * np.log(probs)).sum()


# =========================================================================
# B. 산업 특성 - 26번(업종비중 가중) 데이터로 직접 계산
# =========================================================================
di = pd.read_csv("26_industry_baseline_by_district.csv", dtype={"TRDAR_CD": str})
di = di[di["실제상권여부"]]  # 상권외 그룹은 여기선 제외 (우리 1,614개만 다룸)

industry_wide = di.pivot(index="TRDAR_CD", columns="업종", values="위치수").fillna(0)
industry_total = industry_wide.sum(axis=1)
industry_share = industry_wide.div(industry_total, axis=0)

industry_feat = pd.DataFrame(index=industry_share.index)
for col in industry_share.columns:
    industry_feat[f"업종비중_{col}"] = industry_share[col]

industry_feat["업종다양성_엔트로피"] = industry_share.apply(entropy, axis=1)
industry_feat["업종집중도_HHI"] = (industry_share ** 2).sum(axis=1)
industry_feat["상위3업종비중"] = industry_share.apply(lambda r: r.sort_values(ascending=False).head(3).sum(), axis=1)
industry_feat["전체점포수_로그"] = np.log1p(industry_total)
industry_feat = industry_feat.reset_index()

print(f"B(산업 특성): {len(industry_feat):,}개 상권, {len(industry_feat.columns) - 1}개 변수")

# =========================================================================
# A + B 결합, 우리 1,614개로 제한
# =========================================================================
visitor = pd.read_csv("37_clustering_ready_features.csv", dtype={"TRDAR_CD": str})
our = pd.read_csv("25_district_full_baseline.csv", dtype={"TRDAR_CD": str})
our_codes = set(our.loc[our["실제상권여부"], "TRDAR_CD"])

combined = visitor.merge(industry_feat, on="TRDAR_CD", how="left")
combined = combined[combined["TRDAR_CD"].isin(our_codes)].reset_index(drop=True)

print(f"\nA+B 결합, 우리 1,614개로 제한: {len(combined):,}행 "
      f"(1,614개와 같아야 함)")

# --- 클러스터링에 실제로 쓸 최종 피처 컬럼 목록 ---
# (문서 1번의 A/B 변수 중, 이 시점에 실제로 확보 가능한 것들만)
feature_cols = [
    # A. 방문자 특성
    "총생활인구_로그", "여성비율",
    "연령비율_10대", "연령비율_20대", "연령비율_30대", "연령비율_40대", "연령비율_50대", "연령비율_60대이상",
    "주말인구비율",
    "시간대비율_00_06", "시간대비율_06_11", "시간대비율_11_14", "시간대비율_14_17", "시간대비율_17_21", "시간대비율_21_24",
    "연령다양성_엔트로피", "시간대다양성_엔트로피",
    "총직장인구_로그", "직장인구_여성비율",
    "총상주인구_로그", "상주인구_여성비율",
    "업무형비율",
    "분기총매출_로그", "건당매출액", "주말매출비율", "야간매출비율", "매출성장률", "매출변동성_CV",
    "생활인구당매출",
    # B. 산업 특성
    "전체점포수_로그", "업종다양성_엔트로피", "업종집중도_HHI", "상위3업종비중",
] + [c for c in combined.columns if c.startswith("업종비중_")]

missing_cols = [c for c in feature_cols if c not in combined.columns]
if missing_cols:
    print(f"⚠️ 피처 목록에 있는데 실제 컬럼에 없는 것: {missing_cols}")
feature_cols = [c for c in feature_cols if c in combined.columns]

combined.to_csv("38_district_profile_full.csv", index=False, encoding="utf-8-sig")

# --- 정확한 손실 행수 계산 ---
before_n = len(combined)
clustering_ready = combined.dropna(subset=feature_cols)
after_n = len(clustering_ready)

print(f"\n=== 최종 피처({len(feature_cols)}개) 기준 dropna 결과 ===")
print(f"드롭 전: {before_n:,}개 상권")
print(f"드롭 후: {after_n:,}개 상권")
print(f"손실: {before_n - after_n:,}개 ({(before_n - after_n) / before_n * 100:.2f}%)")

# --- 어떤 컬럼이 손실의 원인인지 다시 한번 명확히 ---
per_col_missing = combined[feature_cols].isna().sum()
per_col_missing = per_col_missing[per_col_missing > 0].sort_values(ascending=False)
print("\n손실에 기여한 컬럼별 결측 행수:")
print(per_col_missing)

clustering_ready.to_csv("38_clustering_ready_final.csv", index=False, encoding="utf-8-sig")
print(f"\n저장 완료: 38_district_profile_full.csv (결측 포함 전체 {before_n:,}행), "
      f"38_clustering_ready_final.csv (dropna 후 {after_n:,}행, 클러스터링 입력용)")
