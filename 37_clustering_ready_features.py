import numpy as np
import pandas as pd

# =========================================================================
# 37번: 클러스터링용 최종 상권 프로파일 테이블 정리.
# =========================================================================
# 36번 산출물에서 문제됐던 부분을 정리:
#   - 아파트가구비율: 전체 0(상권 단위 API에선 항상 0) -> 드롭
#   - 생활인구_상주인구_비율: 생활인구가 "특정 시점 누적"(요일/시간대별로
#     쪼개져 합산되는 구조, 공식 정의도 "특정 시점" 스냅샷들의 집합)이라
#     상주인구(등록 기반 스냅샷 1장)와 단위가 다를 가능성이 커서 드롭
#   - 직장인구_상주인구_비율: 단순 나눗셈 대신, 직장인구/(직장인구+상주인구)
#     "업무형 비율"로 교체. 0~1로 자동 고정되고, 분모가 0에 가까워지는
#     문제도 없어짐(직장인구·상주인구는 둘 다 등록 기반 스냅샷이라 단위도
#     맞을 가능성이 높음 - 요일/시간대 분해가 없는 구조가 그 근거)

df = pd.read_csv("36_district_profile_features.csv", dtype={"TRDAR_CD": str})

# --- 문제 컬럼 정리 ---
df = df.drop(columns=["아파트가구비율", "생활인구_상주인구_비율", "직장인구_상주인구_비율"])

df["업무형비율"] = df["총직장인구"] / (df["총직장인구"] + df["총상주인구"])
# 둘 다 0인 극소수 상권은 정의상 계산 불가 -> NaN 유지 (억지로 0.5 채우지 않음)

df.to_csv("37_clustering_ready_features.csv", index=False, encoding="utf-8-sig")

print(f"저장 완료: 37_clustering_ready_features.csv ({len(df):,}개 상권, {len(df.columns)}개 컬럼)")

print("\n=== 업무형비율 분포 ===")
print(df["업무형비율"].describe())

print("\n=== 남은 컬럼별 결측치 ===")
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n=== 분산 0인 컬럼(있으면 안 됨, 있으면 더 빼야 함) ===")
numeric = df.drop(columns=["TRDAR_CD"]).select_dtypes("number")
zero_var = numeric.columns[numeric.std(skipna=True) == 0]
print(list(zero_var) if len(zero_var) else "없음")
