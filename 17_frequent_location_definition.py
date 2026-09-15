import pandas as pd

# =========================================================================
# 17번: "자주 바뀌는 위치" 정의 (중앙값 2년=8분기 기준)
# =========================================================================
# 기준: 절단(좌측/우측) 제거한 순수 폐업 episode들의 생존분기수 중앙값이 8분기(2년)였음.
# 이 8분기를 "이 상권/이 자리에서 통상적으로 버티는 기간"의 기준값으로 삼아서,
# 위치(location) 단위로 "이 자리는 세입자들이 평균적으로 그 기준보다 짧게 버틴다"를
# 정의한다.

df_loc = pd.read_csv("location_metrics_corrected.csv")

MEDIAN_SURVIVAL = 8  # 분기 (=2년), 절단 제거 후 계산된 전체 중앙값

reliable = df_loc[df_loc["관측분기수"] >= 15].copy()  # 표본 신뢰가능(15분기 이상 관측)

reliable["자주바뀜여부"] = reliable["평균생존분기수"] < MEDIAN_SURVIVAL

frequent = reliable[reliable["자주바뀜여부"]].copy()

print(f"신뢰가능 위치 수: {len(reliable):,}")
print(f"자주 바뀌는 위치 수(평균생존 < {MEDIAN_SURVIVAL}분기): {len(frequent):,} "
      f"({len(frequent)/len(reliable)*100:.2f}%)")

frequent.to_csv("17_frequent_change_locations.csv", index=False, encoding="utf-8-sig")
print("저장 완료: 17_frequent_change_locations.csv")
