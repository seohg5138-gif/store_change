import pandas as pd
import numpy as np

# =========================================================================
# 22번: 위치ID(PK) 기준으로 공실을 직접 계산
# =========================================================================
# 정의: 위치ID의 [첫 episode 시작 ~ 마지막 episode 종료] 사이에서,
#       '점유되지 않은' 분기 = 공실 분기
#
# 중요(생존율 계산과의 일관성): 13번에서 구조적 공백(bad_periods) 안에 낀
# 가짜 교체를 병합할 때, "앞뒤가 같은 가게면 그 공백 구간에도 계속 있었다"고
# 가정하고 생존기간에 포함시켰다. 공실률도 같은 가정을 그대로 적용해야
# 일관성이 생긴다. 즉:
#   - tenancy_episode_corrected.parquet의 episode가 덮고 있는 구간 -> '점유됨'
#     (bad_periods 안이라도 13번이 이미 병합해서 '있었다'고 판단한 구간이면 점유로 인정)
#   - episode가 안 덮고 있으면서 + bad_periods에 속한 분기만 -> '알 수 없음'
#     (분모/분자 양쪽에서 제외, 억지로 공실도 점유도 아니라고 단정하지 않음)
#   - episode가 안 덮고 있으면서 + bad_periods가 아닌 분기 -> 진짜 '공실'로 계산

ep = pd.read_parquet("tenancy_episode_corrected.parquet")

raw = pd.read_parquet("shop_period_raw.parquet", columns=["period", "층정보_신뢰가능여부"])
raw = raw[raw["층정보_신뢰가능여부"] == "검증완료"]
period_grid = sorted(raw["period"].unique())

try:
    bad_periods = set(pd.read_csv("13_bad_periods.csv")["bad_period"])
except FileNotFoundError:
    bad_periods = set(pd.read_csv("bad_periods.csv")["bad_period"])

print(f"전역 분기 수: {len(period_grid)}개")
print(f"구조적 공백 분기: {sorted(bad_periods)}")

records = []
for loc_id, group in ep.groupby("위치ID"):
    start = group["시작분기"].min()
    end = group["종료분기"].max()

    # '점유된 분기' = 이 위치의 (보정된) episode들이 덮고 있는 모든 분기.
    # 13번이 병합한 episode라면, bad_periods 안이라도 여기 포함되어 '점유'로 잡힌다.
    occupied = set()
    for row in group.itertuples(index=False):
        occupied.update(p for p in period_grid if row.시작분기 <= p <= row.종료분기)

    # 전체 구간 중, "bad_periods인데 점유로도 안 잡힌"(=진짜 알 수 없는) 분기만 계산에서 제외
    span = [
        p for p in period_grid
        if start <= p <= end and not (p in bad_periods and p not in occupied)
    ]
    vacant = [p for p in span if p not in occupied]

    records.append(dict(
        위치ID=loc_id,
        첫분기=start,
        끝분기=end,
        관측span분기수=len(span),
        공실분기수=len(vacant),
    ))

vac_df = pd.DataFrame(records)
vac_df["공실률"] = vac_df["공실분기수"] / vac_df["관측span분기수"].replace(0, np.nan)

print(f"\n위치 수: {len(vac_df):,}")
print(vac_df["공실률"].describe())

vac_df.to_csv("22_location_vacancy.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: 22_location_vacancy.csv")
