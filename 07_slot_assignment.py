# -*- coding: utf-8 -*-
"""
7단계: 호정보가 100% 결측이라서, 같은 PNU19+층_최종에 동시에 여러 상가업소번호가
있는 경우를 구분할 "슬롯번호"를 인위적으로 배정한다. (검증 결과 33%에서 발생 확인됨)

배정 규칙:
- 같은 (PNU19, 층_최종) 그룹 안에서, period 순서대로 훑으면서
  - 그 분기에 없어진 상가업소번호는 슬롯을 반납
  - 새로 등장한 상가업소번호는 "가장 작은 빈 슬롯 번호"를 받음 (없으면 새 번호 발급)
- 이러면 슬롯 번호가 "그 층의 물리적 자리" 개념에 가장 가깝게 유지됨 (완벽하진 않음)

⚠️ 중요: 이 스크립트는 그룹별 순차 처리라서 데이터가 크면 느릴 수 있습니다.
먼저 SAMPLE_FRAC를 작게 두고 로직이 맞는지 확인한 뒤, 1.0으로 바꿔서 전체 실행하세요.

실행:
    python 07_slot_assignment.py

출력:
    intermediate/df_all_slotted.parquet (슬롯번호 컬럼 추가)
"""

import heapq
import os
import sys
import time

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")

# 먼저 0.02(2%) 정도로 돌려서 몇 분 안에 끝나는지, 결과가 말이 되는지 확인하세요.
# 확인되면 1.0으로 바꿔서 전체 데이터에 적용하세요.
SAMPLE_FRAC = 1.0


def assign_slots_for_group(sub: pd.DataFrame) -> pd.Series:
    """
    sub: 이미 같은 (PNU19, 층_최종)로 묶인 부분 데이터프레임.
    반환: sub.index와 같은 인덱스를 가진 슬롯번호 Series.
    """
    sub_sorted = sub.sort_values("period")
    active = {}          # 상가업소번호 -> 슬롯번호
    free_slots = []      # min-heap
    next_new_slot = 1
    result = pd.Series(index=sub.index, dtype="int64")

    for period, period_df in sub_sorted.groupby("period"):
        current_ids = set(period_df["상가업소번호"])

        # 이번 분기에 없어진 애들 슬롯 반납
        gone = [sid for sid in active if sid not in current_ids]
        for sid in gone:
            heapq.heappush(free_slots, active.pop(sid))

        # 새로 등장한 애들 슬롯 배정
        for sid in current_ids:
            if sid not in active:
                if free_slots:
                    slot = heapq.heappop(free_slots)
                else:
                    slot = next_new_slot
                    next_new_slot += 1
                active[sid] = slot

        for idx, sid in zip(period_df.index, period_df["상가업소번호"]):
            result.loc[idx] = active[sid]

    return result


def main():
    df_all = pd.read_parquet(config.out_path("df_all_floor_split.parquet"))

    if SAMPLE_FRAC < 1.0:
        print(f"⚠️ SAMPLE_FRAC={SAMPLE_FRAC}로 표본만 처리합니다. 검증 후 1.0으로 바꿔서 재실행하세요.")
        pnu_floor_keys = (
            df_all[["PNU19", "층_최종"]].drop_duplicates().sample(frac=SAMPLE_FRAC, random_state=42)
        )
        df_target = df_all.merge(pnu_floor_keys, on=["PNU19", "층_최종"], how="inner")
    else:
        df_target = df_all

    n_groups = df_target.groupby(["PNU19", "층_최종"]).ngroups
    print(f"처리할 (PNU19, 층_최종) 그룹 수: {n_groups:,}, 대상 행수: {len(df_target):,}")

    t0 = time.time()
    slot_series = (
        df_target.groupby(["PNU19", "층_최종"], group_keys=False)
        .apply(assign_slots_for_group)
    )
    print(f"슬롯 배정 소요 시간: {time.time() - t0:.1f}초")

    df_target = df_target.copy()
    df_target["슬롯번호"] = slot_series

    print("\n한 그룹 안의 최대 슬롯번호 분포 (상위 10개):")
    max_slot_per_group = df_target.groupby(["PNU19", "층_최종"])["슬롯번호"].max()
    print(max_slot_per_group.value_counts().sort_index(ascending=False).head(10))

    suffix = "_sample" if SAMPLE_FRAC < 1.0 else ""
    out = config.out_path(f"df_all_slotted{suffix}.parquet")
    df_target.to_parquet(out, index=False)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
