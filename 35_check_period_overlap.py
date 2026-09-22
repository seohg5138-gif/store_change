import pandas as pd

# =========================================================================
# 34번: 우리 위치 데이터(201512~202512)와 33번이 받아온 API 데이터의 기간이
# 실제로 얼마나 겹치는지 확인한다.
# =========================================================================
# OA-15568 페이지 안내에 따르면 API 쪽은 2021년 이후 자료만 제공된다고
# 했는데, 이게 4개 서비스(추정매출/길단위인구/직장인구/상주인구) 모두
# 동일한지, 최신 쪽 끝은 어디까지인지 실제 데이터로 확인.

# --- 우리 쪽 위치 데이터 기간 ---
our_min, our_max = 201512, 202512
print(f"우리 위치 데이터 기간: {our_min} ~ {our_max}")

SERVICES = ["추정매출", "길단위인구", "직장인구", "상주인구"]

for name in SERVICES:
    path = f"34_{name}_raw.csv"
    try:
        df = pd.read_csv(path, usecols=["STDR_YYQU_CD", "분기코드_통일"])
    except FileNotFoundError:
        print(f"\n{name}: {path} 없음 (33번 아직 안 받았으면 건너뜀)")
        continue
    except ValueError:
        # 컬럼명이 다르면 전체를 읽어서 확인
        df = pd.read_csv(path)
        print(f"\n{name}: 예상 컬럼(STDR_YYQU_CD, 분기코드_통일)이 없음. "
              f"실제 컬럼: {list(df.columns)[:10]}...")
        continue

    api_min, api_max = df["분기코드_통일"].min(), df["분기코드_통일"].max()
    n_quarters = df["분기코드_통일"].nunique()

    overlap_min = max(our_min, api_min)
    overlap_max = min(our_max, api_max)
    overlap_ok = overlap_min <= overlap_max

    print(f"\n{name}: API 기간 {api_min} ~ {api_max} ({n_quarters}개 분기)")
    if overlap_ok:
        print(f"  겹치는 구간: {overlap_min} ~ {overlap_max}")
    else:
        print("  ⚠️ 겹치는 구간 없음 (데이터 기간이 서로 완전히 다름)")
