# -*- coding: utf-8 -*-
"""
51번: 점포(가게) 단위 테이블을 만든다. "어떤 업종인지 / 프랜차이즈인지"를 보기 위한 단계.

왜 위치 단위가 아니라 점포 단위인가:
  45~50번의 분석 단위는 "위치"(10년 동안 여러 가게가 거쳐간 자리)라서 업종·프랜차이즈를 붙일 수가 없다.
  (한 자리에 카페 -> 치킨집 -> 편의점이 거쳐갔다면 그 자리의 업종은?)
  업종과 프랜차이즈는 "가게"의 속성이므로, 가게 하나(= tenancy episode 하나)를 단위로 놓고
  "이 자리에, 이 업종으로, 프랜차이즈/개인으로 들어온 가게가 2년 안에 폐업했나"를 본다.

만드는 것:
  1) 업종 분류 통일
     상가업소 데이터는 조사 기간 중 업종 분류 체계가 개편된 적이 있어서, 옛날에 폐업한 가게와 최근 가게의
     업종 이름이 서로 다른 체계일 수 있다. 개편 시점을 데이터에서 자동으로 찾고, 개편 전후로 모두 관측된
     가게들의 "옛 중분류 -> 새 중분류" 대응(가장 흔한 대응)을 만들어 새 체계로 통일한다.
     (개편이 감지되지 않으면 원래 이름 그대로 씀)
  2) 프랜차이즈 판별 (외부 가맹 목록이 없어서 데이터 기반 규칙)
     브랜드명 = 상호명에서 지점명/괄호/공백/'OO점' 꼬리를 떼어낸 이름
     프랜차이즈 = 같은 브랜드명 가게가 서울에 FR_MIN_STORES곳 이상, FR_MIN_GU개 이상 자치구에 있고,
                  그중 지점명이 채워진 비율이 FR_MIN_BRANCH_SHARE 이상
     -> "행복부동산"처럼 흔하지만 서로 무관한 개인 가게 이름은 지점명이 거의 없어서 걸러짐
     * 공정위 가맹사업 정보공개서 브랜드 목록이 있으면 FRANCHISE_LIST_PATH에 넣어 합칠 수 있음
  3) 점포 테이블: 45번 분석대상 위치의 episode + 업종 + 프랜차이즈 + 47번 입지 변수
     2년 폐업 판정 대상:
       - 2023년 1분기(ENTRY_CUTOFF) 전에 처음 관측된 가게만 (폐업 + 영업중 모두). 2023년 이후 첫 관측은 제외
       - 좌측절단(데이터 첫 분기부터 영업 중 = 개업 시점 모름) 제외
  4) 기술통계: 업종별 / 프랜차이즈 여부별 / 업종 x 층구분 / 업종 x 프랜차이즈 2년 폐업률

입력:  shop_period_raw.parquet, tenancy_episode_corrected.parquet, 45_location_risk_base.csv, 47_location_features.csv
출력:  51_store_episodes.parquet, 51_industry_crosswalk.csv, 51_franchise_brands.csv,
       51_rate_by_industry.csv, 51_rate_by_franchise.csv, 51_rate_industry_floor.csv, 51_rate_industry_franchise.csv
"""

import os
import re

import numpy as np
import pandas as pd

import risk_common as rc

FR_MIN_STORES = 10
FR_MIN_GU = 3
FR_MIN_BRANCH_SHARE = 0.3
# 대형 브랜드 보조 규칙: 서울 전역에 널리 퍼진 브랜드는 지점명을 덜 적어도 프랜차이즈로 봄
#   (1차 실행에서 네네치킨 262곳/25개구/지점명 25%가 빠졌음. 김밥천국 6.5%처럼 독립 가게가 이름을 같이 쓰는 경우는 계속 제외)
FR_BIG_MIN_STORES = 100
FR_BIG_MIN_GU = 20
FR_BIG_MIN_BRANCH_SHARE = 0.15
# 공정위 가맹사업 정보공개서 브랜드 목록 (공공데이터포털 '공정거래위원회_가맹정보_브랜드 ...' 등에서 받은 CSV)
#   '브랜드명'이 들어간 컬럼을 자동으로 찾고, 우리 데이터에 FR_EXT_MIN_STORES곳 이상 있는 브랜드만 추가
FRANCHISE_LIST_PATH = None          # 예: r"...\공정위_브랜드목록.csv"
FR_EXT_MIN_STORES = 3
MIN_N_RATE = 300                    # 기술통계에서 표본이 이보다 작은 칸은 표시하지 않음

# 어떤 가게로 2년 폐업을 판정할지
#  "2023이전"(기본): 좌측절단 제외 + risk_common.ENTRY_CUTOFF(2023년 1분기) 전에 처음 관측된 가게 전부
#                    (폐업한 가게 + 아직 영업 중인 가게 모두. 2년 이상 지켜볼 수 있었던 가게들)
#  "폐업만"         : 좌측절단 제외 + 폐업한 가게만 (비교용. 업종/프랜차이즈 비교에 쓰면 왜곡됨)
EPISODE_SCOPE = "2023이전"

# [규칙] 폐업 판정은 상가업소번호만 근거로 한다 (코드북 14번 기각 사유).
#   14번에서 "번호는 다르지만 상호명이 같으면 같은 가게"로 병합을 시도했으나
#   네네치킨·세븐일레븐·OO공인중개사사무소 같은 프랜차이즈·관행형 이름에 병합이 몰려
#   서로 다른 사업자를 같은 점포로 오판하는 문제로 폐기됨. 그래서 같은 상호 승계도 폐업으로 센다.


# ---------------------------------------------------------------------------
# 1) 업종 분류 통일
# ---------------------------------------------------------------------------
def harmonize_industry(raw):
    """raw: 상가업소번호, period, 대분류, 중분류 (1,900만 행 규모라 먼저 가게x업종 단위로 압축해서 처리)
    반환: 가게별 최종 (업종대분류, 업종중분류), 대응표, 개편분기"""
    sets = raw.groupby("period", observed=True)["대분류"].agg(lambda s: frozenset(map(str, s.dropna().unique())))
    periods = list(sets.index)
    print("\n[분기별 업종 대분류 이름 수]")
    print(sets.apply(len).to_string())

    revision = None
    for prev, cur in zip(periods[:-1], periods[1:]):
        if len(sets[cur] ^ sets[prev]) >= 3:    # 대분류 이름이 3개 이상 바뀌면 체계 개편으로 봄
            revision = cur
            break

    comp = (raw.groupby(["상가업소번호", "대분류", "중분류"], observed=True)["period"]
            .agg(첫분기="min",끝분기="max").reset_index())
    comp["대분류"] = comp["대분류"].astype(str)
    comp["중분류"] = comp["중분류"].astype(str)
    latest = comp.sort_values("끝분기").drop_duplicates("상가업소번호", keep="last")

    if revision is None:
        print("  업종 분류 체계 개편이 감지되지 않았습니다 -> 원래 이름 사용")
        out = latest[["상가업소번호", "대분류", "중분류"]].rename(
            columns={"대분류": "업종대분류", "중분류": "업종중분류"})
        return out, pd.DataFrame(), None

    prev = periods[periods.index(revision) - 1]
    print(f"  업종 분류 개편 감지: {revision} 분기부터")
    print(f"    사라진 대분류: {sorted(sets[prev] - sets[revision])}")
    print(f"    새 대분류    : {sorted(sets[revision] - sets[prev])}")

    old_rec = comp[comp["첫분기"] < revision].sort_values("끝분기").drop_duplicates("상가업소번호", keep="last")
    new_rec = comp[comp["끝분기"] >= revision].sort_values("첫분기").drop_duplicates("상가업소번호", keep="first")
    both = old_rec.merge(new_rec, on="상가업소번호", suffixes=("_old", "_new"))
    print(f"  개편 전후 모두 관측된 가게: {len(both):,}곳 (대응표를 만드는 표본)")

    cw = (both.groupby(["중분류_old", "중분류_new"]).size().rename("가게수").reset_index()
          .sort_values(["중분류_old", "가게수"], ascending=[True, False]))
    cw["비중"] = cw["가게수"] / cw.groupby("중분류_old")["가게수"].transform("sum")
    best = cw.drop_duplicates("중분류_old").rename(columns={"중분류_old": "옛중분류", "중분류_new": "새중분류"})
    weak = best[best["비중"] < 0.5]
    print(f"  옛 중분류 {len(best)}개 대응 완료, 그중 대응 비중 50% 미만(애매) {len(weak)}개")
    if len(weak):
        print(weak.head(10).to_string(index=False))

    new_side = comp[comp["끝분기"] >= revision]
    new_mid2major = new_side.groupby("중분류")["대분류"].agg(lambda s: s.value_counts().idxmax())

    # 가게별 최종 업종: 개편 후에도 관측된 가게는 개편 후 마지막 값, 개편 전에만 있던 가게는 대응표로 변환
    last_new = new_side.sort_values("끝분기").drop_duplicates("상가업소번호", keep="last")[["상가업소번호", "중분류"]]
    only_old = latest[~latest["상가업소번호"].isin(last_new["상가업소번호"])][["상가업소번호", "중분류"]].copy()
    only_old["중분류"] = only_old["중분류"].map(best.set_index("옛중분류")["새중분류"]).fillna(only_old["중분류"])
    final = pd.concat([last_new, only_old], ignore_index=True)
    final["업종대분류"] = final["중분류"].map(new_mid2major)
    final = final.rename(columns={"중분류": "업종중분류"})
    print(f"  최종 업종: {final['업종중분류'].nunique()}개 중분류 / 대분류 매핑 실패 "
          f"{final['업종대분류'].isna().mean()*100:.2f}%")
    final["업종대분류"] = final["업종대분류"].fillna("기타")
    return final, best, revision


# ---------------------------------------------------------------------------
# 2) 프랜차이즈
# ---------------------------------------------------------------------------
_PAREN = re.compile(r"[\(\[].*?[\)\]]")
_TAIL = re.compile(r"\s+\S*점$")


def brand_key(name, branch):
    s = str(name) if pd.notna(name) else ""
    if pd.notna(branch) and str(branch).strip():
        s = s.replace(str(branch), " ")
    s = _PAREN.sub(" ", s)
    s = _TAIL.sub("", s.strip())
    s = re.sub(r"[\s\-_·.,&']+", "", s).lower()
    return s


def flag_franchise(stores):
    """stores: 상가업소번호, 상호명, 지점명, 시군구명 (가게당 1행)"""
    stores = stores.copy()
    stores["브랜드"] = [brand_key(a, b) for a, b in zip(stores["상호명"], stores["지점명"])]
    stores["지점명있음"] = stores["지점명"].notna() & (stores["지점명"].astype(str).str.strip() != "")
    print(f"\n[프랜차이즈 판별] 지점명이 채워진 가게 비율: {stores['지점명있음'].mean()*100:.1f}%")

    b = stores[stores["브랜드"].str.len() >= 2].groupby("브랜드", observed=True).agg(
        가게수=("상가업소번호", "size"), 자치구수=("시군구명", "nunique"), 지점명비율=("지점명있음", "mean"),
        예시상호=("상호명", "first"))
    rule1 = ((b["가게수"] >= FR_MIN_STORES) & (b["자치구수"] >= FR_MIN_GU)
             & (b["지점명비율"] >= FR_MIN_BRANCH_SHARE))
    rule2 = ((b["가게수"] >= FR_BIG_MIN_STORES) & (b["자치구수"] >= FR_BIG_MIN_GU)
             & (b["지점명비율"] >= FR_BIG_MIN_BRANCH_SHARE))
    b["프랜차이즈"] = rule1 | rule2
    b["판별근거"] = np.where(rule1, "지점명규칙", np.where(rule2, "대형브랜드규칙", ""))
    print(f"  대형브랜드 규칙으로 추가: {(rule2 & ~rule1).sum():,}개 브랜드 "
          f"{b.loc[rule2 & ~rule1].sort_values('가게수', ascending=False)['예시상호'].head(10).tolist()}")

    if FRANCHISE_LIST_PATH and os.path.exists(FRANCHISE_LIST_PATH):
        ext_df = rc.read_csv_any(FRANCHISE_LIST_PATH)
        col = next((c for c in ext_df.columns if "브랜드명" in c), ext_df.columns[0])
        ext = set(ext_df[col].dropna().map(lambda x: brand_key(x, None)))
        added = b.index.isin(ext) & ~b["프랜차이즈"] & (b["가게수"] >= FR_EXT_MIN_STORES)
        b.loc[added, "프랜차이즈"] = True
        b.loc[added, "판별근거"] = "공정위목록"
        print(f"  공정위 목록({col}, {len(ext):,}개 브랜드)으로 추가: {added.sum():,}개 브랜드")

    fr = b[b["프랜차이즈"]].sort_values("가게수", ascending=False)
    print(f"  프랜차이즈 브랜드 {len(fr):,}개, 해당 가게 {fr['가게수'].sum():,}곳")
    print("  [가게수 상위 브랜드]")
    print(fr.head(20)[["예시상호", "가게수", "자치구수", "지점명비율"]].to_string())
    miss = b[(~b["프랜차이즈"]) & (b["가게수"] >= 30)].sort_values("가게수", ascending=False)
    print("  [흔하지만 프랜차이즈로 안 본 이름 (확인용, 상위 15)]")
    print(miss.head(15)[["예시상호", "가게수", "자치구수", "지점명비율"]].to_string())

    stores["프랜차이즈"] = stores["브랜드"].map(b["프랜차이즈"]).fillna(False).astype(int)
    return stores[["상가업소번호", "브랜드", "프랜차이즈"]], b.reset_index()


# ---------------------------------------------------------------------------
def rate_table(df, keys, min_n=MIN_N_RATE):
    g = df.groupby(keys, observed=True)["2년내폐업"].agg(가게수="size", 폐업률="mean").reset_index()
    return g[g["가게수"] >= min_n]


def main():
    t0 = rc.start_timer()
    cols = ["상가업소번호", "period", "상호명", "지점명", "시군구명", "상권업종대분류명", "상권업종중분류명"]
    raw = pd.read_parquet("shop_period_raw.parquet", columns=cols)
    raw = raw.rename(columns={"상권업종대분류명": "대분류", "상권업종중분류명": "중분류"})
    for c in ["대분류", "중분류", "시군구명"]:
        raw[c] = raw[c].astype("category")
    rc.log(f"shop_period_raw {len(raw):,}행, 가게 {raw['상가업소번호'].nunique():,}곳", t0)

    # 1) 업종
    ind, crosswalk, revision = harmonize_industry(raw[["상가업소번호", "period", "대분류", "중분류"]])
    if len(crosswalk):
        rc.save_csv(crosswalk, "51_industry_crosswalk.csv")
    rc.log("업종 통일 완료", t0)

    # 2) 프랜차이즈
    stores = (raw.sort_values("period").drop_duplicates("상가업소번호", keep="last")
              [["상가업소번호", "상호명", "지점명", "시군구명"]])
    fr, brands = flag_franchise(stores)
    rc.save_csv(brands[brands["가게수"] >= FR_MIN_STORES].sort_values("가게수", ascending=False),
                "51_franchise_brands.csv")
    del raw
    rc.log("프랜차이즈 판별 완료", t0)

    # 3) 점포 테이블
    base = pd.read_csv("45_location_risk_base.csv", usecols=["위치ID", "TRDAR_CD", "위험여부"],
                       dtype={"TRDAR_CD": str})
    feat = pd.read_csv("47_location_features.csv", low_memory=False)
    ep = pd.read_parquet("tenancy_episode_corrected.parquet",
                         columns=["위치ID", "episode_id", "상가업소번호", "시작분기", "종료분기",
                                  "생존분기수_보정", "생존여부"])
    grid = sorted(set(ep["시작분기"]) | set(ep["종료분기"]))
    pos = {p: i for i, p in enumerate(grid)}
    ep = ep[ep["위치ID"].isin(set(base["위치ID"]))].copy()
    ep = ep.sort_values(["위치ID", "시작분기"])
    ep["이전입점수"] = ep.groupby("위치ID").cumcount()           # 이 가게 전에 이 자리를 거쳐간 가게 수
    ep["좌측절단"] = ep["시작분기"] == grid[0]
    ep["2년관측가능"] = ep["시작분기"].map(pos) <= len(grid) - rc.SURVIVAL_2Y_QUARTERS
    ep["2년내폐업"] = ((ep["생존여부"] == "폐업") & (ep["생존분기수_보정"] < rc.SURVIVAL_2Y_QUARTERS)).astype(int)
    ep["입점연도"] = (ep["시작분기"] // 100).astype(int)
    if EPISODE_SCOPE == "폐업만":
        ep["분석대상"] = (~ep["좌측절단"]) & (ep["생존여부"] == "폐업")
    else:
        ep["분석대상"] = (~ep["좌측절단"]) & (ep["시작분기"] < rc.ENTRY_CUTOFF)
    n_all = len(ep)
    print(f"  전체 episode {n_all:,} / 좌측절단 제외 {int(ep['좌측절단'].sum()):,} / "
          f"{rc.ENTRY_CUTOFF} 이후 첫 관측 제외 {int((~ep['좌측절단'] & (ep['시작분기'] >= rc.ENTRY_CUTOFF)).sum()):,}")
    print(f"\n[2년 폐업 판정 대상 기준: {EPISODE_SCOPE}]")

    ep = (ep.merge(ind, on="상가업소번호", how="left")
            .merge(fr[["상가업소번호", "프랜차이즈", "브랜드"]], on="상가업소번호", how="left")
            .merge(base, on="위치ID", how="left")
            .merge(feat, on="위치ID", how="left"))
    ep["프랜차이즈"] = ep["프랜차이즈"].fillna(0).astype(int)

    ep["업종대분류"] = ep["업종대분류"].fillna("정보없음")
    ep["업종중분류"] = ep["업종중분류"].fillna("정보없음")
    ep.to_parquet("51_store_episodes.parquet", index=False)
    a = ep[ep["분석대상"]]
    print(f"\n[점포 테이블] episode {len(ep):,}개 중 2년 판정 가능 {len(a):,}개, "
          f"2년 내 폐업률 {a['2년내폐업'].mean()*100:.1f}%, 프랜차이즈 비율 {a['프랜차이즈'].mean()*100:.1f}%")
    print("[완료] 저장: 51_store_episodes.parquet")

    # 4) 기술통계
    by_ind = rate_table(a, ["업종대분류", "업종중분류"]).sort_values("폐업률", ascending=False)
    rc.print_table(by_ind.head(15), "2년 내 폐업률 높은 업종(중분류)")
    rc.print_table(by_ind.tail(10), "2년 내 폐업률 낮은 업종(중분류)")
    rc.save_csv(by_ind, "51_rate_by_industry.csv")

    by_fr = rate_table(a, ["프랜차이즈"])
    by_fr["구분"] = by_fr["프랜차이즈"].map({1: "프랜차이즈", 0: "개인"})
    rc.print_table(by_fr[["구분", "가게수", "폐업률"]], "프랜차이즈 vs 개인 (전체)")
    print("  * 전체 비교는 업종 구성 차이가 섞여 있음 -> 아래 업종대분류 x 프랜차이즈 표로 볼 것")
    rc.save_csv(by_fr, "51_rate_by_franchise.csv")

    ind_fr = rate_table(a, ["업종대분류", "프랜차이즈"], min_n=100).pivot(
        index="업종대분류", columns="프랜차이즈", values="폐업률").rename(columns={0: "개인", 1: "프랜차이즈"})
    print("\n[업종대분류 x 프랜차이즈 2년 내 폐업률 (%)]")
    print((ind_fr * 100).round(1).to_string())
    rc.save_csv(rate_table(a, ["업종대분류", "업종중분류", "프랜차이즈"], min_n=100), "51_rate_industry_franchise.csv")

    ind_floor = rate_table(a, ["업종대분류", "층구분"], min_n=100).pivot(
        index="업종대분류", columns="층구분", values="폐업률")
    print("\n[업종대분류 x 층구분 2년 내 폐업률 (%)]")
    print((ind_floor * 100).round(1).to_string())
    rc.save_csv(rate_table(a, ["업종대분류", "업종중분류", "층구분"], min_n=100), "51_rate_industry_floor.csv")

    print("\n  * 프랜차이즈 효과는 '인과'가 아니라 '관련'임: 프랜차이즈 본사가 좋은 자리를 골라 들어가는 효과가 섞여 있음."
          " 입지를 통제한 비교는 52번 모델에서 봄")
    rc.log("51번 완료", t0)


if __name__ == "__main__":
    main()
