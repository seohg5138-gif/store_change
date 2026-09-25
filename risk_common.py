# -*- coding: utf-8 -*-
"""
45~50번(위치 단위 위험도 분석) 공통 설정/함수.

- 17~27번과 동일하게 "코드 폴더(= 00_config.OUTPUT_FOLDER)"에서 실행하는 것을 전제로
  입력/출력 파일은 상대경로로 읽고 쓴다.
- 레포에 utils.py가 올라가 있지 않아서, 여기서 쓰는 함수는 전부 이 파일 안에서 자체 정의한다
  (utils에 의존하지 않음).
- print()에 이모지 쓰지 않음 (Windows cp949 리다이렉트 시 UnicodeEncodeError 방지).
"""

import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
try:
    config = import_module("00_config")
    BUILDING_PATH = config.BUILDING_PATH
except Exception:  # 00_config가 없는 환경(테스트 등)에서도 import는 되게
    config = None
    BUILDING_PATH = r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\서울시 건축물대장 표제부.csv"

BASE_DIR = r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest"
SUBWAY_PATH = os.path.join(BASE_DIR, "지하철역_GEOM (역사마스터).csv")
SHP_PATH = os.path.join(BASE_DIR, "area", "서울시 상권분석서비스(영역-상권).shp")

# 지하철역 CSV 컬럼명을 직접 지정하고 싶으면 여기에 (None이면 자동 탐지)
SUBWAY_LAT_COL = None
SUBWAY_LON_COL = None
SUBWAY_NAME_COL = None

# 환경변수로 덮어쓸 수 있게 (테스트/다른 PC용)
BUILDING_PATH = os.environ.get("RISK_BUILDING_PATH", BUILDING_PATH)
SUBWAY_PATH = os.environ.get("RISK_SUBWAY_PATH", SUBWAY_PATH)
SHP_PATH = os.environ.get("RISK_SHP_PATH", SHP_PATH)

# ---------------------------------------------------------------------------
# 분석 설정 (프로젝트 관례값)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MIN_OBS_QUARTERS = 15          # 신뢰가능 위치 기준 (17/24/25/27번과 동일)
MIN_LOC_PER_DISTRICT = 10      # 상권 평균이 안정적이려면 최소 위치 수 (27번 MIN_LOCATIONS_PER_CELL과 동일)
RISK_QUANTILE = 0.90           # 상권대비 교체 상위 10% = 위험 위치
TOP5_QUANTILE = 0.95           # 발표자료 히스토그램의 상위 5% 선
SURVIVAL_2Y_QUARTERS = 8       # 2년 = 8분기 (17번 MEDIAN_SURVIVAL과 동일)
# 가게 단위 2년 폐업 판정에 넣을 가게: 이 분기 "이전"에 처음 관측된 가게만 (2023년 이후 첫 관측 가게는 제외)
#  -> 2023년 전에 들어온 가게는 2년 이상 지켜볼 수 있었으므로 결과(2년 내 폐업/생존)를 확실히 앎
#  * 데이터 첫 분기부터 이미 있던 가게(좌측절단)는 개업 시점을 모르므로 계속 제외
ENTRY_CUTOFF = 202301

# "개인 교체" 지표를 무엇으로 볼지.
#  - "교체율"  : 교체횟수 / (관측분기수-1). 관측기간 길이를 통제함 (27번 상대교체도와 같은 정의) -> 기본값
#  - "교체횟수": 발표자료 1차본 방식. 관측기간이 긴 위치일수록 커지는 편향이 있음 -> 비교용
RISK_BASIS = "교체율"

UNASSIGNED_CODE = "UNASSIGNED"

# 위험등급
#  - "quantile"(기본): 위험도 점수의 순위로 나눔. 하위 10% 매우안전 / 10~30% 안전 / 30~70% 보통 /
#                      70~90% 위험 / 상위 10% 매우위험.
#    48번은 class weight를 Balanced로 줘서 점수가 "실제 확률"이 아니라, 고정 확률 경계(20/35/50/65%)로
#    자르면 등급 크기가 한쪽으로 쏠림(1차 실행: '위험' 등급에 46%). 순위 기준이면 등급 뜻이 명확함.
#  - "fixed": 발표자료 1차본의 확률 경계
GRADE_METHOD = "quantile"
GRADE_QUANTILES = [0, 0.10, 0.30, 0.70, 0.90, 1.0]
GRADE_BINS = [-0.001, 0.20, 0.35, 0.50, 0.65, 1.0]
GRADE_LABELS = ["매우안전", "안전", "보통", "위험", "매우위험"]


def assign_grade(score):
    """위험도 점수 -> 5등급 (GRADE_METHOD에 따라)."""
    score = pd.Series(score)
    if GRADE_METHOD == "quantile":
        pct = score.rank(pct=True, method="first")
        return pd.cut(pct, bins=[-0.001] + GRADE_QUANTILES[1:], labels=GRADE_LABELS).astype(str).values
    return pd.cut(score, bins=GRADE_BINS, labels=GRADE_LABELS).astype(str).values

# 모델 입력 변수 (발표자료 8개 변수). 47번에서 추가로 만드는 변수(건물연령, 지하층수 등)는
# EXTRA_FEATURES에 넣으면 48번 모델에 같이 들어간다.
CAT_FEATURES = ["층구분", "주용도", "상권유형"]
NUM_FEATURES = ["연면적", "건물내위치수", "건물지상층수", "주변상가수_200m", "지하철거리_m"]
EXTRA_FEATURES = ["층번호"]    # 47-1 PCA에서 위험과 가장 강하게 연결된 축(PC5)이 층번호라서 추가. 2층이상을 층별로 구분


# ---------------------------------------------------------------------------
# 로그 / 출력
# ---------------------------------------------------------------------------
def start_timer():
    return time.time()


def log(msg, t0=None):
    if t0 is None:
        print(f"[진행] {msg}", flush=True)
    else:
        print(f"[진행] {msg} ({time.time() - t0:.1f}초)", flush=True)


def print_table(df, title=None):
    """27번과 같은 표 출력 함수. tabulate 있으면 github 표, 없으면 to_string."""
    if title:
        print(f"\n=== {title} ===")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except Exception:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))


def save_csv(df, path, index=False):
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"[완료] 저장: {path} ({len(df):,}행)")


def set_korean_font():
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ---------------------------------------------------------------------------
# 키 정규화
# ---------------------------------------------------------------------------
def normalize_trdar(series):
    """인수인계서 규칙: TRDAR_CD dtype이 파일마다 달라서(float vs str) 반드시 정규화 후 조인.
    숫자로 못 바꾸는 값(UNASSIGNED 등)은 결측으로 돌린다."""
    out = pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)
    return out.replace({"<NA>": np.nan})


def parse_location_id(loc_ids):
    """위치ID = PNU19 + '_' + 층_최종 + '_' + 슬롯번호 (08번).
    층 토큰 안에 '_'가 들어갈 가능성까지 고려해서 앞/뒤에서 하나씩만 자른다."""
    s = loc_ids.astype(str)
    pnu = s.str.split("_", n=1).str[0]
    rest = s.str.split("_", n=1).str[1]
    floor = rest.str.rsplit("_", n=1).str[0]
    slot = rest.str.rsplit("_", n=1).str[1]
    return pd.DataFrame({"PNU19": pnu, "층_최종": floor, "슬롯번호": slot}, index=loc_ids.index)


def floor_category(floor_token):
    """06번 normalize_floor_token 결과('1','2',...,'지하1','반지하')를 4범주로.
    11번 이후 분석 대상은 '검증완료'뿐이라 '층정보없음'은 원칙적으로 안 나온다
    (나오면 데이터 흐름이 바뀐 것이니 확인 필요)."""
    if floor_token is None or (isinstance(floor_token, float) and np.isnan(floor_token)):
        return "층정보없음"
    t = str(floor_token).strip()
    if t in ("", "nan", "None"):
        return "층정보없음"
    if t.startswith("지하") or t.startswith("반지") or t.upper().startswith("B"):
        return "지하"
    num = pd.to_numeric(t, errors="coerce")
    if pd.isna(num):
        return "기타"
    if num == 1:
        return "1층"
    if num >= 2:
        return "2층이상"
    return "지하"  # 0 이하 숫자


def floor_number(floor_token):
    """층 토큰 -> 숫자 (지하1 -> -1, 반지하 -> -0.5). 숫자로 못 바꾸면 NaN."""
    t = str(floor_token).strip()
    if t.startswith("반지"):
        return -0.5
    if t.startswith("지하"):
        n = pd.to_numeric(t.replace("지하", ""), errors="coerce")
        return -n if pd.notna(n) else -1.0
    return pd.to_numeric(t, errors="coerce")


# ---------------------------------------------------------------------------
# 공간 계산 (위경도 -> haversine BallTree)
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_008.8


def to_radians(lat, lon):
    return np.deg2rad(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))


def build_balltree(lat, lon):
    from sklearn.neighbors import BallTree
    return BallTree(to_radians(lat, lon), metric="haversine")


def count_within(tree, lat, lon, radius_m):
    return tree.query_radius(to_radians(lat, lon), r=radius_m / EARTH_RADIUS_M, count_only=True)


def nearest_distance_m(tree, lat, lon):
    dist, idx = tree.query(to_radians(lat, lon), k=1)
    return dist[:, 0] * EARTH_RADIUS_M, idx[:, 0]


def read_csv_any(path, **kw):
    """cp949 -> utf-8-sig -> utf-8 순서로 시도 (공공데이터 CSV 인코딩이 제각각이라)."""
    last = None
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError as e:
            last = e
    return pd.read_csv(path, encoding="cp949", encoding_errors="replace", low_memory=False, **kw)
