# -*- coding: utf-8 -*-
"""
공통 유틸 함수 모음.
- build_pnu19_vectorized: 벡터화된 PNU19 생성 (astype 등 dtype 이슈 없이 안전하게)
- normalize_floor_token: 층정보 표기 통일 (01->1, B1/지->지하1, 반/반지층->반지하)
- split_floor_unit: 3자리 이상 숫자를 총층수 기준으로 층/호로 동적 분할
"""

import re
import pandas as pd


def build_pnu19_vectorized(법정동코드, 대지구분코드, 지번본번지, 지번부번지):
    """
    네 개의 pandas Series를 받아서 PNU19(19자리 문자열) Series를 반환.
    반드시 fillna 후 정수로 캐스팅해서 소수점(.0)이 안 붙게 처리한다.
    (과거 MemoryError/매칭 0% 원인이었던 dtype 문제를 여기서 한 번에 막음)
    """
    def _clean_int_str(s, width, default=0):
        s = pd.to_numeric(s, errors="coerce").fillna(default).astype("int64")
        return s.astype(str).str.zfill(width)

    bjd = _clean_int_str(법정동코드, 10)
    daeji = _clean_int_str(대지구분코드, 1, default=1)
    bonbeon = _clean_int_str(지번본번지, 4)
    bubeon = _clean_int_str(지번부번지, 4)

    return bjd + daeji + bonbeon + bubeon


FLOOR_ALIASES = {
    "지": "지하1",
    "지1": "지하1",
    "지하": "지하1",
    "B1": "지하1",
    "B2": "지하2",
    "B3": "지하3",
    "B4": "지하4",
    "반": "반지하",
    "반지층": "반지하",
    "반지하": "반지하",
}


def normalize_floor_token(floor):
    """층정보 원본 문자열 하나를 표준 표기로 정규화."""
    if pd.isna(floor):
        return "nan"
    f = str(floor).strip()
    if f == "" or f.lower() == "nan":
        return "nan"

    # 01 -> 1 처럼 앞자리 0 제거 (단, "0"만 있는 경우는 그대로 둠)
    if re.fullmatch(r"0+\d+", f):
        f = re.sub(r"^0+", "", f)

    if f in FLOOR_ALIASES:
        return FLOOR_ALIASES[f]

    m = re.fullmatch(r"B(\d+)", f)
    if m:
        return f"지하{m.group(1)}"

    return f


def split_floor_unit(code, total_floors):
    """
    층정보가 3자리 이상 숫자(예: '106', '1204')일 때, 건물의 실제 총층수를 기준으로
    앞부분이 총층수를 넘지 않는 가장 긴 자릿수를 층으로 판단해서 (층, 호)로 분할.
    총층수를 모르거나 숫자가 아니면 원본 그대로 반환하고 호는 None.
    """
    code = str(code)
    if not code.isdigit():
        return code, None
    if total_floors is None or pd.isna(total_floors) or total_floors <= 0:
        return code, None

    total_floors = int(total_floors)
    for floor_digits in range(len(code) - 1, 0, -1):
        floor_part = int(code[:floor_digits])
        if 1 <= floor_part <= total_floors:
            return str(floor_part), code[floor_digits:]
    return code[0], code[1:]


# ---------------------------------------------------------------------------
# 진행 상황 로깅 헬퍼
# 무거운 CSV/parquet 처리 중간에 아무 출력이 없으면 멈춘 건지 알 수 없어서,
# 각 스크립트가 "지금 이 단계 시작함" / "N초 걸려서 끝남"을 찍을 때 쓰는 함수.
# 사용법:
#   t0 = utils.start_timer()
#   utils.log("df_all 로딩 시작...")
#   df = pd.read_parquet(...)
#   utils.log("df_all 로딩 완료", t0)
# ---------------------------------------------------------------------------
import time


def start_timer():
    return time.time()


def log(msg, t0=None):
    if t0 is not None:
        print(f"  [{time.time() - t0:.1f}초 경과] {msg}")
    else:
        print(f"[시작] {msg}")
