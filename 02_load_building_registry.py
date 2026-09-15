# -*- coding: utf-8 -*-
"""
2단계: 서울시 건축물대장 표제부 CSV를 로드한다.

- 파일 중간에 cp949로 못 읽는 바이트가 소수 섞여 있어서, strict 모드 실패 시
  encoding_errors='replace'로 재시도한다 (해당 셀 몇 개만 깨진 문자로 대체됨, 무시 가능한 수준).
- 필요한 컬럼만 남겨서 저장한다.

실행:
    python 02_load_building_registry.py

출력:
    intermediate/building_raw.parquet
"""

import os
import sys

import pandas as pd

from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")

NEED_COLS = [
    "대지위치", "시군구코드명", "법정동코드명", "대지구분코드명",
    "주지번", "부지번", "주용도코드명",
    "세대수", "지상층수", "지하층수",
]


def main():
    try:
        df = pd.read_csv(config.BUILDING_PATH, encoding="cp949", low_memory=False)
    except UnicodeDecodeError:
        print("cp949 strict 실패 -> encoding_errors='replace'로 재시도")
        df = pd.read_csv(
            config.BUILDING_PATH, encoding="cp949",
            encoding_errors="replace", low_memory=False,
        )

    print(f"건축물대장 shape: {df.shape}")

    missing = [c for c in NEED_COLS if c not in df.columns]
    if missing:
        print(f"⚠️ 다음 컬럼이 건축물대장에 없습니다: {missing}")

    df = df[[c for c in NEED_COLS if c in df.columns]].copy()

    out = config.out_path("building_raw.parquet")
    df.to_parquet(out, index=False)
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
