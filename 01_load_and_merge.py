# -*- coding: utf-8 -*-
"""
1단계: 40개 원본 상가 CSV를 읽어서 하나로 합친다.

핵심 원칙:
- 필요한 컬럼만 읽는다 (usecols) -> 메모리 3~5배 절약
- period(YYYYMM)를 파일명에서 추출해서 컬럼으로 남긴다
- 인코딩은 cp949를 우선 시도하고 실패하면 utf-8로 재시도
- 파일 간 컬럼 구성이 다른지 반드시 체크하고 로그로 남긴다

실행:
    python 01_load_and_merge.py

출력:
    intermediate/df_all.parquet
"""

import glob
import os
import re

import pandas as pd

from importlib import import_module
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
config = import_module("00_config")

# 이 컬럼들은 25자리짜리 큰 숫자 문자열(건물관리번호 등)이라, pandas가 자동으로
# 정수 dtype으로 잘못 추론하면 int64 범위를 넘어서서 parquet 저장 시
# "Python int too large to convert to C long" 에러가 납니다.
# 반드시 처음부터 문자열로 강제 지정해서 읽어야 합니다.
FORCE_STR_COLS = {
    "상가업소번호": str, "건물관리번호": str, "지번코드": str,
    "법정동코드": str, "대지구분코드": str, "구우편번호": str, "신우편번호": str,
    # 층정보/호정보/동정보는 파일에 따라 순수 숫자만 있으면 pandas가 float으로
    # 잘못 추론해서 "01"의 앞자리 0이 날아가거나("01"->1.0), 40개 파일을 합칠 때
    # dtype이 섞이는 문제가 생김. 반드시 문자열로 고정.
    "층정보": str, "호정보": str, "동정보": str,
}


def read_one(path: str) -> pd.DataFrame:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"\d{6}", name)
    if not m:
        raise ValueError(f"파일명에서 period(YYYYMM)를 못 찾았습니다: {path}")
    period = int(m.group())

    dtype = {k: v for k, v in FORCE_STR_COLS.items() if k in config.USE_COLS}

    try:
        df = pd.read_csv(
            path, low_memory=False, encoding="cp949",
            usecols=lambda c: c in config.USE_COLS,
            dtype=dtype,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path, low_memory=False, encoding="utf-8",
            usecols=lambda c: c in config.USE_COLS,
            dtype=dtype,
        )

    df["period"] = period
    return df


def main():
    files = sorted(glob.glob(os.path.join(config.DATA_FOLDER, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV 파일을 못 찾았습니다: {config.DATA_FOLDER}")
    print(f"총 파일 수: {len(files)}")

    dfs = []
    col_sets = []
    for f in files:
        df = read_one(f)
        dfs.append(df)
        col_sets.append(frozenset(df.columns))
        print(f"  {os.path.basename(f)}: {len(df):,}행 로드 완료")

    base = col_sets[0]
    mismatches = [
        (files[i], base.symmetric_difference(c))
        for i, c in enumerate(col_sets)
        if c != base
    ]
    if mismatches:
        print("\n⚠️ 컬럼이 다른 파일이 있습니다:")
        for fname, diff in mismatches:
            print(f"  {os.path.basename(fname)}: {diff}")
    else:
        print("\n✅ 모든 파일 컬럼 동일")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n최종 합친 행수: {len(df_all):,}")
    print(f"메모리 사용량: {df_all.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    out = config.out_path("df_all.parquet")
    df_all.to_parquet(out, index=False)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
