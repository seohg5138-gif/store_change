# -*- coding: utf-8 -*-
"""
공통 설정 파일. 경로만 본인 환경에 맞게 수정하면 됩니다.
다른 스크립트들은 이 파일을 import해서 씁니다.
"""

# 원본 40개 CSV가 들어있는 폴더
DATA_FOLDER = r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\store_name_1525\data"

# 건축물대장 표제부 CSV 경로
BUILDING_PATH = r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\서울시 건축물대장 표제부.csv"

# 중간 산출물(parquet)을 저장할 폴더 (없으면 자동 생성됨)
OUTPUT_FOLDER = r"C:\Users\seohg\OneDrive\바탕 화면\2026\seoul datalob contest\store_name_1525\code"

# 원본 상가 데이터에서 실제로 사용할 컬럼만 지정 (메모리 절약 핵심)
USE_COLS = [
    "상가업소번호", "상호명", "지점명",
    "상권업종대분류명", "상권업종중분류명", "상권업종소분류명",
    "시군구명", "법정동명", "법정동코드",
    "대지구분코드", "지번본번지", "지번부번지", "지번주소",
    "건물관리번호", "건물명", "도로명주소",
    "동정보", "층정보", "호정보",
    "경도", "위도",
]

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def out_path(filename: str) -> str:
    """OUTPUT_FOLDER 밑에 파일 경로를 만들어주는 헬퍼."""
    return os.path.join(OUTPUT_FOLDER, filename)
