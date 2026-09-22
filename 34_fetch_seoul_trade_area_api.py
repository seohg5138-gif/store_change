import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

# =========================================================================
# 33번: 서울시 상권분석서비스 Open API 4종을 끝까지 페이지네이션해서 긁어온다.
# =========================================================================
# 이 스크립트는 openapi.seoul.go.kr에 접근해야 하므로 로컬 환경에서 실행해야
# 함(샌드박스에서는 이 도메인 접근이 막혀 있어 내가 직접 실행할 수 없음).
#
# 분기 포맷 주의: 이 API의 STDR_YYQU_CD는 5자리(예: 20222 = 2022년 2분기)인데,
# 우리 내부 파이프라인(24~30번)의 분기 컬럼은 6자리(예: 202206 = 2022년+월,
# 월은 03/06/09/12)임. convert_quarter()로 변환해서 저장해두면 나중에 조인이
# 훨씬 쉬워짐.

API_KEY = "444843556373656f3131386c5472474d"  # sample 아님, 실제 발급받은 키로 교체
PAGE_SIZE = 1000
SLEEP_SEC = 0.2  # API 서버 부담을 줄이기 위한 호출 간 대기시간

SERVICES = {
    "추정매출": "VwsmTrdarSelngQq",
    #"길단위인구": "VwsmTrdarFlpopQq",
    # "직장인구": "VwsmTrdarWrcPopltnQq",
    # "상주인구": "VwsmTrdarRepopQq",
}


def convert_quarter(yyqu5):
    """API의 5자리 분기코드(YYYYQ)를 내부 6자리 분기코드(YYYYMM, MM은
    03/06/09/12)로 변환. 예: 20222 -> 202206"""
    yyqu5 = str(yyqu5)
    year, q = yyqu5[:4], yyqu5[4]
    month = {"1": "03", "2": "06", "3": "09", "4": "12"}.get(q)
    if month is None:
        return None
    return int(year + month)


def fetch_service(service_name, service_code, max_pages=None):
    """서비스 하나를 끝까지 페이지네이션해서 전체 row를 리스트로 반환."""
    all_rows = []
    start = 1
    total = None
    page_num = 0
    while True:
        end = start + PAGE_SIZE - 1
        url = f"http://openapi.seoul.go.kr:8088/{API_KEY}/xml/{service_code}/{start}/{end}/"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ⚠️ 요청 실패 ({start}-{end}): {e} - 이 페이지 건너뛰고 계속")
            start += PAGE_SIZE
            page_num += 1
            if max_pages and page_num >= max_pages:
                break
            continue

        root = ET.fromstring(resp.content)

        # 에러 응답 체크 (예: INFO-200 = 해당 데이터 없음, 필수값 누락 등)
        code_el = root.find("./RESULT/CODE")
        if code_el is not None and not code_el.text.startswith("INFO-000"):
            msg_el = root.find("./RESULT/MESSAGE")
            print(f"  ⚠️ API 응답 코드 {code_el.text}: "
                  f"{msg_el.text if msg_el is not None else ''} (start={start})")
            break

        if total is None:
            total_el = root.find("list_total_count")
            total = int(total_el.text) if total_el is not None else 0
            print(f"  {service_name}: 전체 {total:,}행, "
                  f"{-(-total // PAGE_SIZE):,}페이지 예정")

        rows = root.findall("row")
        if not rows:
            # 총 개수(total)에 아직 못 미쳤는데 빈 응답이 왔다면, 데이터가
            # 끝난 게 아니라 일시적 오류일 가능성이 큼 -> 조용히 멈추지 않고
            # 몇 번 재시도한 뒤에도 안 되면 "이 페이지만" 건너뛰고 계속 진행.
            if total is not None and start <= total:
                retry_ok = False
                for retry_i in range(3):
                    print(f"  ⚠️ {start}-{end} 빈 응답(재시도 {retry_i + 1}/3)")
                    time.sleep(1.0)
                    try:
                        resp = requests.get(url, timeout=30)
                        resp.raise_for_status()
                        root = ET.fromstring(resp.content)
                        rows = root.findall("row")
                    except (requests.RequestException, ET.ParseError):
                        rows = []
                    if rows:
                        retry_ok = True
                        break
                if not retry_ok:
                    print(f"  ⚠️ {start}-{end} 재시도 3회 실패 - 이 페이지 건너뛰고 계속 진행 "
                          f"(전체 {total:,}행 중 일부 누락됨, 나중에 이 구간만 다시 받아야 함)")
                    start += PAGE_SIZE
                    page_num += 1
                    if start > total:
                        break
                    continue
            else:
                break

        for row in rows:
            all_rows.append({child.tag: child.text for child in row})

        print(f"    {start}-{min(end, total)} 완료 ({len(all_rows):,}/{total:,})")

        start += PAGE_SIZE
        page_num += 1
        if start > total:
            break
        if max_pages and page_num >= max_pages:
            print(f"  (max_pages={max_pages} 도달, 중단)")
            break
        time.sleep(SLEEP_SEC)

    return all_rows


if __name__ == "__main__":
    if API_KEY == "여기에_본인_인증키_입력":
        raise SystemExit("API_KEY를 실제 인증키로 바꾼 뒤 다시 실행해줘.")

    for name, code in SERVICES.items():
        print(f"\n=== {name} ({code}) 수집 시작 ===")
        # 추정매출은 482페이지라 오래 걸림 - 먼저 인구 3개부터 받고 싶으면
        # 이 줄의 max_pages를 조정하거나, 이 서비스만 따로 나중에 돌리면 됨.
        rows = fetch_service(name, code)

        if not rows:
            print(f"  {name}: 수집된 행 없음, 스킵")
            continue

        df = pd.DataFrame(rows)
        if "STDR_YYQU_CD" in df.columns:
            df["분기코드_통일"] = df["STDR_YYQU_CD"].apply(convert_quarter)

        out_path = f"34_{name}_raw.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  저장 완료: {out_path} ({len(df):,}행, {len(df.columns)}컬럼)")
