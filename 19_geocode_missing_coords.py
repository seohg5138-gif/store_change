import pandas as pd
import requests
import time

# =========================================================================
# 19번: 좌표가 결측인 위치만 VWorld 지오코딩 API로 채운다.
# =========================================================================
# 18번이 만든 location_coords.csv를 읽어서, 아직 결측인 것만 채우고
# 같은 파일(location_coords.csv)을 그대로 갱신한다. 이렇게 파일명을 하나로
# 고정해두면, 20번/23번은 "18번만 돌렸는지 19번까지 돌렸는지"를 신경 쓸 필요
# 없이 항상 location_coords.csv 하나만 읽으면 된다 (조건분기 없음).
#
# VWorld 연결 전이라 아직 안 돌렸다면 건너뛰어도 된다 - 20번/23번은 18번까지만
# 돌린 상태의 location_coords.csv(결측 일부 포함)로도 정상 동작한다. 다만
# 그만큼 상권 매칭률이 19번을 돌렸을 때보다 낮게 나올 수 있다.
#
# 실행 순서: 반드시 18번 다음, 20번보다 먼저 돌릴 것.
#
# 도로명주소만 사용한다. shop_period_raw.parquet에는 지번주소 컬럼이 애초에 없고
# (08번이 분석용 최소 컬럼만 남기면서 뺐음), 위치ID 자체도 지번 "체계"(PNU19)로
# 만들어진 것이지 지번주소 "텍스트"를 사용한 게 아니라서, 지번주소를 따로 안
# 가져와도 지오코딩 정확도에 차이가 없다.

VWORLD_API_KEY = ""


def geocode_vworld(address):
    url = "https://api.vworld.kr/req/address"
    params = {
        "service": "address", "request": "getcoord", "version": "2.0",
        "crs": "epsg:4326", "address": address, "refine": "true",
        "simple": "false", "format": "json", "type": "road",
        "key": VWORLD_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data["response"]["status"] == "OK":
            point = data["response"]["result"]["point"]
            return float(point["x"]), float(point["y"])
    except Exception:
        pass
    return None, None


def main():
    coords = pd.read_csv("location_coords.csv")
    targets = coords[coords["경도"].isna()].copy()
    print(f"좌표 결측(지오코딩 대상) 수: {len(targets):,}")

    if len(targets) == 0:
        print("결측 없음 - 지오코딩 불필요.")
        return

    results = []
    for i, row in enumerate(targets.itertuples(index=False)):
        lon, lat = geocode_vworld(row.도로명주소)
        results.append(dict(위치ID=row.위치ID, 경도_지오코딩=lon, 위도_지오코딩=lat))
        time.sleep(0.1)
        if i % 200 == 0:
            print(f"  {i}/{len(targets)} 진행 중...")

    geocoded = pd.DataFrame(results)
    success = geocoded["경도_지오코딩"].notna().sum()
    print(f"\n지오코딩 성공: {success:,} / {len(geocoded):,}")

    # --- location_coords.csv를 같은 파일명으로 갱신 저장 (덮어씀) ---
    coords = coords.merge(geocoded, on="위치ID", how="left")
    coords["경도"] = coords["경도"].combine_first(coords["경도_지오코딩"])
    coords["위도"] = coords["위도"].combine_first(coords["위도_지오코딩"])
    coords = coords.drop(columns=["경도_지오코딩", "위도_지오코딩"])

    coords.to_csv("location_coords.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: location_coords.csv (남은 결측: {coords['경도'].isna().sum():,}개)")


if __name__ == "__main__":
    main()
