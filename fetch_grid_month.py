# -*- coding: utf-8 -*-
"""
fetch_grid_month.py
- VSRT Grid API에서 SKY/TMP/REH/WSD/PCP/PTY를 '지정 월' 윈도우로 수집
- 수집 구간: [YYYY-MM-01 00:00, (YYYY-MM + 1달)-01 00:00)  (KST)
- 파일명: sky_30day_YYYYMM.csv (기본 outdir에 저장)

기존 fetch_grid_30day.py의 병렬 요청/재시도/QPS 제한 로직을 재사용합니다.
"""

import argparse, os, sys, tempfile
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 재사용: fetch_grid_30day 모듈의 도우미들 (같은 폴더에 있어야 함)
try:
    import fetch_grid_30day as fg30
except Exception as e:
    print("ERROR: fetch_grid_30day.py 모듈을 import할 수 없습니다. 같은 폴더에 존재하는지 확인하세요.", file=sys.stderr)
    raise

# KST 타임존
KST = timezone(timedelta(hours=9))

def compute_month_window(ym: Optional[str], year: Optional[int], month: Optional[int]) -> Tuple[datetime, datetime, str]:
    """
    Returns (start_ts_kst, end_ts_kst, tag) for the specified month.
    tag = "YYYYMM"
    """
    if ym:
        y, m = map(int, ym.split("-"))
    else:
        if not (year and month):
            raise ValueError("--ym 또는 --year/--month를 지정하세요.")
        y, m = int(year), int(month)

    start = datetime(y, m, 1, 0, 0, tzinfo=KST)
    end = datetime(y + (m // 12), (m % 12) + 1, 1, 0, 0, tzinfo=KST)  # 다음달 1일 00:00
    tag = f"{y:04d}{m:02d}"
    return start, end, tag

def main():
    ap = argparse.ArgumentParser(description="Grid API 월별 수집 → sky_30day_YYYYMM.csv 저장")
    ap.add_argument("--auth", help="기상청 API 키 (.env의 KMA_API_KEY 대체 가능)")
    ap.add_argument("--lat", type=float, default=35.1595)
    ap.add_argument("--lon", type=float, default=126.8526)
    ap.add_argument("--ym", help="수집할 월 (YYYY-MM 형식, 예: 2025-08)")
    ap.add_argument("--year", type=int, help="연도(YYYY)")
    ap.add_argument("--month", type=int, help="월(1-12)")
    ap.add_argument("--outdir", default=r"D:\data", help="CSV 저장 폴더 (기본 D:\\data)")
    ap.add_argument("--out_csv", help="저장 파일 경로 지정(선택). 지정 시 파일명 규칙을 무시함.")
    ap.add_argument("--base", default="https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd")

    # 성능/안정성 옵션 (fetch_grid_30day와 동일)
    ap.add_argument("--workers", type=int, default=16, help="병렬 워커 수 (8~32 권장)")
    ap.add_argument("--retries", type=int, default=3, help="요청 재시도 횟수")
    ap.add_argument("--timeout", type=int, default=10, help="요청 타임아웃(초)")
    ap.add_argument("--qps", type=float, default=8.0, help="초당 요청 제한(서버 배려용)")
    args = ap.parse_args()

    # API 키
    auth_key = args.auth or os.getenv("KMA_API_KEY")
    if not auth_key:
        print("ERROR: API 키가 없습니다. --auth 또는 .env(KMA_API_KEY)를 설정하세요.", file=sys.stderr); sys.exit(1)

    # 기간 설정
    start_dt, end_dt, tag = compute_month_window(args.ym, args.year, args.month)
    print(f"[WINDOW] {start_dt.isoformat()} ~ {end_dt.isoformat()}  (KST)  tag={tag}")

    # 격자 좌표
    nx, ny = fg30.latlon_to_xy(args.lat, args.lon)
    print(f"위도 {args.lat}, 경도 {args.lon} -> 격자 ({nx},{ny})")

    # 파일 경로
    if args.out_csv:
        out_csv_path = args.out_csv
    else:
        os.makedirs(args.outdir or ".", exist_ok=True)
        out_csv_path = os.path.join(args.outdir, f"sky_30day_{tag}.csv")
    print(f"[OUT] {out_csv_path}")

    # 수집 변수/슬롯
    tmfc_slots = ["02","05","08","11","14","17","20","23"]
    wanted_vars = ["SKY","TMP","REH","WSD","PCP","PTY"]

    # 수집 대상 작업 생성
    tasks = []
    cur = start_dt
    while cur < end_dt:  # 반열림: end_dt 미포함
        dstr = cur.strftime("%Y%m%d")
        for hh in tmfc_slots:
            tmfc = f"{dstr}{hh}"
            tmfc_dt = datetime.strptime(tmfc, "%Y%m%d%H").replace(tzinfo=KST)
            for i in range(1, 6+1):  # +1h ~ +6h
                tmef_dt = tmfc_dt + timedelta(hours=i)
                if not (start_dt <= tmef_dt < end_dt):
                    continue
                tmef = tmef_dt.strftime("%Y%m%d%H")
                for v in wanted_vars:
                    tasks.append((v, tmfc, tmef))
        cur += timedelta(days=1)

    total = len(tasks)
    if total == 0:
        print("[WARN] 수집 대상이 없습니다. 빈 CSV를 생성합니다.")
        os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
        pd.DataFrame(columns=["tmfc","tmef","nx","ny"] + wanted_vars).to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 0개 데이터 저장 완료 -> {out_csv_path}")
        return

    print(f"수집 예정 요청 수: {total} (vars={len(wanted_vars)})")

    # 병렬 요청
    limiter = fg30.QPSLimiter(args.qps) if args.qps and args.qps > 0 else None
    results: Dict[Tuple[str,str], Dict[str, Optional[float]]] = {}
    success = 0; fail = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "pv-fetch/month"})
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_key = {}
            for (v, tmfc, tmef) in tasks:
                fut = ex.submit(
                    fg30.fetch_one_var, session, args.base, auth_key,
                    v, tmfc, tmef, nx, ny,
                    args.timeout, args.retries, 0.5,  # backoff_base=0.5
                    limiter
                )
                future_to_key[fut] = (tmfc, tmef, v)

            for i, fut in enumerate(as_completed(future_to_key), 1):
                tmfc, tmef, v = future_to_key[fut]
                try:
                    val = fut.result()
                    if (tmef, tmfc) not in results:
                        results[(tmef, tmfc)] = {"tmfc": tmfc, "tmef": tmef, "nx": nx, "ny": ny}
                    results[(tmef, tmfc)][v] = val
                    if val is not None:
                        success += 1
                except Exception:
                    fail += 1
                if i % 200 == 0 or i == total:
                    print(f"진행: {i}/{total}  (성공≈{success}, 실패≈{fail})")

    if not results:
        print("수집된 데이터가 없습니다. 헤더만 포함한 CSV를 생성합니다.")
        os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
        pd.DataFrame(columns=["tmfc","tmef","nx","ny"] + wanted_vars).to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 0개 데이터 저장 완료 -> {out_csv_path}")
        return

    # tmef별 최신 tmfc만 유지, 시각 정렬
    rows = list(results.values())
    df = pd.DataFrame(rows)
    df = df.sort_values(["tmef","tmfc"]).drop_duplicates(subset=["tmef"], keep="last")
    df["tmef_dt"] = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce")
    df = df.sort_values("tmef_dt").drop(columns=["tmef_dt"])

    # 안전 저장(임시파일 -> 원자적 교체)
    outdir = os.path.dirname(out_csv_path) or "."
    os.makedirs(outdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="sky_tmp_", suffix=".csv", dir=outdir)
    os.close(fd)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, out_csv_path)

    print(f"[OK] {len(df)}개 데이터 저장 완료 -> {out_csv_path}")
    print(f"[STATS] 총요청={total}, 값획득≈{success}, 실패≈{fail}, 성공률≈{(success/max(total,1))*100:.1f}%")

if __name__ == "__main__":
    main()
