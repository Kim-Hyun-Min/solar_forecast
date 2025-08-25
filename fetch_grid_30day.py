# -*- coding: utf-8 -*-
r"""
fetch_grid_day.py  (빠른 병렬 수집 버전)
- VSRT Grid API에서 SKY/TMP/REH/WSD/PCP/PTY를 시점(+1~+6h)별로 병렬 수집
- 하루 8개 발표시각 × 6시간 × 6변수 = 288요청/일 (멀티팩터)
- 기능:
  * ThreadPoolExecutor 병렬 요청 (--workers)
  * 재시도/지수백오프 (--retries)
  * rate limit (--qps) 간단 제어
  * 임시파일 저장 후 os.replace로 원자적 덮어쓰기
  * 격자 크기 안전모드(불일치시 에러)
사용 예:
  python fetch_grid_day.py --days 30 --include-today --out_csv C:\Users\admin\Desktop\sky_30d.csv --workers 16
"""

import argparse, os, sys, math, time, threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

KST = timezone(timedelta(hours=9))

# -----------------------
# 좌표 변환 (위경도 -> 격자 nx,ny)
# -----------------------
def latlon_to_xy(lat, lon):
    RE=6371.00877; GRID=5.0; SLAT1=30.0; SLAT2=60.0; OLON=126.0; OLAT=38.0; XO=43; YO=136
    rad=math.pi/180.0
    re=RE/GRID
    sl1,sl2=SLAT1*rad,SLAT2*rad; olon=OLON*rad; olat=OLAT*rad
    sn = math.log(math.cos(sl1)/math.cos(sl2)) / math.log(math.tan(math.pi*0.25+sl2*0.5)/math.tan(math.pi*0.25+sl1*0.5))
    sf = (math.tan(math.pi*0.25+sl1*0.5)**sn) * (math.cos(sl1)/sn)
    ro = (re*sf) / (math.tan(math.pi*0.25+olat*0.5)**sn)
    ra = (re*sf) / (math.tan(math.pi*0.25+lat*rad*0.5)**sn)
    theta = (lon*rad-olon)*sn
    x = ra*math.sin(theta)+XO; y=ro-ra*math.cos(theta)+YO
    return int(round(x)), int(round(y))

# -----------------------
# 격자 형태 (안전모드)
# -----------------------
def infer_grid_shape_strict(count: int, nx_expected=149, ny_expected=253) -> Tuple[int,int]:
    if count == nx_expected * ny_expected:
        return nx_expected, ny_expected
    # 일부 API가 행우선/열우선 상관없이 동일 곱을 주는 경우만 허용
    if count == ny_expected * nx_expected:
        return nx_expected, ny_expected
    raise ValueError(f"[ERROR] Unexpected grid size: got {count} values, expected {nx_expected}x{ny_expected}.")

def tokenize_numbers(text: str) -> List[float]:
    toks=[]
    for ln in text.splitlines():
        ln=ln.strip()
        if not ln or ln.startswith("#"): continue
        ln=ln.replace(","," ").replace("\t"," ")
        for t in ln.split():
            try: toks.append(float(t))
            except: pass
    return toks

def extract_cell_value(tokens, nx, ny, cell_nx, cell_ny):
    x0 = int(cell_nx)-1; y0 = int(cell_ny)-1
    if not (0<=x0<nx and 0<=y0<ny): return None
    idx = y0*nx + x0  # 행우선
    if not (0<=idx<len(tokens)): return None
    return float(tokens[idx])

# -----------------------
# 발표 시각 슬롯
# -----------------------
def latest_tmfc_slot_hh(now_kst: datetime) -> Optional[str]:
    slots = ["02","05","08","11","14","17","20","23"]
    hh = now_kst.strftime("%H")
    past = [s for s in slots if s <= hh]
    return past[-1] if past else None

# -----------------------
# 간단 rate limit (QPS)
# -----------------------
class QPSLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.1, qps)
        self.lock = threading.Lock()
        self.next_time = time.perf_counter()

    def wait(self):
        with self.lock:
            now = time.perf_counter()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            # 다음 허용 시각
            self.next_time = max(self.next_time + 1.0/self.qps, time.perf_counter())

# -----------------------
# 요청 함수(재시도/백오프)
# -----------------------
def fetch_one_var(session: requests.Session, base_url: str, auth_key: str,
                  var_name: str, tmfc: str, tmef: str, nx: int, ny: int,
                  timeout: int, retries: int, backoff_base: float,
                  limiter: Optional[QPSLimiter]) -> Optional[float]:
    params = {"authKey": auth_key, "tmfc": tmfc, "tmef": tmef, "vars": var_name}
    for attempt in range(retries+1):
        try:
            if limiter: limiter.wait()
            r = session.get(base_url, params=params, timeout=timeout)
            r.raise_for_status()
            tokens = tokenize_numbers(r.text)
            if not tokens:
                return None
            gx, gy = infer_grid_shape_strict(len(tokens))
            val = extract_cell_value(tokens, gx, gy, nx, ny)
            if val is None:
                return None
            # -99, -999 등 결측 / PCP=0은 유효
            if int(round(val)) in (-99, -999) and var_name != "PCP":
                return None
            return float(val)
        except Exception as e:
            if attempt < retries:
                sleep_s = backoff_base * (2 ** attempt) * (1 + 0.2*attempt)
                time.sleep(sleep_s)
                continue
            return None

# -----------------------
# 메인
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", help="기상청 API 키 (.env의 KMA_API_KEY 대체 가능)")
    ap.add_argument("--lat", type=float, default=35.1595)
    ap.add_argument("--lon", type=float, default=126.8526)
    ap.add_argument("--days", type=int, default=1, help="어제부터 N일(오늘 제외). 기본 1")
    ap.add_argument("--include-today", action="store_true", help="오늘 00시~현재 이전 슬롯 포함")
    ap.add_argument("--out_csv", default="D:\data\sky_30day.csv")
    ap.add_argument("--base", default="https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_vsrt_grd")
    # 성능/안정성 옵션
    ap.add_argument("--workers", type=int, default=16, help="병렬 워커 수 (8~32 권장)")
    ap.add_argument("--retries", type=int, default=3, help="요청 재시도 횟수")
    ap.add_argument("--timeout", type=int, default=10, help="요청 타임아웃(초)")
    ap.add_argument("--qps", type=float, default=8.0, help="초당 요청 제한(서버 배려용)")
    args = ap.parse_args()

    load_dotenv()
    auth_key = args.auth or os.getenv("KMA_API_KEY")
    if not auth_key:
        print("ERROR: API 키가 없습니다. --auth 또는 .env(KMA_API_KEY)를 설정하세요.", file=sys.stderr); sys.exit(1)

    nx, ny = latlon_to_xy(args.lat, args.lon)
    print(f"위도 {args.lat}, 경도 {args.lon} -> 격자 ({nx},{ny})")

    now = datetime.now(KST)
    today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if args.include_today:
        start_dt = today0 - timedelta(days=args.days)
        end_dt   = today0
    else:
        end_dt   = today0 - timedelta(days=1)
        start_dt = end_dt - timedelta(days=args.days-1)

    tmfc_slots = ["02","05","08","11","14","17","20","23"]
    today_str = now.strftime("%Y%m%d")
    last_ok_today = latest_tmfc_slot_hh(now)

    wanted_vars = ["SKY","TMP","REH","WSD","PCP","PTY"]

    # 수집 대상 (tmfc, tmef, var) 리스트 구성
    tasks = []
    cur = start_dt
    while cur <= end_dt:
        dstr = cur.strftime("%Y%m%d")
        for hh in tmfc_slots:
            if args.include_today and dstr == today_str:
                if (last_ok_today is None) or (hh > last_ok_today):
                    continue
            tmfc = f"{dstr}{hh}"
            tmfc_dt = datetime.strptime(tmfc, "%Y%m%d%H").replace(tzinfo=KST)
            for i in range(1, 6+1):
                tmef_dt = tmfc_dt + timedelta(hours=i)
                tmef = tmef_dt.strftime("%Y%m%d%H")
                for v in wanted_vars:
                    tasks.append((v, tmfc, tmef))
        cur += timedelta(days=1)

    total = len(tasks)
    if total == 0:
        print("수집 대상이 없습니다."); sys.exit(0)
    print(f"수집 예정 요청 수: {total} (days={args.days}, include_today={args.include_today}, vars={len(wanted_vars)})")

    # 병렬 요청
    limiter = QPSLimiter(args.qps) if args.qps > 0 else None
    results: Dict[Tuple[str,str], Dict[str, Optional[float]]] = {}
    success = 0; fail = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "pv-fetch/parallel"})
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_key = {}
            for v, tmfc, tmef in tasks:
                key = (tmfc, tmef, v)
                future = ex.submit(
                    fetch_one_var, session, args.base, auth_key,
                    v, tmfc, tmef, nx, ny,
                    args.timeout, args.retries, 0.5,
                    limiter
                )
                future_to_key[future] = key

            # 결과 취합
            for i, fut in enumerate(as_completed(future_to_key), 1):
                tmfc, tmef, v = None, None, None
                try:
                    tmfc, tmef, v = future_to_key[fut]
                    val = fut.result()
                    if (tmef, tmfc) not in results:
                        results[(tmef, tmfc)] = {"tmfc": tmfc, "tmef": tmef, "nx": nx, "ny": ny}
                    results[(tmef, tmfc)][v] = val
                    success += 1 if val is not None else 0
                except Exception:
                    fail += 1
                if i % 200 == 0 or i == total:
                    print(f"진행: {i}/{total}  (성공≈{success}, 실패≈{fail})")

    if not results:
        print("수집된 데이터가 없습니다. 헤더만 포함한 CSV를 생성합니다.")
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        pd.DataFrame(columns=["tmfc","tmef","nx","ny","SKY","TMP","REH","WSD","PCP","PTY"]).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] 0개 데이터 저장 완료 -> {args.out_csv}")
        return

    # tmef 최신 tmfc 우선, 시각 정렬
    rows = list(results.values())
    df = pd.DataFrame(rows)
    df = df.sort_values(["tmef","tmfc"]).drop_duplicates(subset=["tmef"], keep="last")
    df["tmef_dt"] = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce")
    df = df.sort_values("tmef_dt").drop(columns=["tmef_dt"])

    # 안전 저장(임시파일 -> 원자적 교체)
    outdir = os.path.dirname(args.out_csv) or "."
    os.makedirs(outdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="sky_tmp_", suffix=".csv", dir=outdir)
    os.close(fd)
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, args.out_csv)

    print(f"[OK] {len(df)}개 데이터 저장 완료 -> {args.out_csv}")
    print(f"[STATS] 총요청={total}, 값획득≈{success}, 실패≈{fail}, 성공률≈{(success/max(total,1))*100:.1f}%")

if __name__ == "__main__":
    main()
