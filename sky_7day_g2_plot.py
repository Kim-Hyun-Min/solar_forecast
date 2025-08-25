# -*- coding: utf-8 -*-
# sky_7day_g2_plot.py — 중복 제거: utils_weather만 사용
# 기능:
#  - 최근 n일(or 지정 구간) 발전량 계산(Clear/Full)
#  - 그래프 저장(pv_g2.png)
#  - CSV 저장(pv_power_last7days.csv: dt_kst, pv_clear_kw, pv_full_kw)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 공통 유틸 (중복 함수 없음)
from utils_weather import (
    FactorParams,
    load_weather_csv,
    build_power_scenarios,
)

# 기본값
CSV_DEFAULT = r"D:\data\sky_30day.csv"
OUTDIR_DEFAULT = "pv_simple"
LAT_DEFAULT, LON_DEFAULT = 35.1511, 126.8902
CAPACITY_KW_DEFAULT, PR_DEFAULT = 3.0, 0.75

def _setup_korean_font():
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

def main():
    _setup_korean_font()

    ap = argparse.ArgumentParser(description="최근 기간 Clear/Full 발전량 계산 및 g2 플롯/CSV 생성")
    ap.add_argument("--csv", default=CSV_DEFAULT, help="입력 기상 CSV (tmef, SKY, TMP, REH, WSD, PCP, PTY)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT, help="출력 폴더")
    ap.add_argument("--lat", type=float, default=LAT_DEFAULT)
    ap.add_argument("--lon", type=float, default=LON_DEFAULT)
    ap.add_argument("--capacity-kw", type=float, default=CAPACITY_KW_DEFAULT)
    ap.add_argument("--pr", type=float, default=PR_DEFAULT)
    # 기간
    ap.add_argument("--last-days", type=int, default=7, help="최근 n일 (기본 7)")
    ap.add_argument("--start", help="시작 YYYY-MM-DD")
    ap.add_argument("--end", help="종료 YYYY-MM-DD (포함)")
    # 감쇠/모델 파라미터
    ap.add_argument("--atten-model", choices=["table","linear"], default="table")
    ap.add_argument("--atten-k", type=float, default=0.9)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 기상 CSV 로드(중복 tmfc 정리, dt_kst 생성, 컬럼 보정은 utils에서 처리)
    df = load_weather_csv(args.csv).sort_values("dt_kst")

    # 2) 기간 필터
    t_all = df["dt_kst"]
    if args.start or args.end:
        mask = pd.Series(True, index=df.index)
        if args.start:
            mask &= (t_all >= pd.to_datetime(args.start))
        if args.end:
            # 하루 끝까지 포함
            mask &= (t_all <= pd.to_datetime(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        df = df.loc[mask].copy()
        tag = f"custom_{(args.start or '')}_{(args.end or '')}".replace(':','').replace('/','-')
    else:
        start_ts = t_all.max() - pd.Timedelta(days=args.last_days)
        df = df[t_all >= start_ts].copy()
        tag = f"last{args.last_days}days"

    if df.empty:
        raise ValueError("선택된 기간에 데이터가 없습니다.")

    # 3) 발전량(kW) 시나리오 계산 (Clear/Full만 사용)
    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen = build_power_scenarios(
        df=df,
        lat=args.lat,
        lon=args.lon,
        capacity_kw=args.capacity_kw,
        pr=args.pr,
        params=params
    )
    pv_clear = scen["clear"]
    pv_full  = scen["full"]

    # 4) g2 플롯 저장 (Clear vs Full)
    t = pd.to_datetime(df["dt_kst"])
    plt.figure(figsize=(14,5))
    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.plot(t, pv_clear.values, label="Clear (이론)", linestyle="--", linewidth=1.6)
    plt.plot(t, pv_full.values,  label="Full (모든 요소)", linewidth=1.8)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Power (kW)")
    plt.xlabel("KST Time")
    plt.title("PV Power — Clear vs Full")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "pv_g2.png")
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 그래프 저장: {out_png}")

    # 5) CSV 저장 (파일명은 기존 유지)
    #    pv_power_last7days.csv (dt_kst, pv_clear_kw, pv_full_kw)
    out_csv = os.path.join(args.outdir, "pv_power_last7days.csv")
    df_out = pd.DataFrame({
        "dt_kst": df["dt_kst"].values,
        "pv_clear_kw": pv_clear.values,
        "pv_full_kw":  pv_full.values,
    })
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV 저장: {out_csv}")

if __name__ == "__main__":
    main()
