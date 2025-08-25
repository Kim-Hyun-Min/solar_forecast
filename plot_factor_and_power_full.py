
# -*- coding: utf-8 -*-
# plot_factor_and_power_full.py — utils_weather 의존 버전(중복 제거)
# 기능:
#   1) 최근 n일(or 지정 기간) 기상 요소를 각각 PNG로 저장
#      - SKY(%), TMP(°C), REH(%), WSD(m/s), PCP(mm), PTY(라벨)
#   2) 같은 기간 발전량 CSV 저장 (dt_kst, pv_clear_kw, pv_full_kw)
#
# 출력:
#   pv_factors/factor_sky.png
#   pv_factors/factor_tmp.png
#   pv_factors/factor_reh.png
#   pv_factors/factor_wsd.png
#   pv_factors/factor_pcp.png
#   pv_factors/factor_pty.png
#   pv_factors/pv_hourly_power.csv

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
    parse_sky_to_cloudfrac,  # SKY % 표시에만 사용
)

# 기본값
CSV_DEFAULT = r"D:\data\sky_30day.csv"
OUTDIR_DEFAULT = "pv_factors"
LAT_DEFAULT, LON_DEFAULT = 35.1511, 126.8902
CAPACITY_KW_DEFAULT, PR_DEFAULT = 3.0, 0.75

def _setup_korean_font():
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

def _style_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, alpha=0.3)

def _plot_and_save(t, y, ylabel, title, out_path, kind="line", step_where="post", yticks=None, yticklabels=None):
    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    if kind == "line":
        plt.plot(t, y, linewidth=1.8)
    elif kind == "step":
        plt.step(t, y, where=step_where, linewidth=1.6)
    _style_time_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.set_xlabel("KST Time")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[OK] 저장: {out_path}")

def main():
    _setup_korean_font()

    ap = argparse.ArgumentParser(description="최근 기간 요소별 PNG + 발전량 CSV (Clear vs Full)")
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

    # 1) 기상 CSV 로드(중복 tmef → 최신 tmfc 유지, dt_kst 생성, 결측 컬럼 표준화)
    df = load_weather_csv(args.csv).sort_values("dt_kst")

    # 2) 기간 필터
    t_all = df["dt_kst"]
    if args.start or args.end:
        mask = pd.Series(True, index=df.index)
        if args.start:
            mask &= (t_all >= pd.to_datetime(args.start))
        if args.end:
            mask &= (t_all <= pd.to_datetime(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        df = df.loc[mask].copy()
    else:
        start_ts = t_all.max() - pd.Timedelta(days=args.last_days)
        df = df[t_all >= start_ts].copy()

    if df.empty:
        raise ValueError("선택된 기간에 데이터가 없습니다.")

    # 3) 요소별 PNG 저장
    t = pd.to_datetime(df["dt_kst"])

    # SKY (%)
    sky_pct = df.get("SKY").apply(parse_sky_to_cloudfrac) * 100.0
    _plot_and_save(
        t, sky_pct, "운량(%)", "최근 운량(%)",
        os.path.join(args.outdir, "factor_sky.png"), kind="line"
    )

    # TMP (°C)
    _plot_and_save(
        t, df.get("TMP"), "기온(°C)", "최근 기온(°C)",
        os.path.join(args.outdir, "factor_tmp.png"), kind="line"
    )

    # REH (%)
    _plot_and_save(
        t, df.get("REH"), "상대습도(%)", "최근 상대습도(%)",
        os.path.join(args.outdir, "factor_reh.png"), kind="line"
    )

    # WSD (m/s)
    _plot_and_save(
        t, df.get("WSD"), "풍속(m/s)", "최근 풍속(m/s)",
        os.path.join(args.outdir, "factor_wsd.png"), kind="line"
    )

    # PCP (mm)
    _plot_and_save(
        t, df.get("PCP"), "강수량(mm)", "최근 강수량(mm)",
        os.path.join(args.outdir, "factor_pcp.png"), kind="line"
    )

    # PTY (라벨)
    def to_int_or_nan(x):
        try: return int(float(x))
        except: return float("nan")
    pty_int = df.get("PTY").apply(to_int_or_nan)
    pty_labels = {0:"없음",1:"비",2:"비/눈",3:"눈",5:"빗방울",6:"빗방울/눈날림",7:"눈날림"}
    yticks = [k for k in [0,1,2,3,5,6,7] if k in pd.Series(pty_int).dropna().unique()]
    yticklabels = [pty_labels.get(k, str(k)) for k in yticks]
    _plot_and_save(
        t, pty_int, "강수형태", "최근 강수형태(PTY)",
        os.path.join(args.outdir, "factor_pty.png"),
        kind="step", step_where="post", yticks=yticks, yticklabels=yticklabels
    )

    # 4) 발전량(kW) 시나리오 계산 (utils_weather)
    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen = build_power_scenarios(
        df=df,
        lat=args.lat,
        lon=args.lon,
        capacity_kw=args.capacity_kw,
        pr=args.pr,
        params=params
    )

    # 5) CSV 저장 (시간별 kW) — Clear vs Full
    out_csv = os.path.join(args.outdir, "pv_hourly_power.csv")
    df_out = pd.DataFrame({
        "dt_kst": df["dt_kst"].values,
        "pv_clear_kw": scen["clear"].values,
        "pv_full_kw":  scen["full"].values,
    })
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 발전량 CSV 저장: {out_csv}")

    print("[DONE] 요소별 PNG(단위 명시) + 발전량 CSV 완료.")

if __name__ == "__main__":
    main()
