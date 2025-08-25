# -*- coding: utf-8 -*-
# plot_14panel.py (utils_weather 의존 버전)
# 최근 N일(기본 7일): 요소별 시계열 + 해당 요소 반영 전력(only) 그래프를 바로 아래에 배치 (총 14패널)
# 1: SKY(%), 2: Power—Cloud only
# 3: TMP(°C), 4: Power—Temp only
# 5: REH(%), 6: Power—Humidity only
# 7: WSD(m/s), 8: Power—Temp only(풍속 영향 포함)
# 9: PCP(mm), 10: Power—Soiling only
# 11: PTY(라벨), 12: Power—Rain only
# 13: Power—Clear(이론), 14: Power—Full(모든 요소)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 공통 유틸 모듈
from utils_weather import (
    KST,
    FactorParams,
    load_weather_csv,
    build_power_scenarios,
    parse_sky_to_cloudfrac,
)

# ===== 기본 파라미터 =====
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

def main():
    _setup_korean_font()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT, help="입력 CSV 경로 (tmef, SKY, TMP, REH, WSD, PCP, PTY)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT, help="출력 폴더")
    ap.add_argument("--lat", type=float, default=LAT_DEFAULT)
    ap.add_argument("--lon", type=float, default=LON_DEFAULT)
    ap.add_argument("--capacity-kw", type=float, default=CAPACITY_KW_DEFAULT)
    ap.add_argument("--pr", type=float, default=PR_DEFAULT)
    ap.add_argument("--last-days", type=int, default=7, help="최근 n일(기본 7)")
    # 고급: 감쇠모델/파라미터 바꾸고 싶을 때
    ap.add_argument("--atten-model", choices=["table","linear"], default="table")
    ap.add_argument("--atten-k", type=float, default=0.9)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) CSV 로드 (중복 tmef→최신 tmfc 유지, dt_kst 생성, 결측 컬럼 표준화)
    df = load_weather_csv(args.csv)  # utils_weather에서 일관 처리
    df = df.sort_values("dt_kst")

    # 2) 최근 N일 필터
    last = df["dt_kst"].max()
    start_ts = last - pd.Timedelta(days=args.last_days)
    df = df[df["dt_kst"] >= start_ts].copy()

    # 3) 시나리오 계산 (kW)
    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen = build_power_scenarios(
        df=df,
        lat=args.lat,
        lon=args.lon,
        capacity_kw=args.capacity_kw,
        pr=args.pr,
        params=params
    )

    # 4) 14패널 플롯
    fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(18, 36), sharex=True)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)

    t = pd.to_datetime(df["dt_kst"])

    def stylize(ax, ylabel, title):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # 1–2 SKY & cloud_only
    sky_pct = df.get("SKY").apply(parse_sky_to_cloudfrac) * 100.0
    axes[0].plot(t, sky_pct, linewidth=1.6)
    stylize(axes[0], "운량(%)", "최근 운량(%)")
    axes[1].plot(t, scen["cloud_only"].values, linewidth=1.8)
    stylize(axes[1], "kW", "예측전력 — Cloud only")

    # 3–4 TMP & temp_only
    axes[2].plot(t, df.get("TMP"), linewidth=1.6)
    stylize(axes[2], "기온(°C)", "최근 기온(°C)")
    axes[3].plot(t, scen["temp_only"].values, linewidth=1.8)
    stylize(axes[3], "kW", "예측전력 — Temp only")

    # 5–6 REH & humid_only
    axes[4].plot(t, df.get("REH"), linewidth=1.6)
    stylize(axes[4], "상대습도(%)", "최근 상대습도(%)")
    axes[5].plot(t, scen["humid_only"].values, linewidth=1.8)
    stylize(axes[5], "kW", "예측전력 — Humidity only")

    # 7–8 WSD & temp_only (풍속 영향 포함)
    axes[6].plot(t, df.get("WSD"), linewidth=1.6)
    stylize(axes[6], "풍속(m/s)", "최근 풍속(m/s)")
    axes[7].plot(t, scen["temp_only"].values, linewidth=1.8)
    stylize(axes[7], "kW", "예측전력 — Temp only (풍속 영향 포함)")

    # 9–10 PCP & soil_only
    axes[8].plot(t, df.get("PCP"), linewidth=1.6)
    stylize(axes[8], "강수량(mm)", "최근 강수량(mm)")
    axes[9].plot(t, scen["soil_only"].values, linewidth=1.8)
    stylize(axes[9], "kW", "예측전력 — Soiling only")

    # 11–12 PTY(라벨) & rain_only
    def to_int_or_nan(x):
        try: return int(float(x))
        except: return float("nan")
    pty_int = df.get("PTY").apply(to_int_or_nan)
    axes[10].step(t, pty_int, where="post", linewidth=1.6)
    pty_labels = {0:"없음",1:"비",2:"비/눈",3:"눈",5:"빗방울",6:"빗방울/눈날림",7:"눈날림"}
    yticks = [k for k in [0,1,2,3,5,6,7] if k in pd.Series(pty_int).dropna().unique()]
    axes[10].set_yticks(yticks)
    axes[10].set_yticklabels([pty_labels.get(k, str(k)) for k in yticks])
    stylize(axes[10], "강수형태", "최근 강수형태(PTY)")
    axes[11].plot(t, scen["rain_only"].values, linewidth=1.8)
    stylize(axes[11], "kW", "예측전력 — Rain only")

    # 13 Clear, 14 Full
    axes[12].plot(t, scen["clear"].values, linewidth=1.8, linestyle="--")
    stylize(axes[12], "kW", "예측전력 — Clear(이론)")
    axes[13].plot(t, scen["full"].values, linewidth=2.0)
    stylize(axes[13], "kW", "예측전력 — Full(모든 요소 반영)")
    axes[13].set_xlabel("KST 시각")

    plt.tight_layout()
    out_path = os.path.join(args.outdir, "pv_14panel.png")
    plt.savefig(out_path, dpi=150)
    print(f"[OK] 14패널 저장: {out_path}")

if __name__ == "__main__":
    main()
