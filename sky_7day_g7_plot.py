# -*- coding: utf-8 -*-
# sky_7day_g7_plot.py — 7개 시나리오(kW) + 선택적 SOC, utils_weather 의존 버전
import os
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

# 공통 유틸
from utils_weather import (
    KST,
    FactorParams,
    load_weather_csv,
    build_power_scenarios,
)

# ===== 기본 설정 =====
LAT_DEFAULT, LON_DEFAULT = 35.1511, 126.8902
CAPACITY_KW_DEFAULT, PR_DEFAULT = 3.0, 0.75
OUTDIR_DEFAULT = "pv_simple"
CSV_DEFAULT = r"D:\data\sky_30day.csv"

# 배터리(선택 시뮬레이션용)
BATTERY_KWH = 10.0
MAX_CHARGE_KW = 3.0
CHG_EFF = 0.95
INV_EFF = 0.96
INIT_SOC = 0.50

def _setup_korean_font():
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

# ===== 플롯 유틸 =====
def plot_7_separate(t, series_map, outdir, prefix="pv", ylab="Power (kW)"):
    os.makedirs(outdir, exist_ok=True)
    names = [
        ("clear",      f"{prefix}_clear.png",      f"{ylab} — Clear (이론)"),
        ("full",       f"{prefix}_full.png",       f"{ylab} — All factors"),
        ("cloud_only", f"{prefix}_factor_cloud.png",  f"{ylab} — Cloud only"),
        ("temp_only",  f"{prefix}_factor_temp.png",   f"{ylab} — Temp only"),
        ("humid_only", f"{prefix}_factor_humid.png",  f"{ylab} — Humidity only"),
        ("soil_only",  f"{prefix}_factor_soil.png",   f"{ylab} — Soiling only"),
        ("rain_only",  f"{prefix}_factor_rain.png",   f"{ylab} — Rain only"),
    ]
    for key, fname, ttl in names:
        plt.figure(figsize=(12,4))
        if key not in ("clear", "full"):
            plt.plot(t, series_map["clear"].values, label="Clear", linestyle="--", alpha=0.6)
        plt.plot(t, series_map[key].values, label=key.replace("_", " ").title(), linewidth=1.8)
        plt.title(ttl)
        plt.xlabel("KST Time"); plt.ylabel(ylab)
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=140)
        print(f"[OK] 저장: {path}")

def plot_7in1(t, series_map, outdir, fname, ylab, ylim=None):
    import matplotlib.dates as mdates
    plt.figure(figsize=(16,6))
    labels = [
        ("clear",      "Clear (이론)"),
        ("full",       "All factors"),
        ("cloud_only", "Cloud only"),
        ("temp_only",  "Temp only"),
        ("humid_only", "Humidity only"),
        ("soil_only",  "Soiling only"),
        ("rain_only",  "Rain only"),
    ]
    for key, lab in labels:
        plt.plot(t, series_map[key].values, label=lab, linewidth=1.6)
    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if ylim: plt.ylim(*ylim)
    plt.ylabel(ylab); plt.xlabel("KST Time")
    plt.title(f"{ylab} — 7 Scenarios")
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=150)
    print(f"[OK] 저장: {out_path}")

# ===== 메인 =====
def main():
    _setup_korean_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT, help="입력 CSV 경로 (tmef, SKY, TMP, REH, WSD, PCP, PTY)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT, help="출력 폴더")
    ap.add_argument("--lat", type=float, default=LAT_DEFAULT)
    ap.add_argument("--lon", type=float, default=LON_DEFAULT)
    ap.add_argument("--capacity-kw", type=float, default=CAPACITY_KW_DEFAULT)
    ap.add_argument("--pr", type=float, default=PR_DEFAULT)
    # 기간 옵션
    ap.add_argument("--last-days", type=int, help="최근 n일만 (예: 7)")
    ap.add_argument("--start", help="시작 날짜 YYYY-MM-DD")
    ap.add_argument("--end", help="종료 날짜 YYYY-MM-DD (포함)")
    # SOC 출력 여부
    ap.add_argument("--with-soc", action="store_true", help="SOC(축전량) 그래프/CSV도 함께 저장")
    # 감쇠 모델 파라미터 (utils_weather.FactorParams로 전달)
    ap.add_argument("--atten-model", choices=["table","linear"], default="table")
    ap.add_argument("--atten-k", type=float, default=0.9)
    args = ap.parse_args()

    # CSV 읽기(중복 tmef → 최신 tmfc 유지, dt_kst 생성, 결측 컬럼 표준화는 utils에서 처리)
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV 파일이 없습니다: {args.csv}", file=sys.stderr); sys.exit(1)
    df = load_weather_csv(args.csv)  # tz-aware KST 열 'dt_kst' 생성, 필수 컬럼 존재 보장
    df = df.sort_values("dt_kst")

    # 기간 필터
    t_all = df["dt_kst"]
    mask = pd.Series(True, index=df.index)
    title_suffix = ""
    if args.last_days:
        start_ts = t_all.max() - pd.Timedelta(days=args.last_days)
        mask &= (t_all >= start_ts)
        title_suffix = f" (Last {args.last_days} days)"
    elif args.start or args.end:
        if args.start:
            mask &= (t_all >= pd.to_datetime(args.start))
        if args.end:
            mask &= (t_all <= pd.to_datetime(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        title_suffix = f" ({args.start or ''}~{args.end or ''})".strip(" ()")
    df = df.loc[mask].copy()

    # 시나리오(kW) — 전부 utils_weather로 계산
    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen_kw = build_power_scenarios(df, args.lat, args.lon, args.capacity_kw, args.pr, params)

    # 평균 보정계수 로그
    try:
        ratio = scen_kw["full"].sum() / max(scen_kw["clear"].sum(), 1e-9)
        print(f"[INFO] 평균 보정계수{title_suffix}: {ratio:.2f} (Clear 대비 {ratio*100:.1f}%)")
    except Exception as e:
        print(f"[WARN] 평균 보정계수 계산 실패: {e}")

    # 결과 CSV 구성 (kW)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    df_out = pd.DataFrame({"dt_kst": df["dt_kst"]})
    df_out["pv_clear_kw"]      = scen_kw["clear"].values
    df_out["pv_full_kw"]       = scen_kw["full"].values
    df_out["pv_cloud_only_kw"] = scen_kw["cloud_only"].values
    df_out["pv_temp_only_kw"]  = scen_kw["temp_only"].values
    df_out["pv_humid_only_kw"] = scen_kw["humid_only"].values
    df_out["pv_soil_only_kw"]  = scen_kw["soil_only"].values
    df_out["pv_rain_only_kw"]  = scen_kw["rain_only"].values

    # CSV 저장
    out_csv = os.path.join(outdir, "pv_from_sky_power_soc.csv" if args.with_soc else "pv_from_sky_power.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV 저장: {out_csv}")

    # --- 그래프 저장 ---
    t = pd.to_datetime(df_out["dt_kst"])

    # (A) kW 개별 7장
    plot_7_separate(t, scen_kw, outdir, prefix="pv", ylab="Power (kW)")
    # (B) kW 7-in-1 한 장
    plot_7in1(t, scen_kw, outdir, "pv_kw_7in1.png", "Power (kW)")

    if args.with_soc:
        # (C) SOC 개별 7장
        plot_7_separate(t, scen_soc, outdir, prefix="battery_soc", ylab="SOC (0–1)")
        # (D) SOC 7-in-1 한 장
        plot_7in1(t, scen_soc, outdir, "battery_soc_7in1.png", "SOC (0–1)", ylim=(0,1.05))

    print("[DONE] 그래프 저장 완료.")

if __name__ == "__main__":
    main()
