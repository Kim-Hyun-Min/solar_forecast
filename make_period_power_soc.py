# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np

from utils_weather import (
    KST, FactorParams, load_weather_csv, build_power_scenarios, ensure_kst
)

def integrate_kwh(t_kst, kw):
    """가변 간격 시계열(kW) -> 각 구간별 kWh (사다리꼴이 아닌 직사각 적분: kw * dt_h)"""
    t = pd.to_datetime(t_kst)
    kw = np.asarray(kw, dtype=float)
    if len(t) < 2:
        return np.zeros_like(kw)
    dt_min = np.diff(t.values).astype('timedelta64[m]').astype(float)
    dt_h = np.append(dt_min/60.0, dt_min[-1]/60.0 if len(dt_min)>0 else 1.0)
    return kw * dt_h, dt_h

def simulate_soc(gen_kwh, dt_h, cap_kwh=10.0, init_soc=0.5, chg_eff=0.95, max_charge_kw=3.0):
    """
    간단한 충전 전용 SOC 모델 (방전/부하 미반영).
    - 시간당 최대 충전 전력 제한(max_charge_kw)과 효율(chg_eff) 반영
    """
    n = len(gen_kwh)
    soc = np.zeros(n, dtype=float)
    soc[0] = float(init_soc)
    for i in range(1, n):
        max_in_kwh = max_charge_kw * dt_h[i] * chg_eff
        add_kwh = float(np.clip(gen_kwh[i], 0.0, max_in_kwh))
        soc[i] = min(1.0, soc[i-1] + add_kwh / cap_kwh)
    return soc

def main():
    ap = argparse.ArgumentParser(description="기간별 pv_full_kw, kWh, SOC 계산 후 CSV 저장")
    ap.add_argument("--csv", default=r"D:\data\sky_30day.csv", help="입력 기상 CSV (tmef, SKY..)")
    ap.add_argument("--start", required=True, help="시작일 YYYY-MM-DD (예: 2025-07-22)")
    ap.add_argument("--end",   required=True, help="종료일 YYYY-MM-DD (예: 2025-08-20, 포함)")
    ap.add_argument("--out", default=r"D:\data\pv_20250722_20250820_features.csv", help="출력 CSV 경로")

    # 발전/위치 파라미터
    ap.add_argument("--lat", type=float, default=35.1511)
    ap.add_argument("--lon", type=float, default=126.8902)
    ap.add_argument("--capacity-kw", type=float, default=3.0)
    ap.add_argument("--pr", type=float, default=0.75)

    # 감쇠모델 파라미터 (plot_factor_and_power_full.py와 동일 엔진)
    ap.add_argument("--atten-model", choices=["table","linear"], default="table")
    ap.add_argument("--atten-k", type=float, default=0.9)

    # 배터리(간단 모델)
    ap.add_argument("--batt-kwh", type=float, default=10.0)
    ap.add_argument("--init-soc", type=float, default=0.5)
    ap.add_argument("--chg-eff", type=float, default=0.95)
    ap.add_argument("--max-charge-kw", type=float, default=3.0)
    args = ap.parse_args()

    # 1) 원본 CSV 로드 + 기간 필터(KST)
    df = load_weather_csv(args.csv).sort_values("dt_kst")
    if df.empty:
        raise SystemExit("입력 CSV가 비어 있습니다.")

    start_kst = ensure_kst(pd.Timestamp(args.start))
    # 종료일 '하루 끝'까지 포함
    end_kst   = ensure_kst(pd.Timestamp(args.end)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    m = (df["dt_kst"] >= start_kst) & (df["dt_kst"] <= end_kst)
    df = df.loc[m].copy()
    if df.empty:
        raise SystemExit("선택한 기간에 데이터가 없습니다.")

    # 2) 발전 시나리오(kW) 계산 (pv_full_kw, pv_clear_kw)
    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen = build_power_scenarios(
        df=df, lat=args.lat, lon=args.lon,
        capacity_kw=args.capacity_kw, pr=args.pr, params=params
    )
    pv_full_kw  = scen["full"].values
    pv_clear_kw = scen["clear"].values

    # 3) kWh 적분 + 간단 SOC
    gen_full_kwh, dt_h = integrate_kwh(df["dt_kst"], pv_full_kw)
    gen_clear_kwh, _   = integrate_kwh(df["dt_kst"], pv_clear_kw)
    soc = simulate_soc(
        gen_kwh=gen_full_kwh, dt_h=dt_h,
        cap_kwh=args.batt_kwh, init_soc=args.init_soc,
        chg_eff=args.chg_eff, max_charge_kw=args.max_charge_kw
    )

    # 4) 저장 (머신러닝 학습에 바로 쓰기 좋은 피처 포함)
    out = pd.DataFrame({
        "dt_kst": df["dt_kst"],
        "SKY": df.get("SKY"), "TMP": df.get("TMP"), "REH": df.get("REH"),
        "WSD": df.get("WSD"), "PCP": df.get("PCP"), "PTY": df.get("PTY"),
        "pv_full_kw":  pv_full_kw,
        "pv_clear_kw": pv_clear_kw,
        "gen_full_kwh":  gen_full_kwh,
        "gen_clear_kwh": gen_clear_kwh,
        "soc": soc
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {os.path.abspath(args.out)}  rows={len(out)}")

if __name__ == "__main__":
    main()
