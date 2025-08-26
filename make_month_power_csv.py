# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np
from utils_weather import FactorParams, load_weather_csv, build_power_scenarios

def integrate_kwh(t, kw):
    """가변 간격 시간대 시계열(kW) -> kWh 적분"""
    t = pd.to_datetime(t)
    kw = np.asarray(kw, dtype=float)
    if len(t) < 2:
        return np.zeros_like(kw)
    dt_h = np.diff(t.values).astype('timedelta64[m]').astype(float)/60.0
    dt_h = np.append(dt_h, dt_h[-1])  # 마지막 구간은 직전 간격으로 근사
    return kw * dt_h  # 구간 kWh

def simulate_soc(kwh, cap_kwh=10.0, init_soc=0.5, chg_eff=0.95, inv_eff=0.96,
                 max_charge_kw=3.0, dt_hours=None):
    """생산 kWh만 반영한 간단 충전 SOC(방전 없음, 상한=1.0)"""
    n = len(kwh)
    soc = np.zeros(n, dtype=float)
    soc[0] = init_soc
    # dt_hours: 각 샘플 구간 길이(시간) (없으면 동일 간격으로 가정)
    if dt_hours is None:
        dt_hours = np.ones(n, dtype=float) * (1.0 if n==0 else 1.0)
    for i in range(1, n):
        # 시간당 최대 충전량 제한(전력 한계 고려)
        max_in_kwh = max_charge_kw * dt_hours[i] * chg_eff
        charge_kwh = min(kwh[i], max_in_kwh)
        soc[i] = min(1.0, soc[i-1] + charge_kwh / cap_kwh)
    return soc

def main():
    ap = argparse.ArgumentParser(description="월간 kW→kWh & SOC 산출 CSV")
    ap.add_argument("--csv", required=True, help="입력 기상 CSV(merge_sky_months 결과 또는 월 CSV)")
    ap.add_argument("--out", default="pv_month_features.csv", help="출력 CSV 경로")
    ap.add_argument("--lat", type=float, default=35.1511)
    ap.add_argument("--lon", type=float, default=126.8902)
    ap.add_argument("--capacity-kw", type=float, default=3.0)
    ap.add_argument("--pr", type=float, default=0.75)
    # 모델 파라미터
    ap.add_argument("--atten-model", choices=["table","linear"], default="table")
    ap.add_argument("--atten-k", type=float, default=0.9)
    # 배터리 파라미터(간단)
    ap.add_argument("--batt-kwh", type=float, default=10.0)
    ap.add_argument("--init-soc", type=float, default=0.5)
    ap.add_argument("--chg-eff", type=float, default=0.95)
    ap.add_argument("--inv-eff", type=float, default=0.96)
    ap.add_argument("--max-charge-kw", type=float, default=3.0)
    args = ap.parse_args()

    dfw = load_weather_csv(args.csv).sort_values("dt_kst")
    if dfw.empty:
        raise SystemExit("입력 CSV에 유효한 데이터가 없습니다.")

    params = FactorParams(atten_model=args.atten_model, atten_k=args.atten_k)
    scen = build_power_scenarios(
        df=dfw, lat=args.lat, lon=args.lon,
        capacity_kw=args.capacity_kw, pr=args.pr, params=params
    )
    t = pd.to_datetime(dfw["dt_kst"])
    pv_full_kw  = scen["full"].values
    pv_clear_kw = scen["clear"].values

    # kWh 적분(샘플 간격 반영)
    t_diff_min = np.diff(t.values).astype('timedelta64[m]').astype(float)
    dt_h = np.append(t_diff_min/60.0, t_diff_min[-1]/60.0 if len(t_diff_min)>0 else 1.0)
    gen_full_kwh  = integrate_kwh(t, pv_full_kw)
    gen_clear_kwh = integrate_kwh(t, pv_clear_kw)

    # 간이 SOC(충전만) 시뮬레이션
    soc = simulate_soc(gen_full_kwh, cap_kwh=args.batt_kwh, init_soc=args.init_soc,
                       chg_eff=args.chg_eff, inv_eff=args.inv_eff,
                       max_charge_kw=args.max_charge_kw, dt_hours=dt_h)

    out = pd.DataFrame({
        "dt_kst": t,
        "SKY": dfw["SKY"],
        "TMP": dfw["TMP"], "REH": dfw["REH"], "WSD": dfw["WSD"], "PCP": dfw["PCP"], "PTY": dfw["PTY"],
        "pv_full_kw":  pv_full_kw,
        "pv_clear_kw": pv_clear_kw,
        "gen_full_kwh":  gen_full_kwh,
        "gen_clear_kwh": gen_clear_kwh,
        "soc": soc
    })
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {os.path.abspath(args.out)}  (rows={len(out)})")

if __name__ == "__main__":
    main()
