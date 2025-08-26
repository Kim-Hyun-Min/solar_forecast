# -*- coding: utf-8 -*-
import argparse, os, sys, subprocess, tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from xgboost import XGBRegressor

KST = timezone(timedelta(hours=9))

def run_fetch_recent6(out_csv):
    """
    fetch_grid_30day.py를 재사용해 오늘 날짜의 tmfc 슬롯들 중 '가능한 최신' 발표의 +1~+6h를 가져오게 합니다.
    간단 버전: --days 1 --include-today로 최근 것까지 뽑고, tmef 내림차순->상위 6개만 추립니다.
    """
    # 사용자 환경에 맞게 Python 경로/옵션 조정
    cmd = [sys.executable, "fetch_grid_30day.py", "--days", "1", "--include-today", "--out_csv", out_csv]
    print("[CMD]", " ".join(cmd))
    subprocess.check_call(cmd, shell=False)

def load_xgb_model(model_path, feat_path):
    m = XGBRegressor()
    m.load_model(model_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        feats = [ln.strip() for ln in f if ln.strip()]
    return m, feats

def add_time_features(df, ts_col="dt_kst"):
    t = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dow"]>=5).astype(int)
    return df

def add_rolling_feats(df, cols=("TMP","REH","WSD","PCP"), windows=(3,6)):
    df = df.copy()
    for c in cols:
        for w in windows:
            df[f"{c}_ma{w}"] = df[c].rolling(w, min_periods=1).mean()
            df[f"{c}_sd{w}"] = df[c].rolling(w, min_periods=2).std().fillna(0)
    return df

def main():
    ap = argparse.ArgumentParser(description="최근 6시간 예보 예측")
    ap.add_argument("--model", required=True, help="train_distill.py에서 저장한 모델(.json)")
    ap.add_argument("--features", required=True, help="모델 피처 목록(.feat)")
    ap.add_argument("--lat", type=float, default=35.1511)
    ap.add_argument("--lon", type=float, default=126.8902)
    ap.add_argument("--tmpdir", default=None, help="임시 CSV 폴더")
    ap.add_argument("--out", default="pred_recent6.csv")
    args = ap.parse_args()

    tmpdir = args.tmpdir or tempfile.gettempdir()
    tmp_csv = os.path.join(tmpdir, f"sky_recent_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}.csv")

    # 1) 최근 데이터 수집
    run_fetch_recent6(tmp_csv)

    # 2) 로드 & 최신 tmfc만 유지(중복 제거)
    df = pd.read_csv(tmp_csv)
    if df.empty:
        raise SystemExit("수집 결과가 비었습니다.")
    # 최신 tmfc 유지
    df = df.sort_values(["tmef","tmfc"]).drop_duplicates(subset=["tmef"], keep="last")
    # tmef 시각으로 정렬 후, 가장 최근 6개(= +1~+6h)만 사용
    df["dt_kst"] = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce")
    df = df.sort_values("dt_kst").tail(6).reset_index(drop=True)

    # 3) 피처 생성
    use_cols = ["SKY","TMP","REH","WSD","PCP","PTY","dt_kst"]
    for c in use_cols:
        if c not in df: df[c] = np.nan
    feat = add_time_features(df[use_cols].copy(), ts_col="dt_kst")
    feat = add_rolling_feats(feat)

    # 4) 예측
    model, features = load_xgb_model(args.model, args.features)
    X = feat[features].replace([np.inf,-np.inf], np.nan).fillna(0)
    yhat = model.predict(X)

    out = pd.DataFrame({
        "dt_kst": df["dt_kst"],
        "SKY": df["SKY"], "TMP": df["TMP"], "REH": df["REH"], "WSD": df["WSD"], "PCP": df["PCP"], "PTY": df["PTY"],
        "pred_pv_kw": yhat
    })
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 예측 6개 저장: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
