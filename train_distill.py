# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime

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
    ap = argparse.ArgumentParser(description="의사라벨 학습(XGBoost)")
    ap.add_argument("--feat_csv", required=True, help="make_month_power_csv.py 결과")
    ap.add_argument("--target", choices=["pv_full_kw","gen_full_kwh"], default="pv_full_kw")
    ap.add_argument("--model_out", default="distill_xgb.json")
    args = ap.parse_args()

    df = pd.read_csv(args.feat_csv, parse_dates=["dt_kst"])
    df = add_time_features(df)
    df = add_rolling_feats(df)

    features = [
        "hour","dow","month","is_weekend",
        "SKY","TMP","REH","WSD","PCP","PTY",
        "TMP_ma3","REH_ma3","WSD_ma3","PCP_ma3",
        "TMP_sd3","REH_sd3","WSD_sd3","PCP_sd3",
        "TMP_ma6","REH_ma6","WSD_ma6","PCP_ma6",
    ]
    X = df[features].replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df[args.target].astype(float).values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(df))
    models = []
    for tr, va in kf.split(X):
        m = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, tree_method="hist"
        )
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              verbose=False)
        preds[va] = m.predict(X.iloc[va])
        models.append(m)

    mae = mean_absolute_error(y, preds)
    print(f"[CV] MAE={mae:.4f} ({args.target})")

    # 최종 전체 적합 후 저장(간단 버전: 첫 fold 모델 저장)
    models[0].save_model(args.model_out)
    # 피처 목록도 저장
    with open(args.model_out + ".feat", "w", encoding="utf-8") as f:
        f.write("\n".join(features))
    print(f"[OK] 모델 저장: {os.path.abspath(args.model_out)}")

if __name__ == "__main__":
    main()
