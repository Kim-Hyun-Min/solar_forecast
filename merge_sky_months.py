# -*- coding: utf-8 -*-
"""
merge_sky_months.py
- 월별로 수집한 VSRT Grid CSV들을 하나로 병합
- 같은 tmef에 대해 최신 tmfc만 유지 (중복 제거)
- 출력: 결합 CSV (자동 파일명 또는 --out 지정)

입력 CSV 예상 컬럼:
  tmfc, tmef, nx, ny, SKY, TMP, REH, WSD, PCP, PTY
(일부 컬럼이 없으면 NaN으로 채워집니다)

사용 예:
  python merge_sky_months.py --inputs D:\data\sky_30day_202505.csv D:\data\sky_30day_202506.csv --out D:\data\sky_60day_202505-202506.csv
  python merge_sky_months.py --indir D:\data --pattern "sky_30day_2025*.csv" --outdir D:\data
"""

import os
import glob
import argparse
from typing import List, Tuple

import pandas as pd
import numpy as np

REQ_COLS = ["tmfc", "tmef", "nx", "ny", "SKY", "TMP", "REH", "WSD", "PCP", "PTY"]

def _collect_inputs(args) -> List[str]:
    paths: List[str] = []
    if args.inputs:
        paths.extend(args.inputs)
    if args.indir:
        pattern = args.pattern or "sky_30day_*.csv"
        paths.extend(glob.glob(os.path.join(args.indir, pattern)))
    # 중복 제거 & 존재 확인
    uniq = []
    for p in paths:
        if p not in uniq and os.path.exists(p):
            uniq.append(p)
    return uniq

def _infer_out_name(files: List[str], outdir: str) -> str:
    # 파일명에서 YYYYMM 추출해서 범위로 이름 생성
    tags = []
    for f in files:
        base = os.path.basename(f)
        # sky_30day_YYYYMM.csv 형태 가정
        for tok in base.replace(".", "_").split("_"):
            if len(tok) == 6 and tok.isdigit():
                tags.append(tok)
                break
    tags = sorted(set(tags))
    if tags:
        tag = f"{tags[0]}-{tags[-1]}" if len(tags) > 1 else tags[0]
        name = f"sky_merged_{tag}.csv"
    else:
        name = "sky_merged.csv"
    os.makedirs(outdir or ".", exist_ok=True)
    return os.path.join(outdir or ".", name)

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 필요한 컬럼 보장
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # tmef/tmfc 문자열화 (정렬/중복처리에 안전)
    df["tmef"] = df["tmef"].astype(str)
    df["tmfc"] = df["tmfc"].astype(str)
    return df

def _dedupe_latest_tmfc(df: pd.DataFrame) -> pd.DataFrame:
    # 같은 tmef 내에서 tmfc 최신(가장 큰 문자열)이 마지막이 되도록 정렬
    df2 = df.sort_values(["tmef", "tmfc"]).drop_duplicates(subset=["tmef"], keep="last")
    # 시간 정렬
    df2["tmef_dt"] = pd.to_datetime(df2["tmef"], format="%Y%m%d%H", errors="coerce")
    df2 = df2.sort_values("tmef_dt").drop(columns=["tmef_dt"])
    return df2

def _preview(df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] 병합 결과: 0 rows")
        return
    tmin = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce").min()
    tmax = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce").max()
    print(f"[PREVIEW] rows={len(df)}, range={tmin} ~ {tmax}")
    print("[HEAD]")
    print(df.head(3).to_string(index=False))
    print("[TAIL]")
    print(df.tail(3).to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description="월별 VSRT CSV 병합 + 최신 tmfc 유지")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="+", help="직접 지정한 CSV들 (여러개)")
    g.add_argument("--indir", help="CSV 폴더 (pattern과 함께 사용)")
    ap.add_argument("--pattern", help='--indir 사용 시 글롭 패턴 (기본 "sky_30day_*.csv")')
    ap.add_argument("--out", help="저장할 CSV 경로")
    ap.add_argument("--outdir", default=".", help="out 미지정 시 저장 폴더 (기본 .)")
    ap.add_argument("--no-dedupe", action="store_true", help="tmef 중복 제거/최신 tmfc 유지 과정을 끄기")
    args = ap.parse_args()

    files = _collect_inputs(args)
    if not files:
        raise SystemExit("입력 CSV가 없습니다. --inputs 또는 --indir/--pattern을 확인하세요.")
    print(f"[INPUT] {len(files)} files")
    for p in files:
        print(" -", p)

    out_path = args.out or _infer_out_name(files, args.outdir)
    print("[OUTPUT]", out_path)

    # 로드/병합
    dfs = []
    total_rows = 0
    for p in files:
        df = _load_csv(p)
        total_rows += len(df)
        dfs.append(df[REQ_COLS])  # 필요한 컬럼만 사용(순서 통일)
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[MERGE] raw_rows={total_rows} -> concat_rows={len(merged)}")

    # 중복 제거(같은 tmef 최신 tmfc만)
    if not args.no_dedupe:
        before = len(merged)
        merged = _dedupe_latest_tmfc(merged)
        print(f"[DEDUPE] {before} -> {len(merged)} (drop dup tmef, keep latest tmfc)")
    else:
        # 그래도 시간 정렬만
        merged["tmef_dt"] = pd.to_datetime(merged["tmef"], format="%Y%m%d%H", errors="coerce")
        merged = merged.sort_values("tmef_dt").drop(columns=["tmef_dt"])

    _preview(merged)

    # 저장(안전 저장: 임시파일 후 교체)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    merged.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, out_path)
    print(f"[OK] 병합 저장 완료 -> {out_path}")

if __name__ == "__main__":
    main()
