# -*- coding: utf-8 -*-
"""
utils_weather.py
- 기상 데이터 공통 유틸 함수 모음
- 시간대(KST) 일관 처리, 운량 해석, 태양 고도/클리어스카이, 온도/습도/소일링/강수 벌점,
  시나리오(kW) 계산, CSV 로딩/정규화 등
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import timezone, timedelta, datetime
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ===== 공통 상수 =====
KST = timezone(timedelta(hours=9))

# VSRT 격자 상수 (위경도->격자 변환과 호환)
GRID_NX, GRID_NY = 149, 253

# ===== 파라미터 컨테이너 =====
@dataclass
class FactorParams:
    # 구름(운량) 감쇠
    atten_model: str = "table"  # "table" | "linear"
    atten_k: float = 0.9        # linear일 때 기울기

    # 모듈/온도
    noct_c: float = 45.0
    temp_coeff_pct_per_c: float = -0.4  # [%/°C], STC 25°C 기준

    # 습도
    humidity_threshold: float = 90.0    # REH ≥ threshold → 패널티 적용
    humidity_penalty_pct: float = 3.0   # [%]

    # 소일링(먼지) — 일일 누적(%)과 최대 손실(%)
    soiling_pct_per_day: float = 0.15
    soiling_max_pct: float = 20.0
    # 강우에 따른 세척 임계값(mm)
    rain_clean_full_mm: float = 5.0     # 이 이상이면 완전 세척
    rain_clean_partial_mm: float = 1.0  # 이 이하면 세척 효과 없음

    # 강수 진행 중 패널티(PTY 코드별)
    # 0: 없음, 1: 비, 2: 비/눈, 3: 눈, 5: 빗방울, 6: 빗방울/눈날림, 7: 눈날림
    rain_running_penalties: Dict[int, float] = field(default_factory=lambda: {
        0: 1.00,  # 없음
        1: 0.85,  # 비
        2: 0.83,  # 비/눈
        3: 0.80,  # 눈 (덮임 영향 가정)
        5: 0.90,  # 빗방울
        6: 0.88,  # 빗방울/눈날림
        7: 0.82,  # 눈날림
    })

# ===== 시간대 & 안전 변환 =====
def ensure_kst(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    """입력 시각을 KST tz-aware Timestamp로 보정."""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            return ts.tz_localize(KST)
        return ts.tz_convert(KST)
    # python datetime
    if ts.tzinfo is None:
        return pd.Timestamp(ts, tz=KST)
    return pd.Timestamp(ts).tz_convert(KST)

# ===== 태양 위치/클리어스카이 =====
def solar_elevation_kst(dt_kst: pd.Timestamp | datetime, lat: float, lon: float) -> float:
    """KST 기준 태양 고도각(deg) 근사."""
    t = ensure_kst(dt_kst).tz_convert("UTC").to_pydatetime()
    n = t.timetuple().tm_yday
    frac = t.hour + t.minute/60 + t.second/3600
    gamma = 2*math.pi/365 * (n - 1 + (frac - 12)/24)

    decl = (
        0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
        - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
        - 0.002697*math.cos(3*gamma) + 0.00148*math.sin(3*gamma)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
        - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma)
    )
    time_offset = eqtime + 4*lon
    tst = frac * 60 + time_offset
    ha = math.radians((tst / 4) - 180)
    latr = math.radians(lat)

    elev = math.degrees(math.asin(
        math.sin(latr)*math.sin(decl) + math.cos(latr)*math.cos(decl)*math.cos(ha)
    ))
    return elev

def shape_clear_sky(dt_kst: pd.Timestamp | datetime, lat: float, lon: float) -> float:
    """클리어스카이(정규화) 형태 ~ sin(elevation)+clip."""
    elev = solar_elevation_kst(dt_kst, lat, lon)
    return max(0.0, math.sin(math.radians(max(0.0, elev))))

# ===== 운량 해석 & 감쇠 =====
def parse_sky_to_cloudfrac(v) -> float:
    """
    SKY 코드/수치 → 0~1 운량 비율.
    - DB01/02/03/04, 00/01/02/03 등 코드 매핑
    - 0~1/0~10/0~100 범위 자동 스케일
    - 1/3/4 정수코드(과거 관측)도 처리
    """
    if pd.isna(v): return np.nan
    s = str(v).strip().upper()
    table = {"DB01": 0.0, "00": 0.0, "DB02": 0.3, "01": 0.3, "DB03": 0.6, "02": 0.6, "DB04": 0.9, "03": 0.9}
    if s in table:
        return table[s]
    try:
        x = float(s)
        if x in (1, 3, 4):
            return {1: 0.1, 3: 0.6, 4: 0.9}[int(x)]
        if 0 <= x <= 1:
            return float(x)
        if 1 < x <= 10:
            return float(np.clip(x / 10.0, 0.0, 1.0))
        if 0 <= x <= 100:
            return float(np.clip(x / 100.0, 0.0, 1.0))
    except Exception:
        pass
    return np.nan

def cloudfrac_to_multiplier(cf: float, params: FactorParams) -> float:
    """운량→감쇠 배율. table/linear 선택."""
    if np.isnan(cf): return np.nan
    if params.atten_model == "linear":
        return float(np.clip(1.0 - params.atten_k * cf, 0.05, 1.0))
    # 표 테이블(기본)
    if cf <= 0.15: return 0.95
    if cf <= 0.35: return 0.80
    if cf <= 0.70: return 0.55
    return 0.22

# ===== 온도/풍속 보정 =====
def estimate_tcell_from_poa(poa_wm2: float, temp_air_c: float, wind_ms: float, params: FactorParams) -> float:
    """
    평면 일사량 근사(POA) → 모듈온도 추정.
    - 풍속이 증가하면 냉각(선형 근사)
    """
    if poa_wm2 <= 0:
        return float(temp_air_c if not np.isnan(temp_air_c) else 25.0)
    ta = temp_air_c if not np.isnan(temp_air_c) else 20.0
    v = wind_ms if not np.isnan(wind_ms) else 1.0
    base = ta + (params.noct_c - 20.0) / 800.0 * poa_wm2
    cool = max(0.6, 1.0 - 0.06 * max(0.0, v - 1.0))  # 0.6~1.0 범위
    return float(base * cool + (1.0 - cool) * ta)

def temp_coeff_multiplier(t_cell_c: float, params: FactorParams) -> float:
    """모듈 온도계수(%)를 배율로 환산."""
    if t_cell_c is None or np.isnan(t_cell_c):
        return 1.0
    return float(1.0 + (params.temp_coeff_pct_per_c / 100.0) * (t_cell_c - 25.0))

# ===== 습도/강수/소일링 =====
def humidity_penalty_multiplier(reh_pct: float, params: FactorParams) -> float:
    if reh_pct is None or np.isnan(reh_pct): return 1.0
    return 1.0 - (params.humidity_penalty_pct / 100.0) if reh_pct >= params.humidity_threshold else 1.0

def ptype_rain_penalty_multiplier(pty, params: FactorParams) -> float:
    """강수 진행 중 배율(PTY 코드별). 미지정 코드는 1.0."""
    try:
        k = int(float(pty))
    except Exception:
        return 1.0
    return float(params.rain_running_penalties.get(k, 1.0))

def rainfall_cleaning_soiling_multiplier(
    pcp_mm_series: pd.Series,
    params: FactorParams,
    index_timestamps: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    소일링 누적 → 강수로 세척.
    - 연속 무강수 기간 동안 하루당 soiling_pct_per_day 만큼 선형 증가(최대 soiling_max_pct)
    - 강수량이 rain_clean_full_mm 이상이면 완전 세척(0% 손실)
      rain_clean_partial_mm~rain_clean_full_mm 사이는 선형 세척(부분 복구)
    """
    if index_timestamps is None:
        idx = pcp_mm_series.index
    else:
        idx = pd.to_datetime(index_timestamps)

    m = []
    days_since_rain = 0.0
    loss_pct = 0.0
    last_ts = None

    for ts, p in zip(idx, pcp_mm_series.values):
        ts = pd.Timestamp(ts)
        if last_ts is None:
            last_ts = ts

        dt_days = (ts - last_ts).total_seconds() / (3600 * 24.0)
        last_ts = ts

        # 강수 세척 로직
        rr = 0.0 if p is None or np.isnan(p) else float(p)
        if rr >= params.rain_clean_full_mm:
            loss_pct = 0.0  # 완전 세척
            days_since_rain = 0.0
        elif rr >= params.rain_clean_partial_mm:
            # 부분 세척: 선형으로 일부 복구 (partial_mm ~ full_mm 구간)
            frac = (rr - params.rain_clean_partial_mm) / (params.rain_clean_full_mm - params.rain_clean_partial_mm + 1e-9)
            loss_pct *= max(0.0, 1.0 - float(np.clip(frac, 0.0, 1.0)))
            days_since_rain = 0.0
        else:
            days_since_rain += max(0.0, dt_days)
            loss_pct = min(params.soiling_max_pct, loss_pct + params.soiling_pct_per_day * dt_days)

        m.append(1.0 - loss_pct / 100.0)

    return np.array(m, dtype=float)

# ===== 데이터 정규화/보조 =====
def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """필수 컬럼 확보 및 타입 표준화."""
    df = df.copy()
    for col in ["SKY", "TMP", "REH", "WSD", "PCP", "PTY"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def dedupe_latest_by_tmfc(df: pd.DataFrame) -> pd.DataFrame:
    """동일 tmef에 대해 최신 tmfc만 유지."""
    if "tmef" not in df.columns:
        raise ValueError("CSV에 'tmef' 컬럼이 필요합니다. 예: 202507010200")
    if "tmfc" in df.columns:
        df = df.copy()
        df["tmfc"] = df["tmfc"].astype(str)
        df = df.sort_values(["tmef", "tmfc"]).drop_duplicates(subset=["tmef"], keep="last")
    else:
        df = df.drop_duplicates(subset=["tmef"], keep="last")
    return df

def load_weather_csv(path: str, tz_to_kst: bool = True) -> pd.DataFrame:
    """CSV 로드 → tmef를 dt_kst로 변환(naive라도 KST 가정)."""
    df = pd.read_csv(path)
    df = dedupe_latest_by_tmfc(df)
    dt = pd.to_datetime(df["tmef"], format="%Y%m%d%H", errors="coerce")
    if tz_to_kst:
        # naive → KST 부여
        df["dt_kst"] = pd.to_datetime([ensure_kst(t) for t in dt])
    else:
        df["dt_kst"] = dt
    return normalize_weather_columns(df)

# ===== 전력(kW) 시나리오 =====
def build_power_scenarios(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    capacity_kw: float,
    pr: float,
    params: Optional[FactorParams] = None,
) -> Dict[str, pd.Series]:
    """Clear/Cloud/Temp/Humid/Soil/Rain/Full 시나리오(kW) 반환."""
    params = params or FactorParams()
    work = df.copy()

    # 운량 & 시간형상
    work["cloud_frac"] = work["SKY"].apply(parse_sky_to_cloudfrac).interpolate().bfill().ffill().fillna(0.6)
    work["clear_shape"] = [shape_clear_sky(t, lat, lon) for t in work["dt_kst"]]
    work["m_cloud"] = work["cloud_frac"].apply(lambda x: cloudfrac_to_multiplier(x, params))

    # POA & 모듈온도
    poa_clear = 1000.0 * work["clear_shape"].astype(float)
    tcell_clear = [estimate_tcell_from_poa(poa, ta, ws, params) for poa, ta, ws in zip(poa_clear, work["TMP"], work["WSD"])]
    m_temp_clear = np.array([temp_coeff_multiplier(tc, params) for tc in tcell_clear], dtype=float)

    poa_cloud = 1000.0 * work["clear_shape"].astype(float) * work["m_cloud"].astype(float)
    tcell_cloud = [estimate_tcell_from_poa(poa, ta, ws, params) for poa, ta, ws in zip(poa_cloud, work["TMP"], work["WSD"])]
    work["m_temp"] = np.array([temp_coeff_multiplier(tc, params) for tc in tcell_cloud], dtype=float)

    # 기타 벌점
    work["m_humid"] = work["REH"].apply(lambda x: humidity_penalty_multiplier(x, params))
    idx = pd.to_datetime(work["dt_kst"])
    work["m_soil"] = rainfall_cleaning_soiling_multiplier(pd.Series(work["PCP"].values, index=idx), params, index_timestamps=idx)
    work["m_rain"] = work["PTY"].apply(lambda x: ptype_rain_penalty_multiplier(x, params))

    # 기본 전력 (kW)
    base_clear = capacity_kw * pr * work["clear_shape"].astype(float)

    scen = {
        "clear":      pd.Series(base_clear.values, index=work.index, name="pv_clear_kw"),
        "cloud_only": pd.Series((base_clear * work["m_cloud"]).values, index=work.index, name="pv_cloud_only_kw"),
        "temp_only":  pd.Series((base_clear * m_temp_clear).values, index=work.index, name="pv_temp_only_kw"),
        "humid_only": pd.Series((base_clear * work["m_humid"]).values, index=work.index, name="pv_humid_only_kw"),
        "soil_only":  pd.Series((base_clear * work["m_soil"]).values, index=work.index, name="pv_soil_only_kw"),
        "rain_only":  pd.Series((base_clear * work["m_rain"]).values, index=work.index, name="pv_rain_only_kw"),
        "full":       pd.Series((base_clear * work["m_cloud"] * work["m_temp"] * work["m_humid"] * work["m_soil"] * work["m_rain"]).values,
                                index=work.index, name="pv_full_kw"),
    }
    return scen

# ===== (선택) 격자 변환 =====
def latlon_to_xy(lat: float, lon: float) -> Tuple[int, int]:
    """기상청 LCC 좌표계 변환(격자)."""
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

__all__ = [
    "KST",
    "FactorParams",
    "ensure_kst",
    "solar_elevation_kst",
    "shape_clear_sky",
    "parse_sky_to_cloudfrac",
    "cloudfrac_to_multiplier",
    "estimate_tcell_from_poa",
    "temp_coeff_multiplier",
    "humidity_penalty_multiplier",
    "ptype_rain_penalty_multiplier",
    "rainfall_cleaning_soiling_multiplier",
    "normalize_weather_columns",
    "dedupe_latest_by_tmfc",
    "load_weather_csv",
    "build_power_scenarios",
    "latlon_to_xy",
]
