# 태양광 발전량 예측 (운량 기반)

기상청 VSRT 격자 운량/기상 데이터를 수집해, Clear-sky 대비 보정 요인(운량, 온도, 습도, 소일링, 강수)을 적용한 시간별 발전량(kW)을 추정하고 그래프로 시각화합니다.

## 폴더 구성
.
├─ fetch_grid_30day.py # 최근 N일(슬라이딩) 수집
├─ fetch_grid_month.py # 지정 '월' 윈도우 수집
├─ merge_sky_months.py # 월별 CSV 여러 개 병합
├─ utils_weather.py # 공통 유틸(클리어스카이/보정/시나리오)
├─ plot_14panel.py # 요소별↔전력 14패널 그리기
├─ sky_7day_g2_plot.py # Clear vs Full 2개 그래프 + CSV
├─ sky_7day_g7_plot.py # 7개 시나리오 그래프 세트
├─ requirements.txt
└─ .env.example # 환경변수 예시(실제 키는 .env에)

bash
복사
편집

## 환경 준비
```bash
# (선택) 가상환경 권장
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
환경변수(.env)
프로젝트 루트에 .env 파일을 만들고 API 키를 넣습니다. (깃에는 올리지 마세요)

ini
KMA_API_KEY=YOUR_KMA_KEY
필요 시 데이터 폴더도 환경변수로 관리할 수 있습니다.

ini
DATA_DIR=./data
데이터 수집
1) 최근 N일 (예: 7일)

bash
python fetch_grid_30day.py --days 7 --include-today --out_csv ./data/sky_30day.csv
2) 특정 월 전체 (예: 2025-08)
bash
python fetch_grid_month.py --ym 2025-08 --outdir ./data
# 결과: ./data/sky_30day_202508.csv
3) 여러 달 병합

bash
python merge_sky_months.py --inputs ./data/sky_30day_202505.csv ./data/sky_30day_202506.csv --out ./data/sky_merged_202505-202506.csv

# 또는 폴더 패턴으로
python merge_sky_months.py --indir ./data --pattern "sky_30day_2025*.csv" --outdir ./data
시각화 & 결과 저장
A) 14패널 (최근 7일 기본)

bash
python plot_14panel.py --csv ./data/sky_30day.csv --outdir ./out_14
# ./out_14/pv_14panel.png 생성
B) Clear vs Full (기본 최근 7일)

bash
python sky_7day_g2_plot.py --csv ./data/sky_30day.csv --outdir ./out_g2
# ./out_g2/pv_g2.png, ./out_g2/pv_power_last7days.csv
C) 7개 시나리오 세트

bash
python sky_7day_g7_plot.py --csv ./data/sky_30day.csv --outdir ./out_g7
# ./out_g7/ 하위에 여러 PNG + CSV
공통 옵션 예시

위치: --lat 35.1511 --lon 126.8902

설비용량/PR: --capacity-kw 3.0 --pr 0.75

감쇠모델: --atten-model table|linear --atten-k 0.9

기간 지정: --last-days 7 또는 --start 2025-08-01 --end 2025-08-07

주의사항
.env(비밀키)와 대용량 CSV는 Git에 올리지 않습니다. .gitignore로 제외하세요.

Windows 절대경로(D:\data\...) 대신 ./data 같이 상대경로 사용을 권장합니다.

트러블슈팅
API 키 없음: .env의 KMA_API_KEY 확인

인증/서버 에러: 잠시 후 재시도(--retries, --qps 옵션)

폰트 깨짐: Windows는 기본 말굽고딕, macOS/Linux는 한글 폰트 설치 필요

yaml
---

# requirements.txt (예시)

```txt
python-dotenv>=1.0.0
requests>=2.31.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
추후 라이브러리가 늘면 이 파일만 업데이트하면 됩니다.
(현재 스크립트에서 사용하는 외부 패키지 기준: requests, pandas, numpy, matplotlib, .env 로딩용 python-dotenv)


## 머신러닝 (ML) — gen_full_kwh 기반 예측

### 1) 학습 데이터 준비
수집된 기상 CSV(`sky_30day.csv` 등)를 바탕으로 kWh·SOC를 계산해 학습용 데이터셋을 생성합니다.

```bash
python ML/make_period_power_soc.py --csv ./data/sky_30day.csv --start 2025-07-22 --end 2025-08-20 --out ./data/pv_20250722_20250820_features.csv
출력 CSV 컬럼:

dt_kst, SKY, TMP, REH, WSD, PCP, PTY

pv_full_kw, pv_clear_kw (전력 kW)

gen_full_kwh, gen_clear_kwh (에너지 kWh)

soc (배터리 축전량, 단순 모델)

2) 모델 학습
gen_full_kwh(구간별 에너지)를 타깃으로 XGBoost 모델을 학습합니다.

bash
복사
편집
python ML/train_distill.py --feat_csv ./data/pv_20250722_20250820_features.csv --target gen_full_kwh --model_out ./data/distill_xgb.json
출력: distill_xgb.json (모델), distill_xgb.json.feat (피처 목록)

교차검증 MAE 로그 확인 가능 (예: MAE ≈ 0.04 kWh)

3) 최근 6시간 예보 예측
API 호출로 가장 최근 발표 슬롯의 +1~+6h 예보를 수집하고, 학습된 모델로 에너지를 예측합니다.

bash
복사
편집
python ML/predict_recent6.py --model ./data/distill_xgb.json --features ./data/distill_xgb.json.feat --out ./data/pred_recent6.csv
출력: pred_recent6.csv

dt_kst, SKY, TMP, REH, WSD, PCP, PTY

pred_gen_kwh (예측 에너지)