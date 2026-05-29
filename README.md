![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-AnomalyDetection-purple)

<img width="1686" height="854" alt="스크린샷 2026-05-26 13 27 18" src="https://github.com/user-attachments/assets/933de58d-626d-4d50-98c6-57e15c694503" />

# tep-anomaly-detection
### 공정 센서 데이터 기반 실시간 AI 이상탐지 및 Explainable Process Monitoring System

---

## Demo
https://tep-anomaly-detection.streamlit.app/

---

## 프로젝트 개요

Tennessee Eastman Process(TEP)는 화학공정 분야에서 이상탐지와 공정 제어 연구에 널리 활용되는 대표적인 벤치마크 데이터셋이다. 실제 화학공장의 운전 환경을 기반으로 다양한 이상 상황(Fault)을 포함하고 있으며, 본 프로젝트에서는 22개의 공정 측정 변수(XMEAS)와 11개의 조작 변수(XMV)를 활용하여 공정 이상탐지 시스템을 구현하였다.

실제 공정 현장에서 이상탐지 시스템은 두 가지 상충되는 요구사항을 동시에 만족해야 한다.

- Recall 극대화: 이상 상황을 놓칠 경우 생산 손실이나 안전사고로 이어질 수 있다.
- False Alarm 최소화: 과도한 오경보는 현장 신뢰도를 저하시켜 경보 피로(Alarm Fatigue)를 유발한다.

본 프로젝트는 이러한 두 가지 요구사항을 동시에 고려하면서, 나아가 "왜 이상으로 판단되었는가" 를 설명할 수 있는 시스템 구현을 목표로 하였다. 단순히 이상 여부를 분류하는 데 그치지 않고, SHAP(Shapley Additive Explanations)을 활용하여 각 공정 변수가 이상 판단에 기여한 정도를 정량적으로 분석하였다. 이를 통해 현장 엔지니어와 데이터 분석가가 동일한 기준으로 이상 원인을 해석하고 활용할 수 있는 Explainable AI 기반 공정 이상탐지 시스템을 구현하였다.

---

## 공정 구조 이해

```

[피드 A / D / E]      [피드 C]       [피드 B(gas)]
       │                  │                 │
       └──────────────────┴─────────────────┘
                          ↓
              ┌──────────────────────┐
              │    반응기 (Reactor)  │  ← 냉각수 입구온도 제어
              │  xmeas_7 (압력)      │    xmv_10: 냉각수 유량
              │  xmeas_9 (온도)      │    xmv_11: Agitator 속도
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │  냉각기 (Condenser)  │  ← 냉각수 입구온도 제어
              │  xmeas_12~14         │    Fault 5: 냉각기 냉각수 이상
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │기액분리기(Separator) │
              │  xmeas_15~17         │
              │  xmeas_17 (온도) ★   │  ← 평균 변화 작음 → 동적 피처 핵심
              └──────────┬───────────┘
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
      재순환 압축기              [퍼지 스트림]
      xmv_5 (속도)
             ↓
    ┌─────────────────┐
    │ 분리탑(Stripper) │
    │  xmeas_18~22    │  ← 생성물 G/H 분리
    │  xmv_7~9        │
    └─────────────────┘

```

---

## 데이터셋

| 파일 | 설명 | 행 수 |
|------|------|-------|
| `TEP_FaultFree_Training.csv` | 정상 운전 학습 데이터 | 500 runs × 22 = 11,000 |
| `TEP_FaultFree_Testing.csv` | 정상 운전 테스트 데이터 | 960 runs × 22 |
| `TEP_Faulty_Training.csv` | 이상 유형 1~20 학습 데이터 | 500 runs × 20 |
| `TEP_Faulty_Testing.csv` | 이상 유형 1~20 테스트 데이터 | 960 runs × 20 |

**다운로드:** [Kaggle — TEP Simulation Dataset](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset)
→ `data/raw/` 폴더에 4개 CSV를 넣으면 된다.

### 이상 유형 (fault) 설명

| faultNumber | 이상 내용 | 유형 |
|-------------|-----------|------|
| 1 | A/C 피드 비율 이상 | Step |
| 2 | B 성분 조성 이상 | Step |
| 4 | 반응기 냉각수 입구온도 이상 | Step |
| 5 | 냉각기 냉각수 입구온도 이상 | Step |
| 7 | C 헤더 압력 손실 | Step |

---

## 프로젝트 구조

```
tep-anomaly-detection/
│
├── data/
│   ├── raw/                        # 원본 CSV (Kaggle 다운로드 → 여기에 배치)
│   │   ├── TEP_FaultFree_Training.csv
│   │   ├── TEP_FaultFree_Testing.csv
│   │   ├── TEP_Faulty_Training.csv
│   │   └── TEP_Faulty_Testing.csv
│   │
│   ├── processed/                  # 02_preprocessing.ipynb 산출물
│   │   ├── X_tr.npy / X_va.npy / X_te.npy    # StandardScaler 적용 피처 배열
│   │   ├── y_tr.npy / y_va.npy / y_te.npy    # 레이블 (0=정상, 1=이상)
│   │   ├── group_tr.npy / ...                 # simulationRun ID (Detection Delay용)
│   │   ├── feature_names.csv                  # 피처 이름 목록 (원본 33 + 동적 피처)
│   │   ├── xgb_threshold.npy                  # val set 최적 threshold
│   │   └── if_threshold.npy                   # IF 최적 threshold
│   │
│   ├── metrics.json                # 모델 성능 지표 (대시보드 연동)
│   ├── result_df.parquet           # 테스트셋 예측 결과 전체 (대시보드 연동)
│   ├── top_features.json           # SHAP 상위 피처 목록 (대시보드 연동)
│   └── shap_values.npy             # SHAP 값 배열 (대시보드 연동)
│
├── models/
│   ├── isolation_forest.pkl        # 학습된 IF 모델
│   ├── xgboost_baseline.pkl        # 학습된 XGBoost 모델
│   └── scaler.pkl                  # StandardScaler (train fit → 추론 재사용)
│
├── notebooks/
│   ├── 01_eda.ipynb                # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb     # 동적 피처 생성 + 전처리
│   ├── 03_isolation_forest.ipynb  # 비지도 이상탐지 baseline
│   ├── 04_xgboost_baseline.ipynb  # 지도학습 이상탐지 + 모델 비교
│   └── 05_shap_interpretability.ipynb  # SHAP 기반 XAI
│
├── src/
│   └── data_loader.py              # TEP 데이터 로딩 유틸 (load_tep 등)
│
├── outputs/                        # 시각화 저장
│   ├── 01_class_distribution.png
│   ├── 01_timeseries_comparison.png
│   ├── 01_fault_comparison_heatmap.png
│   └── ...
│
├── app.py                          # Streamlit 대시보드
├── generate_demo_data.py           # 데모용 합성 데이터 생성 + 모델 학습
├── requirements.txt
└── README.md

```

---

## 모델 비교 전략

| 모델 | 특징 | 활용 목적 |
|------|------|------|
| Isolation Forest | 비지도 이상탐지 | 초기 기준선 성능 확보 |
| XGBoost | 지도학습 기반 분류 | SHAP 기반 해석 가능 |
| Autoencoder | 정상 패턴 재구성 기반 탐지 | 비지도 이상탐지 성능 비교 |
| LSTM | 시계열 의존성 학습 | 시간 흐름 기반 이상 탐지 |

---

## 핵심 구현 내용

- TEP 공정 데이터 기반 이상탐지 파이프라인 구축
- Isolation Forest 및 XGBoost 기반 이상탐지 모델 구현
- SHAP 기반 공정 변수 중요도 해석
- Streamlit 기반 실시간 공정 모니터링 대시보드 구현
- Fault 유형별 탐지 성능 및 탐지 지연 시각화
- 공정 변수 상관관계 및 이상 분포 EDA 제공
- Fault 발생 이후 탐지 지연(Detection Delay) 분석
- Explainable AI 기반 공정 변수 영향도 시각화

---

## 실행 방법

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 데이터 준비
mkdir -p data/raw
# Kaggle에서 4개 CSV 다운로드 후 data/raw/ 에 배치

# 4. 노트북 순서 실행
jupyter notebook notebooks/
# 01_eda.ipynb → 02_preprocessing.ipynb → 03_isolation_forest.ipynb
# → 04_xgboost_baseline.ipynb → 05_shap_interpretability.ipynb

# 5. 대시보드 실행 (실제 데이터 기반)
streamlit run app.py

# 데이터 없이 빠른 확인 (데모 합성 데이터)
python generate_demo_data.py
streamlit run app.py

```

---

## 공정 분석 인사이트

Fault 1 — A/C Feed Ratio 이상
반응기 압력(xmeas_7)이 정상 구간 대비 급격히 변동한다. 피드 비율 변화가 반응기 내 성분 조성에 영향을 미치고, 압력 변화로 이어지는 경로가 SHAP으로 확인된다.

Fault 2 — B 성분 조성 이상
B 가스 성분 변화가 반응기 압력 및 재순환 계통 전체에 영향을 준다. 단일 변수보다 여러 변수의 동시 변화 패턴(rolling_std)이 탐지에 효과적이다.

Fault 4 — 반응기 냉각수 이상
냉각수 입구온도 이상이 반응기 온도(xmeas_9)에 영향을 주지만, 평균값 변화는 작다. xmeas_9_rstd(반응기 온도 변동성)가 SHAP 상위에 등장 — 온도 자체보다 변동성이 이상의 신호임을 정량적으로 확인하였다.

Fault 5 — 냉각기 냉각수 이상
분리기 온도(xmeas_17) 관련 동적 피처가 상위 기여 변수로 확인된다. 정적 피처만으로는 탐지가 어려운 대표적인 케이스로, 동적 피처 도입의 효과가 가장 두드러진다.

동적 피처의 효과
SHAP 상위 20개 피처 중 동적 피처(_rmean, _rstd, _diff, _zscore)가 다수를 차지한다. 이는 공정 이상이 단순 측정값의 변화보다 시계열 패턴의 변화로 먼저 나타남을 의미하며, 특히 rolling_std(변동성)가 효과적이다.

---

## 기술 스택

`Python` `pandas` `scikit-learn` `XGBoost` `PyTorch` `SHAP` `Streamlit` `Plotly`
