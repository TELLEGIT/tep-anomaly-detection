![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-AnomalyDetection-purple)

# tep-anomaly-detection
### 공정 센서 데이터 기반 실시간 AI 이상탐지 및 Explainable Process Monitoring System

---

## 프로젝트 배경

**Tennessee Eastman Process(TEP)** 는 1993년 Downs & Vogel이 발표한
화학공정 제어 벤치마크 시뮬레이터입니다.
반응기 → 냉각기 → 기액분리기 → 재순환 → 분리탑으로 이어지는
실제 화학공장 구조를 그대로 재현하며,
41개 공정 측정 변수(XMEAS)와 11개 조작 변수(XMV)로 구성된
총 52개 변수 시스템이다.

이 프로젝트는 TEP 공정 데이터를 기반으로
실시간 이상 상태를 탐지하고,
이상 판단의 원인을 설명 가능한 형태(XAI)로 해석하며,
이를 시각적으로 모니터링할 수 있는
공정 데이터를 활용한 AI 기반 공정 모니터링 및 이상탐지 시스템 구현 프로젝트이다.

---

## 공정 구조 이해

```
[피드 A/D/E/C/F]
       ↓
  [반응기 Reactor]  ← 냉각수 제어 (xmv_10, xmv_11)
       ↓
  [냉각기 Condenser]
       ↓
  [기액분리기 Separator] ← 온도·압력 (xmeas_12~17)
       ↓
  [재순환 압축기]
       ↓
  [분리탑 Stripper]     ← 생성물 G/H 분리
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
→ `data/raw/` 폴더에 4개 CSV를 넣으면 됩니다.

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
├── data/
│   ├── raw/                   # 원본 CSV (용량 문제로 Git 제외)
│   ├── processed/             # 전처리 데이터 저장
│   ├── metrics.json
│   ├── result_df.parquet
│   ├── top_features.json
│   └── shap_values.npy
│
├── models/
│   ├── isolation_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_xgboost_baseline.ipynb
│   └── 05_shap_interpretability.ipynb
│
├── app.py                     # Streamlit 대시보드
├── generate_demo_data.py
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
source venv/bin/activate        # Mac/Linux

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Jupyter 노트북 실행
jupyter notebook notebooks/

# 4. 대시보드 실행
streamlit run app.py

```

---

## 화공 도메인 인사이트

- **fault 1**: A/C 피드 비율 이상 → 반응기 압력(xmeas_7) 급변
- **fault 4**: 반응기 냉각수 밸브 이상 → 반응기 온도(xmeas_9) 급등
- SHAP 분석 결과, 반응기 구역 변수(xmeas_7, 9, 21)가 이상 탐지에 가장 크게 기여

---

## 기술 스택

`Python` `pandas` `scikit-learn` `XGBoost` `PyTorch` `SHAP` `Streamlit` `Plotly`
