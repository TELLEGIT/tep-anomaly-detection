# tep-anomaly-detection
### 화학공정 센서 데이터 기반 AI 이상탐지 및 실시간 공정 모니터링 시스템

---

## 프로젝트 배경

**Tennessee Eastman Process(TEP)** 는 1993년 Downs & Vogel이 발표한
화학공정 제어 벤치마크 시뮬레이터입니다.
반응기 → 냉각기 → 기액분리기 → 재순환 → 분리탑으로 이어지는
실제 화학공장 구조를 그대로 재현하며,
52개 센서(XMEAS) + 11개 조작 변수(XMV) = 총 52개 변수로 구성됩니다.

이 프로젝트는 TEP 데이터를 활용해:
- 비지도·지도 학습 기반 **공정 이상탐지 모델** 개발
- **SHAP XAI** 로 이상 발생 원인 변수를 화공 원리와 연결해 해석
- **Streamlit 실시간 모니터링 대시보드** 구현

을 목표로 합니다.

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
│   ├── raw/                  # 원본 CSV (Git 미포함)
│   └── processed/            # 전처리 후 npy 저장
│
├── notebooks/
│   ├── 01_eda.ipynb           # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb # 전처리 실험
│   ├── 03_baseline.ipynb      # Isolation Forest + XGBoost
│   ├── 04_autoencoder.ipynb   # Autoencoder 이상탐지
│   ├── 05_lstm.ipynb          # LSTM 시계열 분류
│   └── 06_xai_shap.ipynb      # SHAP 해석
│
├── src/
│   ├── data_loader.py         # 데이터 로딩·분리·라벨 생성
│   ├── preprocessor.py        # 정규화·윈도우 생성
│   ├── evaluator.py           # 성능 평가 공통 함수
│   ├── xai.py                 # SHAP 분석
│   └── models/
│       ├── isolation_forest.py
│       ├── autoencoder.py
│       └── lstm_classifier.py
│
├── models/                    # 학습된 모델 저장 (.pkl, .pt)
├── outputs/                   # 그래프·리포트 PNG
├── app/
│   └── streamlit_app.py       # 실시간 모니터링 대시보드
├── requirements.txt
└── README.md
```

---

## 모델 비교 전략

| 모델 | 학습 방식 | 시계열 반영 | 포트폴리오 역할 |
|------|-----------|------------|----------------|
| Isolation Forest | 비지도 | X | 빠른 베이스라인 |
| XGBoost | 지도 | X | 성능 기준선 + SHAP |
| Autoencoder | 비지도 | △ | **메인 모델** (현장 현실적) |
| LSTM | 지도 | O | 시계열 강점 실험 |

---

## 핵심 성과 (목표)

| 모델 | F1-Score | ROC-AUC | 탐지 지연(타임스텝) |
|------|----------|---------|-----------------|
| Isolation Forest | - | - | - |
| XGBoost | - | - | - |
| Autoencoder | - | - | - |
| LSTM | - | - | - |

*실험 완료 후 채워넣기*

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
streamlit run app/streamlit_app.py
```

---

## 화공 도메인 인사이트

- **fault 1**: A/C 피드 비율 이상 → 반응기 압력(xmeas_7) 급변
- **fault 4**: 반응기 냉각수 밸브 이상 → 반응기 온도(xmeas_9) 급등
- SHAP 분석 결과, 반응기 구역 변수(xmeas_7, 9, 21)가 이상 탐지에 가장 크게 기여

---

## 기술 스택

`Python` `pandas` `scikit-learn` `XGBoost` `PyTorch` `SHAP` `Streamlit` `Plotly`
