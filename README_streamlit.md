# ⚗️ TEP 공정 이상탐지 대시보드

**Tennessee Eastman Process — Isolation Forest + XGBoost + SHAP**

## 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데모 데이터 생성 (최초 1회)
python generate_demo_data.py

# 3. 대시보드 실행
streamlit run app.py
```

## 페이지 구성

| 페이지 | 내용 |
|--------|------|
| 🏠 시스템 대시보드 | KPI 카드, 이상 탐지 시계열, 모델 비교 |
| 📈 실시간 모니터링 | 공정 변수 시계열, Anomaly Score |
| 🔬 모델 성능 분석 | ROC/PR Curve, Confusion Matrix, Fault별 히트맵 |
| 🧠 XAI — SHAP 해석 | Feature importance, Violin plot |
| ⚙️ 공정 변수 탐색 | 분포 비교, 산점도, 상관관계 |

## 모델 구성

- **Isolation Forest**: 비지도 이상탐지 (정상 데이터만 학습)
- **XGBoost**: 지도학습 기반 이상탐지 (scale_pos_weight로 불균형 처리)
- **동적 피처**: rolling_mean, rolling_std, diff, z-score (window=10)
- **SHAP**: TreeExplainer로 개별 샘플 설명

## 이상 유형

| Fault | 설명 |
|-------|------|
| Fault 1 | A/C Feed Ratio 변화 |
| Fault 2 | B Composition 변화 |
| Fault 4 | Reactor Cooling 이상 |
| Fault 5 | Condenser Cooling 이상 |
