"""
TEP 데모 데이터 생성 스크립트
실제 모델/데이터가 없을 때 대시보드 시연용 합성 데이터를 만든다.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import os, json

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─── 변수 정의 ───────────────────────────────────────────────
XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 23)]
XMV_COLS   = [f"xmv_{i}"   for i in range(1, 12)]
BASE_COLS  = XMEAS_COLS + XMV_COLS

# Rolling 피처
FEATURE_COLS = BASE_COLS.copy()
for c in BASE_COLS:
    FEATURE_COLS += [f"{c}_rmean", f"{c}_rstd", f"{c}_diff", f"{c}_zscore"]

FAULT_TYPES = [0, 1, 2, 4, 5]
FAULT_DESC  = {0:"정상", 1:"A/C Feed Ratio", 2:"B Composition", 4:"Reactor Cooling", 5:"Condenser Cooling"}
N_NORMAL    = 800
N_FAULT     = 200
N_TIMESTEPS = 500

# ─── 데이터 생성 ─────────────────────────────────────────────
def gen_run(fault_type, n_steps):
    t = np.arange(n_steps)
    data = {}
    for i, c in enumerate(XMEAS_COLS):
        base = np.random.randn(n_steps).cumsum() * 0.05 + np.sin(t * 0.02 + i) * 0.3
        if fault_type > 0 and n_steps > 160:
            shift = np.zeros(n_steps)
            shift[160:] = np.random.choice([1,-1]) * (0.8 + fault_type * 0.2)
            base += shift
        data[c] = base + np.random.randn(n_steps) * 0.1

    for i, c in enumerate(XMV_COLS):
        base = np.random.randn(n_steps).cumsum() * 0.03 + np.cos(t * 0.015 + i) * 0.2
        if fault_type > 0 and n_steps > 160:
            shift = np.zeros(n_steps)
            shift[160:] = np.random.choice([1,-1]) * (0.5 + fault_type * 0.15)
            base += shift
        data[c] = base + np.random.randn(n_steps) * 0.08

    df = pd.DataFrame(data)
    df["faultNumber"] = fault_type
    df["label"]       = 0 if fault_type == 0 else \
                        np.where(np.arange(n_steps) >= 160, 1, 0)
    df["simulationRun"] = np.nan
    df["time"] = t
    return df

rows = []
run_id = 0
for ft in [0] * 20 + [1,2,4,5] * 5:
    df = gen_run(ft, N_TIMESTEPS)
    df["simulationRun"] = run_id
    rows.append(df)
    run_id += 1

all_df = pd.concat(rows, ignore_index=True)

# ─── 동적 피처 ───────────────────────────────────────────────
WINDOW = 10
for c in BASE_COLS:
    grp = all_df.groupby("simulationRun")[c]
    all_df[f"{c}_rmean"]  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    all_df[f"{c}_rstd"]   = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
    all_df[f"{c}_diff"]   = grp.transform(lambda x: x.diff().fillna(0))
    m = grp.transform("mean"); s = grp.transform("std").replace(0, 1)
    all_df[f"{c}_zscore"] = (all_df[c] - m) / s

all_df = all_df.fillna(0)

# train / test 분리
train_runs = all_df["simulationRun"].unique()[:30]
test_runs  = all_df["simulationRun"].unique()[30:]
train_df = all_df[all_df["simulationRun"].isin(train_runs)]
test_df  = all_df[all_df["simulationRun"].isin(test_runs)]

# ─── 스케일러 ────────────────────────────────────────────────
scaler = StandardScaler()
X_train = train_df[FEATURE_COLS].values
X_test  = test_df[FEATURE_COLS].values
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
y_train = train_df["label"].values
y_test  = test_df["label"].values

# ─── Isolation Forest ────────────────────────────────────────
normal_mask = train_df["faultNumber"] == 0
IF = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
IF.fit(X_train_s[normal_mask])
if_scores = -IF.score_samples(X_test_s)   # 높을수록 이상
if_preds  = (if_scores > np.percentile(if_scores, 85)).astype(int)

# ─── XGBoost ─────────────────────────────────────────────────
ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
XGB = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=ratio, eval_metric="logloss",
    use_label_encoder=False, random_state=42
)
XGB.fit(X_train_s, y_train)
xgb_proba = XGB.predict_proba(X_test_s)[:, 1]
xgb_preds = (xgb_proba > 0.5).astype(int)

# ─── 결과 데이터프레임 저장 ───────────────────────────────────
result_df = test_df[["time", "simulationRun", "faultNumber", "label"] + BASE_COLS].copy()
result_df["if_score"]   = if_scores
result_df["if_pred"]    = if_preds
result_df["xgb_proba"]  = xgb_proba
result_df["xgb_pred"]   = xgb_preds

result_df.to_parquet("data/result_df.parquet", index=False)
train_df[["simulationRun","faultNumber","label"] + BASE_COLS].to_parquet("data/train_df.parquet", index=False)
test_df[["simulationRun","faultNumber","label","time"] + list(dict.fromkeys(BASE_COLS + FEATURE_COLS))].to_parquet("data/test_df_full.parquet", index=False)

# ─── SHAP 값 계산 ────────────────────────────────────────────
import shap
explainer = shap.TreeExplainer(XGB)
sample_idx = np.random.choice(len(X_test_s), min(500, len(X_test_s)), replace=False)
shap_values = explainer.shap_values(X_test_s[sample_idx])

top_feat_idx = np.abs(shap_values).mean(0).argsort()[::-1][:20]
top_features = [FEATURE_COLS[i] for i in top_feat_idx]
top_shap     = np.abs(shap_values).mean(0)[top_feat_idx].tolist()

np.save("data/shap_values.npy", shap_values[:, top_feat_idx])
with open("data/top_features.json", "w") as f:
    json.dump({"features": top_features, "mean_abs_shap": top_shap}, f, ensure_ascii=False)

# ─── 성능 지표 저장 ──────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

def get_metrics(y_true, y_score, y_pred, name):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "model": name,
        "auroc":  round(roc_auc_score(y_true, y_score), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": round(average_precision_score(y_true, y_score), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "cm": confusion_matrix(y_true, y_pred).tolist()
    }

metrics = [
    get_metrics(y_test, if_scores, if_preds, "Isolation Forest"),
    get_metrics(y_test, xgb_proba, xgb_preds, "XGBoost"),
]

# Fault 별 metrics
fault_metrics = {}
for ft in [1, 2, 4, 5]:
    mask = result_df["faultNumber"].isin([0, ft])
    sub  = result_df[mask]
    y_t  = sub["label"].values
    if len(np.unique(y_t)) < 2: continue
    p, r, f1, _ = precision_recall_fscore_support(y_t, sub["xgb_pred"].values, average="binary", zero_division=0)
    fault_metrics[str(ft)] = {
        "desc": FAULT_DESC[ft],
        "auroc": round(roc_auc_score(y_t, sub["xgb_proba"].values), 4),
        "recall": round(r, 4),
        "f1": round(f1, 4)
    }

# Detection delay (samples after fault onset)
delays = {}
for ft in [1, 2, 4, 5]:
    sub = result_df[result_df["faultNumber"] == ft]
    runs = sub["simulationRun"].unique()
    d_list = []
    for run in runs:
        r_df = sub[sub["simulationRun"] == run].sort_values("time")
        onset = r_df[r_df["label"] == 1].index.min()
        detect = r_df[(r_df["xgb_pred"] == 1) & (r_df.index >= onset)].index.min()
        if pd.notna(onset) and pd.notna(detect):
            d_list.append(detect - onset)
    delays[str(ft)] = int(np.median(d_list)) if d_list else 0

with open("data/metrics.json", "w", encoding="utf-8") as f:
    json.dump({"overall": metrics, "fault": fault_metrics, "delay": delays,
               "feature_cols": FEATURE_COLS, "base_cols": BASE_COLS,
               "fault_desc": {str(k): v for k, v in FAULT_DESC.items()}}, f)

joblib.dump(IF, "models/isolation_forest.pkl")
joblib.dump(XGB, "models/xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")
with open("data/col_info.json", "w") as f:
    json.dump({"feature_cols": FEATURE_COLS, "base_cols": BASE_COLS,
               "xmeas_cols": XMEAS_COLS, "xmv_cols": XMV_COLS}, f)

print("✅ 데모 데이터 생성 완료")
print(f"  train: {train_df.shape}, test: {test_df.shape}")
print(f"  IF  AUROC={metrics[0]['auroc']:.3f}  F1={metrics[0]['f1']:.3f}")
print(f"  XGB AUROC={metrics[1]['auroc']:.3f}  F1={metrics[1]['f1']:.3f}")
