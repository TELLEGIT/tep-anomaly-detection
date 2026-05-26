"""
TEP 공정 이상탐지 대시보드
Tennessee Eastman Process — Isolation Forest + XGBoost + SHAP
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os
from datetime import datetime

# ─── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="TEP 공정 이상탐지 시스템",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 컬러 팔레트 ──────────────────────────────────────────────
C = {
    "bg_dark":    "#F5F7FA",   
    "bg_card":    "#FFFFFF",   
    "bg_panel":   "#EEF2F6",   
    "border":     "#D0D7DE",

    "text":       "#1F2328",
    "text_dim":   "#57606A",

    "normal":     "#1D9E75",
    "fault":      "#E24B4A",
    "xgb":        "#7B4FBF",
    "if_c":       "#378ADD",
    "threshold":  "#EF9F27",
    "shap_pos":   "#E24B4A",
    "shap_neg":   "#378ADD",
    "accent":     "#218BFF",
}
def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# ─── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

.stApp {{
    background: {C['bg_dark']};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {C['text']};
}}

/* 사이드바 */
section[data-testid="stSidebar"] {{
    background: {C['bg_card']} !important;
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: {C['text']} !important;
}}

/* 헤더 */
.dash-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    background: {C['bg_card']};
    padding: 24px;
    border: 1px solid {C['border']};
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
.dash-header .badge {{
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2640 100%);
    border: 1px solid {C['accent']}40;
    color: {C['accent']};
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.dash-header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: {C['text']};
    margin: 0;
    letter-spacing: -0.3px;
}}
.dash-header .sub {{
    font-size: 13px;
    color: {C['text_dim']};
    margin: 0;
    font-weight: 400;
}}

/* KPI 카드 */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}}
.kpi-card {{
    background: #FFFFFF;
    border: 1px solid #D0D7DE;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, {C['accent']});
}}
.kpi-label {{
    font-size: 11px;
    color: {C['text_dim']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--accent-color, {C['text']});
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}}
.kpi-delta {{
    font-size: 12px;
    margin-top: 6px;
    font-weight: 500;
}}
.kpi-delta.up   {{ color: {C['normal']}; }}
.kpi-delta.down {{ color: {C['fault']}; }}
.kpi-delta.neutral {{ color: {C['text_dim']}; }}

/* 섹션 제목 */
.section-title {{
    font-size: 14px;
    font-weight: 600;
    color: {C['text']};
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-title .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {C['accent']};
    display: inline-block;
}}

/* 상태 배지 */
.status-normal {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {C['normal']}18;
    border: 1px solid {C['normal']}40;
    color: {C['normal']};
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px;
}}
.status-fault {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {C['fault']}18;
    border: 1px solid {C['fault']}40;
    color: {C['fault']};
    font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px;
    animation: pulse 1.5s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 {C['fault']}40; }}
    50%       {{ box-shadow: 0 0 0 6px {C['fault']}00; }}
}}

/* 표 */
.metrics-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.metrics-table th {{
    background: {C['bg_panel']};
    color: {C['text_dim']};
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 14px;
    border-bottom: 1px solid {C['border']};
    text-align: left;
}}
.metrics-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid {C['border']}60;
    color: {C['text']};
}}
.metrics-table tr:last-child td {{ border-bottom: none; }}
.metrics-table tr:hover td {{ background: {C['bg_panel']}; }}

/* 알림 박스 */
.alert-box {{
    background: {C['fault']}12;
    border: 1px solid {C['fault']}30;
    border-left: 3px solid {C['fault']};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
}}
.info-box {{
    background: {C['accent']}12;
    border: 1px solid {C['accent']}30;
    border-left: 3px solid {C['accent']};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
}}

/* SHAP 바 */
.shap-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}}
.shap-feat {{
    font-size: 12px;
    color: {C['text']};
    font-family: 'JetBrains Mono', monospace;
    width: 170px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.shap-bar-outer {{
    flex: 1;
    height: 8px;
    background: {C['border']};
    border-radius: 4px;
    overflow: hidden;
}}
.shap-bar-inner {{
    height: 100%;
    border-radius: 4px;
    background: var(--bar-color);
}}
.shap-val {{
    font-size: 11px;
    color: {C['text_dim']};
    font-family: 'JetBrains Mono', monospace;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['bg_card']};
    border-bottom: 1px solid {C['border']};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C['text_dim']} !important;
    font-weight: 500;
    font-size: 13px;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {C['accent']} !important;
    border-bottom-color: {C['accent']} !important;
    background: transparent !important;
}}

/* 버튼 */
.stButton > button {{
    background: {C['bg_panel']};
    color: {C['text']};
    border: 1px solid {C['border']};
    font-size: 13px;
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    border-color: {C['accent']};
    color: {C['accent']};
    background: {C['accent']}12;
}}

/* selectbox */
.stSelectbox > div > div {{
    background: {C['bg_card']};
    border-color: {C['border']};
    color: {C['text']};
}}

/* 구분선 */
hr {{ border-color: {C['border']}; margin: 20px 0; }}

/* 카드 wrapper */
.card {{
    background: #FFFFFF;
    border: 1px solid #D0D7DE;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}

/* 범례 dot */
.legend-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}
</style>
""", unsafe_allow_html=True)

# ─── 데이터 로딩 ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    result_df  = pd.read_parquet(f"{base}/data/result_df.parquet")
    with open(f"{base}/data/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(f"{base}/data/top_features.json") as f:
        shap_info = json.load(f)
    shap_vals = np.load(f"{base}/data/shap_values.npy")
    return result_df, metrics, shap_info, shap_vals

try:
    result_df, metrics, shap_info, shap_vals = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"데이터 로딩 실패: {e}\n\n`python generate_demo_data.py` 를 먼저 실행하세요.")
    st.stop()

# ─── Plotly 공통 레이아웃 ─────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=C["bg_card"],
    plot_bgcolor=C["bg_panel"],
    font=dict(family="Inter, sans-serif", color=C["text"], size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(
        gridcolor=C["border"], gridwidth=0.5,
        linecolor=C["border"], tickcolor=C["border"],
        zerolinecolor=C["border"],
    ),
    yaxis=dict(
        gridcolor=C["border"], gridwidth=0.5,
        linecolor=C["border"], tickcolor=C["border"],
        zerolinecolor=C["border"],
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor=C["bg_card"],
        bordercolor=C["border"],
        font=dict(size=12),
    ),
)

def plot_layout(**overrides):
    layout = PLOT_LAYOUT.copy()
    for key in ("xaxis", "yaxis", "legend", "hoverlabel"):
        if key in overrides and isinstance(overrides[key], dict):
            layout[key] = {**PLOT_LAYOUT.get(key, {}), **overrides.pop(key)}
    layout.update(overrides)
    return layout

fault_desc = {int(k): v for k, v in metrics["fault_desc"].items()}
fault_desc[0] = "정상"
FAULT_COLOR = {0: C["normal"], 1: "#F97316", 2: C["xgb"], 4: C["fault"], 5: C["accent"]}

# ─── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 16px 0 20px 0;">
        <div style="font-size:32px; margin-bottom:6px;">⚗️</div>
        <div style="font-size:14px; font-weight:700; color:{C['text']}; letter-spacing:-0.3px;">TEP 이상탐지 시스템</div>
        <div style="font-size:11px; color:{C['text_dim']}; margin-top:3px;">Tennessee Eastman Process DX</div>
    </div>
    <hr style="margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "📌 페이지 선택",
        ["🏠 시스템 대시보드", "📈 실시간 모니터링", "🔬 모델 성능 분석", "🧠 XAI — SHAP 해석", "⚙️ 공정 변수 탐색"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px; color:{C['text_dim']}; font-weight:600; letter-spacing:0.6px; text-transform:uppercase; margin-bottom:10px;'>필터</div>", unsafe_allow_html=True)

    fault_options = {0:"전체", 1:"Fault 1", 2:"Fault 2", 4:"Fault 4", 5:"Fault 5"}
    sel_fault = st.selectbox("이상 유형", list(fault_options.values()))
    sel_fault_key = [k for k, v in fault_options.items() if v == sel_fault][0]

    runs = sorted(result_df["simulationRun"].unique())
    sel_run = st.selectbox("시뮬레이션 런", runs, index=min(5, len(runs)-1))

    threshold_pct = st.slider("IF 임계값 백분위수 (%)", 70, 99, 85)
    xgb_threshold = st.slider("XGBoost 임계값", 0.1, 0.9, 0.5, step=0.05)

    st.markdown("<hr>", unsafe_allow_html=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div style='font-size:11px; color:{C['text_dim']};'>
        <div style='margin-bottom:5px;'>🕐 {now}</div>
        <div style='margin-bottom:5px;'>📦 테스트 샘플: <span style='color:{C['text']};'>{len(result_df):,}</span></div>
        <div style='margin-bottom:5px;'>🔧 피처 수: <span style='color:{C['text']};'>{len(metrics['feature_cols'])}</span></div>
        <div>🏷️ 이상 유형: <span style='color:{C['text']};'>4종</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── 데이터 필터 ──────────────────────────────────────────────
if sel_fault_key == 0:
    view_df = result_df.copy()
else:
    view_df = result_df[result_df["faultNumber"].isin([0, sel_fault_key])].copy()

run_df = result_df[result_df["simulationRun"] == sel_run].sort_values("time")

# 동적 임계값 적용
if_thresh = np.percentile(result_df["if_score"], threshold_pct)
view_df["if_pred_dyn"] = (view_df["if_score"] > if_thresh).astype(int)
run_df["if_pred_dyn"]  = (run_df["if_score"] > if_thresh).astype(int)
view_df["xgb_pred_dyn"] = (view_df["xgb_proba"] > xgb_threshold).astype(int)
run_df["xgb_pred_dyn"]  = (run_df["xgb_proba"] > xgb_threshold).astype(int)

# 기본 metrics
m_if  = metrics["overall"][0]
m_xgb = metrics["overall"][1]

# ══════════════════════════════════════════════════════════════
# PAGE 1 — 시스템 대시보드
# ══════════════════════════════════════════════════════════════
if "대시보드" in page:
    # 헤더
    st.markdown(f"""
    <div class="dash-header">
        <div>
            <p style='margin:0 0 3px 0;'><span class="badge">LIVE DEMO</span></p>
            <h1>TEP 공정 이상탐지 대시보드</h1>
            <p class="sub">Tennessee Eastman Process · IF + XGBoost + SHAP · 공정 DX</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI ──────────────────────────────────────────────────
    total = len(view_df)
    n_fault_detected = view_df["xgb_pred_dyn"].sum()
    n_true_fault = view_df["label"].sum()

    tp_live = ((view_df["label"] == 1) & (view_df["xgb_pred_dyn"] == 1)).sum()
    recall_live = tp_live / max(n_true_fault, 1)
    xgb_auroc = m_xgb["auroc"]

    kpi_data = [
        ("XGBoost AUROC",  f"{xgb_auroc:.3f}",  C["xgb"],      f"▲ IF 대비 +{xgb_auroc - m_if['auroc']:.3f}", "up"),
        ("탐지된 이상 수",  f"{n_fault_detected:,}", C["fault"],  f"실제 이상: {n_true_fault:,}건", "neutral"),
        ("실시간 Recall",  f"{recall_live:.1%}",  C["normal"],   f"F1 = {m_xgb['f1']:.3f}", "up"),
        ("모니터링 변수",  "33",                  C["accent"],   "XMEAS 22 + XMV 11", "neutral"),
    ]

    cols = st.columns(4)
    for col, (label, val, color, delta, delta_cls) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-delta {delta_cls}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 상태 배지 ──────────────────────────────────────────
    run_fault = run_df["faultNumber"].iloc[-1] if len(run_df) > 0 else 0
    run_pred  = run_df["xgb_pred_dyn"].iloc[-1] if len(run_df) > 0 else 0
    if run_pred == 1:
        status_html = f'<div class="status-fault">🔴 이상 감지됨 — Run {int(sel_run)} | Fault {int(run_fault)}: {fault_desc.get(int(run_fault),"Unknown")}</div>'
    else:
        status_html = f'<div class="status-normal">🟢 정상 운전 중 — Run {int(sel_run)}</div>'
    st.markdown(status_html, unsafe_allow_html=True)
    st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

    # ── 행 1: Anomaly Score 시계열 + Fault 분포 ──────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<p class="section-title"><span class="dot"></span>Anomaly Score 시계열 (선택 런)</p>', unsafe_allow_html=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.45],
                            vertical_spacing=0.05)

        # Fault 구간 음영
        fault_start = run_df[run_df["label"] == 1]["time"].min() if (run_df["label"] == 1).any() else None
        if fault_start is not None:
            for r in [1, 2]:
                fig.add_vrect(x0=fault_start, x1=run_df["time"].max(),
                              fillcolor=C["fault"], opacity=0.07,
                              line_width=0, row=r, col=1)
                fig.add_vline(x=fault_start, line_dash="dash",
                              line_color=C["fault"], line_width=1,
                              row=r, col=1)

        # IF score
        fig.add_trace(go.Scatter(
            x=run_df["time"], y=run_df["if_score"],
            name="IF Anomaly Score",
            line=dict(color=C["if_c"], width=1.5),
            fill="tozeroy", fillcolor="rgba(55, 138, 221, 0.07)",
        ), row=1, col=1)
        fig.add_hline(y=if_thresh,
                      line_dash="dot", line_color=C["threshold"], line_width=1.5,
                      row=1, col=1)

        # XGBoost proba
        fig.add_trace(go.Scatter(
            x=run_df["time"], y=run_df["xgb_proba"],
            name="XGB 이상 확률",
            line=dict(color=C["xgb"], width=2),
            fill="tozeroy", fillcolor="rgba(123, 79, 191, 0.09)",
        ), row=2, col=1)
        fig.add_hline(y=xgb_threshold,
                      line_dash="dot", line_color=C["threshold"], line_width=1.5,
                      row=2, col=1)

        fig.update_layout(**PLOT_LAYOUT,
                          height=320,
                          title=dict(text=f"Run {int(sel_run)} | {fault_desc.get(int(run_fault), '알 수 없음')}",
                                     font=dict(size=12), x=0))
        fig.update_yaxes(title_text="IF Score", row=1, col=1,
                         gridcolor=C["border"], tickfont=dict(size=10))
        fig.update_yaxes(title_text="XGB Prob", row=2, col=1, range=[0, 1],
                         gridcolor=C["border"], tickfont=dict(size=10))
        fig.update_xaxes(title_text="Time Step", row=2, col=1,
                         gridcolor=C["border"], tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<p class="section-title"><span class="dot"></span>이상 유형별 탐지 성능</p>', unsafe_allow_html=True)

        fault_m = metrics.get("fault", {})
        ft_keys = [k for k in ["1", "2", "4", "5"] if k in fault_m]

        fig2 = go.Figure()
        fts    = [f"Fault {k}" for k in ft_keys]
        aurocs  = [fault_m[k]["auroc"] for k in ft_keys]
        recalls = [fault_m[k]["recall"] for k in ft_keys]
        f1s     = [fault_m[k]["f1"] for k in ft_keys]
        colors  = [FAULT_COLOR.get(int(k), C["accent"]) for k in ft_keys]

        fig2.add_trace(go.Bar(
            name="AUROC", x=fts, y=aurocs,
            marker_color=[hex_to_rgba(c, 0.80) for c in colors],
            marker_line_color=colors, marker_line_width=1.5,
            text=[f"{v:.3f}" for v in aurocs],
            textposition="outside", textfont=dict(size=11),
        ))
        fig2.add_trace(go.Scatter(
            name="Recall", x=fts, y=recalls,
            mode="markers+lines",
            marker=dict(color=C["normal"], size=8, symbol="diamond"),
            line=dict(color=C["normal"], width=1.5, dash="dot"),
        ))
        fig2.update_layout(
            height=320,
            title=dict(text="Fault별 XGBoost 성능", font=dict(size=12), x=0),
            barmode="group"
        )
        fig2.update_layout(
            paper_bgcolor=C["bg_card"],
            plot_bgcolor=C["bg_panel"],
            font=dict(family="Inter, sans-serif", color=C["text"], size=12),
            margin=dict(l=10, r=10, t=36, b=10),
            legend=PLOT_LAYOUT["legend"],
            hoverlabel=PLOT_LAYOUT["hoverlabel"],
        )
        fig2.update_yaxes(range=[0, 1.12], **PLOT_LAYOUT["yaxis"])
        fig2.update_xaxes(**PLOT_LAYOUT["xaxis"])
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── 행 2: Confusion Matrix + Detection Delay + 모델 비교 ──
    c3, c4, c5 = st.columns(3)

    with c3:
        st.markdown('<p class="section-title"><span class="dot"></span>XGBoost Confusion Matrix</p>', unsafe_allow_html=True)
        cm_data = np.array(m_xgb["cm"])
        cm_labels = ["정상", "이상"]
        cm_text = [[f"<b>{v}</b><br>{'TN' if i==0 and j==0 else 'FP' if i==0 else 'FN' if j==0 else 'TP'}"
                    for j, v in enumerate(row)] for i, row in enumerate(cm_data)]

        fig3 = go.Figure(go.Heatmap(
            z=cm_data, x=cm_labels, y=cm_labels,
            text=cm_text, texttemplate="%{text}",
            colorscale=[[0, C["bg_panel"]], [1, hex_to_rgba(C["xgb"], 0.50)]],
            showscale=False, textfont=dict(size=13),
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=240)
        fig3.update_xaxes(title_text="예측")
        fig3.update_yaxes(title_text="실제")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with c4:
        st.markdown('<p class="section-title"><span class="dot"></span>Detection Delay (중앙값)</p>', unsafe_allow_html=True)
        delays = metrics.get("delay", {})
        ft_d   = [f"F{k}" for k in delays]
        delay_v = list(delays.values())

        delay_colors = [C["fault"], C["xgb"], C["threshold"], C["accent"]]

        fig4 = go.Figure(go.Bar(
            x=ft_d,
            y=delay_v,
            marker_color=[delay_colors[i % len(delay_colors)] for i in range(len(delay_v))],
            text=[f"{v}s" for v in delay_v],
            textposition="outside",
            textfont=dict(size=12, color=C["text"]),
        ))

        fig4.update_layout(**plot_layout(
            height=240,
            yaxis=dict(title="탐지 지연 샘플 수"),
            title=dict(text="이상 발생 → 첫 탐지 지연", font=dict(size=12), x=0),
        ))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    with c5:
        st.markdown('<p class="section-title"><span class="dot"></span>IF vs XGBoost 성능 비교</p>', unsafe_allow_html=True)
        metrics_keys = ["auroc", "pr_auc", "precision", "recall", "f1"]
        m_keys_label = ["AUROC", "PR-AUC", "Precision", "Recall", "F1"]

        if_vals  = [m_if[k] for k in metrics_keys]
        xgb_vals = [m_xgb[k] for k in metrics_keys]

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Isolation Forest", x=m_keys_label, y=if_vals,
            marker_color=hex_to_rgba(C["if_c"], 0.60),
            marker_line_color=C["if_c"], marker_line_width=1.5,
        ))
        fig5.add_trace(go.Bar(
            name="XGBoost", x=m_keys_label, y=xgb_vals,
            marker_color=hex_to_rgba(C["xgb"], 0.60),
            marker_line_color=C["xgb"], marker_line_width=1.5,
        ))
        fig5.update_layout(**plot_layout(
            height=240,
            yaxis=dict(range=[0, 1.05]),
            barmode="group",
        ))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# PAGE 2 — 실시간 모니터링
# ══════════════════════════════════════════════════════════════
elif "모니터링" in page:
    st.markdown(f"""
    <div class="dash-header">
        <h1>📈 실시간 공정 모니터링</h1>
        <p class="sub">공정 변수 시계열 · Anomaly Score · 이상 구간 표시</p>
    </div>
    """, unsafe_allow_html=True)

    base_cols = metrics.get("base_cols", [f"xmeas_{i}" for i in range(1, 23)] + [f"xmv_{i}" for i in range(1, 12)])
    available_vars = [c for c in base_cols if c in run_df.columns]

    col_l, col_r = st.columns([1, 3])
    with col_l:
        selected_vars = st.multiselect(
            "모니터링 변수 선택",
            available_vars,
            default=available_vars[:4],
            max_selections=8,
        )
        show_fault_zone = st.checkbox("이상 구간 표시", value=True)
        show_rolling    = st.checkbox("Rolling Mean 오버레이", value=False)

    with col_r:
        if not selected_vars:
            st.info("왼쪽에서 변수를 선택하세요.")
        else:
            n = len(selected_vars)
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                                vertical_spacing=0.02)

            fault_start = run_df[run_df["label"] == 1]["time"].min() if (run_df["label"] == 1).any() else None

            for i, var in enumerate(selected_vars, 1):
                if var not in run_df.columns:
                    continue
                color = list(FAULT_COLOR.values())[i % len(FAULT_COLOR)]
                fig.add_trace(go.Scatter(
                    x=run_df["time"], y=run_df[var],
                    name=var, line=dict(color=color, width=1.5),
                    hovertemplate=f"<b>{var}</b><br>t=%{{x}}<br>v=%{{y:.4f}}<extra></extra>"
                ), row=i, col=1)

                if show_rolling and f"{var}_rmean" in run_df.columns:
                    fig.add_trace(go.Scatter(
                        x=run_df["time"], y=run_df[f"{var}_rmean"],
                        name=f"{var} rmean", showlegend=False,
                        line=dict(color=color, width=1, dash="dot"),
                    ), row=i, col=1)

                if show_fault_zone and fault_start is not None:
                    fig.add_vrect(x0=fault_start, x1=run_df["time"].max(),
                                  fillcolor=C["fault"], opacity=0.07,
                                  line_width=0, row=i, col=1)

                fig.update_yaxes(title_text=var, row=i, col=1,
                                 gridcolor=C["border"], tickfont=dict(size=9),
                                 title_font=dict(size=10))

            fig.update_xaxes(title_text="Time Step", row=n, col=1,
                             gridcolor=C["border"])
            fig.update_layout(**PLOT_LAYOUT, height=max(180 * n, 300),
                              title=dict(text=f"Run {int(sel_run)} 공정 변수 시계열", font=dict(size=13), x=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 하단: 이상 구간 상세
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="dot"></span>이상 탐지 상세 — Anomaly Score & 예측</p>', unsafe_allow_html=True)

    fig6 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5], vertical_spacing=0.06)

    fault_start = run_df[run_df["label"] == 1]["time"].min() if (run_df["label"] == 1).any() else None
    if fault_start is not None:
        for r in [1, 2]:
            fig6.add_vrect(x0=fault_start, x1=run_df["time"].max(),
                          fillcolor=C["fault"], opacity=0.08,
                          line_width=0, row=r, col=1)

    fig6.add_trace(go.Scatter(
        x=run_df["time"], y=run_df["if_score"],
        name="IF Anomaly Score",
        line=dict(color=C["if_c"], width=1.8),
        fill="tozeroy", fillcolor="rgba(55, 138, 221, 0.08)"
    ), row=1, col=1)
    fig6.add_hline(y=if_thresh, line_dash="dot",
                   line_color=C["threshold"], line_width=1.5, row=1, col=1)
    fig6.add_annotation(x=run_df["time"].max(), y=if_thresh,
                        text=f"THR={if_thresh:.3f}", showarrow=False,
                        font=dict(size=10, color=C["threshold"]),
                        xanchor="right", row=1, col=1)

    fig6.add_trace(go.Scatter(
        x=run_df["time"], y=run_df["xgb_proba"],
        name="XGB P(이상)",
        line=dict(color=C["xgb"], width=2),
        fill="tozeroy", fillcolor="rgba(123, 79, 191, 0.09)"
    ), row=2, col=1)
    fig6.add_hline(y=xgb_threshold, line_dash="dot",
                   line_color=C["threshold"], line_width=1.5, row=2, col=1)

    # 탐지 지점 마커
    detected = run_df[run_df["xgb_pred_dyn"] == 1]
    if len(detected) > 0:
        fig6.add_trace(go.Scatter(
            x=detected["time"], y=detected["xgb_proba"],
            name="탐지됨", mode="markers",
            marker=dict(color=C["fault"], size=4, symbol="circle"),
        ), row=2, col=1)

    fig6.update_layout(**plot_layout(height=320))
    fig6.update_yaxes(title_text="IF Score", row=1, col=1, gridcolor=C["border"])
    fig6.update_yaxes(title_text="XGB Prob", row=2, col=1, range=[0, 1], gridcolor=C["border"])
    fig6.update_xaxes(title_text="Time Step", row=2, col=1, gridcolor=C["border"])
    st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# PAGE 3 — 모델 성능 분석
# ══════════════════════════════════════════════════════════════
elif "성능" in page:
    st.markdown(f"""
    <div class="dash-header">
        <h1>🔬 모델 성능 분석</h1>
        <p class="sub">ROC Curve · PR Curve · Confusion Matrix · Fault별 분석</p>
    </div>
    """, unsafe_allow_html=True)

    # 성능 지표 테이블
    st.markdown('<p class="section-title"><span class="dot"></span>전체 성능 요약</p>', unsafe_allow_html=True)

    headers = ["모델", "AUROC", "PR-AUC", "Precision", "Recall", "F1"]
    rows_html = ""
    for m in [m_if, m_xgb]:
        name_color = C["if_c"] if "Forest" in m["model"] else C["xgb"]
        rows_html += f"""
        <tr>
            <td><span style='color:{name_color}; font-weight:600;'>{m['model']}</span></td>
            <td><b>{m['auroc']:.4f}</b></td>
            <td>{m['pr_auc']:.4f}</td>
            <td>{m['precision']:.4f}</td>
            <td>{m['recall']:.4f}</td>
            <td>{m['f1']:.4f}</td>
        </tr>"""
    st.markdown(f"""
    <div class="card">
    <table class="metrics-table">
        <thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ROC + PR Curve
    c_roc, c_pr = st.columns(2)

    with c_roc:
        st.markdown('<p class="section-title"><span class="dot"></span>ROC Curve (재현)</p>', unsafe_allow_html=True)
        from sklearn.metrics import roc_curve, precision_recall_curve

        y_true = view_df["label"].values
        fig_roc = go.Figure()

        for model_name, scores, color in [
            ("Isolation Forest", view_df["if_score"].values, C["if_c"]),
            ("XGBoost",          view_df["xgb_proba"].values, C["xgb"]),
        ]:
            if len(np.unique(y_true)) < 2: continue
            fpr, tpr, _ = roc_curve(y_true, scores)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, scores)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{model_name} (AUC={auc:.3f})",
                line=dict(color=color, width=2),
            ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color=C["text_dim"], width=1, dash="dash"),
        ))
        fig_roc.update_layout(**plot_layout(
            height=300,
            xaxis=dict(title="FPR"),
            yaxis=dict(title="TPR", range=[0, 1.02]),
            title=dict(text="ROC Curve", font=dict(size=12), x=0),
        ))
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

    with c_pr:
        st.markdown('<p class="section-title"><span class="dot"></span>PR Curve (재현)</p>', unsafe_allow_html=True)
        fig_pr = go.Figure()

        for model_name, scores, color in [
            ("Isolation Forest", view_df["if_score"].values, C["if_c"]),
            ("XGBoost",          view_df["xgb_proba"].values, C["xgb"]),
        ]:
            if len(np.unique(y_true)) < 2: continue
            prec, rec, _ = precision_recall_curve(y_true, scores)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, scores)
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, name=f"{model_name} (AP={ap:.3f})",
                line=dict(color=color, width=2),
            ))

        fig_pr.update_layout(**plot_layout(
            height=300,
            xaxis=dict(title="Recall"),
            yaxis=dict(title="Precision", range=[0, 1.02]),
            title=dict(text="PR Curve", font=dict(size=12), x=0),
        ))
        st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

    # Fault별 성능 히트맵
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="dot"></span>Fault 유형별 XGBoost 성능</p>', unsafe_allow_html=True)

    fault_m = metrics.get("fault", {})
    ft_keys = [k for k in ["1", "2", "4", "5"] if k in fault_m]

    heat_data = {
        "AUROC":   [fault_m[k]["auroc"]  for k in ft_keys],
        "Recall":  [fault_m[k]["recall"] for k in ft_keys],
        "F1":      [fault_m[k]["f1"]     for k in ft_keys],
    }
    ft_labels = [f"Fault {k}\n({fault_m[k]['desc']})" for k in ft_keys]

    fig_heat = go.Figure(go.Heatmap(
        z=list(heat_data.values()),
        x=ft_labels, y=list(heat_data.keys()),
        text=[[f"{v:.3f}" for v in row] for row in heat_data.values()],
        texttemplate="%{text}",
        colorscale=[
            [0,   hex_to_rgba(C["fault"], 0.38)],
            [0.5, hex_to_rgba(C["xgb"], 0.38)],
            [1,   hex_to_rgba(C["normal"], 0.67)],
        ],
        zmin=0, zmax=1,
        showscale=True,
        textfont=dict(size=13, color="white"),
    ))
    fig_heat.update_layout(**plot_layout(height=200))
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# PAGE 4 — SHAP 해석
# ══════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown(f"""
    <div class="dash-header">
        <h1>🧠 XAI — SHAP 해석</h1>
        <p class="sub">SHapley Additive exPlanations · 변수 중요도 · 공정 도메인 해석</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b>SHAP 해석</b>은 모델이 어떤 공정 변수를 근거로 이상을 판단했는지 보여줍니다.
        이를 통해 단순히 “이상 발생 여부”만 확인하는 것이 아니라,
        이상 판단에 크게 영향을 준 변수와 공정 변화 패턴까지 함께 파악할 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

    top_feats = shap_info["features"][:15]
    mean_shap = shap_info["mean_abs_shap"][:15]
    max_shap  = max(mean_shap) if mean_shap else 1

    # SHAP 바 차트
    tab1, tab2 = st.tabs(["📊 평균 |SHAP| 순위", "🔥 SHAP Beeswarm (Plotly 재현)"])

    with tab1:
        col_shap, col_info = st.columns([3, 2])

        with col_shap:
            st.markdown('<p class="section-title"><span class="dot"></span>Top 15 Feature Importance (SHAP)</p>', unsafe_allow_html=True)

            fig_shap = go.Figure()
            feat_type_colors = []
            for f in top_feats:
                if "_rmean" in f or "_rstd" in f or "_diff" in f or "_zscore" in f:
                    feat_type_colors.append(C["xgb"])
                elif "xmeas" in f:
                    feat_type_colors.append(C["if_c"])
                else:
                    feat_type_colors.append(C["normal"])

            fig_shap.add_trace(go.Bar(
                x=mean_shap[::-1],
                y=top_feats[::-1],
                orientation="h",
                marker_color=feat_type_colors[::-1],
                marker_line_color=feat_type_colors[::-1],
                marker_line_width=1,
                text=[f"{v:.4f}" for v in mean_shap[::-1]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_shap.update_layout(**plot_layout(
                height=420,
                xaxis=dict(title="Mean |SHAP Value|"),
                yaxis=dict(tickfont=dict(size=10, family="JetBrains Mono")),
                title=dict(text="XGBoost Feature Importance by SHAP", font=dict(size=12), x=0),
            ))
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

        with col_info:
            st.markdown('<p class="section-title"><span class="dot"></span>피처 유형 분류</p>', unsafe_allow_html=True)

            type_counts = {"동적 피처\n(rolling/diff/zscore)": 0, "XMEAS\n(측정변수)": 0, "XMV\n(조작변수)": 0}
            for f in top_feats:
                if "_rmean" in f or "_rstd" in f or "_diff" in f or "_zscore" in f:
                    type_counts["동적 피처\n(rolling/diff/zscore)"] += 1
                elif "xmeas" in f:
                    type_counts["XMEAS\n(측정변수)"] += 1
                else:
                    type_counts["XMV\n(조작변수)"] += 1

            fig_pie = go.Figure(go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                hole=0.5,
                marker_colors=[C["xgb"], C["if_c"], C["normal"]],
                textfont=dict(size=11),
            ))
            fig_pie.update_layout(**plot_layout(
                height=200,
                showlegend=True,
                legend=dict(font=dict(size=10)),
            ))
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div class="card" style="margin-top:0;">
                <div style="font-size:12px; color:{C['text_dim']}; margin-bottom:10px; font-weight:600;">피처 색상 범례</div>
                <div style="font-size:12px; margin-bottom:7px;">
                    <span class="legend-dot" style="background:{C['xgb']};"></span>
                    동적 피처 (rolling/diff/zscore)
                </div>
                <div style="font-size:12px; margin-bottom:7px;">
                    <span class="legend-dot" style="background:{C['if_c']};"></span>
                    XMEAS — 공정 측정변수 (온도, 압력, 유량 등)
                </div>
                <div style="font-size:12px; margin-bottom:12px;">
                    <span class="legend-dot" style="background:{C['normal']};"></span>
                    XMV — 조작변수 (밸브, 속도 등)
                </div>
                <div style="font-size:11px; color:{C['text_dim']}; line-height:1.6;">
                    💡 <b>동적 피처</b>가 상위에 많을수록, 공정의 <b>시계열 변화 패턴</b>이 이상탐지에 중요함을 의미합니다.
                    단순 측정값보다 <b>rolling std · diff</b>가 이상을 먼저 포착합니다.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-title"><span class="dot"></span>SHAP Value 분포 (Violin)</p>', unsafe_allow_html=True)
        # shap_vals shape: (n_samples, top_k)
        n_show = min(10, shap_vals.shape[1])

        fig_bee = go.Figure()
        for i in range(n_show):
            feat_name = top_feats[i]
            vals = shap_vals[:, i]
            color = C["xgb"] if any(s in feat_name for s in ["_rmean","_rstd","_diff","_zscore"]) \
                    else C["if_c"] if "xmeas" in feat_name else C["normal"]

            fig_bee.add_trace(go.Violin(
                x=[feat_name] * len(vals),
                y=vals,
                name=feat_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=hex_to_rgba(color, 0.18),
                line_color=color,
                showlegend=False,
            ))

        fig_bee.add_hline(y=0, line_dash="dash",
                          line_color=C["text_dim"], line_width=1)
        fig_bee.update_layout(**plot_layout(
            height=380,
            xaxis=dict(tickangle=-30, tickfont=dict(size=9, family="JetBrains Mono")),
            yaxis=dict(title="SHAP Value → 이상 방향 (+)"),
            title=dict(text="각 피처의 SHAP 분포 (양수=이상 기여, 음수=정상 기여)",
                       font=dict(size=12), x=0),
        ))
        st.plotly_chart(fig_bee, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# PAGE 5 — 공정 변수 탐색
# ══════════════════════════════════════════════════════════════
elif "공정 변수" in page:
    st.markdown(f"""
    <div class="dash-header">
        <h1>⚙️ 공정 변수 탐색 (EDA)</h1>
        <p class="sub">정상 vs 이상 분포 비교 · 상관관계 · 변수 간 산점도</p>
    </div>
    """, unsafe_allow_html=True)

    base_cols = metrics.get("base_cols", [])
    available = [c for c in base_cols if c in view_df.columns]

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        var_x = st.selectbox("변수 X", available, index=0)
    with c_sel2:
        var_y = st.selectbox("변수 Y", available, index=min(3, len(available)-1))

    col_dist, col_scatter = st.columns(2)

    with col_dist:
        st.markdown('<p class="section-title"><span class="dot"></span>정상 vs 이상 분포 비교</p>', unsafe_allow_html=True)
        normal_vals = view_df[view_df["label"] == 0][var_x].dropna()
        fault_vals  = view_df[view_df["label"] == 1][var_x].dropna()

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=normal_vals, name="정상",
            marker_color=hex_to_rgba(C["normal"], 0.53),
            nbinsx=40,
            histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=fault_vals, name="이상",
            marker_color=hex_to_rgba(C["fault"], 0.53),
            nbinsx=40,
            histnorm="probability density",
        ))
        fig_dist.update_layout(**plot_layout(
            height=280,
            barmode="overlay",
            xaxis=dict(title=var_x),
            yaxis=dict(title="밀도"),
            title=dict(text=f"{var_x} 분포", font=dict(size=12), x=0),
        ))
        fig_dist.update_traces(opacity=0.75)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    with col_scatter:
        st.markdown('<p class="section-title"><span class="dot"></span>변수 간 산점도</p>', unsafe_allow_html=True)
        sample_df = view_df.sample(min(2000, len(view_df)), random_state=42)

        fig_sc = go.Figure()
        for label, color, name in [(0, C["normal"], "정상"), (1, C["fault"], "이상")]:
            sub = sample_df[sample_df["label"] == label]
            if len(sub) == 0: continue
            fig_sc.add_trace(go.Scatter(
                x=sub[var_x], y=sub[var_y],
                mode="markers", name=name,
                marker=dict(color=hex_to_rgba(color, 0.53), size=3),
            ))
        fig_sc.update_layout(**plot_layout(
            height=280,
            xaxis=dict(title=var_x),
            yaxis=dict(title=var_y),
            title=dict(text=f"{var_x} vs {var_y}", font=dict(size=12), x=0),
        ))
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # 상관관계 히트맵
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="dot"></span>핵심 변수 상관관계 (정상 데이터 기준)</p>', unsafe_allow_html=True)

    hmap_vars = available[:12]
    normal_sub = view_df[view_df["label"] == 0][hmap_vars].dropna()
    corr_mat = normal_sub.corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale=[
            [0,   hex_to_rgba(C["fault"], 0.85)],
            [0.5, C["bg_panel"]],
            [1,   hex_to_rgba(C["normal"], 0.85)],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
        texttemplate="%{text}",
        showscale=True,
        textfont=dict(size=8),
    ))
    fig_corr.update_layout(**plot_layout(
        height=380,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        title=dict(text="Pearson Correlation (정상 운전 구간)",
                   font=dict(size=12), x=0),
    ))
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

# ─── 푸터 ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 24px 0 8px; border-top: 1px solid {C['border']}; margin-top: 24px;">
    <div style="font-size:11px; color:{C['text_dim']};">
        ⚗️ TEP 공정 이상탐지 시스템 · IF + XGBoost + SHAP · 공정 DX 포트폴리오
    </div>
</div>
""", unsafe_allow_html=True)
