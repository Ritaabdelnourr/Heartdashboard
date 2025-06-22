# app.py ────────────────────────────────────────────────────────────────
"""
Heart-Disease Dashboard  ·  Streamlit 1-file version
– Adds tabs, richer analytics, coordinated views, and a multivariate
  bleeding-risk model with a live threshold slider.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix
)

# ───────────────────────────────────────────────────────────────────────
#  0.  PAGE CONFIG & PALETTE
# ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart-Disease Dashboard",
    page_icon="🚑",
    layout="wide"
)
px.defaults.template = "plotly_white"
DARK, LIGHT = "#1f77b4", "#aec7e8"

# ───────────────────────────────────────────────────────────────────────
#  1.  LOAD / CLEAN
# ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "heart_disease_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Dates
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y",
                                errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Quick one-hot for binary yes/no columns
    yn = {"yes": 1, "no": 0}
    for c in ["Smoker", "HTN", "Bleeding"]:
        df[c + "_Num"] = (
            df[c].astype(str).str.strip().str.lower().map(yn)
        )

    # Extra helpers
    df["Year"]   = df["Date"].dt.year
    df["Month"]  = df["Date"].dt.to_period("M").astype(str)
    df["AgeBin"] = pd.cut(
        df["Age"].astype(float),
        bins=[0, 49, 59, 69, 79, 120],
        labels=["<50", "50-59", "60-69", "70-79", "80+"]
    )

    # Keep rows we can analyse
    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num"
    ])
    return df

df = load_data()

# ───────────────────────────────────────────────────────────────────────
#  2.  SIDEBAR GLOBAL FILTERS
# ───────────────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Global filters")

yrs = sorted(df["Year"].unique())
y_from, y_to = st.sidebar.slider("Year range", int(yrs[0]), int(yrs[-1]),
                                 (int(yrs[0]), int(yrs[-1])))

areas = sorted(df["Residence"].unique())
sel_area = st.sidebar.multiselect("Residence", areas, default=areas)

genders = sorted(df["Sex"].unique())
sel_gender = st.sidebar.multiselect("Gender", genders, default=genders)

smk_opts = ["yes", "no"]
sel_smoke = st.sidebar.multiselect("Smoker?", smk_opts, default=smk_opts)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
a_from, a_to = st.sidebar.slider("Age range", age_min, age_max,
                                 (age_min, age_max))

filtered = df[
    (df["Year"].between(y_from, y_to)) &
    (df["Residence"].isin(sel_area)) &
    (df["Sex"].isin(sel_gender)) &
    (df["Smoker"].isin(sel_smoke)) &
    (df["Age"].between(a_from, a_to))
].copy()

# ───────────────────────────────────────────────────────────────────────
#  3.  KPI HEADER
# ───────────────────────────────────────────────────────────────────────
st.title("🚑 Heart-Disease Dashboard")
st.markdown("#### Open-Heart Surgeries Cohort")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Patients", f"{len(filtered):,}")
k2.metric("Smokers %", f"{filtered['Smoker_Num'].mean()*100:0.1f}")
k3.metric("HTN %", f"{filtered['HTN_Num'].mean()*100:0.1f}")
k4.metric("Bleeding %", f"{filtered['Bleeding_Num'].mean()*100:0.1f}")

# ───────────────────────────────────────────────────────────────────────
#  4.  TABBED LAYOUT
# ───────────────────────────────────────────────────────────────────────
tab_over, tab_demo, tab_risk, tab_raw = st.tabs(
    ["Overview", "Demographics", "Risk model", "Raw data"]
)

# ══════════════════════════════════════════════════════════════════════
#  4-A  OVERVIEW  – trend line + quick distribution
# ══════════════════════════════════════════════════════════════════════
with tab_over:
    st.subheader("📈 Monthly surgery volume")
    vol = (
        filtered.groupby("Month")["Bleeding_Num"]
        .count()
        .reset_index(name="Surgeries")
    )
    fig = px.line(vol, x="Month", y="Surgeries", markers=True,
                  color_discrete_sequence=[DARK])
    fig.update_layout(yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  4-B  DEMOGRAPHICS  – stacked bar, age dist, geography, HTN×Age heat
# ══════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("👥 Demographic & clinical breakdown")
    c11, c12 = st.columns(2)
    c21, c22 = st.columns(2)

    # Smoking×Sex
    smoke_sex = (
        filtered.groupby(["Sex", "Smoker_Num"])
        .size()
        .reset_index(name="Count")
    )
    fig1 = px.bar(
        smoke_sex, x="Sex", y="Count", color="Smoker_Num",
        color_discrete_map={1: DARK, 0: LIGHT},
        barmode="stack", title="Smoking by sex"
    )
    c11.plotly_chart(fig1, use_container_width=True)

    # Age histogram
    fig2 = px.histogram(
        filtered, x="Age", nbins=20, color_discrete_sequence=[DARK],
        title="Age distribution"
    )
    c12.plotly_chart(fig2, use_container_width=True)

    # Surgeries by area (click → sets a session filter)
    if "clicked_area" not in st.session_state:
        st.session_state.clicked_area = None

    counts = (
        filtered["Residence"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Residence", "Residence": "Count"})
    )
    fig3 = px.bar(counts, x="Residence", y="Count",
                  title="Surgeries by residence")
    fig3.update_layout(xaxis_tickangle=-45)
    sel = c21.plotly_chart(
        fig3, use_container_width=True, click_data=True, key="res_bar"
    )
    if sel and sel.selected_data:
        st.session_state.clicked_area = sel.selected_data["points"][0]["x"]

    # Heat-map AgeBin × HTN
    heat = (
        filtered.groupby(["AgeBin", "HTN_Num"])
        .size()
        .reset_index(name="Patients")
    )
    fig4 = px.density_heatmap(
        heat, x="AgeBin", y="HTN_Num", z="Patients",
        color_continuous_scale="Blues",
        title="HTN prevalence across age bands",
        labels={"HTN_Num": "HTN (0=no, 1=yes)"}
    )
    c22.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  4-C  RISK MODEL  – multivariate logistic, ROC, threshold slider
# ══════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("🤖 Predicting post-op bleeding")

    use_df = filtered.dropna(subset=["Smoker_Num", "HTN_Num", "Age"])
    if len(use_df) < 60:
        st.info("Not enough rows in current filter to train a model.")
    else:
        X = pd.get_dummies(
            use_df[["Age", "Sex", "Smoker_Num", "HTN_Num"]],
            drop_first=True
        )
        y = use_df["Bleeding_Num"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, stratify=y, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        roc_auc = auc(fpr, tpr)

        st.metric("AUC", f"{roc_auc:0.2f}")

        # ROC plot
        roc_fig = px.area(
            x=fpr, y=tpr,
            labels={"x": "False positive rate",
                    "y": "True positive rate"},
        )
        roc_fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(roc_fig, use_container_width=True)

        # Threshold slider
        thr = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
        y_pred = (y_proba >= thr).astype(int)
        acc = accuracy_score(y_te, y_pred)
        st.write(f"**Accuracy @ {thr:0.2f}:** {acc:0.2f}")

        # Confusion matrix (heat-map style)
        cm = confusion_matrix(y_te, y_pred, labels=[1, 0])
        cm_fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=["Pred yes", "Pred no"],
            y=["True yes", "True no"],
            color_continuous_scale="Blues",
            title="Confusion matrix"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  4-D  RAW DATA + DOWNLOAD
# ══════════════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("🗂️ Filtered data")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_heart_disease.csv"
    )
