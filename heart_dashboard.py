import streamlit as st
import pandas as pd
import plotly.express as px

# ─────────  PAGE SET-UP  ───────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
TITLE_COL, *_ = st.columns([0.35, 0.22, 0.22, 0.22])
with TITLE_COL:
    st.write("### 🚑 Heart-Disease Dashboard — Open-Heart Surgeries")

# ─────────  LOAD & PREP  ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col + "_Num"] = (
            df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        )
    return df.dropna(subset=[
        "Sex","Age","Residence",
        "Smoker_Num","HTN_Num","Bleeding_Num","Year"
    ])
df = load_data()

# ─────────  FILTER BAR  ────────────────────────────────────────────────
with st.container():
    fc1, fc2 = st.columns([0.5, 0.5], gap="small")
    yrs = sorted(df["Year"].unique())
    with fc1:
        yr_start, yr_end = st.slider("Year range", int(yrs[0]), int(yrs[-1]),
                                     (int(yrs[0]), int(yrs[-1])))
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    with fc2:
        a_from, a_to = st.slider("Age range", a_min, a_max, (a_min, a_max))

    fc3, fc4, fc5, fc6 = st.columns(4, gap="small")
    with fc3:
        areas = sorted(df["Residence"].unique())
        sel_area = st.multiselect("Residence", areas, default=areas)
    with fc4:
        genders = sorted(df["Sex"].unique())
        sel_gender = st.multiselect("Gender", genders, default=genders)
    with fc5:
        smk_vals = sorted(df["Smoker"].unique())
        sel_smk = st.multiselect("Smoker", smk_vals, default=smk_vals)
    with fc6:
        htn_vals = sorted(df["HTN"].unique())
        sel_htn = st.multiselect("HTN", htn_vals, default=htn_vals)

df_f = df[
    df["Year"].between(yr_start, yr_end) &
    df["Residence"].isin(sel_area) &
    df["Sex"].isin(sel_gender) &
    df["Smoker"].isin(sel_smk) &
    df["HTN"].isin(sel_htn) &
    df["Age"].between(a_from, a_to)
]

# ─────────  KPI ROW  ────────────────────────────────────────────────────
_, k1, k2, k3 = st.columns([0.35, 0.22, 0.22, 0.22])
k1.metric("Patients", f"{len(df_f):,}")
k2.metric("Smokers %", f"{df_f['Smoker_Num'].mean()*100:.1f}")
k3.metric("HTN %",     f"{df_f['HTN_Num'].mean()*100:.1f}")

# ─────────  CHART CONFIG  ──────────────────────────────────────────────
H  = 185
M  = dict(t=12, b=5, l=5, r=5)
CFG = {"displayModeBar": False}

# custom colour maps
SEX_COLORS = {"M": "#1f77b4", "F": "#ff7f0e"}          # blue / orange
HTN_COLORS = {"No HTN": "#2ca02c", "HTN": "#d62728"}   # green / red
DARK, LIGHT = "#1f77b4", "#aec7e8"                     # for other charts

# ─────────  2 × 2 GRID  ────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="small")
r2c1, r2c2 = st.columns(2, gap="small")

# ① Surgeries by Gender  (distinct colours)
with r1c1:
    fig = px.histogram(df_f, x="Sex", color="Sex",
                       color_discrete_map=SEX_COLORS,
                       template="plotly_white")
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ② Surgeries by Smoking Status (pie unchanged)
with r1c2:
    fig = p
