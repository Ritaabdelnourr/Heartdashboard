import streamlit as st
import pandas as pd
import plotly.express as px

# ─── PAGE SET-UP ───────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
TITLE_COL, *_ = st.columns([0.35, 0.22, 0.22, 0.22])
with TITLE_COL:
    st.write("### 🚑 Heart-Disease Dashboard — Open-Heart Surgeries")

# ─── LOAD & PREP ───────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col] = df[col].astype(str).str.strip()
        df[col + "_Num"] = df[col].str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
    ])
    df["Sex"]    = df["Sex"].str.strip()
    df["Smoker"] = df["Smoker"].str.strip()
    df["HTN"]    = df["HTN"].str.strip()
    return df

df = load_data()

# ─── COLOUR MAPS ───────────────────────────────────────────────────────
SEX_COLORS = {"M": "#1f77b4", "F": "#ff7f0e"}          # blue / orange
HTN_COLORS = {"No HTN": "#17becf", "HTN": "#9467bd"}   # teal / purple
DARK, LIGHT = "#1f77b4", "#aec7e8"

# ─── FILTER BAR ────────────────────────────────────────────────────────
with st.container():
    fc1, fc2 = st.columns([0.5, 0.5], gap="small")
    yrs = sorted(df["Year"].unique())
    with fc1:
        yr_start, yr_end = st.slider("Year", int(yrs[0]), int(yrs[-1]),
                                     (int(yrs[0]), int(yrs[-1])))
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    with fc2:
        a_from, a_to = st.slider("Age", a_min, a_max, (a_min, a_max))
    fc3, fc4, fc5, fc6 = st.columns(4, gap="small")
    with fc3:
        sel_area = st.multiselect("Residence",
                                  sorted(df["Residence"].unique()),
                                  default=sorted(df["Residence"].unique()))
    with fc4:
        sel_gender = st.multiselect("Gender",
                                    sorted(df["Sex"].unique()),
                                    default=sorted(df["Sex"].unique()))
    with fc5:
        sel_smk = st.multiselect("Smoker",
                                 sorted(df["Smoker"].unique()),
                                 default=sorted(df["Smoker"].unique()))
    with fc6:
        sel_htn = st.multiselect("HTN",
                                 sorted(df["HTN"].unique()),
                                 default=sorted(df["HTN"].unique()))

df_f = df[
    df["Year"].between(yr_start, yr_end)
    & df["Residence"].isin(sel_area)
    & df["Sex"].isin(sel_gender)
    & df["Smoker"].isin(sel_smk)
    & df["HTN"].isin(sel_htn)
    & df["Age"].between(a_from, a_to)
]

# ─── KPI ROW ───────────────────────────────────────────────────────────
_, k1, k2, k3 = st.columns([0.35, 0.22, 0.22, 0.22])
k1.metric("Patients", f"{len(df_f):,}")
k2.metric("Smokers %", f"{df_f['Smoker_Num'].mean()*100:.1f}")
k3.metric("HTN %",     f"{df_f['HTN_Num'].mean()*100:.1f}")

# ─── CHART CONFIG ──────────────────────────────────────────────────────
H   = 140                      # compact height
M   = dict(t=5, b=5, l=5, r=5)
CFG = {"displayModeBar": False}

# ─── ROW 1 (two charts) ────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="small")

with r1c1:   # Gender
    sex_counts = (df_f["Sex"].value_counts()
                  .rename_axis("Sex").reset_index(name="Count"))
    fig = px.bar(sex_counts, x="Sex", y="Count",
                 color="Sex", color_discrete_map=SEX_COLORS,
                 template="plotly_white")
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r1c2:   # Smoking
    fig = px.pie(df_f, names="Smoker", hole=0.35, template="plotly_white")
    fig.update_traces(marker=dict(colors=[DARK, LIGHT]),
                      textinfo="percent+label")
    fig.update_layout(height=H, margin=M)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ─── ROW 2 (histogram + HTN bar) ───────────────────────────────────────
r2c1, r2c2 = st.columns(2, gap="small")

with r2c1:   # Age histogram
    fig = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig.update_traces(marker_color=DARK)
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r2c2:   # Bleeding risk by HTN
    hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
    hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig = px.bar(hr, x="HTN", y="Bleeding_Num",
                 labels={"Bleeding_Num": "Bleeding Rate"},
                 template="plotly_white",
                 color="HTN", color_discrete_map=HTN_COLORS)
    fig.update_layout(height=H, margin=M, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
