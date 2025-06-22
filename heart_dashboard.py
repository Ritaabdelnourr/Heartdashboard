import streamlit as st
import pandas as pd
import plotly.express as px

# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
TITLE_COL, *_ = st.columns([0.35, 0.22, 0.22, 0.22])
with TITLE_COL:
    st.write("### 🚑 Heart-Disease Dashboard — Open-Heart Surgeries")

# ── Load & preprocess (unchanged) ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col + "_Num"] = (
            df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        )

    return df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
    ])

df = load_data()

# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.header("Filters")

yrs = sorted(df["Year"].unique())
yr_start, yr_end = st.sidebar.slider("Year", yrs[0], yrs[-1], (yrs[0], yrs[-1]))

areas = st.sidebar.multiselect(
    "Residence", sorted(df["Residence"].unique()),
    default=sorted(df["Residence"].unique())
)

genders = st.sidebar.multiselect(
    "Gender", sorted(df["Sex"].unique()),
    default=sorted(df["Sex"].unique())
)

smk = st.sidebar.multiselect(
    "Smoker", sorted(df["Smoker"].unique()),
    default=sorted(df["Smoker"].unique())
)

htn = st.sidebar.multiselect(
    "HTN", sorted(df["HTN"].unique()),
    default=sorted(df["HTN"].unique())
)

a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
a_from, a_to = st.sidebar.slider("Age", a_min, a_max, (a_min, a_max))

df_f = df[
    df["Year"].between(yr_start, yr_end) &
    df["Residence"].isin(areas) &
    df["Sex"].isin(genders) &
    df["Smoker"].isin(smk) &
    df["HTN"].isin(htn) &
    df["Age"].between(a_from, a_to)
]

# ── KPIs inline with title ─────────────────────────────────────────────────
_, k1, k2, k3 = st.columns([0.35, 0.22, 0.22, 0.22])
k1.metric("Patients", f"{len(df_f):,}")
k2.metric("Smokers %", f"{df_f['Smoker_Num'].mean()*100:.1f}")
k3.metric("HTN %",     f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── Plotly config ──────────────────────────────────────────────────────────
H = 200                              # chart height
M = dict(t=15, b=5, l=5, r=5)        # tight margins
CFG = {"displayModeBar": False}      # hide mode-bar
DARK, LIGHT = "#1f77b4", "#aec7e8"

# ── 2 × 2 canvas ───────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="small")
r2c1, r2c2 = st.columns(2, gap="small")

with r1c1:
    fig = px.histogram(df_f, x="Sex", template="plotly_white",
                       color_discrete_sequence=[DARK])
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r1c2:
    fig = px.pie(df_f, names="Smoker", hole=0.35, template="plotly_white")
    fig.update_traces(textinfo="percent+label",
                      marker=dict(colors=[DARK, LIGHT]))
    fig.update_layout(height=H, margin=M)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r2c1:
    fig = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig.update_traces(marker_color=DARK)
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r2c2:
    cnt = (df_f["Residence"].value_counts()
           .rename_axis("Residence").reset_index(name="Count"))
    fig = px.bar(cnt, x="Residence", y="Count", template="plotly_white")
    fig.update_traces(marker_color=LIGHT)
    fig.update_layout(height=H, margin=M, xaxis_tickangle=-45,
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ── Collapsible detail ─────────────────────────────────────────────────────
with st.expander("🔮 Bleeding-risk detail", expanded=False):
    c1, c2 = st.columns(2, gap="small")

    with c1:
        hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
        hr["HTN"] = hr["HTN_Num"].map({0: "No", 1: "Yes"})
        fig = px.bar(hr, x="HTN", y="Bleeding_Num",
                     labels={"Bleeding_Num": "Bleed Rate"},
                     template="plotly_white")
        fig.update_traces(marker_color=[LIGHT, DARK])
        fig.update_layout(height=H-10, margin=M, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config=CFG)

    with c2:
        rate = (df_f.groupby(df_f["Age"].round())["Bleeding_Num"]
                .mean().reset_index(name="Rate"))
        fig = px.line(rate, x="Age", y="Rate", template="plotly_white")
        fig.update_traces(line_color=DARK)
        fig.update_layout(height=H-10, margin=M, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config=CFG)
