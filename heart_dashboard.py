import streamlit as st
import pandas as pd
import plotly.express as px

# ─── PAGE CONFIG + ULTRA-COMPACT CSS ─────────────────────────────────
st.set_page_config(page_title="Heart-Disease Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:0.3rem;}
    .block-container > div {margin-top:0.25rem;margin-bottom:0.25rem;}
    div[data-testid="column"] > div:first-child {margin-top:0rem;}
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] span,
    div[data-testid="stMultiSelect"] label,
    div[data-baseweb="select"] * {
        font-size:0.68rem !important; line-height:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── TITLE ───────────────────────────────────────────────────────────
st.write("#### 🚑 Heart-Disease Dashboard – Open-Heart Surgeries")

# ─── DATA LOAD ───────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for c in ["Smoker", "HTN", "Bleeding"]:
        df[c] = df[c].astype(str).str.strip()
        df[c + "_Num"] = df[c].str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
    ])
    df["Sex"]    = df["Sex"].str.strip()
    df["Smoker"] = df["Smoker"].str.strip()
    df["HTN"]    = df["HTN"].str.strip()
    return df

df = load_data()

# ─── COLOUR MAPS ─────────────────────────────────────────────────────
SEX_COLORS = {"M": "#1f77b4", "F": "#ff7f0e"}
HTN_COLORS = {"No HTN": "#17becf", "HTN": "#9467bd"}
DARK, LIGHT = "#1f77b4", "#aec7e8"

# ─── FILTER BAR ──────────────────────────────────────────────────────
with st.container():
    f1, f2 = st.columns([0.5, 0.5], gap="small")
    yrs = sorted(df["Year"].unique())
    with f1:
        yr_start, yr_end = st.slider("Year", yrs[0], yrs[-1],
                                     (yrs[0], yrs[-1]),
                                     label_visibility="collapsed")
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    with f2:
        a_from, a_to = st.slider("Age", a_min, a_max, (a_min, a_max),
                                 label_visibility="collapsed")

    f3, f4, f5, f6 = st.columns(4, gap="small")
    with f3:
        sel_area = st.multiselect(
            "Residence", sorted(df["Residence"].unique()),
            default=sorted(df["Residence"].unique())
        )
    with f4:
        sel_gender = st.multiselect(
            "Gender", sorted(df["Sex"].unique()),
            default=sorted(df["Sex"].unique())
        )
    with f5:
        sel_smk = st.multiselect(
            "Smoker", sorted(df["Smoker"].unique()),
            default=sorted(df["Smoker"].unique())
        )
    with f6:
        sel_htn = st.multiselect(
            "HTN", sorted(df["HTN"].unique()),
            default=sorted(df["HTN"].unique())
        )

df_f = df[
    df["Year"].between(yr_start, yr_end)
    & df["Residence"].isin(sel_area)
    & df["Sex"].isin(sel_gender)
    & df["Smoker"].isin(sel_smk)
    & df["HTN"].isin(sel_htn)
    & df["Age"].between(a_from, a_to)
]

# ─── CHART SETTINGS ──────────────────────────────────────────────────
H   = 120
M   = dict(t=3, b=3, l=3, r=3)
CFG = {"displayModeBar": False}
FONT = {"size": 9}

# ─── ROW 1 ────────────────────────────────────────────────────────────
c11, c12 = st.columns(2, gap="small")

with c11:  # Gender
    g = df_f["Sex"].value_counts().reset_index()
    g.columns = ["Sex", "Count"]
    fig = px.bar(g, x="Sex", y="Count",
                 color="Sex", color_discrete_map=SEX_COLORS,
                 template="plotly_white")
    fig.update_layout(height=H, margin=M, showlegend=False, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with c12:  # Smoking
    fig = px.pie(df_f, names="Smoker", hole=0.35,
                 template="plotly_white")
    fig.update_traces(marker=dict(colors=[DARK, LIGHT]),
                      textinfo="percent+label")
    fig.update_layout(height=H, margin=M, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ─── ROW 2 ────────────────────────────────────────────────────────────
c21, c22 = st.columns(2, gap="small")

with c21:  # Age
    fig = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig.update_traces(marker_color=DARK)
    fig.update_layout(height=H, margin=M, showlegend=False, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with c22:  # Bleeding vs HTN
    hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
    hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig = px.bar(hr, x="HTN", y="Bleeding_Num",
                 labels={"Bleeding_Num": "Bleeding Rate"},
                 template="plotly_white",
                 color="HTN", color_discrete_map=HTN_COLORS)
    fig.update_layout(height=H, margin=M,
                      yaxis_tickformat=".0%", font=FONT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)
