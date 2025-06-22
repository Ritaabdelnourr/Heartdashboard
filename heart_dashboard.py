import streamlit as st
import pandas as pd
import plotly.express as px

# ── PAGE CONFIG & ULTRA-COMPACT CSS ────────────────────────────────
st.set_page_config(page_title="Heart-Disease Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .block-container{padding-top:0.3rem;}
    .block-container>div{margin-top:0.25rem;margin-bottom:0.25rem;}
    div[data-testid="column"]>div:first-child{margin-top:0rem;}
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] span,
    div[data-testid="stMultiSelect"] label,
    div[data-baseweb="select"] *{
        font-size:0.68rem!important;line-height:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── TITLE ───────────────────────────────────────────────────────────
st.write("#### 🚑 Heart-Disease Dashboard – Open-Heart Surgeries")

# ── LOAD DATA ───────────────────────────────────────────────────────
@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for c in ["Smoker", "HTN", "Bleeding"]:
        df[c] = df[c].astype(str).str.strip()
        df[c + "_Num"] = df[c].str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["Sex","Age","Smoker_Num",
                           "HTN_Num","Bleeding_Num","Year"])
    df["Sex"]    = df["Sex"].str.strip()
    df["Smoker"] = df["Smoker"].str.strip()
    df["HTN"]    = df["HTN"].str.strip()
    return df

df = load()

# ── COLOURS ─────────────────────────────────────────────────────────
SEX_COLORS = {"M": "#1f77b4", "F": "#ff7f0e"}
HTN_COLORS = {"No HTN": "#17becf", "HTN": "#9467bd"}
DARK, LIGHT = "#1f77b4", "#aec7e8"

# ── FILTER BAR ──────────────────────────────────────────────────────
with st.container():
    s1, s2 = st.columns(2, gap="small")
    yrs = sorted(df["Year"].unique())
    with s1:
        yr_from, yr_to = st.slider("Year range", yrs[0], yrs[-1],
                                   (yrs[0], yrs[-1]))
    amin, amax = int(df["Age"].min()), int(df["Age"].max())
    with s2:
        age_from, age_to = st.slider("Age range", amin, amax,
                                     (amin, amax))

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        sel_gender = st.multiselect("Gender",
                                    sorted(df["Sex"].unique()),
                                    default=sorted(df["Sex"].unique()))
    with c2:
        sel_smk = st.multiselect("Smoker",
                                 sorted(df["Smoker"].unique()),
                                 default=sorted(df["Smoker"].unique()))
    with c3:
        sel_htn = st.multiselect("HTN",
                                 sorted(df["HTN"].unique()),
                                 default=sorted(df["HTN"].unique()))

df_f = df[
    df["Year"].between(yr_from, yr_to)
    & df["Sex"].isin(sel_gender)
    & df["Smoker"].isin(sel_smk)
    & df["HTN"].isin(sel_htn)
    & df["Age"].between(age_from, age_to)
]

# ── PLOT CONFIG ─────────────────────────────────────────────────────
H   = 160                          # 20 px taller for title space
M   = dict(t=25, b=3, l=3, r=3)    # extra top margin
CFG = {"displayModeBar": False}
FONT = dict(size=9)
TITLE_STYLE = dict(font=dict(size=12), x=0.5)  # centred, 12-pt

# ── ROW 1 ───────────────────────────────────────────────────────────
c11, c12 = st.columns(2, gap="small")

with c11:                          # Gender bar
    g = df_f["Sex"].value_counts().reset_index()
    g.columns = ["Sex", "Count"]
    fig = px.bar(g, x="Sex", y="Count", color="Sex",
                 color_discrete_map=SEX_COLORS, template="plotly_white")
    fig.update_layout(title={**TITLE_STYLE, "text":"Surgeries by Gender"},
                      height=H, margin=M, showlegend=False, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with c12:                          # Smoking pie
    fig = px.pie(df_f, names="Smoker", hole=0.35, template="plotly_white")
    fig.update_traces(marker=dict(colors=[DARK, LIGHT]),
                      textinfo="percent+label")
    fig.update_layout(title={**TITLE_STYLE, "text":"Smokers vs Non-Smokers"},
                      height=H, margin=M, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ── ROW 2 ───────────────────────────────────────────────────────────
c21, c22 = st.columns(2, gap="small")

with c21:                          # Age histogram
    fig = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig.update_traces(marker_color=DARK)
    fig.update_layout(title={**TITLE_STYLE, "text":"Age Distribution"},
                      height=H, margin=M, showlegend=False, font=FONT)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with c22:                          # Bleeding vs HTN
    hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
    hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig = px.bar(hr, x="HTN", y="Bleeding_Num",
                 labels={"Bleeding_Num": "Bleeding Rate"},
                 template="plotly_white",
                 color="HTN", color_discrete_map=HTN_COLORS)
    fig.update_layout(title={**TITLE_STYLE, "text":"Bleeding Risk by Hypertension"},
                      height=H, margin=M, yaxis_tickformat=".0%",
                      font=FONT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)
