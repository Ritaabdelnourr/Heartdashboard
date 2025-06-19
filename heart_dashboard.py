import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚑 Open Heart Disease Dashboard", layout="wide")
st.title("🚑 Open Heart Disease Dashboard")
st.markdown("#### Open Heart Surgeries Cohort Analysis")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"]   = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"]   = df["Date"].dt.year
    df["Age"]    = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col + "_Num"] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year", "Obesity"
    ])
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
yrs = sorted(df["Year"].unique())
yr_start, yr_end = st.sidebar.slider("Year range", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
areas = sorted(df["Residence"].unique())
sel_areas = st.sidebar.multiselect("Area", areas, default=areas)

df_f = df[df["Year"].between(yr_start, yr_end) & df["Residence"].isin(sel_areas)]

# ── KPIs ───────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients",   f"{len(df_f):,}")
c2.metric("Smokers (%)",      f"{df_f['Smoker_Num'].mean()*100:.1f}")
c3.metric("Hypertension (%)", f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── Chart Section ──────────────────────────────────────────────────────────────
st.subheader("Open Heart Surgeries")

dark_blue = "#1f77b4"
light_blue = "#aec7e8"

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("Gender Distribution")
    fig1 = px.histogram(
        df_f, x="Sex",
        template="plotly_white",
        color_discrete_sequence=[dark_blue]
    )
    fig1.update_layout(
        height=260,
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    st.subheader("Smoking Status")
    fig2 = px.pie(
        df_f, names="Smoker", hole=0.4,
        template="plotly_white"
    )
    fig2.update_traces(
        textinfo="percent+label",
        marker=dict(colors=[dark_blue, light_blue])
    )
    fig2.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("Surgeries by Age")
    fig3 = px.histogram(
        df_f, x="Age", nbins=20,
        template="plotly_white",
    )
    fig3.update_traces(marker_color=dark_blue)
    fig3.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    st.subheader("Surgeries by Area")
    cnt = (
        df_f["Residence"]
        .value_counts()
        .rename_axis("Area")
        .reset_index(name="Count")
    )
    fig4 = px.bar(
        cnt, x="Area", y="Count",
        template="plotly_white"
    )
    fig4.update_traces(marker_color=light_blue)
    fig4.update_layout(
        height=260,
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis_tickangle=-45,
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Obesity Insight Panel ───────────────────────────────────────────────────────
st.subheader("🔎 Bleeding Rate by Obesity Status")
# compute bleeding rate per obesity group
bleed_ob = (
    df_f.groupby("Obesity")["Bleeding_Num"]
    .mean()
    .reset_index()
    .rename(columns={"Bleeding_Num": "Bleeding_Rate"})
)
# ensure the order is No → Yes
bleed_ob["Obesity"] = pd.Categorical(bleed_ob["Obesity"], categories=["No", "Yes"], ordered=True)

fig5 = px.bar(
    bleed_ob,
    x="Obesity",
    y="Bleeding_Rate",
    labels={"Bleeding_Rate": "Bleeding Rate"},
    template="plotly_white"
)
# dark_blue for obese, light_blue for non-obese
fig5.update_traces(marker_color=[light_blue, dark_blue])
fig5.update_layout(
    height=260,
    margin=dict(t=30, b=10, l=10, r=10),
    yaxis_tickformat=".0%",
    showlegend=False
)
st.plotly_chart(fig5, use_container_width=True)
