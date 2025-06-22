# Heart-Disease Dashboard  – 2 × 2 layout
# -------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.title("🚑 Heart Disease Dashboard")
st.markdown("#### Open Heart Surgeries Cohort Analysis")

# ── Load & preprocess ────────────────────────────────────────────────────────
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

    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
    ])
    return df

df = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

yrs = sorted(df["Year"].unique())
yr_start, yr_end = st.sidebar.slider(
    "Year range", int(yrs[0]), int(yrs[-1]), (int(yrs[0]), int(yrs[-1]))
)

areas = sorted(df["Residence"].unique())
sel_areas = st.sidebar.multiselect("Residence (Area)", areas, default=areas)

genders = sorted(df["Sex"].unique())
sel_gender = st.sidebar.multiselect("Gender", genders, default=genders)

smoke_opts = sorted(df["Smoker"].unique())
sel_smoke = st.sidebar.multiselect("Smoking Status", smoke_opts, default=smoke_opts)

htn_opts = sorted(df["HTN"].unique())
sel_htn = st.sidebar.multiselect("Hypertension Status", htn_opts, default=htn_opts)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_start, age_end = st.sidebar.slider(
    "Age range", age_min, age_max, (age_min, age_max)
)

# Apply filters
df_f = df[
    df["Year"].between(yr_start, yr_end) &
    df["Residence"].isin(sel_areas) &
    df["Sex"].isin(sel_gender) &
    df["Smoker"].isin(sel_smoke) &
    df["HTN"].isin(sel_htn) &
    df["Age"].between(age_start, age_end)
]

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Total Patients",   f"{len(df_f):,}")
k2.metric("Smokers (%)",      f"{df_f['Smoker_Num'].mean()*100:.1f}")
k3.metric("Hypertension (%)", f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── Plotly config ────────────────────────────────────────────────────────────
plot_config = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": [
        "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "resetScale2d"
    ],
    "scrollZoom": True,
}
dark_blue, light_blue = "#1f77b4", "#aec7e8"

# ── 2 × 2 DASHBOARD CANVAS ───────────────────────────────────────────────────
st.subheader("Open-Heart Surgeries – 2 × 2 Overview")

# Row 1
r1c1, r1c2 = st.columns(2, gap="medium")

with r1c1:
    st.markdown("##### Surgeries by Gender")
    fig1 = px.histogram(
        df_f, x="Sex", template="plotly_white",
        color_discrete_sequence=[dark_blue]
    )
    fig1.update_layout(height=225, margin=dict(t=25, b=10, l=10, r=10),
                       showlegend=False, dragmode="select")
    st.plotly_chart(fig1, use_container_width=True, config=plot_config)

with r1c2:
    st.markdown("##### Surgeries by Smoking Status")
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, template="plotly_white")
    fig2.update_traces(
        textinfo="percent+label",
        marker=dict(colors=[dark_blue, light_blue])
    )
    fig2.update_layout(height=225, margin=dict(t=25, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True, config=plot_config)

# Row 2
r2c1, r2c2 = st.columns(2, gap="medium")

with r2c1:
    st.markdown("##### Surgeries by Age")
    fig3 = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig3.update_traces(marker_color=dark_blue)
    fig3.update_layout(height=225, margin=dict(t=25, b=10, l=10, r=10),
                       dragmode="select")
    st.plotly_chart(fig3, use_container_width=True, config=plot_config)

with r2c2:
    st.markdown("##### Surgeries by Area")
    cnt = (
        df_f["Residence"].value_counts()
        .rename_axis("Area")
        .reset_index(name="Count")
    )
    fig4 = px.bar(cnt, x="Area", y="Count", template="plotly_white")
    fig4.update_traces(marker_color=light_blue)
    fig4.update_layout(
        height=225, margin=dict(t=25, b=10, l=10, r=10),
        xaxis_tickangle=-45, showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True, config=plot_config)

# ── Optional detail in expander ───────────────────────────────────────────────
with st.expander("🔮 Bleeding-risk detail (click to open)", expanded=False):
    p1, p2 = st.columns(2, gap="medium")

    with p1:
        st.markdown("*Bleeding Rate by HTN*")
        hr = (
            df_f.groupby("HTN_Num")["Bleeding_Num"]
            .mean()
            .reset_index()
        )
        hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
        fig5 = px.bar(
            hr, x="HTN", y="Bleeding_Num",
            labels={"Bleeding_Num": "Bleeding Rate"},
            template="plotly_white"
        )
        fig5.update_traces(marker_color=[light_blue, dark_blue])
        fig5.update_layout(
            height=220, margin=dict(t=15, b=10, l=10, r=10),
            yaxis_tickformat=".0%", showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True, config=plot_config)

    with p2:
        st.markdown("*Bleeding Rate vs Age*")
        age_rate = (
            df_f.groupby(df_f["Age"].round())["Bleeding_Num"]
            .mean()
            .reset_index(name="Bleeding_Rate")
        )
        fig6 = px.line(
            age_rate, x="Age", y="Bleeding_Rate",
            labels={"Bleeding_Rate": "Bleeding Rate", "Age": "Age (yrs)"},
            template="plotly_white"
        )
        fig6.update_traces(line_color=dark_blue)
        fig6.update_layout(
            height=220, margin=dict(t=15, b=10, l=10, r=10),
            dragmode="zoom", yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig6, use_container_width=True, config=plot_config)
