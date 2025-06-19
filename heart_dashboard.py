import streamlit as st
import pandas as pd
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚑 Open-Heart Surgery Dashboard", layout="wide")

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

# Load your data (Excel or CSV)
df = pd.read_excel("heart disease.xlsx", engine="openpyxl")
# Keep only the two open-heart surgery reasons
df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]

# Extract year for filtering
df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year

# Year slider
years     = sorted(df["Year"].dropna().astype(int).unique())
yr_start, yr_end = st.sidebar.slider("Year range", years[0], years[-1], (years[0], years[-1]))

# Residence multiselect
residences    = sorted(df["Residence"].dropna().unique())
sel_residences = st.sidebar.multiselect("Residence", residences, default=residences)

# Apply filters
df_f = df[
    (df["Year"]  >= yr_start) &
    (df["Year"]  <= yr_end) &
    (df["Residence"].isin(sel_residences))
]

# ── Title & KPIs ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort")
m1, m2, m3 = st.columns(3)
m1.metric("Patients",         f"{len(df_f):,}")
m2.metric("Smokers (%)",      f"{df_f['Smoker'].mean()*100:.1f}")
m3.metric("Hypertension (%)", f"{df_f['HTN'].mean()*100:.1f}")

# ── 2×2 Grid of Charts ─────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)

with r1c1:
    fig1 = px.histogram(
        df_f,
        x="Sex",
        color="Sex",
        title="Gender"
    )
    fig1.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    fig2 = px.pie(
        df_f,
        names="Smoker",
        hole=0.4,
        title="Smoker vs. Non-Smoker"
    )
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)

with r2c1:
    fig3 = px.box(
        df_f,
        y="Age",
        title="Age Distribution"
    )
    fig3.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Residence", "Count"]
    fig4 = px.bar(
        cnt,
        x="Residence",
        y="Count",
        title="By Residence"
    )
    fig4.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
