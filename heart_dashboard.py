import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="🚑 Open-Heart Surgery Dashboard", layout="wide")

# Load cleaned data
df = pd.read_csv("heart_disease_clean.csv")
df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year

# Sidebar filters
st.sidebar.header("Filters")
years = sorted(df["Year"].dropna().astype(int).unique())
yr_start, yr_end = st.sidebar.slider("Year range", years[0], years[-1], (years[0], years[-1]))
res_opts = sorted(df["Residence"].dropna().unique())
sel_res = st.sidebar.multiselect("Residence", res_opts, default=res_opts)

df_f = df[(df["Year"].between(yr_start, yr_end)) & (df["Residence"].isin(sel_res))]

# **Drop any rows still containing NaNs in key fields**
required = ["Sex", "Age", "Smoker", "HTN", "Bleeding", "Residence", "Year"]
df_f = df_f.dropna(subset=required).copy()

# Title & KPIs
st.title("🚑 Open-Heart Surgery Cohort")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)", f"{df_f.Smoker.mean()*100:.1f}")
c3.metric("Hypertension (%)", f"{df_f.HTN.mean()*100:.1f}")

# 2×2 grid
r1c1, r1c2 = st.columns(2)
with r1c1:
    fig1 = px.histogram(df_f, x="Sex", color="Sex", title="Gender")
    fig1.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
with r1c2:
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, title="Smoker vs Non-Smoker")
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig3 = px.box(df_f, y="Age", title="Age Distribution")
    fig3.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
with r2c2:
    cnt = df_f.Residence.value_counts().reset_index().rename(columns={"index":"Residence","Residence":"Count"})
    fig4 = px.bar(cnt, x="Residence", y="Count", title="By Residence")
    fig4.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
