import streamlit as st
import pandas as pd
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚑 Open-Heart Surgery Dashboard", layout="wide")

# ── Load & Prep Data ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # adjust filename/extension if needed
    df = pd.read_excel("heart disease.xlsx", engine="openpyxl")
    # keep only the two surgery reasons
    df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]
    # extract Year for filtering
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    return df

df = load_data()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")

# 1) Surgery type
surgs    = df["Surgery"].dropna().unique().tolist()
sel_surg = st.sidebar.multiselect("Surgery Reason", surgs, default=surgs)

# 2) Year range (guard against empty or single-value lists)
years = sorted(df["Year"].dropna().astype(int).unique().tolist())
if len(years) > 1:
    yr_start, yr_end = st.sidebar.slider("Year range", years[0], years[-1], (years[0], years[-1]))
elif len(years) == 1:
    yr_start = yr_end = years[0]
    st.sidebar.markdown(f"Year: **{years[0]}**")
else:
    yr_start = yr_end = None  # no valid years

# 3) Residence
res_opts = df["Residence"].dropna().unique().tolist()
sel_res  = st.sidebar.multiselect("Residence", res_opts, default=res_opts)

# ── Apply Filters ───────────────────────────────────────────────────────────────
df_f = df.copy()
df_f = df_f[df_f["Surgery"].isin(sel_surg)]
if yr_start is not None:
    df_f = df_f[df_f["Year"].between(yr_start, yr_end)]
df_f = df_f[df_f["Residence"].isin(sel_res)]

# ── Title & Metrics ─────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort")
m1, m2, m3 = st.columns(3)
m1.metric("Patients",         f"{len(df_f):,}")
m2.metric("Smokers (%)",      f"{df_f['Smoker'].mean()*100:.1f}")
m3.metric("Hypertension (%)", f"{df_f['HTN'].mean()*100:.1f}")

# ── 2×2 Grid of Charts ─────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)
with r1c1:
    fig1 = px.histogram(df_f, x="Sex", color="Sex", title="Gender")
    fig1.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, title="Smoker vs. Non-Smoker")
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig3 = px.box(df_f, y="Age", title="Age Distribution")
    fig3.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Residence", "Count"]
    fig4 = px.bar(cnt, x="Residence", y="Count", title="By Residence")
    fig4.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
