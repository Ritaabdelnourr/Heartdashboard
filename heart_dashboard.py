import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
        "Sex", "Age", "Residence", "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
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

dark_blue  = "#1f77b4"
light_blue = "#aec7e8"

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("Surgeries by Gender")
    fig1 = px.histogram(
        df_f, x="Sex",
        template="plotly_white",
        color_discrete_sequence=[dark_blue]
    )
    fig1.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    st.subheader("Surgeries by Smoking Status")
    fig2 = px.pie(
        df_f, names="Smoker", hole=0.4, template="plotly_white"
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
        df_f, x="Age", nbins=20, template="plotly_white"
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
        cnt, x="Area", y="Count", template="plotly_white"
    )
    fig4.update_traces(marker_color=light_blue)
    fig4.update_layout(
        height=260, margin=dict(t=30, b=10, l=10, r=10),
        xaxis_tickangle=-45, showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Bleeding Prediction Panel ──────────────────────────────────────────────────
st.subheader("🔮 Predict Bleeding Risk Based on HTN")

# Prepare data for modeling
X = df_f[["HTN_Num"]]
y = df_f["Bleeding_Num"]

# Train/test split & model
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_tr, y_tr)
y_proba = model.predict_proba(X_te)[:, 1]

# Build prediction DataFrame
df_pred = X_te.copy()
df_pred["Bleed_Prob"] = y_proba
df_pred["HTN"] = df_pred["HTN_Num"].map({0: "No HTN", 1: "HTN"})

# Plot predicted bleeding probability
fig5 = px.violin(
    df_pred, x="HTN", y="Bleed_Prob", box=True, points="all",
    color="HTN", 
    color_discrete_map={"No HTN": light_blue, "HTN": dark_blue},
    labels={"Bleed_Prob": "Predicted Bleeding Probability"},
    template="plotly_white"
)
fig5.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
st.plotly_chart(fig5, use_container_width=True)
