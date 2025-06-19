import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Open-Heart Surgery Dashboard", layout="wide")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("heart disease.xlsx", engine="openpyxl")
    # parse Date → Year
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

# 1) Surgery type (whatever values you have)
surg_types = sorted(df["Surgery"].dropna().unique())
sel_surg = st.sidebar.multiselect("Surgery Type", surg_types, default=surg_types)

# 2) Year slider (only from non-null years)
years = sorted(df["Year"].dropna().astype(int).unique())
yr_start, yr_end = st.sidebar.slider(
    "Year range", years[0], years[-1], (years[0], years[-1])
)

# 3) Residence
res_opts = sorted(df["Residence"].dropna().unique())
sel_res = st.sidebar.multiselect("Residence", res_opts, default=res_opts)

# apply all filters
df_f = df[
    df["Surgery"].isin(sel_surg) &
    df["Year"].between(yr_start, yr_end) &
    df["Residence"].isin(sel_res)
].copy()

# ── Title & KPIs ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)",     f"{df_f['Smoker'].mean()*100:.1f}")
c3.metric("Hypertension (%)",f"{df_f['HTN'].mean()*100:.1f}")

# ── 2×2 Grid of Charts ─────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)
with r1c1:
    fig1 = px.histogram(df_f, x="Sex", color="Sex", title="Gender")
    fig1.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
with r1c2:
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, title="Smoker vs Non-Smoker")
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig3 = px.box(df_f, y="Age", title="Age Distribution")
    fig3.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
with r2c2:
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Residence","Count"]
    fig4 = px.bar(cnt, x="Residence", y="Count", title="By Residence")
    fig4.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

# ── Bleeding Prediction (HTN → Bleeding) ───────────────────────────────────────
st.subheader("🔮 Predict Bleeding Post-Surgery")
X = df_f[["HTN"]].astype(int)
y = df_f["Bleeding"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_tr, y_tr)
y_pred  = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:,1]

acc = accuracy_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc = auc(fpr, tpr)

st.markdown(f"**Accuracy:** {acc:.2f}   **AUC:** {roc_auc:.2f}")

roc_fig = px.area(
    x=fpr, y=tpr,
    title="ROC Curve",
    labels={"x":"False Positive Rate","y":"True Positive Rate"}
)
roc_fig.add_shape(type="line", line=dict(dash="dash"), x0=0,x1=1,y0=0,y1=1)
roc_fig.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
st.plotly_chart(roc_fig, use_container_width=True)
