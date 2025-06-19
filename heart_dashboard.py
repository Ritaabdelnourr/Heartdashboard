import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Open Heart Surgery Dashboard", layout="wide")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("heart disease.xlsx", engine="openpyxl")
    df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    return df.dropna(subset=["Year"])

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
min_y, max_y = int(df.Year.min()), int(df.Year.max())
yr_start, yr_end = st.sidebar.slider("Year range", min_y, max_y, (min_y, max_y))
res = st.sidebar.multiselect("Residence", options=sorted(df.Residence.unique()), default=sorted(df.Residence.unique()))

df_f = df[(df.Year >= yr_start) & (df.Year <= yr_end) & (df.Residence.isin(res))]

# ── Title & KPIs ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort")
c1, c2, c3 = st.columns(3)
c1.metric("Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)", f"{df_f.Smoker.mean()*100:.1f}")
c3.metric("HTN (%)",      f"{df_f.HTN.mean()*100:.1f}")

# ── 2×2 Grid ────────────────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)

with r1c1:
    fig = px.histogram(df_f, x="Sex", color="Sex", title="Gender")
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig = px.pie(df_f, names="Smoker", hole=0.4, title="Smoker vs. Non-Smoker")
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

r2c1, r2c2 = st.columns(2)

with r2c1:
    fig = px.box(df_f, y="Age", title="Age Distribution")
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    cnt = df_f.Residence.value_counts().reset_index()
    cnt.columns = ["Residence","Count"]
    fig = px.bar(cnt, x="Residence", y="Count", title="By Residence")
    fig.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ── Prediction (HTN → Bleeding) ────────────────────────────────────────────────
st.subheader("Predict Bleeding Post-Surgery")

X = df_f[["HTN"]].astype(int)
y = df_f["Bleeding"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_tr, y_tr)
y_pred = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:,1]

acc = accuracy_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc = auc(fpr, tpr)

st.markdown(f"**Accuracy:** {acc:.2f} &nbsp;&nbsp; **AUC:** {roc_auc:.2f}")

fig_roc = px.area(x=fpr, y=tpr,
                  title="ROC Curve",
                  labels={"x":"FPR","y":"TPR"})
fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
fig_roc.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
st.plotly_chart(fig_roc, use_container_width=True)
