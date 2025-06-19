import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Open-Heart Surgery Dashboard", layout="wide")

# ── Data load & preprocess ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Make sure 'heart disease.xlsx' is in the same folder
    df = pd.read_excel("heart disease.xlsx", engine="openpyxl")
    # Filter to the two surgery types
    df = df[df["Surgery"].isin(["cardiovascular disease", "valvular disease"])]
    # Extract year for filtering
    df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    return df.dropna(subset=["Year"])

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")
y_min, y_max = int(df.Year.min()), int(df.Year.max())
yr_start, yr_end = st.sidebar.slider(
    "Year range", y_min, y_max, (y_min, y_max)
)
residence_opts = sorted(df["Residence"].dropna().unique())
sel_res = st.sidebar.multiselect("Residence", residence_opts, default=residence_opts)

# Apply filters
df_f = df[
    (df.Year >= yr_start) &
    (df.Year <= yr_end) &
    (df.Residence.isin(sel_res))
]

# ── Title & KPIs ───────────────────────────────────────────────────────────────
st.title("🚑 Open-Heart Surgery Cohort")
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients", f"{len(df_f):,}")
c2.metric("Smokers (%)",     f"{df_f.Smoker.mean()*100:.1f}")
c3.metric("Hypertension (%)",f"{df_f.HTN.mean()*100:.1f}")

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
    cnt = df_f.Residence.value_counts().reset_index()
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
y_proba = model.predict_proba(X_te)[:, 1]

acc = accuracy_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc = auc(fpr, tpr)

st.markdown(f"**Accuracy:** {acc:.2f}    **AUC:** {roc_auc:.2f}")

fig5 = px.area(
    x=fpr, y=tpr,
    title="ROC Curve",
    labels={"x":"False Positive Rate","y":"True Positive Rate"}
)
fig5.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
fig5.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
st.plotly_chart(fig5, use_container_width=True)
