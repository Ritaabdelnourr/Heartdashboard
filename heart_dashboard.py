import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚑 Heart Surgery Dashboard", layout="wide")

# ── Load & clean data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    # Parse dates like "14.6.2023"
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    # Numeric age
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    # Map yes/no columns to 0/1
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col+"_Num"] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes":1, "no":0})
        )
    # Drop rows missing anything we need
    df = df.dropna(subset=[
        "Sex", "Age", "Residence",
        "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"
    ])
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
years       = sorted(df["Year"].unique())
yr_start, yr_end = st.sidebar.slider("Year range", years[0], years[-1], (years[0], years[-1]))
residences  = sorted(df["Residence"].unique())
sel_res     = st.sidebar.multiselect("Residence", residences, default=residences)

df_f = df[
    df["Year"].between(yr_start, yr_end) &
    df["Residence"].isin(sel_res)
]

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.title("🚑 Patients Undergoing Open-Heart Surgery")
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients",      f"{len(df_f):,}")
c2.metric("Smokers (%)",         f"{df_f['Smoker_Num'].mean()*100:.1f}")
c3.metric("Hypertension (%)",    f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── 2×2 Grid ───────────────────────────────────────────────────────────────────
r1c1, r1c2 = st.columns(2)
with r1c1:
    fig1 = px.histogram(df_f, x="Sex", color="Sex", title="Gender Distribution")
    fig1.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, title="Smoking Status")
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    fig3 = px.box(df_f, y="Age", title="Age Distribution")
    fig3.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Residence","Count"]
    fig4 = px.bar(cnt, x="Residence", y="Count", title="Patients by Residence")
    fig4.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

# ── Prediction Panel ───────────────────────────────────────────────────────────
st.subheader("🔮 Predict Bleeding Post-Surgery (HTN → Bleeding)")

X = df_f[["HTN_Num"]]
y = df_f["Bleeding_Num"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)

model   = LogisticRegression(solver="liblinear").fit(X_tr, y_tr)
y_pred  = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:,1]

acc = accuracy_score(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc = auc(fpr, tpr)

st.markdown(f"**Accuracy:** {acc:.2f}   **AUC:** {roc_auc:.2f}")

fig5 = px.area(
    x=fpr, y=tpr,
    title="ROC Curve",
    labels={"x":"False Positive Rate","y":"True Positive Rate"}
)
fig5.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
fig5.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
st.plotly_chart(fig5, use_container_width=True)
