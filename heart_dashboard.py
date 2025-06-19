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
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker","HTN","Bleeding"]:
        df[col+"_Num"] = df[col].str.strip().str.lower().map({"yes":1,"no":0})
    df = df.dropna(subset=["Sex","Age","Residence","Smoker_Num","HTN_Num","Bleeding_Num","Year"])
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

# grab a blues palette and pick some shades
shades = px.colors.sequential.Blues
# e.g. shades = ['#eff3ff', '#c6dbef', '#6baed6', '#2171b5', '#084594']

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("Gender Distribution")
    # two bars: male/female → use two distinct blue shades
    fig1 = px.histogram(
        df_f, x="Sex", color="Sex",
        color_discrete_map={"Male": shades[-2], "Female": shades[-4]},
        template="plotly_white"
    )
    fig1.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    st.subheader("Smoking Status")
    # two pie slices → two shades
    fig2 = px.pie(
        df_f, names="Smoker", hole=0.4,
        color_discrete_map={"Yes": shades[-3], "No": shades[-5]},
        template="plotly_white"
    )
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("Surgeries by Age")
    # histogram entirely in one shade
    fig3 = px.histogram(
        df_f, x="Age", nbins=20,
        template="plotly_white",
        color_discrete_sequence=[shades[-3]]
    )
    fig3.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    st.subheader("Surgeries by Area")
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Area","Count"]
    # bar chart with a gradient: map height → shade
    fig4 = px.bar(
        cnt, x="Area", y="Count",
        template="plotly_white",
        color="Count", color_continuous_scale=shades
    )
    fig4.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── Prediction Panel ───────────────────────────────────────────────────────────
st.subheader("🔮 Hypertension vs. Bleeding Risk")

# Bleeding rate by HTN
hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
hr["HTN"] = hr["HTN_Num"].map({0:"No HTN",1:"HTN"})
fig5 = px.bar(
    hr, x="HTN", y="Bleeding_Num",
    labels={"Bleeding_Num":"Bleeding Rate"},
    template="plotly_white",
    color="Bleeding_Num", color_continuous_scale=shades
)
fig5.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), yaxis_tickformat=".0%")
fig5.update_traces(showlegend=False)
# ROC curve
X = df_f[["HTN_Num"]]; y = df_f["Bleeding_Num"]
X_tr, X_te, y_tr, y_te = train_test_split(X,y,stratify=y,random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_tr,y_tr)
y_proba = model.predict_proba(X_te)[:,1]
fpr, tpr, _ = roc_curve(y_te, y_proba)
fig6 = px.area(
    x=fpr, y=tpr,
    labels={"x":"FPR","y":"TPR"},
    template="plotly_white",
    color_discrete_sequence=[shades[-2]]
)
fig6.add_shape(type="line", line=dict(dash="dash"), x0=0,x1=1,y0=0,y1=1)
fig6.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)

p1, p2 = st.columns(2)
p1.plotly_chart(fig5, use_container_width=True)
p2.plotly_chart(fig6, use_container_width=True)
