import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.title("🚑 Heart Disease Dashboard")
st.markdown("#### Open Heart Surgeries Cohort Analysis")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col + "_Num"] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=[
        "Sex","Age","Residence","Smoker_Num","HTN_Num","Bleeding_Num","Year"
    ])
    return df

df = load_data()

# ── Sidebar filters & controls ─────────────────────────────────────────────────
st.sidebar.header("Filters & Controls")

# Year & area filters
yrs = sorted(df["Year"].unique())
yr_start, yr_end = st.sidebar.slider("Year range", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
areas = sorted(df["Residence"].unique())
sel_areas = st.sidebar.multiselect("Area", areas, default=areas)

# Chart type toggle for Age
age_view = st.sidebar.radio("Age chart type", ["Histogram", "Boxplot"])

# Model predictor selection
predictors = st.sidebar.multiselect(
    "Model predictors",
    ["HTN", "Smoker", "Age"],
    default=["HTN"]
)

# Download filtered data
df_f = df[df["Year"].between(yr_start, yr_end) & df["Residence"].isin(sel_areas)]
csv = df_f.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("📥 Download filtered data", csv, "filtered.csv", "text/csv")

# ── KPIs ───────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Patients",   f"{len(df_f):,}")
c2.metric("Smokers (%)",      f"{df_f['Smoker_Num'].mean()*100:.1f}")
c3.metric("Hypertension (%)", f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── Interactive table ──────────────────────────────────────────────────────────
st.subheader("Filtered Data")
st.dataframe(df_f)

# ── Chart Section ──────────────────────────────────────────────────────────────
st.subheader("Open Heart Surgeries")

dark_blue  = "#1f77b4"
light_blue = "#aec7e8"

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("Surgeries by Gender")
    fig1 = px.histogram(
        df_f, x="Sex",
        color_discrete_sequence=[dark_blue],
        template="plotly_white"
    )
    fig1.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
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
    fig2.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("Surgeries by Age")
    if age_view == "Histogram":
        fig3 = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
        fig3.update_traces(marker_color=dark_blue)
    else:
        fig3 = px.box(df_f, y="Age", template="plotly_white")
        fig3.update_traces(marker_color=dark_blue)
    fig3.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    st.subheader("Surgeries by Area")
    cnt = (df_f["Residence"]
           .value_counts()
           .rename_axis("Area")
           .reset_index(name="Count"))
    fig4 = px.bar(cnt, x="Area", y="Count", template="plotly_white")
    fig4.update_traces(marker_color=light_blue)
    fig4.update_layout(
        height=260,
        margin=dict(t=30,b=10,l=10,r=10),
        xaxis_tickangle=-45,
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Bleeding Risk & Prediction ─────────────────────────────────────────────────
st.subheader("🔮 Bleeding Risk & Prediction")

# 1) Actual bleeding rate by HTN
hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
hr["HTN"] = hr["HTN_Num"].map({0:"No HTN",1:"HTN"})
fig5 = px.bar(
    hr, x="HTN", y="Bleeding_Num",
    labels={"Bleeding_Num":"Bleeding Rate"},
    template="plotly_white"
)
fig5.update_traces(marker_color=[light_blue, dark_blue])
fig5.update_layout(
    height=260,
    margin=dict(t=10,b=10,l=10,r=10),
    yaxis_tickformat=".0%",
    showlegend=False
)
st.plotly_chart(fig5, use_container_width=True)

# 2) Model & ROC based on selected predictors
#    build feature matrix
feat_cols = []
for p in predictors:
    if p in ("HTN", "Smoker"):
        feat_cols.append(p + "_Num")
    else:
        feat_cols.append(p)

if feat_cols:
    X = df_f[feat_cols]
    y = df_f["Bleeding_Num"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)
    model = LogisticRegression(solver="liblinear").fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]

    # metrics
    acc = accuracy_score(y_te, model.predict(X_te))
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)

    # display
    st.metric("Model accuracy", f"{acc:.2f}")
    st.metric("ROC AUC",        f"{roc_auc:.2f}")

    fig6 = px.area(
        x=fpr, y=tpr,
        labels={"x":"False Positive Rate","y":"True Positive Rate"},
        template="plotly_white"
    )
    fig6.add_shape(type="line", line=dict(dash="dash"), x0=0,x1=1,y0=0,y1=1)
    fig6.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("Pick at least one predictor in the sidebar to run the model.")
