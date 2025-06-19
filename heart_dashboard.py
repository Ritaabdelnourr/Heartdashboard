import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

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
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    # Map binary fields
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col + "_Num"] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=["Sex", "Age", "Residence", "Smoker_Num", "HTN_Num", "Bleeding_Num", "Year"])
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
yrs = sorted(df["Year"].unique())
start_year, end_year = st.sidebar.slider("Year range", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
areas = sorted(df["Residence"].unique())
sel_areas = st.sidebar.multiselect("Area", areas, default=areas)

df_f = df[df["Year"].between(start_year, end_year) & df["Residence"].isin(sel_areas)]

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Total Patients", f"{len(df_f):,}")
k2.metric("Smokers (%)", f"{df_f['Smoker_Num'].mean()*100:.1f}")
k3.metric("Hypertension (%)", f"{df_f['HTN_Num'].mean()*100:.1f}")

# ── Patient Profiles Section ───────────────────────────────────────────────────
st.subheader("Patient Profiles")
palette_qual = px.colors.qualitative.Pastel

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Gender Distribution**")
    fig1 = px.histogram(
        df_f, x="Sex", color="Sex",
        color_discrete_sequence=palette_qual,
        template="plotly_white"
    )
    fig1.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("**Smoking Status**")
    fig2 = px.pie(
        df_f, names="Smoker", hole=0.4,
        color_discrete_sequence=palette_qual,
        template="plotly_white"
    )
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── Procedural Patterns Section ───────────────────────────────────────────────
st.subheader("Procedural Patterns")
palette_seq = px.colors.sequential.Teal

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Age Distribution by Hypertension**")
    fig3 = px.violin(
        df_f, x="HTN_Num", y="Age", color="HTN_Num",
        color_discrete_sequence=["#E4572E", "#1F77B4"],  # red for HTN, blue for no HTN
        category_orders={"HTN_Num": [0, 1]},
        labels={"HTN_Num": "HTN Status"},
        template="plotly_white", box=True, points="all"
    )
    fig3.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("**Surgeries by Area**")
    area_counts = df_f["Residence"].value_counts().reset_index(name="Count").rename(columns={"index": "Area"})
    fig4 = px.treemap(
        area_counts, path=["Area"], values="Count",
        color="Count", color_continuous_scale=palette_seq,
        template="plotly_white"
    )
    fig4.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig4, use_container_width=True)

# ── Prediction Insights Section ────────────────────────────────────────────────
st.subheader("Prediction Insights")
# Prepare model
X = df_f[["HTN_Num"]]
y = df_f["Bleeding_Num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Display three charts: bleeding rate, confusion matrix, ROC
p1, p2, p3 = st.columns(3)

# Bleeding rate by HTN
with p1:
    st.markdown("** Post surgery bleeding rate HTN**")
    hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
    hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig5 = px.bar(
        hr, x="HTN", y="Bleeding_Num",
        labels={"Bleeding_Num": "Bleeding Rate"},
        color="HTN", color_discrete_map={"No HTN": "#1F77B4", "HTN": "#E4572E"},
        template="plotly_white"
    )
    fig5.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig5, use_container_width=True)

# Confusion matrix
with p2:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig6 = px.imshow(
        cm, text_auto=True,
        x=["No Bleed", "Bleed"], y=["No Bleed", "Bleed"],
        color_continuous_scale="Reds", template="plotly_white"
    )
    fig6.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig6, use_container_width=True)

# ROC curve
with p3:
    st.markdown("**ROC Curve**")
    fig7 = px.area(
        x=fpr, y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        title=None, template="plotly_white"
    )
    fig7.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig7.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

