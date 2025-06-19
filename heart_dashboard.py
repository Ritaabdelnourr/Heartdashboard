import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🚑 Heart Surgery Dashboard", layout="wide")

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_clean.csv")
    df["Date"]      = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"]      = df["Date"].dt.year
    df["Age"]       = pd.to_numeric(df["Age"], errors="coerce")
    for col in ["Smoker", "HTN", "Bleeding"]:
        df[col+"_Num"] = (
            df[col].astype(str)
                  .str.strip().str.lower()
                  .map({"yes":1,"no":0})
        )
    df = df.dropna(subset=[
        "Sex","Age","Residence",
        "Smoker_Num","HTN_Num","Bleeding_Num","Year"
    ])
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
yrs         = sorted(df["Year"].unique())
yr_start,yr_end = st.sidebar.slider("Year range", yrs[0], yrs[-1], (yrs[0],yrs[-1]))
res_opts    = sorted(df["Residence"].unique())
sel_res     = st.sidebar.multiselect("Residence", res_opts, default=res_opts)

df_f = df[
    df["Year"].between(yr_start,yr_end) &
    df["Residence"].isin(sel_res)
]

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.title("🚑 Patients Undergoing Open-Heart Surgery")
k1,k2,k3 = st.columns(3)
k1.metric("Total Patients",   f"{len(df_f):,}")
k2.metric("Smokers (%)",      f"{df_f.Smoker_Num.mean()*100:.1f}")
k3.metric("Hypertension (%)", f"{df_f.HTN_Num.mean()*100:.1f}")

# ── 2×2 Grid ───────────────────────────────────────────────────────────────────
c1,c2 = st.columns(2)
with c1:
    # Gender distribution
    fig1 = px.histogram(df_f, x="Sex", color="Sex", title="Gender")
    fig1.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    # Smoking status
    fig2 = px.pie(df_f, names="Smoker", hole=0.4, title="Smoking Status")
    fig2.update_traces(textinfo="percent+label")
    fig2.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

c3,c4 = st.columns(2)
with c3:
    # Bleeding rate by age-bin
    df_f["AgeBin"] = pd.cut(df_f["Age"], bins=[0,30,45,60,75,100],
                             labels=["<30","30–45","45–60","60–75","75+"])
    br = df_f.groupby("AgeBin")["Bleeding_Num"].mean().reset_index()
    fig3 = px.bar(br, x="AgeBin", y="Bleeding_Num",
                  labels={"Bleeding_Num":"Bleeding Rate"},
                  title="Bleeding Rate by Age Group")
    fig3.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    # Patients by residence map (fallback to bar)
    cnt = df_f["Residence"].value_counts().reset_index()
    cnt.columns = ["Residence","Count"]
    try:
        fig4 = px.choropleth(cnt, locations="Residence",
                             locationmode="country names",
                             color="Count", title="Patients by Residence (Map)")
        fig4.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10))
    except Exception:
        fig4 = px.bar(cnt, x="Residence", y="Count",
                      title="Patients by Residence (Bar)")
        fig4.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10),
                           xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

# ── Prediction panel ──────────────────────────────────────────────────────────
st.subheader("🔮 HTN vs Bleeding Risk")

# 1) Bar: Bleeding rate by HTN status
hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
hr["HTN"] = hr["HTN_Num"].map({0:"No HTN",1:"HTN"})
fig5 = px.bar(hr, x="HTN", y="Bleeding_Num",
              labels={"Bleeding_Num":"Bleeding Rate"},
              title="Bleeding Rate: HTN vs No-HTN")
fig5.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), yaxis_tickformat=".0%")
# 2) ROC Curve
X = df_f[["HTN_Num"]]; y = df_f["Bleeding_Num"]
X_tr,X_te,y_tr,y_te = train_test_split(X,y,stratify=y,random_state=42)
model = LogisticRegression(solver="liblinear").fit(X_tr,y_tr)
y_proba = model.predict_proba(X_te)[:,1]
fpr,tpr,_ = roc_curve(y_te,y_proba)

fig6 = px.area(x=fpr, y=tpr,
               title="ROC Curve",
               labels={"x":"False Positive Rate","y":"True Positive Rate"})
fig6.add_shape(type="line", line=dict(dash="dash"), x0=0,x1=1,y0=0,y1=1)
fig6.update_layout(height=260, margin=dict(t=30,b=10,l=10,r=10), showlegend=False)

p1,p2 = st.columns(2)
p1.plotly_chart(fig5, use_container_width=True)
p2.plotly_chart(fig6, use_container_width=True)
