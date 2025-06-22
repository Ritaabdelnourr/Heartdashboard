# ─────────  2 × 2 GRID  ───────────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="small")
r2c1, r2c2 = st.columns(2, gap="small")

with r1c1:  # Gender
    sex_counts = (df_f["Sex"].value_counts()
                  .rename_axis("Sex").reset_index(name="Count"))
    fig = px.bar(sex_counts, x="Sex", y="Count",
                 color="Sex", color_discrete_map=SEX_COLORS,
                 template="plotly_white")
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r1c2:  # Smoking status
    fig = px.pie(df_f, names="Smoker", hole=0.35, template="plotly_white")
    fig.update_traces(marker=dict(colors=[DARK, LIGHT]),
                      textinfo="percent+label")
    fig.update_layout(height=H, margin=M)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r2c1:  # Age distribution
    fig = px.histogram(df_f, x="Age", nbins=20, template="plotly_white")
    fig.update_traces(marker_color=DARK)
    fig.update_layout(height=H, margin=M, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

with r2c2:  # Bleeding rate by Smoking × HTN
    combo = (df_f.groupby(["Smoker_Num", "HTN_Num"])["Bleeding_Num"]
             .mean().reset_index())
    combo["Smoker"] = combo["Smoker_Num"].map({0: "No", 1: "Yes"})
    combo["HTN"]    = combo["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig = px.bar(combo, x="Smoker", y="Bleeding_Num", color="HTN",
                 barmode="group", template="plotly_white",
                 labels={"Bleeding_Num": "Bleeding Rate", "Smoker": "Smoker"},
                 color_discrete_map=HTN_COLORS)
    fig.update_layout(height=H, margin=M, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config=CFG)

# ─────────  EXPANDER  ─────────────────────────────────────────────────
with st.expander("🔮 Bleeding risk by HTN (detail)", expanded=False):
    hr = df_f.groupby("HTN_Num")["Bleeding_Num"].mean().reset_index()
    hr["HTN"] = hr["HTN_Num"].map({0: "No HTN", 1: "HTN"})
    fig = px.bar(hr, x="HTN", y="Bleeding_Num",
                 labels={"Bleeding_Num": "Bleeding Rate"},
                 template="plotly_white",
                 color="HTN", color_discrete_map=HTN_COLORS)
    fig.update_layout(height=H-10, margin=M, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config=CFG)
