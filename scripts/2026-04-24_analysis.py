import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    outcomes = pd.read_csv("scripts/2026-04-16correlation.csv")
    nodes    = pd.read_csv("scripts/2026-04-16_player_info_with_starter.csv")
    nodes["match_team_id"] = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
    edge_dfs = {k: pd.read_csv(f"scripts/2026-04-13_defensive_network_edge({k}).csv")
                for k in ("average", "min", "product", "sum")}
    return outcomes, nodes, edge_dfs

outcomes, nodes, edge_dfs = load_data()
team_names = nodes.groupby("match_team_id")["team_name"].first()

OUTCOME_COLS    = ["goals_against_real", "shots_against"]
WEIGHT_COLS     = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r",
]
FAULT_METRICS   = ["raw_fault", "valued_fault", "raw_fault_r", "valued_fault_r"]
CONTRIB_METRICS = ["raw_contribution", "raw_contribution_r", "valued_contribution", "valued_contribution_r"]


def gini(x):
    x = np.sort(x[x > 0])
    n = len(x)
    return np.nan if n < 2 else (2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n


def build_match_level(edge_df, metric):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    strength = edge_df.groupby("match_team_id")[metric].sum().rename("strength")

    p1 = edge_df[["match_team_id", "player_1", metric]].rename(columns={"player_1": "player"})
    p2 = edge_df[["match_team_id", "player_2", metric]].rename(columns={"player_2": "player"})
    player_str = pd.concat([p1, p2]).groupby(["match_team_id", "player"])[metric].sum()
    gini_s = player_str.groupby("match_team_id").apply(gini).rename("gini")

    self_col = metric + "_self_inv"
    self_inv = nodes.groupby("match_team_id")[self_col].sum().rename("self_inv")
    shared   = strength.rename("shared_inv")
    ratio    = (self_inv / (self_inv + shared)).rename("self_ratio")

    df = pd.concat([strength, gini_s, self_inv, shared, ratio], axis=1)
    df = df.join(team_names).join(
        outcomes.set_index("match_team_id")[["goals_against_real", "shots_against", "n_tackles"]]
    ).dropna(subset=["strength", "gini"])
    return df.reset_index()


def team_level(df):
    return df.groupby("team_name")[
        ["strength", "gini", "self_inv", "shared_inv", "self_ratio", "goals_against_real", "shots_against"]
    ].mean().reset_index()


def icc_one_way(df, group_col, value_col):
    """ICC(1,1): one-way random effects."""
    grps        = df.groupby(group_col)[value_col]
    n_total     = len(df)
    n_groups    = grps.ngroups
    grand_mean  = df[value_col].mean()
    group_sizes = grps.count()
    group_means = grps.mean()

    SS_b = (group_sizes * (group_means - grand_mean) ** 2).sum()
    SS_w = grps.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    df_b = n_groups - 1
    df_w = n_total - n_groups
    MS_b = SS_b / df_b if df_b > 0 else np.nan
    MS_w = SS_w / df_w if df_w > 0 else np.nan

    k0  = (n_total - (group_sizes ** 2).sum() / n_total) / (n_groups - 1)
    icc = (MS_b - MS_w) / (MS_b + (k0 - 1) * MS_w) if MS_w and MS_b else np.nan
    return round(float(icc), 3), round(float(MS_b), 4), round(float(MS_w), 4)


def residualise(y, x):
    return sm.OLS(y, sm.add_constant(x)).fit().resid


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Defensive Network — Statistical Analysis")

with st.sidebar:
    method  = st.selectbox("Edge weight method", list(edge_dfs))
    metric  = st.selectbox("Primary metric", WEIGHT_COLS)
    outcome = st.selectbox("Outcome variable", OUTCOME_COLS)

df_match = build_match_level(edge_dfs[method], metric)
df_team  = team_level(df_match)

tabs = st.tabs([
    "1 · Correlation & Regression",
    "2 · Quadrant Analysis",
    "3 · Within-team Consistency (ICC)",
    "4 · Clustering",
    "5 · Fault vs Contribution",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Correlation & Regression
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    pred_cols = ["strength", "gini", "self_ratio"]

    st.subheader("Pearson correlations")
    corr_rows = []
    for pc in pred_cols:
        for oc in OUTCOME_COLS:
            sub = df_match[[pc, oc]].dropna()
            r, p = stats.pearsonr(sub[pc], sub[oc])
            corr_rows.append({
                "Predictor": pc, "Outcome": oc,
                "r": round(r, 3), "p": round(p, 4),
                "sig": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
            })
    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(corr_df, use_container_width=True)

    pivot = corr_df.pivot(index="Predictor", columns="Outcome", values="r")
    fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Correlation heatmap (r)")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader(f"OLS regression: {outcome} ~ strength + gini + self_ratio")
    sub_reg = df_match[["strength", "gini", "self_ratio", outcome]].dropna()
    X       = sm.add_constant(sub_reg[["strength", "gini", "self_ratio"]])
    model   = sm.OLS(sub_reg[outcome], X).fit()

    coef_df = pd.DataFrame({
        "coef": model.params, "se": model.bse, "t": model.tvalues, "p": model.pvalues,
    }).reset_index().rename(columns={"index": "term"})
    coef_df = coef_df[coef_df["term"] != "const"]
    coef_df["sig"] = coef_df["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

    st.write(f"R² = {model.rsquared:.3f}  |  adj. R² = {model.rsquared_adj:.3f}  |  n = {len(sub_reg)}")
    st.dataframe(coef_df.round(4), use_container_width=True)

    fig_coef = px.bar(coef_df, x="term", y="coef", error_y="se",
                      title="Regression coefficients (±1 SE)",
                      labels={"term": "Predictor", "coef": "Coefficient"})
    fig_coef.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Scatter plots with OLS trend line")
    for pc in pred_cols:
        sub_sc = df_match[["team_name", pc, outcome]].dropna()
        fig_sc = px.scatter(sub_sc, x=pc, y=outcome, hover_name="team_name",
                            trendline="ols", title=f"{pc} vs {outcome}")
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Quadrant Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    med_str  = df_match["strength"].median()
    med_gini = df_match["gini"].median()

    def assign_quadrant(row):
        s = "High strength" if row["strength"] >= med_str else "Low strength"
        g = "Concentrated"  if row["gini"]     >= med_gini else "Balanced"
        return f"{s} / {g}"

    df_match["quadrant"] = df_match.apply(assign_quadrant, axis=1)
    quad_order = [
        "High strength / Concentrated", "High strength / Balanced",
        "Low strength / Concentrated",  "Low strength / Balanced",
    ]

    st.subheader("Mean outcomes per quadrant")
    quad_agg = df_match.groupby("quadrant")[OUTCOME_COLS + ["n_tackles"]].agg(["mean", "std", "count"])
    st.dataframe(quad_agg.round(3), use_container_width=True)

    for oc in OUTCOME_COLS:
        quad_mean = (df_match.groupby("quadrant")[oc]
                     .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)))
                     .reindex(quad_order).reset_index())
        fig_bar = px.bar(quad_mean, x="quadrant", y="mean", error_y="se",
                         title=f"Mean {oc} by quadrant (±1 SE)",
                         labels={"quadrant": "", "mean": oc})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Violin / box plots by quadrant")
    for oc in OUTCOME_COLS:
        fig_vio = px.violin(
            df_match, x="quadrant", y=oc, box=True, points="all",
            hover_name="team_name", category_orders={"quadrant": quad_order},
            title=f"{oc} distribution by quadrant",
        )
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── Within-strength-bin: controlling for density ──────────────────────────
    st.subheader("Within similar strength: does concentration matter?")
    st.markdown(
        "Teams binned into strength quartiles. Within each bin a trend line shows "
        "how Gini relates to the outcome — answering: *given similar network density, "
        "is it better to be concentrated or balanced?*"
    )
    df_match["strength_bin"] = pd.qcut(
        df_match["strength"], 4,
        labels=["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"],
    )
    fig_facet = px.scatter(
        df_match, x="gini", y=outcome, facet_col="strength_bin",
        trendline="ols", hover_name="team_name",
        title=f"Gini vs {outcome} within strength quartiles",
        labels={"gini": "Gini (concentration)", outcome: outcome},
    )
    st.plotly_chart(fig_facet, use_container_width=True)

    # Partial correlation
    sub_pc = df_match[["gini", "strength", outcome]].dropna()
    gini_resid    = residualise(sub_pc["gini"],    sub_pc["strength"])
    outcome_resid = residualise(sub_pc[outcome],   sub_pc["strength"])
    partial_r, partial_p = stats.pearsonr(gini_resid, outcome_resid)

    col1, col2 = st.columns(2)
    col1.metric("Partial r (gini ↔ outcome, strength partialled out)",
                f"{partial_r:.3f}", f"p = {partial_p:.4f}")

    fig_partial = px.scatter(
        x=gini_resid, y=outcome_resid, trendline="ols",
        labels={"x": "Gini residual (strength removed)",
                "y": f"{outcome} residual (strength removed)"},
        title="Partial correlation: Gini vs outcome controlling for strength",
    )
    st.plotly_chart(fig_partial, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Within-team Consistency (ICC)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Intraclass Correlation Coefficient — ICC(1,1)")
    st.markdown("""
    ICC measures what fraction of total variance is attributable to **between-team** differences.  
    - **ICC ≈ 1** → stable team trait across matches  
    - **ICC ≈ 0** → driven by match context, not team identity  
    - Threshold often used: ICC > 0.5 = moderate, > 0.75 = good stability
    """)

    icc_vars = ["strength", "gini", "self_ratio"]
    icc_rows = []
    for v in icc_vars:
        sub = df_match[["team_name", v]].dropna()
        val, ms_b, ms_w = icc_one_way(sub, "team_name", v)
        icc_rows.append({"Variable": v, "ICC": val, "MS_between": ms_b, "MS_within": ms_w,
                          "Interpretation": "stable trait" if val > 0.5 else "match-driven"})
    icc_table = pd.DataFrame(icc_rows)
    st.dataframe(icc_table, use_container_width=True)

    fig_icc = px.bar(icc_table, x="Variable", y="ICC", color="Interpretation",
                     title="ICC per variable", range_y=[0, 1],
                     color_discrete_map={"stable trait": "#2ca02c", "match-driven": "#d62728"})
    fig_icc.add_hline(y=0.5, line_dash="dash", line_color="grey",
                       annotation_text="0.5 threshold")
    st.plotly_chart(fig_icc, use_container_width=True)

    st.subheader("Match-by-match variation per team")
    var_sel   = st.selectbox("Variable to inspect", icc_vars, key="icc_var")
    top_teams = df_match["team_name"].value_counts().head(16).index.tolist()
    sub_teams = df_match[df_match["team_name"].isin(top_teams)]

    fig_box_team = px.box(
        sub_teams, x="team_name", y=var_sel, points="all",
        title=f"{var_sel} match-by-match variance (top 16 teams by n matches)",
        labels={"team_name": ""},
    )
    fig_box_team.update_xaxes(tickangle=45)
    st.plotly_chart(fig_box_team, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Clustering
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("K-means clustering on defensive profile")
    k = st.slider("Number of clusters (k)", 2, 5, 3)

    cluster_cols = ["strength", "gini", "self_ratio"]
    sub_cl = df_match[["match_team_id", "team_name"] + cluster_cols + OUTCOME_COLS].dropna().copy()

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(sub_cl[cluster_cols])
    sub_cl["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled).astype(str)

    st.subheader("Cluster profiles (mean values)")
    profile = sub_cl.groupby("cluster")[cluster_cols + OUTCOME_COLS].mean().round(3)
    st.dataframe(profile, use_container_width=True)

    fig_cl = px.scatter(sub_cl, x="strength", y="gini", color="cluster",
                         hover_name="team_name", size=outcome,
                         title=f"Clusters in strength × gini space (size = {outcome})")
    st.plotly_chart(fig_cl, use_container_width=True)

    fig_cl2 = px.scatter(sub_cl, x="self_ratio", y="gini", color="cluster",
                          hover_name="team_name", size=outcome,
                          title=f"Clusters in self_ratio × gini space (size = {outcome})")
    st.plotly_chart(fig_cl2, use_container_width=True)

    st.subheader("Outcome distributions by cluster")
    for oc in OUTCOME_COLS:
        fig_vio_cl = px.violin(sub_cl, x="cluster", y=oc, box=True, points="all",
                                hover_name="team_name",
                                title=f"{oc} by cluster")
        st.plotly_chart(fig_vio_cl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Fault vs Contribution
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Fault vs Contribution: which network matters more for defensive success?")
    st.markdown("""
    **Fault** metrics capture where defensive errors occur.  
    **Contribution** metrics capture positive defensive actions.  
    We compare how each network's *Gini* (concentration) and *Strength* correlate with outcomes.
    """)

    method5 = st.selectbox("Edge weight method", list(edge_dfs), key="method5")

    @st.cache_data
    def build_comparison(method):
        rows = []
        for m in FAULT_METRICS + CONTRIB_METRICS:
            try:
                df_m = build_match_level(edge_dfs[method], m)
                for oc in OUTCOME_COLS:
                    sub = df_m[["gini", "strength", oc]].dropna()
                    r_g, p_g = stats.pearsonr(sub["gini"],     sub[oc])
                    r_s, p_s = stats.pearsonr(sub["strength"], sub[oc])
                    rows.append({
                        "metric": m,
                        "type":   "Fault"        if m in FAULT_METRICS else "Contribution",
                        "outcome": oc,
                        "r_gini":     round(r_g, 3), "p_gini":     round(p_g, 4),
                        "r_strength": round(r_s, 3), "p_strength": round(p_s, 4),
                    })
            except Exception:
                pass
        return pd.DataFrame(rows)

    comp_df = build_comparison(method5)
    st.dataframe(comp_df, use_container_width=True)

    # Average r by type
    avg_r = comp_df.groupby(["type", "outcome"])[["r_gini", "r_strength"]].mean().reset_index()

    fig_avg_gini = px.bar(avg_r, x="type", y="r_gini", color="outcome", barmode="group",
                           title="Mean r (Gini ↔ outcome): Fault vs Contribution",
                           labels={"r_gini": "Mean r", "type": "Network type"})
    fig_avg_gini.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_avg_gini, use_container_width=True)

    fig_avg_str = px.bar(avg_r, x="type", y="r_strength", color="outcome", barmode="group",
                          title="Mean r (Strength ↔ outcome): Fault vs Contribution",
                          labels={"r_strength": "Mean r", "type": "Network type"})
    fig_avg_str.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_avg_str, use_container_width=True)

    # Scatter: r_strength vs r_gini per metric
    oc5     = st.selectbox("Outcome", OUTCOME_COLS, key="oc5")
    sub_oc5 = comp_df[comp_df["outcome"] == oc5]
    fig_quad = px.scatter(sub_oc5, x="r_strength", y="r_gini",
                           color="type", text="metric",
                           title=f"Strength r vs Gini r for {oc5} — each point is one metric",
                           labels={"r_strength": "r (strength ↔ outcome)",
                                   "r_gini":     "r (gini ↔ outcome)"})
    fig_quad.update_traces(textposition="top center")
    fig_quad.add_vline(x=0, line_dash="dash", line_color="grey")
    fig_quad.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_quad, use_container_width=True)

    # Head-to-head: pick one fault and one contribution metric and compare
    st.subheader("Head-to-head: pick one fault metric vs one contribution metric")
    col_f, col_c = st.columns(2)
    fault_sel   = col_f.selectbox("Fault metric",        FAULT_METRICS,   key="f_sel")
    contrib_sel = col_c.selectbox("Contribution metric", CONTRIB_METRICS, key="c_sel")

    df_fault   = build_match_level(edge_dfs[method5], fault_sel)
    df_contrib = build_match_level(edge_dfs[method5], contrib_sel)

    for oc in OUTCOME_COLS:
        sub_f = df_fault[["gini",   oc]].dropna().rename(columns={"gini": fault_sel})
        sub_c = df_contrib[["gini", oc]].dropna().rename(columns={"gini": contrib_sel})
        merged = sub_f.join(sub_c[[contrib_sel]], how="inner")
        merged_long = pd.melt(merged, id_vars=[oc],
                               value_vars=[fault_sel, contrib_sel],
                               var_name="metric", value_name="gini")
        fig_h2h = px.scatter(merged_long, x="gini", y=oc, color="metric",
                              trendline="ols",
                              title=f"Gini vs {oc}: {fault_sel} vs {contrib_sel}",
                              labels={"gini": "Gini"})
        st.plotly_chart(fig_h2h, use_container_width=True)
