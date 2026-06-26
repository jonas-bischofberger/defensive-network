import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
nodes    = pd.read_csv("scripts/2026-04-29-player_level_metrics.csv")
edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
            for k in ("average", "min", "product", "sum")}

self_inv_cols     = [c for c in nodes.columns if c.endswith("_self_inv")]
contribution_cols = ["raw_contribution", "valued_contribution", "raw_contribution_r", "valued_contribution_r"]
fault_cols        = ["raw_fault",        "valued_fault",        "raw_fault_r",        "valued_fault_r"]
outcome_cols      = ["goals_against", "shots_against", "xg_against"]

STAGE_ORDER = [
    "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals",
    "Third-place match", "Final",
]

# team-level self-inv: sum across players per match-team, then mean across matches
self_inv_match = nodes.groupby("match_team_id")[self_inv_cols].sum().reset_index()
self_inv_match = self_inv_match.merge(
    outcomes[["match_team_id", "team_name"]], on="match_team_id"
)
self_inv_team = self_inv_match.groupby("team_name")[self_inv_cols].mean()


def best_stage(stages):
    ranked = [s for s in STAGE_ORDER if s in stages.values]
    return ranked[-1] if ranked else stages.iloc[0]


def build_team_df(edge_df, x_col, y_col, size_col):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = (
        edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)
    )
    match_level = edge_df.groupby("match_team_id")[[x_col, y_col]].sum().reset_index()
    match_level = match_level.merge(
        outcomes[["match_team_id", "team_name", "competition_stage"] + outcome_cols],
        on="match_team_id",
    )
    # add self_inv at match-team level
    match_level = match_level.merge(
        self_inv_match[["match_team_id", x_col + "_self_inv", y_col + "_self_inv"]],
        on="match_team_id", how="left",
    )
    match_level["x_total"] = match_level[x_col] + match_level[x_col + "_self_inv"]
    match_level["y_total"] = match_level[y_col] + match_level[y_col + "_self_inv"]

    team_df = (
        match_level.groupby("team_name")
        .agg(
            x=(x_col, "mean"),
            y=(y_col, "mean"),
            x_total=("x_total", "mean"),
            y_total=("y_total", "mean"),
            size=(size_col, "mean"),
            n_matches=("match_team_id", "count"),
            best_stage=("competition_stage", best_stage),
        )
        .reset_index()
    )
    team_df["best_stage"] = pd.Categorical(
        team_df["best_stage"], categories=STAGE_ORDER, ordered=True
    )
    return team_df


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Defensive Style — Team Scatter")

with st.sidebar:
    method     = st.selectbox("Edge weight method", list(edge_dfs))
    x_col      = st.selectbox("X axis (contribution)", contribution_cols, index=0)
    y_col      = st.selectbox("Y axis (fault)",        fault_cols,        index=0)
    size_col   = st.selectbox("Bubble size (outcome)", outcome_cols,      index=0)
    size_scale = st.slider("Bubble size scale", min_value=10, max_value=80, value=40)

team_df = build_team_df(edge_dfs[method], x_col, y_col, size_col)

# ── Plot 1: contribution vs fault, color = best stage ────────────────────────
st.subheader("Network contribution vs fault  |  color = best stage")
fig1 = px.scatter(
    team_df,
    x="x", y="y",
    size="size", size_max=size_scale,
    color="best_stage",
    category_orders={"best_stage": STAGE_ORDER},
    color_discrete_map={
        "Group Stage":       "#adb5bd",
        "Round of 16":       "#4cc9f0",
        "Quarter-finals":    "#4361ee",
        "Semi-finals":       "#f77f00",
        "Third-place match": "#9b2226",
        "Final":             "#2dc653",
    },
    hover_name="team_name",
    hover_data={"x": ":.3f", "y": ":.3f", "size": ":.3f", "n_matches": True, "best_stage": True},
    labels={"x": x_col, "y": y_col, "size": size_col, "best_stage": "Best stage"},
)
fig1.add_vline(x=team_df["x"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig1.add_hline(y=team_df["y"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig1.update_layout(height=600)
st.plotly_chart(fig1, use_container_width=True)

# ── Plot 2: network + self-inv combined ───────────────────────────────────────
st.subheader("Total contribution vs fault (network + self-inv)  |  color = best stage")
fig2 = px.scatter(
    team_df,
    x="x_total", y="y_total",
    size="size", size_max=size_scale,
    color="best_stage",
    category_orders={"best_stage": STAGE_ORDER},
    color_discrete_map={
        "Group Stage":       "#adb5bd",
        "Round of 16":       "#4cc9f0",
        "Quarter-finals":    "#4361ee",
        "Semi-finals":       "#f77f00",
        "Third-place match": "#9b2226",
        "Final":             "#2dc653",
    },
    hover_name="team_name",
    hover_data={"x_total": ":.3f", "y_total": ":.3f", "size": ":.3f", "best_stage": True},
    labels={"x_total": f"{x_col} (network + self-inv)", "y_total": f"{y_col} (network + self-inv)",
            "size": size_col, "best_stage": "Best stage"},
)
fig2.add_vline(x=team_df["x_total"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig2.add_hline(y=team_df["y_total"].median(), line_dash="dash", line_color="grey", opacity=0.5)
fig2.update_layout(height=600)
st.plotly_chart(fig2, use_container_width=True)

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("Team-level values")
st.dataframe(
    team_df[["team_name", "x", "x_total", "y", "y_total", "size", "n_matches", "best_stage"]]
    .rename(columns={"x": f"network_{x_col}", "x_total": f"total_{x_col}",
                     "y": f"network_{y_col}", "y_total": f"total_{y_col}", "size": size_col})
    .sort_values("best_stage", ascending=False)
    .round(3),
    use_container_width=True,
)
