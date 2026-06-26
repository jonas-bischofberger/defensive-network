import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
nodes = pd.read_csv("scripts/2026-04-29-player_level_metrics.csv")
edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
            for k in ("average", "min", "product", "sum")}

nodes["match_team_id"] = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
team_names = nodes.groupby("match_team_id")["team_name"].first()

weight_cols = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r",
]

STAGE_ORDER = {
    "Group Stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "3rd Place Final": 4,
    "Final": 5,
}
STAGE_LABEL = {
    "Group Stage": "Group Stage",
    "Round of 16": "Round of 16",
    "Quarter-finals": "Quarter-finals",
    "Semi-finals": "Semi-finals",
    "3rd Place Final": "Semi-finals",
    "Final": "Final",
}
STAGE_PALETTE = {
    "Group Stage": "#a8d8ea",
    "Round of 16": "#78c1e0",
    "Quarter-finals": "#f4a261",
    "Semi-finals": "#e76f51",
    "Final": "#b5179e",
}
STAGE_CATEGORY_ORDER = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]


def build_match_level(edge_df, metric):
    edge_df = edge_df.copy()
    edge_df["match_team_id"] = edge_df["match_id"].astype(str) + "_" + edge_df["defending_team"].astype(str)

    shared = edge_df.groupby("match_team_id")[metric].sum().rename("shared_inv")

    self_col = metric + "_self_inv"
    self_inv = nodes.groupby("match_team_id")[self_col].sum().rename("self_inv")
    ratio    = (self_inv / (self_inv + shared)).rename("self_ratio")

    df = pd.concat([self_inv, shared, ratio], axis=1)
    df = df.join(team_names).join(
        outcomes.set_index("match_team_id")[["goals_against", "shots_against", "xg_against", "competition_stage"]]
    ).dropna(subset=["self_inv", "shared_inv"])
    return df.reset_index()


def team_level(df):
    agg = df.groupby("team_name")[["self_inv", "shared_inv", "self_ratio",
                                    "goals_against", "shots_against", "xg_against"]].mean().reset_index()
    furthest = (
        df.groupby("team_name")["competition_stage"]
        .apply(lambda s: max(s, key=lambda x: STAGE_ORDER.get(x, 0)))
        .map(STAGE_LABEL)
        .rename("furthest_stage")
        .reset_index()
    )
    return agg.merge(furthest, on="team_name")


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Self vs Shared Defensive Networks")

with st.sidebar:
    method = st.selectbox("Edge weight method", list(edge_dfs))
    metric = st.selectbox("Metric", weight_cols)
    y_metric = st.selectbox("Y-axis (scatter)", ["goals_against", "shots_against", "xg_against"])

df_match = build_match_level(edge_dfs[method], metric)
df_team  = team_level(df_match)
df_team_sorted = df_team.sort_values("self_ratio", ascending=False).reset_index(drop=True)

tab1, tab2 = st.tabs(["Self vs Shared", "Data"])

with tab1:
    fig_bar = px.bar(
        df_team_sorted, x="team_name", y=["self_inv", "shared_inv"],
        title="Self vs Shared Involvement per Team",
        labels={"value": "Total Involvement", "team_name": "Team"},
        barmode="stack",
    )
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_scatter = px.scatter(
        df_team_sorted, x="self_ratio", y=y_metric,
        text="team_name", color="furthest_stage",
        color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
        title="Self-involvement ratio vs Defensive Performance",
        labels={"self_ratio": "Self ratio (self / total)", "furthest_stage": "Furthest Stage"},
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Team-match level")
    st.dataframe(df_match)
    st.subheader("Team level")
    st.dataframe(df_team_sorted)
