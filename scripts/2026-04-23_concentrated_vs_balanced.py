import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

outcomes = pd.read_csv("scripts/2026-04-24_match_level_metrics.csv")
nodes = pd.read_csv("scripts/2026-04-29-player_level_metrics.csv")
edge_dfs = {k: pd.read_csv(f"scripts/2026-04-28_defensive_network_edge({k}).csv")
            for k in ("average", "min", "product", "sum")}

# nodes["match_team_id"] = nodes["match_id"].astype(str) + "_" + nodes["defending_team"].astype(str)
team_names = nodes.groupby("match_team_id")["team_name"].first()

weight_cols = [
    "raw_involvement", "raw_fault", "raw_contribution",
    "valued_involvement", "valued_contribution", "valued_fault",
    "raw_responsibility", "raw_fault_r", "raw_contribution_r",
    "valued_responsibility", "valued_contribution_r", "valued_fault_r",
]


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

    df = pd.concat([strength, gini_s], axis=1)
    df = df.join(team_names).join(
        outcomes.set_index("match_team_id")[["goals_against", "shots_against", "xg_against", "n_tackles", "competition_stage"]]
    ).dropna(subset=["strength", "gini"])
    return df.reset_index()


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


def team_level(df):
    agg = df.groupby("team_name")[
        ["strength", "gini", "goals_against", "shots_against", "xg_against"]
    ].mean().reset_index()

    furthest = (
        df.groupby("team_name")["competition_stage"]
        .apply(lambda s: max(s, key=lambda x: STAGE_ORDER.get(x, 0)))
        .map(STAGE_LABEL)
        .rename("furthest_stage")
        .reset_index()
    )
    return agg.merge(furthest, on="team_name")


def quadrant_scatter(df, title, color_col="shots_against", size_col="goals_against", label_col="team_name"):
    mx, mg = df["strength"].median(), df["gini"].median()
    fig = px.scatter(
        df, x="strength", y="gini",
        color=color_col, size=size_col,
        hover_name=label_col, hover_data=["shots_against", "goals_against", "xg_against"],
        color_continuous_scale="RdYlGn_r", title=title,
        labels={"strength": "Total Strength", "gini": "Gini (concentration)"},
    )
    fig.add_vline(x=mx, line_dash="dash", line_color="grey")
    fig.add_hline(y=mg, line_dash="dash", line_color="grey")
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


STAGE_PALETTE = {
    "Group Stage": "#a8d8ea",
    "Round of 16": "#78c1e0",
    "Quarter-finals": "#f4a261",
    "Semi-finals": "#e76f51",
    "Final": "#b5179e",
}
STAGE_CATEGORY_ORDER = ["Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]


def quadrant_scatter_team(df, title, size_col="goals_against"):
    mx, mg = df["strength"].median(), df["gini"].median()
    fig = px.scatter(
        df, x="strength", y="gini",
        color="furthest_stage", size=size_col,
        color_discrete_map=STAGE_PALETTE,
        category_orders={"furthest_stage": STAGE_CATEGORY_ORDER},
        hover_name="team_name", hover_data=["shots_against", "goals_against", "xg_against", "furthest_stage"],
        title=title,
        labels={"strength": "Total Strength", "gini": "Gini (concentration)", "furthest_stage": "Furthest Stage"},
    )
    fig.add_vline(x=mx, line_dash="dash", line_color="grey")
    fig.add_hline(y=mg, line_dash="dash", line_color="grey")
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].max(),
                       text="High strength<br>Concentrated", showarrow=False, font_size=10)
    fig.add_annotation(x=df["strength"].max(), y=df["gini"].min(),
                       text="High strength<br>Balanced", showarrow=False, font_size=10)
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Concentrated vs Balanced Defensive Networks")

with st.sidebar:
    method = st.selectbox("Edge weight method", list(edge_dfs))
    metric = st.selectbox("Metric", weight_cols)
    size_metric = st.selectbox("Size (team-level plot)", ["goals_against", "shots_against", "xg_against"])

df_match = build_match_level(edge_dfs[method], metric)
df_team = team_level(df_match)

tab1, tab2 = st.tabs(["Concentrated vs Balanced", "Data"])

with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(quadrant_scatter(df_match, "Team-match level"), use_container_width=True)
    c2.plotly_chart(quadrant_scatter_team(df_team, "Team level (by stage)", size_col=size_metric), use_container_width=True)

with tab2:
    st.subheader("Team-match level")
    st.dataframe(df_match)
    st.subheader("Team level")
    st.dataframe(df_team)
